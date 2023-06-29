#include "batch_decode.hpp"
#include "scalar.hpp"
#include "rlp_decode.hpp"

namespace fork_5
{

zkresult BatchDecode(const string &input, BatchData (&batchData))
{
    zkresult result;
    uint64_t totalConsumedBytes = 0;
    while (totalConsumedBytes < input.size())
    {
        uint64_t consumedBytes;
        vector<RLPData> rlpData;
        result = RLPDecode(input.substr(totalConsumedBytes), rlpTypeList, rlpData, consumedBytes); // TODO: avoid string replicas, using [uint8_t *, size] instead
        if (result != ZKR_SUCCESS)
        {
            zklog.error("BatchDecode() failed calling RLPDecode() result=" + zkresult2string(result));
            return result;
        }
        totalConsumedBytes += consumedBytes;
        if (input.size() < totalConsumedBytes + 66)
        {
            zklog.error("BatchDecode() failed input too short");
            return ZKR_UNSPECIFIED;
        }

        // Check that we got a single list with 9 elements
        if (rlpData.size() != 1)
        {
            zklog.error("BatchDecode() called RLPDecode() but found rlpData.size=" + to_string(rlpData.size()) + " different from 1");
            return ZKR_UNSPECIFIED;
        }
        if (rlpData[0].type != rlpTypeList)
        {
            zklog.error("BatchDecode() called RLPDecode() but found rlpData[0].type=" + to_string(rlpData[0].type) + " different from list");
            return ZKR_UNSPECIFIED;
        }
        if (rlpData[0].rlpData.size() != 9)
        {
            zklog.error("BatchDecode() called RLPDecode() but found rlpData[0].rlpData.size=" + to_string(rlpData[0].rlpData.size()) + " different from 1");
            return ZKR_UNSPECIFIED;
        }
        
        // TX data
        TXData txData;

        // RLP data
        txData.rlpData = rlpData[0].dataWithLength;

        // Nonce
        if (rlpData[0].rlpData[0].type != rlpTypeString)
        {
            zklog.error("BatchDecode() called RLPDecode() but found unexpected type");
            return ZKR_UNSPECIFIED;
        }
        result = RLPStringToU64(rlpData[0].rlpData[0].data, txData.nonce);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("BatchDecode() failed calling RLPStringToU64() result=" + zkresult2string(result));
            return result;
        }

        // Gas price
        if (rlpData[0].rlpData[1].type != rlpTypeString)
        {
            zklog.error("BatchDecode() called RLPDecode() but found unexpected type");
            return ZKR_UNSPECIFIED;
        }
        result = RLPStringToU256(rlpData[0].rlpData[1].data, txData.gasPrice);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("BatchDecode() failed calling RLPStringToU256() result=" + zkresult2string(result));
            return result;
        }

        // Gas limit
        if (rlpData[0].rlpData[2].type != rlpTypeString)
        {
            zklog.error("BatchDecode() called RLPDecode() but found unexpected type");
            return ZKR_UNSPECIFIED;
        }
        result = RLPStringToU64(rlpData[0].rlpData[2].data, txData.gasLimit);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("BatchDecode() failed calling RLPStringToU64() result=" + zkresult2string(result));
            return result;
        }

        // To
        if (rlpData[0].rlpData[3].type != rlpTypeString)
        {
            zklog.error("BatchDecode() called RLPDecode() but found unexpected type");
            return ZKR_UNSPECIFIED;
        }
        result = RLPStringToU160(rlpData[0].rlpData[3].data, txData.to);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("BatchDecode() failed calling RLPStringToU160() result=" + zkresult2string(result));
            return result;
        }

        // Value
        if (rlpData[0].rlpData[4].type != rlpTypeString)
        {
            zklog.error("BatchDecode() called RLPDecode() but found unexpected type");
            return ZKR_UNSPECIFIED;
        }
        result = RLPStringToU256(rlpData[0].rlpData[4].data, txData.value);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("BatchDecode() failed calling RLPStringToU256() result=" + zkresult2string(result));
            return result;
        }

        // Data
        if (rlpData[0].rlpData[5].type != rlpTypeString)
        {
            zklog.error("BatchDecode() called RLPDecode() but found unexpected type");
            return ZKR_UNSPECIFIED;
        }
        txData.data = "0x" + rlpData[0].rlpData[5].data;

        // Chain ID
        if (rlpData[0].rlpData[6].type != rlpTypeString)
        {
            zklog.error("BatchDecode() called RLPDecode() but found unexpected type");
            return ZKR_UNSPECIFIED;
        }
        result = RLPStringToU64(rlpData[0].rlpData[6].data, txData.chainId);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("BatchDecode() failed calling RLPStringToU64() result=" + zkresult2string(result));
            return result;
        }

        // Empty string
        if (rlpData[0].rlpData[7].type != rlpTypeString)
        {
            zklog.error("BatchDecode() called RLPDecode() but found unexpected type");
            return ZKR_UNSPECIFIED;
        }
        if (rlpData[0].rlpData[7].data.size() != 0)
        {
            zklog.error("BatchDecode() called RLPDecode() but found to size too big = " + to_string(rlpData[0].rlpData[3].data.size()));
            return ZKR_UNSPECIFIED;
        }

        // Empty string
        if (rlpData[0].rlpData[8].type != rlpTypeString)
        {
            zklog.error("BatchDecode() called RLPDecode() but found unexpected type");
            return ZKR_UNSPECIFIED;
        }
        if (rlpData[0].rlpData[8].data.size() != 0)
        {
            zklog.error("BatchDecode() called RLPDecode() but found to size too big = " + to_string(rlpData[0].rlpData[3].data.size()));
            return ZKR_UNSPECIFIED;
        }

        txData.r.set_str(ba2string(input.substr(totalConsumedBytes, 32)), 16);
        txData.s.set_str(ba2string(input.substr(totalConsumedBytes + 32, 32)), 16);
        txData.v = (uint8_t)input[totalConsumedBytes + 64];
        txData.gasPercentage = (uint8_t)input[totalConsumedBytes + 65];
        batchData.tx.push_back(txData);
        totalConsumedBytes += 66;
    }
    return ZKR_SUCCESS;
}

}