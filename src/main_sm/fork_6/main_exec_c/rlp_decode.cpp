#include "rlp_decode.hpp"
#include "scalar.hpp"
#include "zklog.hpp"
#include "utils.hpp"

//#define RLP_LOGS

namespace fork_6
{

zkresult RLPStringToU64 (const string &input, uint64_t &output)
{
    // Reset output
    output = 0;

    // Check that input is not too big    
    if (input.size() > 8)
    {
        zklog.error("RLPStringToU64() found input string too big size=" + to_string(input.size()));
        return ZKR_UNSPECIFIED;
    }

    // Accumulate bytes into result
    for (uint64_t i=0; i<input.size(); i++)
    {
        output = 256*output + (uint8_t)input[i];
    }

#ifdef RLP_LOGS
    zklog.info("RLPStringToU64() returns " + to_string(output));
#endif

    return ZKR_SUCCESS;
}

zkresult RLPStringToU160 (const string &input, mpz_class &output)
{
    // Reset output
    output = 0;

    // Check that input is not too big    
    if (input.size() > 20)
    {
        zklog.error("RLPStringToU160() found input string too big size=" + to_string(input.size()));
        return ZKR_UNSPECIFIED;
    }

    // Accumulate bytes into result
    for (uint64_t i=0; i<input.size(); i++)
    {
        output = 256*output + (uint8_t)input[i];
    }

#ifdef RLP_LOGS
    zklog.info("RLPStringToU160() returns " + output.get_str());
#endif

    return ZKR_SUCCESS;
}

zkresult RLPStringToU256 (const string &input, mpz_class &output)
{
    // Reset output
    output = 0;

    // Check that input is not too big    
    if (input.size() > 32)
    {
        zklog.error("RLPStringToU256() found input string too big size=" + to_string(input.size()));
        return ZKR_UNSPECIFIED;
    }

    // Accumulate bytes into result
    for (uint64_t i=0; i<input.size(); i++)
    {
        output = 256*output + (uint8_t)input[i];
    }

#ifdef RLP_LOGS
    zklog.info("RLPStringToU256() returns " + output.get_str());
#endif

    return ZKR_SUCCESS;
}

zkresult RLPDecodeLength (const string &input, uint64_t &offset, uint64_t &dataLen, RLPType &type)
{
    zkresult result;

    if (input.size() == 0)
    {
        zklog.error("RLPDecodeLength() called with an empty input");
        return ZKR_UNSPECIFIED;
    }
    uint64_t prefix = (uint8_t)input[0];
#ifdef RLP_LOGS
    zklog.info("RLPDecodeLength() prefix=" + to_string(prefix) + "=0x" + byte2string(prefix) + " input.size=" + to_string(input.size()));
#endif
    if (prefix <= 0x7f)
    {
        offset = 0;
        dataLen = 1;
        type = rlpTypeString;
        return ZKR_SUCCESS;
    }
    if (prefix <= 0xb7)
    {
        offset = 1;
        dataLen = prefix - 0x80;
        if (input.size() > dataLen)
        {
            type = rlpTypeString;
            return ZKR_SUCCESS;
        }
    }
    if (prefix <= 0xbf)
    {
        uint64_t lenOfStrLen = prefix - 0xb7;
        if (input.size() > lenOfStrLen)
        {
            offset = 1 + lenOfStrLen;
            result = RLPStringToU64(input.substr(1, lenOfStrLen), dataLen);
            if (result != ZKR_SUCCESS)
            {
                zklog.error("RLPDecodeLength() failed calling RLPStringToU64() result=" + zkresult2string(result));
                return ZKR_UNSPECIFIED;
            }
            if (input.size() > (lenOfStrLen + dataLen))
            {
                type = rlpTypeString;
                return ZKR_SUCCESS;
            }
        }
    }
    if (prefix <= 0xf7)
    {
        offset = 1;
        dataLen = prefix - 0xc0;
        if (input.size() > dataLen)
        {
            type = rlpTypeList;
            return ZKR_SUCCESS;
        }
    }
    if (prefix <= 0xff)
    {
        uint64_t lenOfListLen = prefix - 0xf7;
        if (input.size() > lenOfListLen)
        {
            offset = 1 + lenOfListLen;
            result = RLPStringToU64(input.substr(1, lenOfListLen), dataLen);
            if (result != ZKR_SUCCESS)
            {
                zklog.error("RLPDecodeLength() failed calling RLPStringToU64() result=" + zkresult2string(result));
                return ZKR_UNSPECIFIED;
            }
            if (input.size() > (lenOfListLen + dataLen))
            {
                type = rlpTypeList;
                return ZKR_SUCCESS;
            }
        }
    }
    zklog.error("RLPDecodeLength() invalid input");
    offset = 0;
    dataLen = 0;
    type = rlpTypeUnknown;
    return ZKR_UNSPECIFIED;
}

zkresult RLPDecode(const string &input, RLPType rlpType, vector<RLPData> (&output), uint64_t &consumedBytes)
{
    if (input.size() == 0)
    {
        return ZKR_SUCCESS;
    }

    consumedBytes = 0;
    uint64_t offset;
    uint64_t dataLength;
    RLPType type;
    zkresult result;
    result = RLPDecodeLength(input, offset, dataLength, type);
    if (result != ZKR_SUCCESS)
    {
        zklog.error("RLPDecode() failed calling RLPDecodeLength() result=" + zkresult2string(result));
        return result;
    }
    switch (type)
    {
        case rlpTypeString:
        {
            if (rlpType == rlpTypeList)
            {
                zklog.error("RLPDecode() found a string but wanted a list");
                return ZKR_UNSPECIFIED;
            }

            RLPData rlpData;
            rlpData.type = rlpTypeString;
            rlpData.data = input.substr(offset, dataLength);
            rlpData.dataWithLength = input.substr(0, offset + dataLength);
            output.push_back(rlpData);
#ifdef RLP_LOGS
            zklog.info("RLPDecode() found string offset=" + to_string(offset) + " dataLength=" + to_string(dataLength) + " data=" + ba2string(rlpData.data));
#endif
            consumedBytes += dataLength + 1;
            break;
        }
        case rlpTypeList:
        {
            if (rlpType == rlpTypeString)
            {
                zklog.error("RLPDecode() found a list but wanted a string");
                return ZKR_UNSPECIFIED;
            }
            RLPData rlpData;
            rlpData.type = rlpTypeList;
            rlpData.data = input.substr(offset, dataLength);
            rlpData.dataWithLength = input.substr(0, offset + dataLength);
            output.push_back(rlpData);
#ifdef RLP_LOGS
            zklog.info("RLPDecode() found list offset=" + to_string(offset) + " dataLength=" + to_string(dataLength));
#endif
            uint64_t aux;
            result = RLPDecode(input.substr(offset, dataLength), rlpTypeUnknown, output[output.size()-1].rlpData, aux);
            if (result != ZKR_SUCCESS)
            {
                zklog.error("RLPDecode() failed calling RLPDecode(list) result=" + zkresult2string(result));
                return result;
            }
            consumedBytes += dataLength + 1;
            break;
        }
        default:
        {
            zklog.error("RLPDecode() called RLPDecodeLength() and got invalid type=" + to_string(type));
            exitProcess();
        }
    }

    // If requested a specific type, simply return
    if (rlpType != rlpTypeUnknown)
    {
        return ZKR_SUCCESS;
    }

    // Otherwise, continue decoding
    result = RLPDecode(input.substr(offset + dataLength), rlpTypeUnknown, output, consumedBytes);
    consumedBytes += dataLength;
    return result;
}

}