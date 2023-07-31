#include "tx_data.hpp"
#include "zklog.hpp"
#include "main_sm/fork_5/main/full_tracer.hpp"

using namespace std;

namespace fork_5
{

void TXData::print (void)
{
    zklog.info(
        "TXData::print() nonce=" + to_string(nonce) +
        " gasPrice=" + gasPrice.get_str(10) +
        " gasLimit=" + to_string(gasLimit) +
        " to=0x" + to.get_str(16) +
        " value=" + value.get_str(10) +
        " data=0x" + ba2string(data) +
        " chainId=" + to_string(chainId) +
        " r=0x" + r.get_str(16) +
        " s=0x" + s.get_str(16) +
        " v=" + to_string(v) +
        " gasPercentage=" + to_string(gasPercentage)
        );
}

string TXData::txHash (void)
{
    if (!bTxHashGenerated)
    {
        string toString = NormalizeTo0xNFormat(to.get_str(16), 40);
        uint64_t v2;
        if (chainId == 0)
        {
            v2 = v;
        }
        else
        {
            v2 = v - 27 + chainId * 2 + 35;
        }
        getTransactionHash(toString, value, nonce, gasLimit, gasPrice, data, r, s, v2, txHashResult, txHashRlp);
    }
    return txHashResult;
}

string TXData::signHash (void)
{
    string signHash = keccak256((const uint8_t *)(rlpData.c_str()), rlpData.length());
    return signHash;
}

}