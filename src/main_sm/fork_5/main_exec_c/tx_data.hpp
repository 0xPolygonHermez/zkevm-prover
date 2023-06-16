#ifndef TX_DATA_HPP_fork_5
#define TX_DATA_HPP_fork_5

#include <string>
#include <vector>
#include <gmpxx.h>
#include "scalar.hpp"
#include "zklog.hpp"

using namespace std;

namespace fork_5
{

class TXData
{
public:

    // Batch L2 data is made of an RLP section followed by a set of concatenated binary values
    // Batch L2 data = RLPList(nonde, gasPrice, gasLimit, to, value, data, chainId, "", "") + r(32) + s(32) + v(1) + gasPercentage(1)

    // Data obtained from batch L2 data RLP section
    uint64_t nonce; // 64 bits max
    mpz_class gasPrice; // 256 bits max
    uint64_t gasLimit; // 64 bits max
    mpz_class to; // 160 bits max
    mpz_class value; // 256 bits max
    string data; // Max batch L2 data length limits this field
    uint64_t chainId; // 64 bits max

    // Data obtained from batch  L2 data raw section (concatenated to RLP section)
    mpz_class r; // 256 bits max
    mpz_class s; // 256 bits max
    uint8_t v;
    uint8_t gasPercentage;

    // Print contents, for debugging purposes
    void print (void)
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
};

}

#endif