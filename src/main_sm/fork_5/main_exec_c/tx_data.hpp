#ifndef TX_DATA_HPP_fork_5
#define TX_DATA_HPP_fork_5

#include <string>
#include <vector>
#include <gmpxx.h>
#include "scalar.hpp"
#include "zklog.hpp"
#include "ecrecover.hpp"

using namespace std;

namespace fork_5
{

class TXData
{
public:

    // Batch L2 data is made of an RLP section followed by a set of concatenated binary values
    // Batch L2 data = RLPList(nonde, gasPrice, gasLimit, to, value, data, chainId, "", "") + r(32) + s(32) + v(1) + gasPercentage(1)
    string rlpData; // Original RLP data, as present in batch L2 data

    // Data obtained from batch L2 data RLP section
    uint64_t nonce; // 64 bits max
    mpz_class gasPrice; // 256 bits max
    uint64_t gasLimit; // 64 bits max
    mpz_class to; // 160 bits max
    mpz_class value; // 256 bits max
    string data; // Max batch L2 data length limits this field
    uint64_t chainId; // 64 bits max

    // Data obtained from batch L2 data raw section (concatenated to RLP section)
    mpz_class r; // 256 bits max
    mpz_class s; // 256 bits max
    uint8_t v;
    uint8_t gasPercentage;

    // Gas related data
    mpz_class effectiveGasPrice;
    mpz_class gas;
    mpz_class fee;

    // Data used to build the TX hash
    bool bTxHashGenerated;
    string txHashRlp;
    string txHashResult;

    // Data obtained from call to ECRecover
    ECRecoverResult ecRecoverResult;
    mpz_class fromPublicKey;

    TXData() : bTxHashGenerated(false) {};

    // Print contents, for debugging purposes
    void print (void);

    // Gets tx hash
    string txHash (void);

    // Get signature hash, to be used to validate the signature and get the from public key
    string signHash (void);
};

}

#endif