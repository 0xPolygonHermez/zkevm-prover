#ifndef BATCH_DECODE_HPP_fork_5
#define BATCH_DECODE_HPP_fork_5

#include <string>
#include <vector>
#include "zklog.hpp"
#include "zkresult.hpp"
#include "utils.hpp"

using namespace std;

namespace fork_5
{

class TXData
{
public:
    uint64_t nonce; // 64 bits max
    mpz_class gasPrice; // 256 bits max
    uint64_t gasLimit; // 64 bits max
    mpz_class to; // 160 bits max
    mpz_class value; // 256 bits max
    string data; // Max batch L2 data length limits this field
    uint64_t chainId; // 64 bits max

    mpz_class r; // 256 bits max
    mpz_class s; // 256 bits max
    uint8_t v;
    uint8_t gasPercentage;
};

class BatchData
{
public:
    vector<TXData> tx;
};

// Decode a batch L2 data buffer
zkresult BatchDecode(const string &input, BatchData (&output));

}

#endif