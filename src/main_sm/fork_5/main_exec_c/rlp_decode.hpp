#ifndef RLP_DECODE_HPP_fork_5
#define RLP_DECODE_HPP_fork_5

#include <string>
#include <vector>
#include <gmpxx.h>
#include "zkresult.hpp"
#include "main_sm/fork_5/main_exec_c/rlp_data.hpp"
#include <stdint.h>

using namespace std;

namespace fork_5
{

// Convert a byte array string to an integer
zkresult RLPStringToU64 (const string &input, uint64_t &output);
zkresult RLPStringToU160 (const string &input, mpz_class &output);
zkresult RLPStringToU256 (const string &input, mpz_class &output);

// Decode an RLP length
zkresult RLPDecodeLength (const string &input, uint64_t &offset, uint64_t &dataLength, RLPType &type);

// Decode an RLP input
zkresult RLPDecode (const string &input, RLPType rlpType, vector<RLPData> (&output), uint64_t &consumedBytes);

}

#endif