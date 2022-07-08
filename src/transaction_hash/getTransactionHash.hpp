#ifndef __GET_TRANSACTION_HASH__HPP__
#define __GET_TRANSACTION_HASH__HPP__
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <gmp.h>
#include <gmpxx.h>
#include <arpa/inet.h>
#include "scalar.hpp"
#include "rlp.hpp"
#include "ecrecover/ecrecover.hpp"

/* TODO: BEGIN remove */
class Context_ {
public:
    mpz_class r;
    mpz_class s;
    mpz_class v;
};

/* END remove */
using namespace std;

inline string getTransactionHash(Context_ &ctx, string &to, uint64_t value, uint64_t nonce, uint64_t gasLimit,
                                 uint64_t gasPrice, string &data, uint64_t chainId)
{
    string raw;

    encodeUInt64(raw, nonce);
    encodeUInt64(raw, gasPrice);
    encodeUInt64(raw, gasLimit);
    encodeLen(raw, getHexValueLen(to));
    if (!encodeHexValue(raw, to)) {
        cout << "ERROR encoding to" << endl;
    }
    encodeUInt64(raw, value);
    encodeLen(raw, getHexValueLen(data));
    if (!encodeHexValue(raw, data)) {
        cout << "ERROR encoding data" << endl;
    }

    uint64_t recoveryParam;
    uint64_t ctx_v = ctx.v.get_ui();

    if (ctx.v == 0 || ctx.v == 1) {
        recoveryParam = ctx_v;
    } else {
        recoveryParam = 1 - (ctx_v % 2);
    }
    uint64_t _v = recoveryParam + 27;

    if (chainId) {
        _v += chainId * 2 + 8;
    }

    encodeUInt64(raw, _v);
    string r = ctx.r.get_str(16);
    encodeLen(raw, getHexValueLen(r));
    if (!encodeHexValue(raw, r)) {
        cout << "ERROR encoding r" << endl;
    }

    string s = ctx.s.get_str(16);
    encodeLen(raw, getHexValueLen(s));
    if (!encodeHexValue(raw, s)) {
        cout << "ERROR encoding s" << endl;
    }

    string res;
    encodeLen(res, raw.length(), true);
    res += raw;

    return keccak256((const uint8_t *)(res.c_str()), res.length());
}
#endif