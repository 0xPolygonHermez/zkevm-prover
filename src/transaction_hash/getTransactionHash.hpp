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

/*
// Legacy Transaction Fields
const transactionFields = [
    { name: "nonce",    maxLength: 32, numeric: true },
    { name: "gasPrice", maxLength: 32, numeric: true },
    { name: "gasLimit", maxLength: 32, numeric: true },
    { name: "to",          length: 20 },
    { name: "value",    maxLength: 32, numeric: true },
    { name: "data" },
];
*/
inline int codingUInt64(string &data, uint64_t value, uint8_t codingBase = 0)
{
    unsigned char blen[9] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    int index;
    for (index = 0; index < 8 && value; ++index) {
        blen[7-index] = value & 0xFF;
        value = value >> 8;
    }
    const char *pdata = (char *)blen;
    pdata += (8 - index);
    if (codingBase) {
        codingBase += index;
        data += codingBase;
    }
    data.append(pdata, index);
    return index;
}

inline void encodeLen(string &data, uint32_t len, bool composed = false)
{
    unsigned char encodeType = (composed ? 0xc0 : 0x80);
    if (len <= 55) {
        encodeType += len;
        data.push_back(encodeType);
        return;
    }
    int bytes = codingUInt64(data, len, encodeType + 55);
}

inline void encodeUInt64(string &data, uint64_t value)
{
    if (value && value <= 127) {
        data.push_back(value);
        return;
    }
    codingUInt64(data, value, 0x80);
}

const int ASCIIHexToInt[] =
{
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
        0,     1,     2,    3,      4,     5,     6,     7,     8,     9, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000,    10,    11,    12,    13,    14,    15, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000,    10,    11,    12,    13,    14,    15, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,

    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
};

inline bool encodeHexValue(string &data, const string &hex)
{
    int len = hex.length();
    const char *phex = hex.c_str();
    if (len >= 2 && phex[0] == '0' && phex[1] == 'x') {
        len -= 2;
        phex += 2;
    }

    int index = 0;

    while (index < len)
    {
        int value = ASCIIHexToInt[phex[index]];
        if (index || !(len % 2)) {
            ++index;
            value = value * 16 + ASCIIHexToInt[phex[index]];
        }
        ++index;
        if (value < 0) return false;
        data.push_back(value);
    }
    return true;
}

inline int getHexValueLen(const string &hex)
{
    int len = hex.length();
    const char *phex = hex.c_str();
    if (len >= 2 && phex[0] == '0' && phex[1] == 'x') {
        len -= 2;
    }
    if (len % 2) {
        ++len;
    }
    return (len >> 1);
}

inline string getTransactionHash(Context_ &ctx, string &from, string &to, uint64_t value, uint64_t nonce, uint64_t gasLimit,
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
    /*
        // unsigned transaction (r == 0 && v == 0)

        // signed transaction
        tx.chainId = Math.floor((tx.v - 35) / 2);
        if (tx.chainId < 0) {
            tx.chainId = 0;
        }
    */
    string rawWoChainId = raw;
    if (chainId) {
        encodeUInt64(raw, chainId);
        encodeUInt64(raw, 0);
        encodeUInt64(raw, 0);
    }

    string rawToHash;
    encodeLen(rawToHash, raw.length(), true);
    rawToHash += raw;
    raw = rawWoChainId;

    cout << "Point-A" << endl;
    string hexR = NormalizeToNFormat(ctx.r.get_str(16),64);
    string hexS = NormalizeToNFormat(ctx.s.get_str(16),64);
    string hexV = NormalizeToNFormat(ctx.v.get_str(16),2);
    string signature = "0x" + hexR + hexS + hexV;
    cout << "hexR: " << hexR << endl;
    cout << "hexS: " << hexS << endl;
    cout << "hexV: " << hexV << endl;
    cout << "SIGNATURE: " << signature << endl;

    string hash = keccak256((const uint8_t *)(rawToHash.c_str()), rawToHash.length());
    cout << "hash: " << hash << endl;
    string ecResult = ecrecover(signature, hash);
    cout << "ecResult: " << ecResult << endl;
    uint64_t recoveryParam;

    // TODO: check ctx_v range
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


    hash = keccak256((const uint8_t *)(res.c_str()), res.length());
    cout << "hash: " << hash << endl;

    return res;
}

#endif