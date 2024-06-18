#ifndef CBOR_HPP
#define CBOR_HPP

#include "zkresult.hpp"
#include "scalar.hpp"

// CBOR result
class CborResult
{
public:
    enum ResultType
    {
        UNDEFINED = 0,
        U64 = 1,
        BA = 2,
        TEXT = 3,
        ARRAY = 4,
        TAG = 6
    };
    zkresult result;
    ResultType type;
    uint64_t u64;
    string ba;
    string text;
    vector<CborResult> array;
    uint64_t tagCount;
    vector<CborResult> tag;
    CborResult() : result(ZKR_UNSPECIFIED), type(UNDEFINED), u64(0), tagCount(0) {};
};

string cborType2string (CborResult::ResultType type);

// This function parses CBOR field and stores it in a CborResult
void cbor2result (const string &s, uint64_t &p, CborResult &result);

// This CBOR function expects a simple integer < 24; otherwise it fails
zkresult cbor2u64 (const string &s, uint64_t &p, uint64_t &value);

// This CBOR function expects a byte array; otherwise it fails
zkresult cbor2ba (const string &s, uint64_t &p, string &value);

// This CBOR function expects a text string; otherwise it fails
zkresult cbor2text (const string &s, uint64_t &p, string &value);

// This function expects an integer, which can be long, and returns a scalar
zkresult cbor2scalar (const string &s, uint64_t &p, mpz_class &value);

#endif