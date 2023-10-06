#ifndef SMT_GET_RESULT_HPP
#define SMT_GET_RESULT_HPP

#include <vector>
#include <map>

#include "poseidon_goldilocks.hpp"
#include "goldilocks_base_field.hpp"

using namespace std;

// SMT get method result data
class SmtGetResult
{
public:
    Goldilocks::Element root[4]; // merkle-tree root
    Goldilocks::Element key[4]; // key to look for
    map< uint64_t, vector<Goldilocks::Element> > siblings; // array of siblings // array of fields??
    Goldilocks::Element insKey[4]; // key found
    mpz_class insValue; // value found
    bool isOld0; // is new insert or delete
    mpz_class value; // value retrieved
    uint64_t proofHashCounter;
    string toString (Goldilocks &fr);
};

#endif