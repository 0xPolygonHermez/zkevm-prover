#ifndef SMT_SET_RESULT_HPP
#define SMT_SET_RESULT_HPP

#include <vector>
#include <map>

#include "poseidon_goldilocks.hpp"
#include "goldilocks_base_field.hpp"

using namespace std;

// SMT set method result data
class SmtSetResult
{
public:
    Goldilocks::Element oldRoot[4];
    Goldilocks::Element newRoot[4];
    Goldilocks::Element key[4];
    map< uint64_t, vector<Goldilocks::Element> > siblings;
    Goldilocks::Element insKey[4];
    mpz_class insValue;
    bool isOld0;
    mpz_class oldValue;
    mpz_class newValue;
    string mode;
    uint64_t proofHashCounter;
    string toString (Goldilocks &fr);
};

#endif