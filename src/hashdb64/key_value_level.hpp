#ifndef KEY_VALUE_LEVEL_HPP
#define KEY_VALUE_LEVEL_HPP

#include "scalar.hpp"

class KeyValueLevel
{
public:
    Goldilocks::Element key[4];
    mpz_class value;
    uint64_t level;
};

#endif