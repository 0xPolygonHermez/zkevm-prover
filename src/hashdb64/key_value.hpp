#ifndef KEY_VALUE_HPP
#define KEY_VALUE_HPP

#include "scalar.hpp"

class KeyValue
{
public:
    Goldilocks::Element key[4];
    mpz_class value;
};

#endif