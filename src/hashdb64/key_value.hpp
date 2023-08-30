#ifndef KEY_VALUE_HPP
#define KEY_VALUE_HPP

#include "key.hpp"
#include "scalar.hpp"

class KeyValue
{
public:
    Key key;
    mpz_class value;
};

#endif