#ifndef HASH_VALUE_GL_HPP
#define HASH_VALUE_GL_HPP

#include <string>
#include "goldilocks_base_field.hpp"

using namespace std;

class HashValueGL
{
public:
    Goldilocks::Element hash[4];
    Goldilocks::Element value[12];
    HashValueGL() {};
};

#endif