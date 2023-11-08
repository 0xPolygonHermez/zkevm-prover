#ifndef VERSION_VALUE_HPP
#define VERSION_VALUE_HPP

#include <gmpxx.h>

class VersionValue
{
public:
    uint64_t version;
    mpz_class value;
};

#endif