#ifndef BLAKE_HPP
#define BLAKE_HPP

#include "goldilocks_base_field.hpp"
#include "config.hpp"


void Blake2b256_String (const string &s, string &hash);
void Blake2b256_Ba (const string &ba, string &hash);
void Blake2b256 (const uint8_t * pData, uint64_t dataSize, string &hash);

void Blake2b256_Test (Goldilocks &fr, Config &config);

#endif