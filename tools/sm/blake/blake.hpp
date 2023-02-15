#ifndef BLAKE_HPP
#define BLAKE_HPP

#include <string>
#include "goldilocks_base_field.hpp"
#include "config.hpp"

using namespace std;

void Blake2b256_String (const string &s, string &hash);
void Blake2b256 (const uint8_t * pData, uint64_t dataSize, string &hash);

#endif