#ifndef SHA256_HPP
#define SHA256_HPP

#include "goldilocks_base_field.hpp"
#include "config.hpp"


void SHA256String (const string &s, string &hash);
void SHA256 (const uint8_t * pData, uint64_t dataSize, string &hash);

void SHA256Test (Goldilocks &fr, Config &config);

#endif