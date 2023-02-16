#ifndef SHA256_HPP
#define SHA256_HPP

#include <string>

using namespace std;

void SHA256 (const uint8_t * pData, uint64_t dataSize, string &hash);
void SHA256String (const string &s, string &hash);

#endif