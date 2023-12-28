#ifndef SHA256_HPP
#define SHA256_HPP

#include <string>
#include <stdint.h>
#include "config.hpp"
#include "scalar.hpp"

using namespace std;

void SHA256String (const string &s, string &hash);
void SHA256 (const uint8_t * pData, uint64_t dataSize, string &hash);
void SHA256 (const uint8_t * pData, uint64_t dataSize, mpz_class &hash);
void SHA256F (const uint8_t inR[64], const uint32_t stIn[8], uint32_t stOut[8]);

/* Generate script */
void SHA256Gen (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput, string scriptFile="", string polsFile="", string connectionsFile="");
void SHA256GenerateScript (const Config &config);

#endif