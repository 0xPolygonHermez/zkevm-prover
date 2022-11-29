#ifndef KECCAK_HPP
#define KECCAK_HPP

typedef unsigned char u8;
typedef unsigned long long int u64;
typedef unsigned int ui;

void Keccak(ui r, ui c, const u8 *in, u64 inLen, u8 sfx, u8 *out, u64 outLen);

void FIPS202_SHAKE128(const u8 *in, u64 inLen, u8 *out, u64 outLen);
void FIPS202_SHAKE256(const u8 *in, u64 inLen, u8 *out, u64 outLen);
void FIPS202_SHA3_224(const u8 *in, u64 inLen, u8 *out);
void FIPS202_SHA3_256(const u8 *in, u64 inLen, u8 *out);
void FIPS202_SHA3_384(const u8 *in, u64 inLen, u8 *out);
void FIPS202_SHA3_512(const u8 *in, u64 inLen, u8 *out);

void KeccakF1600(void *s);

#endif