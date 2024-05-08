#ifndef ZKGLOBALS_HPP
#define ZKGLOBALS_HPP

#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include "ffiasm/fec.hpp"
#include "ffiasm/fnec.hpp"
#include "ffiasm/fr.hpp"
#include "ffiasm/fq.hpp"
#include "ffiasm/bls12_381.hpp"
#include "ffiasm/bls12_381_384.hpp"
#include "config.hpp"
#include <string>

using namespace std;

extern Goldilocks fr;
extern PoseidonGoldilocks poseidon;
extern RawFec fec;
extern RawFnec fnec;
extern RawFr bn128;
extern RawFq bn254;
extern RawFq fq;
extern RawBLS12_381 bls12_381;
extern RawBLS12_381_384 bls12_381_384;
extern Config config;
extern const string version;

#include "scalar.hpp"
extern mpz_class BLS_12_381_prime;
extern mpz_class BLS_12_381_384_prime;

// Spec names:

const  string     goldilocks_prime_string = "0xffffffff00000001";
extern mpz_class  goldilocks_prime;
extern Goldilocks goldilocks;

const  string     BN254p_prime_string = "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47";
extern mpz_class  BN254p_prime;
extern RawFq      BN254p;

const  string     BN254r_prime_string = "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001";
extern mpz_class  BN254r_prime;
extern RawFr      BN254r;

const  string     Secp256k1p_prime_string = "0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f";
extern mpz_class  Secp256k1p_prime;
extern RawFec     Secp256k1p;

const  string     Secp256k1r_prime_string = "0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141";
extern mpz_class  Secp256k1r_prime;
extern RawFnec    Secp256k1r;

const string      BLS12_381p_prime_string = "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab";
extern mpz_class  BLS12_381p_prime;
extern RawBLS12_381_384 BLS12_381p;

const  string     BLS12_381r_prime_string = "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001";
extern mpz_class  BLS12_381r_prime;
extern RawBLS12_381 BLS12_381r;

#endif