#include "zkglobals.hpp"
#include "version.hpp"

Goldilocks fr;
PoseidonGoldilocks poseidon;
RawFec fec;
RawFnec fnec;
RawFr bn128;
RawFq bn254;
RawFq fq;
RawBLS12_381 bls12_381;
RawBLS12_381_384 bls12_381_384;
RawBLS12_381 fp;
Config config;
const string version(ZKEVM_PROVER_VERSION);
mpz_class BLS_12_381_prime("52435875175126190479447740508185965837690552500527637822603658699938581184513");
mpz_class BLS_12_381_384_prime("0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");

// Spec names:

mpz_class  goldilocks_prime(goldilocks_prime_string);
Goldilocks goldilocks;

mpz_class  BN254p_prime(BN254p_prime_string);
RawFq      BN254p;

mpz_class  BN254r_prime(BN254r_prime_string);
RawFr      BN254r;

mpz_class  Secp256k1p_prime(Secp256k1p_prime_string);
RawFec     Secp256k1p;

mpz_class  Secp256k1r_prime(Secp256k1r_prime_string);
RawFnec    Secp256k1r;

mpz_class  BLS12_381p_prime(BLS12_381p_prime_string);
RawBLS12_381_384 BLS12_381p;

mpz_class  BLS12_381r_prime(BLS12_381r_prime_string);
RawBLS12_381 BLS12_381r;

mpz_class  aSecp256r1(aSecp256r1_string);
RawpSecp256r1::Element aSecp256r1_fe;

mpz_class  pSecp256r1_prime(pSecp256r1_prime_string);
RawpSecp256r1 pSecp256r1;

mpz_class  nSecp256r1_prime(nSecp256r1_prime_string);
RawnSecp256r1 nSecp256r1;

void zkGlobalsInit (void)
{
    pSecp256r1.fromMpz(aSecp256r1_fe, aSecp256r1.get_mpz_t());
}