#include "zkglobals.hpp"
#include "version.hpp"

Goldilocks fr;
PoseidonGoldilocks poseidon;
RawFec fec;
RawFnec fnec;
RawFr bn128;
RawFq fq;
RawBLS12_381 bls12_381;
Config config;
const string version(ZKEVM_PROVER_VERSION);
mpz_class BLS_12_381_prime("52435875175126190479447740508185965837690552500527637822603658699938581184513");