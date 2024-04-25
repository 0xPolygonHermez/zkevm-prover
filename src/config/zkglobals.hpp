#ifndef ZKGLOBALS_HPP
#define ZKGLOBALS_HPP

#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include "ffiasm/fec.hpp"
#include "ffiasm/fnec.hpp"
#include "ffiasm/fr.hpp"
#include "ffiasm/fq.hpp"
#include "ffiasm/bls12_381.hpp"
#include "config.hpp"
#include <string>

using namespace std;

extern Goldilocks fr;
extern PoseidonGoldilocks poseidon;
extern RawFec fec;
extern RawFnec fnec;
extern RawFr bn128;
extern RawFq fq;
extern RawBLS12_381 bls12_381;
extern Config config;
extern const string version;

#include "scalar.hpp"
extern mpz_class BLS_12_381_prime;

#endif