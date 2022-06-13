#ifndef POSEIDON_LINEAR_HPP
#define POSEIDON_LINEAR_HPP

#include <string>
#include <vector>
#include <gmpxx.h>
#include "poseidon_goldilocks.hpp"
#include "goldilocks/goldilocks_base_field.hpp"

using namespace std;

void PoseidonLinear (Goldilocks &fr, Poseidon_goldilocks &poseidon, vector<uint8_t> &bytes, mpz_class &hash);

#endif