#ifndef POSEIDON_LINEAR_HPP
#define POSEIDON_LINEAR_HPP

#include <string>
#include <vector>
#include "ff/ff.hpp"
#include "poseidon_goldilocks.hpp"

using namespace std;

void PoseidonLinear (Poseidon_goldilocks &poseidon, vector<uint8_t> &bytes, mpz_class &hash);

#endif