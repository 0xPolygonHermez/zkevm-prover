#ifndef UNIT_TEST_HPP
#define UNIT_TEST_HPP

#include <stdint.h>
#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include "config.hpp"

// Returns the total number of failed tests
uint64_t UnitTest (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config);

#endif