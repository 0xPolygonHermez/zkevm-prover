#ifndef STORAGE_TEST_HPP
#define STORAGE_TEST_HPP

#include "config.hpp"
#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"

uint64_t StorageSMTest (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config);

#endif