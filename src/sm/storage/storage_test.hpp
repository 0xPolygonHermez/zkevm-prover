#ifndef STORAGE_TEST_HPP
#define STORAGE_TEST_HPP

#include "config.hpp"
#include "goldilocks/goldilocks_base_field.hpp"
#include "poseidon_opt/poseidon_goldilocks_old.hpp"

void StorageSMTest (Goldilocks &fr, Poseidon_goldilocks &poseidon, Config &config);

#endif