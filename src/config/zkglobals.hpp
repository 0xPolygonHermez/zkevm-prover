#ifndef ZKGLOBALS_HPP
#define ZKGLOBALS_HPP

#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include "ffiasm/fec.hpp"
#include "ffiasm/fnec.hpp"
#include "config.hpp"

extern Goldilocks fr;
extern PoseidonGoldilocks poseidon;
extern RawFec fec;
extern RawFnec fnec;
extern Config config;

#endif