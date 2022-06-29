#ifndef STATEDB_TEST_CASE_HPP
#define STATEDB_TEST_CASE_HPP

#include "config.hpp"

void runStateDBTestClient (const Config& config);
void* stateDBTestClientThread (const Config& config);

#endif