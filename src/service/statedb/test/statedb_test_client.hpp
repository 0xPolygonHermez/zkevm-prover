#ifndef STATEDB_TEST_CLIENT_HPP
#define STATEDB_TEST_CLIENT_HPP

#include "config.hpp"

void runStateDBTestClient (const Config& config);
void* stateDBTestClientThread (const Config& config);

#endif