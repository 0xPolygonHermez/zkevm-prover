#ifndef STATEDB_TEST_PERF_HPP
#define STATEDB_TEST_PERF_HPP

#include "config.hpp"

void runStateDBPerfTest (const Config& config);
void* stateDBPerfTestThread (const Config& config);

#endif
