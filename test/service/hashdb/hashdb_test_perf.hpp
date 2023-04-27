#ifndef HASHDB_TEST_PERF_HPP
#define HASHDB_TEST_PERF_HPP

#include "config.hpp"

void runHashDBPerfTest (const Config& config);
void* hashDBPerfTestThread (const Config& config);

#endif
