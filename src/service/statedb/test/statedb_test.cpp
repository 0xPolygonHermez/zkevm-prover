#include "statedb_test.hpp"
#include "statedb_test_load.hpp"
#include "statedb_test_perf.hpp"
#include "statedb_test_client.hpp"

#define STATEDB_TEST_CLIENT 1
#define STATEDB_TEST_LOAD 2
#define STATEDB_TEST_PERF 3
#define STATEDB_TEST STATEDB_TEST_CLIENT

void runStateDBTest (const Config& config)
{
    #if STATEDB_TEST == STATEDB_TEST_CLIENT
        runStateDBTestClient(config);
    #elif STATEDB_TEST == STATEDB_TEST_LOAD
        runStateDBTestLoad(config);
    #elif STATEDB_TEST == STATEDB_TEST_PERF
        runStateDBPerfTest(config);
    #endif    
}

