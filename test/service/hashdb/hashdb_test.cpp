#include <thread>
#include <unistd.h>
#include "hashdb_test.hpp"
#include "hashdb_test_load.hpp"
#include "hashdb_test_perf.hpp"
#include "hashdb_test_client.hpp"
#include "hashdb_test_big_tree.hpp"
#include "hashdb_test_multi_write.hpp"
#include "../../../src/service/hashdb/hashdb_factory.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "hashdb64_workflow_test.hpp"
#include "hashdb_program_test.hpp"

uint64_t HashDBTest (const Config& config)
{
    std::this_thread::sleep_for(1500ms);

    uint64_t numberOfFailedTests = 0;

    //runHashDBTestClient(config);
    //runHashDBTestLoad(config);
    //runHashDBPerfTest(config);
    //runHashDBBigTree(config);
    
    //numberOfFailedTests += HashDBTestMultiWrite(config);
    //numberOfFailedTests += HashDBProgramTest(config);
    numberOfFailedTests += HashDB64WorkflowTest(config);
    return numberOfFailedTests;
}

