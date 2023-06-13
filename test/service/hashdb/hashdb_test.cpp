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

void runHashDBTest (const Config& config)
{
    std::this_thread::sleep_for(1500ms);

    //runHashDBTestClient(config);
    //runHashDBTestLoad(config);
    //runHashDBPerfTest(config);
    //runHashDBBigTree(config);
    runHashDBTestMultiWrite(config);
}

