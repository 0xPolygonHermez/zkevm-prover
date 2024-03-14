#include <unistd.h>
#include "unit_test.hpp"
#include "timer.hpp"
#include "blake_test.hpp"
#include "sha256_test.hpp"
#include "mem_align_test.hpp"
#include "binary_test.hpp"
#include "storage_test.hpp"
#include "keccak_executor_test.hpp"
#include "get_string_increment_test.hpp"
#include "database_cache_test.hpp"
#include "hashdb_test.hpp"

uint64_t UnitTest (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config)
{
    TimerStart(UNIT_TEST);

    uint64_t numberOfErrors = 0;
    
    TimerStart(UNIT_TEST_BLAKE2B256);
    numberOfErrors += Blake2b256_Test(fr, config);
    TimerStopAndLog(UNIT_TEST_BLAKE2B256);
    
    TimerStart(UNIT_TEST_SHA256);
    numberOfErrors += SHA256Test(fr, config);
    TimerStopAndLog(UNIT_TEST_SHA256);
    
    //TimerStart(UNIT_TEST_MEMALIGN);
    //numberOfErrors += MemAlignSMTest(fr, config);
    //TimerStopAndLog(UNIT_TEST_MEMALIGN);
    
    TimerStart(UNIT_TEST_BINARY);
    numberOfErrors += BinarySMTest(fr, config);
    TimerStopAndLog(UNIT_TEST_BINARY);
    
    //TimerStart(UNIT_TEST_STORAGESM);
    //numberOfErrors += StorageSMTest(fr, poseidon, config);
    //TimerStopAndLog(UNIT_TEST_STORAGESM);

    //TimerStart(UNIT_TEST_KECCAKSM);
    //numberOfErrors += KeccakSMExecutorTest(fr, config);
    //TimerStopAndLog(UNIT_TEST_KECCAKSM);

    TimerStart(UNIT_TEST_GET_STRING_INCREMENT);
    numberOfErrors += GetStringIncrementTest();
    TimerStopAndLog(UNIT_TEST_GET_STRING_INCREMENT);

    TimerStart(UNIT_TEST_DATABASE_CACHE);
    numberOfErrors += DatabaseCacheTest();
    TimerStopAndLog(UNIT_TEST_DATABASE_CACHE);

    TimerStart(UNIT_TEST_HASH_DB);
    numberOfErrors += HashDBTest(config);
    TimerStopAndLog(UNIT_TEST_HASH_DB);
    
    TimerStopAndLog(UNIT_TEST);
    
    if (numberOfErrors == 0)
    {
        zklog.info("UnitTest() successfully completed without errors");
    }
    else
    {
        zklog.error("UnitTest() completed with errors=" + to_string(numberOfErrors));
        sleep(1);
        exit(-1);
    }
    
    return numberOfErrors;
}