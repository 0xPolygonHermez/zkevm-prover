#include "statedb_test.hpp"
#include "statedb_test_load.hpp"
#include "statedb_test_perf.hpp"
#include "statedb_test_client.hpp"
#include <thread>
#include "statedb_factory.hpp"
#include "scalar.hpp"
#include "utils.hpp"

//#define STATEDB_TEST_CLIENT
//#define STATEDB_TEST_LOAD
//#define STATEDB_TEST_PERF
#define STATEDB_TEST_MEMORY_LEAK
void runStateDBTest (const Config& config)
{
    std::this_thread::sleep_for(1500ms);

    #ifdef STATEDB_TEST_CLIENT
        runStateDBTestClient(config);
    #endif

    #ifdef STATEDB_TEST_LOAD
        runStateDBTestLoad(config);
    #endif

    #ifdef STATEDB_TEST_PERF
        runStateDBPerfTest(config);
    #endif

    #ifdef STATEDB_TEST_MEMORY_LEAK

    Goldilocks fr;
    StateDBInterface * pStateDB = StateDBClientFactory::createStateDBClient(fr,config);

    Goldilocks::Element root[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};
    for (uint64_t i=0; i<10000000; i++)
    {
        
        mpz_class value = 1;
        value = uint64_t(random())*uint64_t(random());
        value = ScalarTwoTo64*value + uint64_t(random())*uint64_t(random());
        value = ScalarTwoTo64*value + uint64_t(random())*uint64_t(random());
        value = ScalarTwoTo64*value + uint64_t(random())*uint64_t(random());
        Goldilocks::Element key[4];
        key[0] = fr.fromU64(i);
        key[1] = fr.zero();
        key[2] = fr.zero();
        key[3] = fr.zero();
        //key[0] = fr.fromU64(uint64_t(random())*uint64_t(random()));
        //key[1] = fr.fromU64(uint64_t(random())*uint64_t(random()));
        //key[2] = fr.fromU64(uint64_t(random())*uint64_t(random()));
        //key[3] = fr.fromU64(uint64_t(random())*uint64_t(random()));

        zkresult zkr = pStateDB->set(root, key, value, true, root, NULL, NULL );
        if (zkr != ZKR_SUCCESS)
        {
            cerr << "Error: i=" << i << " zkr=" << zkr << "=" << zkresult2string(zkr) << endl;
            exitProcess();
        }
        if (i%10000 == 0)
        {
            cout << getTimestamp() << " i=" << i << endl;
            printMemoryInfo(true);
        }
        if (i%100==0) pStateDB->flush();
    }

    #endif
}

