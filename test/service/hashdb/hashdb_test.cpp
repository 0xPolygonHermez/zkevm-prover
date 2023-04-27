#include <thread>
#include "hashdb_test.hpp"
#include "hashdb_test_load.hpp"
#include "hashdb_test_perf.hpp"
#include "hashdb_test_client.hpp"
#include "../../../src/service/hashdb/hashdb_factory.hpp"
#include "scalar.hpp"
#include "utils.hpp"

//#define HASHDB_TEST_CLIENT
//#define HASHDB_TEST_LOAD
//#define HASHDB_TEST_PERF
#define HASHDB_TEST_MEMORY_LEAK
void runHashDBTest (const Config& config)
{
    std::this_thread::sleep_for(1500ms);

    #ifdef HASHDB_TEST_CLIENT
        runHashDBTestClient(config);
    #endif

    #ifdef HASHDB_TEST_LOAD
        runHashDBTestLoad(config);
    #endif

    #ifdef HASHDB_TEST_PERF
        runHashDBPerfTest(config);
    #endif

    #ifdef HASHDB_TEST_MEMORY_LEAK

    Goldilocks fr;
    HashDBInterface * pHashDB = HashDBClientFactory::createHashDBClient(fr,config);

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

        zkresult zkr = pHashDB->set(root, key, value, true, root, NULL, NULL );
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
        if (i%100==0) pHashDB->flush();
    }

    #endif
}

