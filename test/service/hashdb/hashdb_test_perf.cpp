#include "hashdb_test.hpp"
#include <nlohmann/json.hpp>
#include "scalar.hpp"
#include "zkassert.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include "database.hpp"
#include <thread>
#include "timer.hpp"
#include "goldilocks_base_field.hpp"
#include "hashdb_interface.hpp"
#include "hashdb_factory.hpp"
#include "hashdb_test_perf.hpp"
#include "hashdb_test_load.hpp"
#include "utils.hpp"

using namespace std;

#define PERF_SET 1
#define PERF_GET 2
#define PERF_TEST PERF_SET
const uint64_t TEST_COUNT = 50000;

void runHashDBPerfTest (const Config& config)
{
    new thread {hashDBPerfTestThread, config};
}

void* hashDBPerfTestThread (const Config& config)
{
    cout << "HashDB performance test started" << endl;
    Goldilocks fr;

    string uuid = getUUID();
    uint64_t tx = 0;

    string sTest;
    #if PERF_TEST == PERF_SET
        sTest = "SET";
    #elif PERF_TEST == PERF_GET
        sTest = "GET";
    #endif

    HashDBInterface* client = HashDBClientFactory::createHashDBClient(fr, config);

    SmtSetResult setResult;
    SmtGetResult getResult;

    Goldilocks::Element root[4]={0,0,0,0};
    Goldilocks::Element newRoot[4]={0,0,0,0};
    Goldilocks::Element key[4]={0,0,0,0};
    mpz_class value;
    mpz_class keyScalar;
    uint64_t r;

    pqxx::connection* pConnection = NULL;

    // Random generator
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned long long> distrib(0, std::llround(std::pow(2,64)));

    try
    {
        string uri = "postgresql://zkhashdb:zkhashdb@127.0.0.1:5532/perf_db";
        pConnection = new pqxx::connection{uri};
        if (!loadRoot(fr, pConnection, 100, root)) return NULL;
        cout << "Root=[" << fr.toString(root[0]) << "," << fr.toString(root[1]) << "," << fr.toString(root[2]) << "," << fr.toString(root[3]) << "]" << endl;
    }
    catch (const std::exception &e)
    {
        cerr << "hashDBPerfTestThread: database.exception: " << e.what() << endl;
        if (pConnection!=NULL) delete pConnection;
        return NULL;
    }

    if (config.hashDBURL=="local") {
        cout << "Executing " << TEST_COUNT << " " << sTest << " operations using local client..." << endl;
    } else {
        cout << "Executing " << TEST_COUNT << " " << sTest << " operations using remote client..." << endl;
    }

    struct timeval tset;
    gettimeofday(&tset, NULL);
    for (uint64_t i=1; i<=TEST_COUNT; i++) {
        keyScalar = 0;
        for (int k=0; k<4; k++) {
            r = distrib(gen);
            keyScalar = (keyScalar << 64) + r;
        }

        scalar2key(fr, keyScalar, key);
        value=i;

        #if PERF_TEST == PERF_SET
            client->set(uuid, tx, root, key, value, PERSISTENCE_DATABASE, newRoot, &setResult, NULL);
            for (int j=0; j<4; j++) root[j] = setResult.newRoot[j];
        #elif PERF_TEST == PERF_GET
            client->get(root, key, value, &getResult);
        #endif
    }
    uint64_t totalTimeUS = TimeDiff(tset);

    #if PERF_TEST == PERF_SET
        cout << "Saving new root..." << endl;
        saveRoot (fr, pConnection, 100, root);
    #endif

    cout << "Total Execution time (us): " << totalTimeUS << endl;
    cout << "Time per " << sTest << ": " << totalTimeUS/TEST_COUNT << "us" << endl;
    cout << sTest << "s per second: " << (float)1000000/(totalTimeUS/TEST_COUNT) << endl;

    cout << "HashDB performance test done" << endl;
    delete pConnection;
    return NULL;
}
