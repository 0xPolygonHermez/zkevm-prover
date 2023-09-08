#include "database_kv_remote_test.hpp"
#include "goldilocks_base_field.hpp"
#include "hashdb.hpp"
#include "hashdb_singleton.hpp"

#define DBKV_REMOTE_TEST_NUMBER_OF_WRITES 1000


uint64_t DatabaseKVRemoteTest (const Config &config){
    
    TimerStart(DATABASE_KV_REMOTE_TEST);

    if(config.dbMultiWrite == true){
        zklog.error("DatabaseKVRemoteTest() this test must be run with config.dbMultiWrite=false");
        exitProcess();
    }
    HashDB * pHashDB = (HashDB *)hashDBSingleton.get();
    if (pHashDB == NULL)
    {
        zklog.error("DatabaseKVRemoteTest() failed calling HashDBSingleton::get()");
        exitProcess();
    }
    Database64 * pDatabase64 = &pHashDB->db64;

    uint64_t numberOfFailedTests = 0;
    Goldilocks fr;
    
    //
    // Lattest version
    //
    uint64_t version = 33;
    pDatabase64->writeLatestVersion(version, true);
    pDatabase64->readLatestVersion(version);
    if(version != 33)
    {
        zklog.error("DatabaseKVRemoteTest() failed calling Database64.readLatestVersion()");
        numberOfFailedTests += 1;
    }
    pDatabase64->writeLatestVersion(0, false);
    pDatabase64->readLatestVersion(version);
    if(version != 33)
    {
        zklog.error("DatabaseKVRemoteTest() failed calling Database64.readLatestVersion()");
        numberOfFailedTests += 1;
    }
    //
    // Version
    //
    Goldilocks::Element root[4];

    mpz_class rootScalar;
    string rootString;
    version = 1;

    for (uint64_t i=0; i<DBKV_REMOTE_TEST_NUMBER_OF_WRITES; i++)
    {
        rootScalar = i;
        scalar2fea(fr, rootScalar, root);
        pDatabase64->writeVersion(root, version, true);
        version+=1;
    }
    /*for (uint64_t i=0; i<NUMBER_OF_DB_CACHE_ADDS; i++)
    {
        rootScalar = i;
        rootString = PrependZeros(rootScalar.get_str(16), 64);
        scalar2fea(fr, rootScalar, root);
        bResult = dbVersionACache.findKey(root, version);
        
        if (!bResult || version != i+1)
        {
            zklog.error("DatabaseAssociativeCacheTest() failed calling DatabaseVersionsAssociativeCache.find() of root=" + rootString + " version=" + to_string(version));
            numberOfFailed++;
        }
    }*/

    //
    // Check number of failed tests
    //
    if(numberOfFailedTests != 0)
    {
        zklog.error("DatabaseKVRemoteTest() failed with " + to_string(numberOfFailedTests) + " errors"); 
        exitProcess();   
    }
   
    TimerStopAndLog(DATABASE_KV_REMOTE_TEST);
    return numberOfFailedTests;
}


