#include "database_kv_remote_test.hpp"
#include "goldilocks_base_field.hpp"
#include "hashdb.hpp"
#include "hashdb_singleton.hpp"

#define DBKV_REMOTE_TEST_NUMBER_OF_WRITES 1000


uint64_t DatabaseKVRemoteTest (const Config &config){
    
    
    uint64_t numberOfFailedTests = 0;
    
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
    pDatabase64->init(); //problematic!
    
    //
    // Lattest version
    //
    uint64_t version = 1;
    pDatabase64->writeLatestVersion(version, true);
    version = pDatabase64->readLatestVersion(version);
    if(version != 1)
    {
        zklog.error("DatabaseKVRemoteTest() failed calling Database64.readLatestVersion()");
        numberOfFailedTests += 1;
    }
    pDatabase64->writeLatestVersion(0, false);
    version = pDatabase64->readLatestVersion(version);
    if(version != 1)
    {
        zklog.error("DatabaseKVRemoteTest() failed calling Database64.readLatestVersion()");
        numberOfFailedTests += 1;
    }

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


