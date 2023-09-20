#include "database_kv_multiwrite_test.hpp"
#include "goldilocks_base_field.hpp"
#include "hashdb.hpp"
#include "hashdb_singleton.hpp"


uint64_t DatabaseKVMultiWriteTest(const Config &config){
    
    uint64_t numberOfFailedTests = 0;
    
    TimerStart(DATABASE_KV_MULTIWRITE_TEST);

    if(config.dbMultiWrite == false){
        zklog.error("DatabaseKVMultiWriteTest() this test must be run with config.dbMultiWrite=true");
        exitProcess();
    }
    HashDB * pHashDB = (HashDB *)hashDBSingleton.get();
    if (pHashDB == NULL)
    {
        zklog.error("DatabaseKVMultiWriteTest() failed calling HashDBSingleton::get()");
        exitProcess();
    }
    Database64 * pDatabase64 = &pHashDB->db64;
     //
    // Lattest version
    //
    uint64_t versionIn = 33;
    uint64_t versionOut;
    bool bfound;
    bfound = pDatabase64->multiWrite.findLatestVersion(versionOut);
    if(bfound == true)
    {
        zklog.error("DatabaseKVMultiWriteTest() failed calling Database64.multiWrite.findLatestVersion() version should not be found");
        numberOfFailedTests += 1;
    } 
    pDatabase64->writeLatestVersion(versionIn, true);
    pDatabase64->readLatestVersion(versionOut);
    if(versionOut != versionIn)
    {
        zklog.error("DatabaseKVTest() failed calling Database64.readLatestVersion()");
        numberOfFailedTests += 1;
    }
    pDatabase64->clearCache(); 
    pDatabase64->multiWrite.findLatestVersion(versionOut);
    if(versionOut != versionIn)
    {
        zklog.error("DatabaseKVTest() failed calling Database64.readLatestVersion(), version should be found in multiWrite");
        numberOfFailedTests += 1;
    }    
    //
    // Check number of failed tests
    //
    if(numberOfFailedTests != 0)
    {
        zklog.error("DatabaseKVMultiWriteTest() failed with " + to_string(numberOfFailedTests) + " errors"); 
        exitProcess();   
    }
   
    TimerStopAndLog(DATABASE_KV_MULTIWRITE_TEST);
    return numberOfFailedTests;
}


