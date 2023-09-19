#include "database_kv_test.hpp"
#include "goldilocks_base_field.hpp"
#include "hashdb.hpp"
#include "hashdb_singleton.hpp"

#define DBKV_REMOTE_TEST_NUMBER_OF_WRITES 1000


uint64_t DatabaseKVTest (const Config &config){
    
    Goldilocks fr;
    uint64_t numberOfFailedTests = 0;
    uint64_t level;
    
    TimerStart(DATABASE_KV_REMOTE_TEST);

    if(config.dbMultiWrite == true){
        zklog.error("DatabaseKVTest() this test must be run with config.dbMultiWrite=false");
        exitProcess();
    }
    HashDB * pHashDB = (HashDB *)hashDBSingleton.get();
    if (pHashDB == NULL)
    {
        zklog.error("DatabaseKVTest() failed calling HashDBSingleton::get()");
        exitProcess();
    }
    Database64 * pDatabase64 = &pHashDB->db64;    
    //
    // Lattest version
    //
    uint64_t versionIn = 33;
    uint64_t versionOut;
    pDatabase64->writeLatestVersion(versionIn, true);
    pDatabase64->readLatestVersion(versionOut);
    if(versionOut != versionIn)
    {
        zklog.error("DatabaseKVTest() failed calling Database64.readLatestVersion()");
        numberOfFailedTests += 1;
    }
    pDatabase64->clearCache(); 
    pDatabase64->readLatestVersion(versionOut);
    if(versionOut != versionIn)
    {
        zklog.error("DatabaseKVTest() failed calling Database64.readLatestVersion()");
        numberOfFailedTests += 1;
    }
    //
    //  Version
    //
    Goldilocks::Element root[4];
    mpz_class rootScalar=1234;
    scalar2fea(fr, rootScalar, root);
    versionIn = 44;
    pDatabase64->writeVersion(root, versionIn, true);
    pDatabase64->readVersion(root, versionOut, NULL);
    if(versionOut != versionIn)
    {
        zklog.error("DatabaseKVTest() failed calling Database64.readVersion()");
        numberOfFailedTests += 1;
    }
    pDatabase64->clearCache(); 
    pDatabase64->readVersion(root, versionOut, NULL);
    if(versionOut != versionIn)
    {
        zklog.error("DatabaseKVTest() failed calling Database64.readVersion()");
        numberOfFailedTests += 1;
    }
    //
    // KeyValue 
    //
    Goldilocks::Element root1[4], root2[4], root3[4];
    rootScalar=1;
    scalar2fea(fr, rootScalar, root1);
    rootScalar=2;
    scalar2fea(fr, rootScalar, root2);
    rootScalar=3;
    scalar2fea(fr, rootScalar, root3);

    versionIn = 10;
    pDatabase64->writeVersion(root1, versionIn, true);
    versionIn = 20;
    pDatabase64->writeVersion(root2, versionIn, true);
    versionIn = 30;
    pDatabase64->writeVersion(root3, versionIn, true);

    Goldilocks::Element key[4];
    mpz_class keyScalar=100;
    scalar2fea(fr, keyScalar, key);
    
    mpz_class valueIn=1000;
    mpz_class valueZero(0);
    mpz_class valueOut;
    pDatabase64->writeKV(root2, key, valueIn, true);
    pDatabase64->readKV(root2, key, valueOut, level, NULL);
    if(valueOut != valueIn)
    {
        zklog.error("DatabaseKVTest() failed calling Database64.readKV()");
        numberOfFailedTests += 1;
    }
    pDatabase64->clearCache();
    pDatabase64->readKV(root2, key, valueOut, level, NULL);
    if(valueOut != valueIn)
    {
        zklog.error("DatabaseKVTest() failed calling Database64.readKV() no cache");
        numberOfFailedTests += 1;
    }

    pDatabase64->readKV(root1, key, valueOut, level, NULL);
    if(valueOut != valueZero)
    {
        zklog.error("DatabaseKVTest() failed calling Database64.readKV() should not be found");
        numberOfFailedTests += 1;
    }
    pDatabase64->readKV(root3, key, valueOut, level, NULL);
    if(valueOut != valueIn)
    {
        zklog.error("DatabaseKVTest() failed calling Database64.readKV() should be found with latter version");
        numberOfFailedTests += 1;
    }

    //
    // Check number of failed tests
    //
    if(numberOfFailedTests != 0)
    {
        zklog.error("DatabaseKVTest() failed with " + to_string(numberOfFailedTests) + " errors"); 
        exitProcess();   
    }
   
    TimerStopAndLog(DATABASE_KV_REMOTE_TEST);
    return numberOfFailedTests;
}


