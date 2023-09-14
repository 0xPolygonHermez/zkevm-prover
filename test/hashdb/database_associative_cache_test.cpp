#include "database_associative_cache_test.hpp"
#include "hashdb/database.hpp"
#include "timer.hpp"
#include "scalar.hpp"
#include "database_versions_associtive_cache.hpp"
#include "database_kv_associative_cache.hpp"

#define NUMBER_OF_DB_CACHE_ADDS 1000

uint64_t DatabaseAssociativeCacheTest (void)
{
    TimerStart(DATABASE_ASSOCIATIVE_CACHE_TEST);
    uint64_t numberOfFailed = 0;
    Goldilocks fr;
    bool update = false;

    //
    // Test DatabaseMTAssociativeCache
    //
    DatabaseMTAssociativeCache dbMTACache;
    dbMTACache.postConstruct(20,17,"MTACache");
    
    Goldilocks::Element key[4];
    mpz_class keyScalar;
    string keyString;
    vector<Goldilocks::Element> value;
    bool bResult;

    for (uint64_t i=0; i<NUMBER_OF_DB_CACHE_ADDS; i++)
    {
        keyScalar = i;
        value.clear();
        for (uint64_t j=0; j<12; j++)
        {
            value.push_back(fr.fromU64(12*i+j));
        }
        scalar2fea(fr, keyScalar, key);
        dbMTACache.addKeyValue(key, value,update);
    }

    for (uint64_t i=0; i<NUMBER_OF_DB_CACHE_ADDS; i++)
    {
        keyScalar = i;
        keyString = PrependZeros(keyScalar.get_str(16), 64);
        scalar2fea(fr, keyScalar, key);
        bResult = dbMTACache.findKey(key, value);

        if (!bResult || value.size() != 12 || value[0].fe != 12*i+0 || value[11].fe != 12*i+11)
        {
            zklog.error("DatabaseAssociativeCacheTest() failed calling DatabaseMTAssociativeCache.find() of key=" + keyString);
            numberOfFailed++;
        }
    }

    dbMTACache.clear();
    for (uint64_t i=0; i<NUMBER_OF_DB_CACHE_ADDS; i++)
    {
        keyScalar = i;
        keyString = PrependZeros(keyScalar.get_str(16), 64);
        scalar2fea(fr, keyScalar, key);
        bResult = dbMTACache.findKey(key, value);
        if (bResult )
        {
            zklog.error("DatabaseAssociativeCacheTest() failed calling DatabaseMTAssociativeCache.find() the cache should be empty");
            numberOfFailed++;
        }
    }


    //
    // Test DatabaseVersionsAssociativeCache
    //
    DatabaseVersionsAssociativeCache dbVersionACache;
    dbVersionACache.postConstruct(20,17,"VersionsACache");
    Goldilocks::Element root[4];

    mpz_class rootScalar;
    string rootString;
    uint64_t version = 1;

    for (uint64_t i=0; i<NUMBER_OF_DB_CACHE_ADDS; i++)
    {
        rootScalar = i;
        scalar2fea(fr, rootScalar, root);
        dbVersionACache.addKeyVersion(root, version);
        version+=1;
    }
    for (uint64_t i=0; i<NUMBER_OF_DB_CACHE_ADDS; i++)
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
    }
    dbVersionACache.clear();
    for (uint64_t i=0; i<NUMBER_OF_DB_CACHE_ADDS; i++)
    {
        rootScalar = i;
        rootString = PrependZeros(rootScalar.get_str(16), 64);
        scalar2fea(fr, rootScalar, root);
        bResult = dbVersionACache.findKey(root, version);
        
        if (bResult)
        {    
            zklog.error("DatabaseAssociativeCacheTest() failed calling DatabaseVersionsAssociativeCache.find() the cache should be empty");
            numberOfFailed++;
        }
    }
    // Test DatabaseKVAssociativeCache
    //
    DatabaseKVAssociativeCache dbKVACache;
    dbKVACache.postConstruct(20,18,"dbKVACache");
    mpz_class valueScalar;

    for(version=0; version<=10; version+=2){
        for (uint64_t i=0; i<NUMBER_OF_DB_CACHE_ADDS; i++){
            keyScalar = i;
            scalar2fea(fr, keyScalar, key);
            valueScalar = (version/2)*NUMBER_OF_DB_CACHE_ADDS + i;
            dbKVACache.addKeyValueVersion(version, key, valueScalar);
        }
    }
    for(version=0; version<=10; version++){
        for (uint64_t i=0; i<NUMBER_OF_DB_CACHE_ADDS; i++){
            keyScalar = i;
            scalar2fea(fr, keyScalar, key);
            bResult = dbKVACache.findKey(version,key, valueScalar);

            if (!bResult || valueScalar != (version/2)*NUMBER_OF_DB_CACHE_ADDS + i)
            {
                zklog.error("DatabaseAssociativeCacheTest() failed calling DatabaseKVAssociativeCache.find() of key=" + keyString + " version=" + to_string(version) + " value=" + valueScalar.get_str(16));
                numberOfFailed++;
            }
        }
    }
    dbKVACache.clear();
    for(version=0; version<=10; version++){
        for (uint64_t i=0; i<NUMBER_OF_DB_CACHE_ADDS; i++){
            keyScalar = i;
            scalar2fea(fr, keyScalar, key);
            bResult = dbKVACache.findKey(version,key, valueScalar);
            if (bResult)
            {
                zklog.error("DatabaseAssociativeCacheTest() failed calling DatabaseKVAssociativeCache.find() the cache should be empty");
                numberOfFailed++;
            }
        }
    }

    //
    // Final result
    //
    if(numberOfFailed != 0)
    {
        zklog.error("DatabaseAssociativeCacheTest() failed with " + to_string(numberOfFailed) + " errors"); 
        exitProcess();   
    }
    TimerStopAndLog(DATABASE_ASSOCIATIVE_CACHE_TEST);
    return numberOfFailed;
}