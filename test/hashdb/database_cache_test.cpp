#include "database_cache_test.hpp"
#include "hashdb/database.hpp"
#include "timer.hpp"
#include "scalar.hpp"

#define NUMBER_OF_DB_CACHE_ADDS 1000

uint64_t DatabaseCacheTest (void)
{
    TimerStart(DATABASE_CACHE_TEST);

    
    //Database::dbMTCache.clear();  //rick pending
    
    uint64_t numberOfFailed = 0;
#ifndef DATABASE_USE_ASSOCIATIVE_CACHE
    Database::dbMTCache.setMaxSize(2000000);
#endif
    Goldilocks fr;
    mpz_class keyScalar;
    string keyString;
    vector<Goldilocks::Element> value;
    bool bResult;
    for (uint64_t i=0; i<NUMBER_OF_DB_CACHE_ADDS; i++)
    {
        keyScalar = i;
        keyString = PrependZeros(keyScalar.get_str(16), 64);
        value.clear();
        for (uint64_t j=0; j<12; j++)
        {
            value.push_back(fr.fromU64(j));
        }
        bool update = false;
#ifndef DATABASE_USE_ASSOCIATIVE_CACHE
        Database::dbMTCache.add(keyString, value,update);
#else
        Database::dbMTCache.add(keyString, value,update, " "," ");
#endif

    }

    //Database::dbMTCache.print(true);

    for (uint64_t i=0; i<NUMBER_OF_DB_CACHE_ADDS; i++)
    {
        keyScalar = i;
        keyString = PrependZeros(keyScalar.get_str(16), 64);
        string leftChildKey;
        string rightChildKey;
        bResult = Database::dbMTCache.find(keyString, value,leftChildKey,rightChildKey);
        if (!bResult)
        {
            zklog.error("DatabaseCacheTest() failed calling Database::dbMTCache.find() of key=" + keyString);
            numberOfFailed++;
        }
    }
    
    //Database::dbMTCache.clear(); //rick pending

    TimerStopAndLog(DATABASE_CACHE_TEST);
    return numberOfFailed;
}