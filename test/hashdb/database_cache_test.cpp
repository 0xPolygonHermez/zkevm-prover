#include "database_cache_test.hpp"
#include "hashdb/database.hpp"
#include "timer.hpp"
#include "scalar.hpp"

#define NUMBER_OF_DB_CACHE_ADDS 1000

uint64_t DatabaseCacheTest (void)
{
    TimerStart(DATABASE_CACHE_TEST);

    
    Database::dbMTCache.clear();
    
    uint64_t numberOfFailed = 0;

    Database::dbMTCache.setMaxSize(2000000);

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
        Database::dbMTCache.add(keyString, value, update);
    }

    //Database::dbMTCache.print(true);

    for (uint64_t i=0; i<NUMBER_OF_DB_CACHE_ADDS; i++)
    {
        keyScalar = i;
        keyString = PrependZeros(keyScalar.get_str(16), 64);
        bResult = Database::dbMTCache.find(keyString, value);
        if (!bResult)
        {
            zklog.error("DatabaseCacheTest() failed calling Database::dbMTCache.find() of key=" + keyString);
            numberOfFailed++;
        }
    }
    
    Database::dbMTCache.clear();

    TimerStopAndLog(DATABASE_CACHE_TEST);
    return numberOfFailed;
}