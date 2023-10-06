#include "database_associative_cache_test.hpp"
#include "hashdb/database.hpp"
#include "timer.hpp"
#include "scalar.hpp"

#define NUMBER_OF_DB_CACHE_ADDS 1000

uint64_t DatabaseAssociativeCacheTest (void)
{
    TimerStart(DATABASE_ASSOCIATIVE_CACHE_TEST);

    uint64_t numberOfFailed = 0;
    Database::dbMTACache.postConstruct(20,17,"MTACache");
    Goldilocks::Element key[4];

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
        scalar2fea(fr, keyScalar, key);
        Database::dbMTACache.addKeyValue(key, value,update);
    }

    //Database::dbMTCache.print(true);

    for (uint64_t i=0; i<NUMBER_OF_DB_CACHE_ADDS; i++)
    {
        keyScalar = i;
        keyString = PrependZeros(keyScalar.get_str(16), 64);
        scalar2fea(fr, keyScalar, key);
        bResult = Database::dbMTACache.findKey(key, value);

        if (!bResult)
        {
            zklog.error("DatabaseCacheTest() failed calling Database::dbMTCache.find() of key=" + keyString);
            numberOfFailed++;
        }
    }
    Database::dbMTCache.clear();
    TimerStopAndLog(DATABASE_ASSOCIATIVE_CACHE_TEST);
    return numberOfFailed;
}