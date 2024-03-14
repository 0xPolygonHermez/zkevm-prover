#include <unistd.h>
#include "database.hpp"
#include "hashdb_singleton.hpp"
#include "poseidon_goldilocks.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "timer.hpp"


#define DATABASE_PERFORMANCE_TEST_SIZE 10000

uint64_t DatabasePerformanceTestSendValues (uint64_t valueSize)
{
    TimerStart(DATABASE_PERFORMANCE_TEST);

    zklog.info("DatabasePerformanceTestSendValues() valueSize=" + to_string(valueSize) + " valueBinaryLength=" + to_string(valueSize*8));

    HashDB *pHashDB = hashDBSingleton.get();
    Database &db = pHashDB->db;
    Goldilocks fr;
    PoseidonGoldilocks poseidon;
    zkresult zkr;

    Goldilocks::Element keyValue[12];
    for (uint64_t i=0; i<12; i++)
    {
        keyValue[i] = fr.zero();
    }

    // Create keys
    Goldilocks::Element key[4];
    string *pKeyString = new string[DATABASE_PERFORMANCE_TEST_SIZE];
    if (pKeyString == NULL)
    {
        zklog.error("Failed allocating pKeyStrig");
        exitProcess();
    }
    for (uint64_t i=0; i<DATABASE_PERFORMANCE_TEST_SIZE; i++)
    {
        keyValue[0] = fr.fromU64(i);
        poseidon.hash(key, keyValue);
        pKeyString[i] = fea2string(fr, key);
    }

    // Create values
    vector<Goldilocks::Element> *pValue = new vector<Goldilocks::Element>[DATABASE_PERFORMANCE_TEST_SIZE];
    if (pValue == NULL)
    {
        zklog.error("Failed allocating pValue");
        exitProcess();
    }
    for (uint64_t i=0; i<DATABASE_PERFORMANCE_TEST_SIZE; i++)
    {
        for (uint64_t j=0; j<valueSize; j++)
        {
            (pValue+i)->push_back(fr.one());
        }
    }

    for (uint64_t i=0; i<DATABASE_PERFORMANCE_TEST_SIZE; i++)
    {
        zkr = db.write(pKeyString[i], NULL, *(pValue + i), true);
        if (zkr != ZKR_SUCCESS)
        {
            cerr << "Error: i=" << i << " zkr=" << zkr << "=" << zkresult2string(zkr) << endl;
            exitProcess();
        }
    }

    // Call flush
    uint64_t flushId, lastSentFlushId;
    zkr = db.flush(flushId, lastSentFlushId);
    if (zkr != ZKR_SUCCESS)
    {
        cerr << "Error: failed calling db.flush() zkr=" << zkr << "=" << zkresult2string(zkr) << endl;
        exitProcess();
    }

    // Wait for data to be stored
    uint64_t storedFlushId;
    uint64_t storingFlushId;
    uint64_t lastFlushId;
    uint64_t pendingToFlushNodes;
    uint64_t pendingToFlushProgram;
    uint64_t storingNodes;
    uint64_t storingProgram;

    do
    {
        sleep(1);
        zkr = db.getFlushStatus(storedFlushId, storingFlushId, lastFlushId, pendingToFlushNodes, pendingToFlushProgram, storingNodes, storingProgram);
        if (zkr != ZKR_SUCCESS)
        {
            cerr << "Error: failed calling db.getFlushStatus() zkr=" << zkr << "=" << zkresult2string(zkr) << endl;
            exitProcess();
        }
    } while (storedFlushId < flushId);

    delete[] pKeyString;
    delete[] pValue;

    TimerStopAndLog(DATABASE_PERFORMANCE_TEST);

    return 0;
}


uint64_t DatabasePerformanceTest (void)
{
    for (uint64_t i=1; i<=256; i*=2)
    {
        DatabasePerformanceTestSendValues(12*i);
    }
    return 0;
}