#include "smt_64_test.hpp"
#include "smt_64.hpp"
#include "smt.hpp"
#include "hashdb_singleton.hpp"
#include "unistd.h"
#include "hashdb_factory.hpp"

#define SMT64_TEST_NUMBER_OF_KEYS 10

uint64_t Smt64Test (const Config &config)
{
    TimerStart(SMT64_TEST);

    uint64_t numberOfFailedTests = 0;
    Goldilocks::Element root[4] = {0, 0, 0, 0};
    Goldilocks fr;
    zkresult zkr;
    vector<KeyValue> keyValues;

    HashDBInterface *pHashDB = HashDBClientFactory::createHashDBClient(fr, config);
    if (pHashDB == NULL)
    {
        zklog.error("Smt64Test() failed calling HashDBClientFactory::createHashDBClient()");
        return 1;
    }

    // Init the keyValues
    for (uint64_t i=0; i<SMT64_TEST_NUMBER_OF_KEYS; i++)
    {
        KeyValue kv;
        kv.key.fe[0] = fr.fromU64(i);
        kv.key.fe[1] = fr.zero();
        kv.key.fe[2] = fr.zero();
        kv.key.fe[3] = fr.zero();
        kv.value = i;
        keyValues.emplace_back(kv);
    }

    // Perform the test, based on the configuration
    if (config.hashDB64)
    {
        // Call writeTree()
        uint64_t flushId = 0;
        uint64_t lastSentFlushId = 0;
        zkr = pHashDB->writeTree(root, keyValues, root, flushId, lastSentFlushId);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Smt64Test() failed calling smt64.writeTree() result=" + zkresult2string(zkr));
            return 1;
        }
        //zklog.info("Smt64Test() smt64.writeTree() returned root=" + fea2string(fr, root) + " flushId=" + to_string(flushId));

        // Wait for the returned flush ID to be sent
        do
        {
            uint64_t storedFlushId, storingFlushId, lastFlushId, pendingToFlushNodes, pendingToFlushProgram, storingNodes, storingProgram;
            string proverId;
            zkr = pHashDB->getFlushStatus(storedFlushId, storingFlushId, lastFlushId, pendingToFlushNodes, pendingToFlushProgram, storingNodes, storingProgram, proverId);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("Smt64Test() failed calling db64.getFlushStatus() result=" + zkresult2string(zkr));
                return 1;
            }
            if (storedFlushId >= flushId)
            {
                //zklog.info("Smt64Test() called db64.getFlushStatus() and got storedFlushId=" + to_string(storedFlushId) + " >= flushId=" + to_string(flushId));
                break;
            }
            sleep(1);
        } while (true);
        
        // Make a copy of the key values, overwriting the value with an invalid value
        vector<KeyValue> readKeyValues = keyValues;
        for (uint64_t i=0; i<readKeyValues.size(); i++)
        {
            readKeyValues[i].value = 0xFFFFFFFFFFFFFFFF;
        }

        // Call readTree() with the returned root
        zkr = pHashDB->readTree(root, readKeyValues);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Smt64Test() failed calling smt64.readTree() result=" + zkresult2string(zkr));
            return 1;
        }

        // Check that both keys and values match the ones we wrote
        for (uint64_t i=0; i<SMT64_TEST_NUMBER_OF_KEYS; i++)
        {
            // Check that the key is still the same
            if ( !fr.equal(readKeyValues[i].key.fe[0], keyValues[i].key.fe[0]) ||
                 !fr.equal(readKeyValues[i].key.fe[1], keyValues[i].key.fe[1]) ||
                 !fr.equal(readKeyValues[i].key.fe[2], keyValues[i].key.fe[2]) ||
                 !fr.equal(readKeyValues[i].key.fe[3], keyValues[i].key.fe[3]) )
            {
                zklog.error("Smt64Test() read different key at i=" + to_string(i) + " read=" + fea2string(fr, readKeyValues[i].key.fe) + " expected=" + fea2string(fr, keyValues[i].key.fe));
                numberOfFailedTests++;
            }

            // Check that the value is the expected one
            if (readKeyValues[i].value != keyValues[i].value)
            {
                zklog.error("Smt64Test() read different key at i=" + to_string(i) + " read=" + readKeyValues[i].value.get_str(10) + " expected=" + keyValues[i].value.get_str(10));
                numberOfFailedTests++;
            }
        }

        // Announce the success
        if (numberOfFailedTests==0)
        {
            zklog.info("Smt64Test() succeeded");
        }
    }
    else
    {
        for (uint64_t i=0; i<SMT64_TEST_NUMBER_OF_KEYS; i++)
        {
            zkr = pHashDB->set("", 0, root, keyValues[i].key.fe, keyValues[i].value, PERSISTENCE_DATABASE, root, NULL, NULL);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("Smt64Test() failed calling pHashDB->set() result=" + zkresult2string(zkr));
                return 1;
            }
            zklog.info("Smt64Test() pHashDB->set() i=" + to_string(i) + " returned root=" + fea2string(fr, root));
        }
    }

    TimerStopAndLog(SMT64_TEST);

    return numberOfFailedTests;
}