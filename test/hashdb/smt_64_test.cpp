#include "smt_64_test.hpp"
#include "smt.hpp"
#include "hashdb_singleton.hpp"
#include "unistd.h"
#include "hashdb_factory.hpp"
#include "utils.hpp"

#define SMT64_TEST_NUMBER_OF_WRITES 1000
#define SMT64_TEST_KEYS_PER_WRITE 10
#define SMT64_TEST_NUMBER_OF_KEYS (SMT64_TEST_NUMBER_OF_WRITES*SMT64_TEST_KEYS_PER_WRITE)


uint64_t Smt64Test (const Config &config)
{
    TimerStart(SMT64_TEST);

    zklog.info("Smt64Test() number of writes=" + to_string(SMT64_TEST_NUMBER_OF_WRITES) + ", keys per write=" + to_string(SMT64_TEST_KEYS_PER_WRITE) + ", number of keys=" + to_string(SMT64_TEST_NUMBER_OF_KEYS));

    uint64_t numberOfFailedTests = 0;
    Goldilocks::Element root[4] = {0, 0, 0, 0};
    Goldilocks fr;
    PoseidonGoldilocks poseidon;
    zkresult zkr;
    bool bRandomKeys = true;
    bool persistent = true;
    
    Goldilocks::Element keyValue[12];
    for (uint64_t i=0; i<12; i++)
    {
        keyValue[i] = fr.zero();
    }

    HashDBInterface *pHashDB = HashDBClientFactory::createHashDBClient(fr, config);
    if (pHashDB == NULL)
    {
        zklog.error("Smt64Test() failed calling HashDBClientFactory::createHashDBClient()");
        return 1;
    }

    TimerStart(SMT64_TEST_KEYVALUES_GENERATION);

    vector<KeyValue> keyValues[SMT64_TEST_NUMBER_OF_WRITES];

    // Init the keyValues
    for (uint64_t i=0; i<SMT64_TEST_NUMBER_OF_KEYS; i++)
    {
        KeyValue kv;
        if (bRandomKeys)
        {
            keyValue[0] = fr.fromU64(i);
            poseidon.hash(kv.key, keyValue);
        }
        else
        {
            kv.key[0] = fr.fromU64(i);
            kv.key[1] = fr.zero();
            kv.key[2] = fr.zero();
            kv.key[3] = fr.zero();
        }
        kv.value = i;
        keyValues[i/SMT64_TEST_KEYS_PER_WRITE].emplace_back(kv);
    }
    for (uint64_t i=0; i<SMT64_TEST_NUMBER_OF_WRITES; i++)
    {
        zkassertpermanent(keyValues[i].size() == SMT64_TEST_KEYS_PER_WRITE);
    }

    TimerStopAndLog(SMT64_TEST_KEYVALUES_GENERATION);

    // Perform the test, based on the configuration
    if (config.hashDB64)
    {
        TimerStart(SMT64_TEST_WRITE_TREE);

        // Call writeTree()
        for (int64_t i=0; i<SMT64_TEST_NUMBER_OF_WRITES; i++)
        {
            zkr = pHashDB->writeTree(root, keyValues[i], root, persistent);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("Smt64Test() failed calling smt64.writeTree() result=" + zkresult2string(zkr));
                return 1;
            }
            //zklog.info("Smt64Test() smt64.writeTree() returned root=" + fea2string(fr, root) + " flushId=" + to_string(flushId));
        }

        TimerStopAndLog(SMT64_TEST_WRITE_TREE);

        TimerStart(SMT64_TEST_FLUSH);
        
        uint64_t flushId = 0;
        uint64_t lastSentFlushId = 0;
        zkr = pHashDB->flush(emptyString, fea2string(fr, root), PERSISTENCE_DATABASE, flushId, lastSentFlushId);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Smt64Test() failed calling phashDB->flush() result=" + zkresult2string(zkr));
            return 1;
        }

        TimerStopAndLog(SMT64_TEST_FLUSH);

        TimerStart(SMT64_TEST_WAIT_FOR_FLUSH);

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

        TimerStopAndLog(SMT64_TEST_WAIT_FOR_FLUSH);
        
        TimerStart(SMT64_TEST_READ_TREE);

        vector<KeyValue> readKeyValues[SMT64_TEST_NUMBER_OF_WRITES];
        vector<HashValueGL> readHashValues[SMT64_TEST_NUMBER_OF_WRITES];

        for (int64_t i=0; i<SMT64_TEST_NUMBER_OF_WRITES; i++)
        {
            // Make a copy of the key values, overwriting the value with an invalid value
            readKeyValues[i] = keyValues[i];
            zkassertpermanent(readKeyValues[i].size() == SMT64_TEST_KEYS_PER_WRITE);
            for (uint64_t j=0; j<readKeyValues[i].size(); j++)
            {
                readKeyValues[i][j].value = 0xFFFFFFFFFFFFFFFF;
            }

            // Call readTree() with the returned root
            zkr = pHashDB->readTree(root, readKeyValues[i], readHashValues[i]);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("Smt64Test() failed calling smt64.readTree() result=" + zkresult2string(zkr));
                return 1;
            }
        }

        TimerStopAndLog(SMT64_TEST_READ_TREE);

        TimerStart(SMT64_TEST_CHECK_READ_TREE_RESULT);

        // Check that both keys and values match the ones we wrote
        for (uint64_t i=0; i<SMT64_TEST_NUMBER_OF_WRITES; i++)
        {
            for (uint64_t j=0; j<SMT64_TEST_KEYS_PER_WRITE; j++)
            {
                // Check that the key is still the same
                if ( !fr.equal(readKeyValues[i][j].key[0], keyValues[i][j].key[0]) ||
                     !fr.equal(readKeyValues[i][j].key[1], keyValues[i][j].key[1]) ||
                     !fr.equal(readKeyValues[i][j].key[2], keyValues[i][j].key[2]) ||
                     !fr.equal(readKeyValues[i][j].key[3], keyValues[i][j].key[3]) )
                {
                    zklog.error("Smt64Test() read different key at i=" + to_string(i) + " read=" + fea2string(fr, readKeyValues[i][j].key) + " expected=" + fea2string(fr, keyValues[i][j].key));
                    numberOfFailedTests++;
                }

                // Check that the value is the expected one
                if (readKeyValues[i][j].value != keyValues[i][j].value)
                {
                    zklog.error("Smt64Test() read different key at i=" + to_string(i) + " read=" + readKeyValues[i][j].value.get_str(10) + " expected=" + keyValues[i][j].value.get_str(10));
                    numberOfFailedTests++;
                }
            }
        }

        TimerStopAndLog(SMT64_TEST_CHECK_READ_TREE_RESULT);

        // Announce the success
        if (numberOfFailedTests==0)
        {
            zklog.info("Smt64Test() succeeded");
        }
    }
    else
    {
        for (uint64_t i=0; i<SMT64_TEST_NUMBER_OF_WRITES; i++)
        {
            for (uint64_t j=0; j<SMT64_TEST_KEYS_PER_WRITE; j++)
            {
                zkr = pHashDB->set("", 0, root, keyValues[i][j].key, keyValues[i][j].value, PERSISTENCE_DATABASE, root, NULL, NULL);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("Smt64Test() failed calling pHashDB->set() result=" + zkresult2string(zkr));
                    return 1;
                }
                zklog.info("Smt64Test() pHashDB->set() i=" + to_string(i) + " returned root=" + fea2string(fr, root));
            }
        }
    }

    TimerStopAndLog(SMT64_TEST);

    return numberOfFailedTests;
}