#include <thread>
#include <unistd.h>
#include "../../../src/service/hashdb/hashdb_factory.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "timer.hpp"

#define HASH_DB_TEST_MULTI_WRITE_NUMBER_OF_TREES 10
#define HASH_DB_TEST_MULTI_WRITE_NUMBER_OF_NODES 1000

uint64_t HashDBTestMultiWrite (const Config& config)
{
    uint64_t numberOfFailedTests = 0;

    string uuid = getUUID();
    uint64_t tx = 0;
    Goldilocks fr;
    PoseidonGoldilocks poseidon;
    HashDBInterface * pHashDB = HashDBClientFactory::createHashDBClient(fr,config);
    uint64_t flushId, storedFlushId;

    TimerStart(HASH_DB_TEST_MULTI_WRITE);

    Goldilocks::Element root[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};
    Goldilocks::Element roots[HASH_DB_TEST_MULTI_WRITE_NUMBER_OF_TREES][4];
    uint64_t tree;
    bool bRandomKeys = false;
    bool bRandomValues = false;

    Goldilocks::Element keyValue[12];
    for (uint64_t i=0; i<12; i++)
    {
        keyValue[i] = fr.zero();
    }
    Goldilocks::Element key[4];

    for (tree=0; tree<HASH_DB_TEST_MULTI_WRITE_NUMBER_OF_TREES; tree++)
    {
        for (uint64_t i=0; i<HASH_DB_TEST_MULTI_WRITE_NUMBER_OF_NODES; i++)
        {
            // Build the value
            mpz_class value;
            if (bRandomValues)
            {
                value =                       uint64_t(random())*uint64_t(random());
                value = ScalarTwoTo64*value + uint64_t(random())*uint64_t(random());
                value = ScalarTwoTo64*value + uint64_t(random())*uint64_t(random());
                value = ScalarTwoTo64*value + uint64_t(random())*uint64_t(random());
            }
            else
            {
                value = tree + 10;
                value = ScalarTwoTo64*value + tree + 10;
                value = ScalarTwoTo64*value + tree + 10;
                value = ScalarTwoTo64*value + tree + 10;
            }

            // Build the key
            if ((tree == (HASH_DB_TEST_MULTI_WRITE_NUMBER_OF_TREES - 1)) || !bRandomKeys)
            {
                keyValue[0] = fr.fromU64(i);
                poseidon.hash(key, keyValue);
            }
            else
            {
                key[0] = fr.fromU64(uint64_t(random())*uint64_t(random()));
                key[1] = fr.fromU64(uint64_t(random())*uint64_t(random()));
                key[2] = fr.fromU64(uint64_t(random())*uint64_t(random()));
                key[3] = fr.fromU64(uint64_t(random())*uint64_t(random()));
                //key[0] = fr.fromU64(5);
                //key[1] = fr.fromU64(i);
                //key[2] = fr.fromU64(i);
                //key[3] = fr.fromU64(i);
            }

            //zklog.info("runHashDBTest() calling set with root=" + fea2string(fr, root) + " key=" + fea2string(fr, key) + " value=" + value.get_str(16));

            // Set the value
            //zklog.info("HashDBTestMultiWrite() tree=" + to_string(tree) + " i=" + to_string(i) + " calling pHashDB->set with root=" + fea2string(fr, root) + " key=" + fea2string(fr,key));
            zkresult zkr = pHashDB->set(uuid, tx, root, key, value, PERSISTENCE_DATABASE, root, NULL, NULL );
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("HashDBTestMultiWrite() set tree=" + to_string(tree) + " i=" + to_string(i) + " result=" + zkresult2string(zkr));
                exitProcess();
            }
            //zklog.info("HashDBTestMultiWrite() tree=" + to_string(tree) + " i=" + to_string(i) + " called pHashDB->set and got root=" + fea2string(fr, root) + " key=" + fea2string(fr,key));
        }

        roots[tree][0] = root[0];
        roots[tree][1] = root[1];
        roots[tree][2] = root[2];
        roots[tree][3] = root[3];

        pHashDB->flush(uuid, fea2string(fr, root), PERSISTENCE_DATABASE, flushId, storedFlushId);

        zklog.info("HashDBTestMultiWrite() after tree=" + to_string(tree) + " root=" + fea2string(fr, root) + " flushId=" + to_string(flushId) + " storedFlushId=" + to_string(storedFlushId));
    }

    TimerStopAndLog(HASH_DB_TEST_MULTI_WRITE);

    do
    {
        uint64_t storingFlushId, lastFlushId, pendingToFlushNodes, pendingToFlushProgram, storingNodes, storingProgram;
        string proverId;
        pHashDB->getFlushStatus(storedFlushId, storingFlushId, lastFlushId, pendingToFlushNodes, pendingToFlushProgram, storingNodes, storingProgram, proverId);
        zklog.info("HashDBTestMultiWrite() after getFlushStatus() flushId=" + to_string(flushId) + " storedFlushId=" + to_string(storedFlushId));
        sleep(1);
    } while (storedFlushId < flushId);

    // All data has been stored on database, so we can clear the cache
    pHashDB->clearCache();

    for (tree=0; tree<HASH_DB_TEST_MULTI_WRITE_NUMBER_OF_TREES; tree++)
    {
        for (uint64_t i=0; i<HASH_DB_TEST_MULTI_WRITE_NUMBER_OF_NODES; i++)
        {
            // Build the expected value
            mpz_class expectedValue;
            if (!bRandomValues)
            {
                expectedValue = tree + 10;
                expectedValue = ScalarTwoTo64*expectedValue + tree + 10;
                expectedValue = ScalarTwoTo64*expectedValue + tree + 10;
                expectedValue = ScalarTwoTo64*expectedValue + tree + 10;
            }

            // Build the key
            keyValue[0] = fr.fromU64(i);
            poseidon.hash(key, keyValue);

            // Read the value
            mpz_class readValue;
            zkresult zkr = pHashDB->get(uuid, roots[tree], key, readValue, NULL, NULL );
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("HashDBTestMultiWrite() get i=" + to_string(i) + " result=" + zkresult2string(zkr));
                exitProcess();
            }

            // Check against the expected value
            if (!bRandomValues && (readValue != expectedValue))
            {
                zklog.error("HashDBTestMultiWrite() found readValue=" + readValue.get_str(16) + " different from expected value=" + expectedValue.get_str(16));
                exitProcess();
            }
            //zklog.info("runHashDBTest() verified i=" + to_string(i));
        }
        zklog.info("HashDBTestMultiWrite() verified tree=" + to_string(tree));
    }

    return numberOfFailedTests;
}

