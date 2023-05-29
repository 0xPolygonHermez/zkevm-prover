#include <thread>
#include <unistd.h>
#include "../../../src/service/hashdb/hashdb_factory.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "timer.hpp"

#define HASH_DB_TEST_MULTI_WRITE_NUMBER_OF_TREES 10
#define HASH_DB_TEST_MULTI_WRITE_NUMBER_OF_NODES 1000

void runHashDBTestMultiWrite (const Config& config)
{
    Goldilocks fr;
    PoseidonGoldilocks poseidon;
    HashDBInterface * pHashDB = HashDBClientFactory::createHashDBClient(fr,config);
    uint64_t flushId, storedFlushId;

    TimerStart(HASH_DB_TEST_MULTI_WRITE);

    Goldilocks::Element root[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};
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
            zkresult zkr = pHashDB->set(root, key, value, true, root, NULL, NULL );
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("runHashDBTest() set tree=" + to_string(tree) + " i=" + to_string(i) + " result=" + zkresult2string(zkr));
                exitProcess();
            }
        }

        zklog.info("runHashDBTest() after tree=" + to_string(tree) + " root=" + fea2string(fr, root));
    }

    TimerStopAndLog(HASH_DB_TEST_MULTI_WRITE);

    pHashDB->flush(flushId, storedFlushId);
    do
    {
        uint64_t storingFlushId, lastFlushId, pendingToFlushNodes, pendingToFlushProgram, storingNodes, storingProgram;
        string proverId;
        pHashDB->getFlushStatus(storedFlushId, storingFlushId, lastFlushId, pendingToFlushNodes, pendingToFlushProgram, storingNodes, storingProgram, proverId);
        sleep(1);
    } while (storedFlushId < flushId);

    pHashDB->clearCache();

    tree = HASH_DB_TEST_MULTI_WRITE_NUMBER_OF_TREES - 1;
    for (uint64_t i=0; i<HASH_DB_TEST_MULTI_WRITE_NUMBER_OF_NODES; i++)
    {
        // Build the expected value
        mpz_class expectedValue;
        expectedValue = tree + 10;
        expectedValue = ScalarTwoTo64*expectedValue + tree + 10;
        expectedValue = ScalarTwoTo64*expectedValue + tree + 10;
        expectedValue = ScalarTwoTo64*expectedValue + tree + 10;

        // Build the key
        keyValue[0] = fr.fromU64(i);
        poseidon.hash(key, keyValue);

        // Read the value
        mpz_class readValue;
        zkresult zkr = pHashDB->get(root, key, readValue, NULL, NULL );
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("runHashDBTest() get i=" + to_string(i) + " result=" + zkresult2string(zkr));
            exitProcess();
        }

        // Check against the expected value
        if (readValue != expectedValue)
        {
            zklog.error("runHashDBTest() found readValue=" + readValue.get_str(16) + " different from expected value=" + expectedValue.get_str(16));
            exitProcess();
        }
        //zklog.info("runHashDBTest() verified i=" + to_string(i));
    }
}

