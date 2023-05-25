#include <thread>
#include <unistd.h>
#include "../../../src/service/hashdb/hashdb_factory.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "timer.hpp"

void runHashDBTestMultiWrite (const Config& config)
{
    Goldilocks fr;
    HashDBInterface * pHashDB = HashDBClientFactory::createHashDBClient(fr,config);
    uint64_t flushId, storedFlushId;

    TimerStart(HASH_DB_TEST_MULTI_WRITE);

    Goldilocks::Element root[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};
    uint64_t tree;
    bool bRandomKeys = true;
    bool bRandomValues = false;
    for (tree=0; tree<10; tree++)
    {
        for (uint64_t i=0; i<1000; i++)
        {
            
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
            Goldilocks::Element key[4];
            if ((tree == 9) || !bRandomKeys)
            {
                key[0] = fr.fromU64(i);
                key[1] = fr.zero();
                key[2] = fr.zero();
                key[3] = fr.zero();
            }
            else
            {
                key[0] = fr.fromU64(uint64_t(random())*uint64_t(random()));
                key[1] = fr.fromU64(uint64_t(random())*uint64_t(random()));
                key[2] = fr.fromU64(uint64_t(random())*uint64_t(random()));
                key[3] = fr.fromU64(uint64_t(random())*uint64_t(random()));
            }

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

    tree = 9;
    for (uint64_t i=0; i<1000; i++)
    {
        
        mpz_class value;
        value = tree + 10;
        value = ScalarTwoTo64*value + tree + 10;
        value = ScalarTwoTo64*value + tree + 10;
        value = ScalarTwoTo64*value + tree + 10;
        Goldilocks::Element key[4];
        key[0] = fr.fromU64(i);
        key[1] = fr.zero();
        key[2] = fr.zero();
        key[3] = fr.zero();
        mpz_class readValue;

        zkresult zkr = pHashDB->get(root, key, readValue, NULL, NULL );
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("runHashDBTest() get i=" + to_string(i) + " result=" + zkresult2string(zkr));
            exitProcess();
        }
        if (readValue != value)
        {
            zklog.error("runHashDBTest() found readValue=" + readValue.get_str(16) + " different from expected value=" + value.get_str(16));
            exitProcess();
        }
        //zklog.info("runHashDBTest() verified i=" + to_string(i));
    }
}

