#include "smt_64_test.hpp"
#include "smt_64.hpp"
#include "smt.hpp"
#include "hashdb_singleton.hpp"
#include "unistd.h"

#define SMT64_TEST_NUMBER_OF_KEYS 10

uint64_t Smt64Test (const Config &config)
{
    Goldilocks::Element root[4] = {0, 0, 0, 0};
    Goldilocks fr;
    zkresult zkr;
    vector<KeyValue> keyValues;
    vector<Key> keys;

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
        keys.emplace_back(kv.key);
    }

    // Perform the test, based on the configuration
    if (config.hashDB64)
    {
        Smt64 smt64(fr);
        Database64 db64(fr, config);
        db64.init();
        uint64_t flushId = 0;
        uint64_t lastSentFlushId = 0;

        zkr = smt64.writeTree(db64, root, keyValues, root, flushId, lastSentFlushId);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Smt64Test() failed calling smt64.writeTree() result=" + zkresult2string(zkr));
            return 1;
        }
        zklog.info("Smt64Test() smt64.writeTree() returned root=" + fea2string(fr, root));
    }
    else
    {
        Smt smt(fr);
        Database db(fr, config);
        db.init();

        for (uint64_t i=0; i<SMT64_TEST_NUMBER_OF_KEYS; i++)
        {
            SmtSetResult smtSetResult;
            zkr = smt.set("", 0, db, root, keyValues[i].key.fe, keyValues[i].value, PERSISTENCE_DATABASE, smtSetResult);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("Smt64Test() failed calling smt.set() result=" + zkresult2string(zkr));
                return 1;
            }
            root[0] = smtSetResult.newRoot[0];
            root[1] = smtSetResult.newRoot[1];
            root[2] = smtSetResult.newRoot[2];
            root[3] = smtSetResult.newRoot[3];
            zklog.info("Smt64Test() smt.set() i=" + to_string(i) + " returned root=" + fea2string(fr, root));
        }
    }

    /*vector<KeyValue> readKeyValues;
    zkr = smt64.readTree(db64, root, keys, readKeyValues);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Smt64Test() failed calling smt64.readTree() result=" + zkresult2string(zkr));
        return 1;
    }*/

    //sleep(2);

    return 0;
}