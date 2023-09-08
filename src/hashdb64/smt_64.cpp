#include "smt_64.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "zkresult.hpp"
#include "zkmax.hpp"
#include "zklog.hpp"
#include <bitset>
#include "state_manager_64.hpp"
#include "key_utils.hpp"
#include "tree_chunk.hpp"

//#define SMT64_PRINT_TREE_CHUNKS


zkresult Smt64::set (const string &batchUUID, uint64_t tx, Database64 &db, const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const Persistence persistence, SmtSetResult &result, DatabaseMap *dbReadLog)
{
#ifdef LOG_SMT
    zklog.info("Smt64::set() called with oldRoot=" + fea2string(fr,oldRoot) + " key=" + fea2string(fr,key) + " value=" + value.get_str(16) + " persistent=" + to_string(persistent));
#endif

    zkresult zkr;

    bool bUseStateManager = db.config.stateManager && (batchUUID.size() > 0);

    if (bUseStateManager)
    {
        // Set the old state root
        string oldRootString;
        oldRootString = fea2string(fr, oldRoot);
        stateManager64.setOldStateRoot(batchUUID, tx, oldRootString, persistence);

        // Write the key-value pair
        string hashString = NormalizeToNFormat(fea2string(fr, key), 64);
        zkr = stateManager64.write(batchUUID, tx, hashString, value, persistence);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Smt64::set() failed calling stateManager.write() key=" + hashString + " result=" + to_string(zkr) + "=" + zkresult2string(zkr));
        }

        // Get a new state root
        Goldilocks::Element newRoot[4]; // TODO: Get a new state root
        string newRootString;
        stateManager64.getVirtualStateRoot(newRoot, newRootString);

        // Set the new sttae root
        stateManager64.setNewStateRoot(batchUUID, tx, newRootString, persistence);

        result.newRoot[0] = newRoot[0];
        result.newRoot[1] = newRoot[1];
        result.newRoot[2] = newRoot[2];
        result.newRoot[3] = newRoot[3];
    }
    else
    {
        // TODO: implementation
    }

    return ZKR_SUCCESS;
}

zkresult Smt64::get (const string &batchUUID, Database64 &db, const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], SmtGetResult &result, DatabaseMap *dbReadLog)
{
#ifdef LOG_SMT
    zklog.info("Smt64::get() called with root=" + fea2string(fr,root) + " and key=" + fea2string(fr,key));
#endif

    bool bUseStateManager = db.config.stateManager && (batchUUID.size() > 0);

    // Read the content of db for entry r: siblings[level] = db.read(r)
    string keyString = fea2string(fr, key);
    mpz_class value;
    zkresult zkr = ZKR_UNSPECIFIED;
    if (bUseStateManager)
    {
        zkr = stateManager64.read(batchUUID, keyString, value, dbReadLog);
    }
    if (zkr != ZKR_SUCCESS)
    {
        zkr = db.readKV(root, key, value, dbReadLog);
    }
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Smt64::get() db.read error: " + to_string(zkr) + " (" + zkresult2string(zkr) + ") root:" + fea2string(fr, root));
        return zkr;
    }
    
    result.value = value;

    return ZKR_SUCCESS;
}