#include "check_tree_test.hpp"
#include "exit_process.hpp"
#include "zklog.hpp"
#include "hashdb_singleton.hpp"
#include "scalar.hpp"
#include "check_tree.hpp"
#include "timer.hpp"

uint64_t CheckTreeTest (Config &config)
{
    TimerStart(CHECK_TREE_TEST);

    HashDB *pHashDB = hashDBSingleton.get();
    Database &db = pHashDB->db;
    zkresult zkr;

    string root = config.checkTreeRoot;

    if (root == "auto")
    {
        vector<Goldilocks::Element> value;
        zkr = db.read(db.dbStateRootKey, db.dbStateRootvKey, value, NULL, true);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("CheckTreeTest() failed calling db.read(stateRootKey) zkr=" + zkresult2string(zkr));
            exitProcess();
        } 
        root = fea2string(db.fr, value[0], value[1], value[2], value[3]);
    }
    zklog.info("CheckTreeTest() going to check tree of root=" + root);

    CheckTreeCounters checkTreeCounters;

    zkresult result = CheckTree(db, root, 0, checkTreeCounters);
    if (result != ZKR_SUCCESS)
    {
        zklog.error("CheckTreeTest() failed calling ClimbTree() result=" + zkresult2string(result));
        return 1;
    }

    zklog.info("intermediateNodes=" + to_string(checkTreeCounters.intermediateNodes));
    zklog.info("leafNodes=" + to_string(checkTreeCounters.leafNodes));
    zklog.info("values=" + to_string(checkTreeCounters.values));
    zklog.info("maxLevel=" + to_string(checkTreeCounters.maxLevel));

    TimerStopAndLog(CHECK_TREE_TEST);

    return 0;
}