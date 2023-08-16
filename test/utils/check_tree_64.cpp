#include "check_tree_64.hpp"
#include "zkmax.hpp"
#include "scalar.hpp"
#include "tree_chunk.hpp"

zkresult CheckTree64 (Database64 &db, const string &key, uint64_t level, CheckTreeCounters64 &checkTreeCounters)
{
    checkTreeCounters.maxLevel = zkmax(checkTreeCounters.maxLevel, level);

    TreeChunk treeChunk(db);
    string2key(db.fr, key, treeChunk.hash);
    zkresult result = db.read(key, treeChunk.hash, treeChunk.data, NULL, false);
    if (result != ZKR_SUCCESS)
    {
        zklog.error("CheckTree() failed key=" + key + " level=" + to_string(level));
        return result;
    }

    zkresult zkr;
    zkr = treeChunk.data2children();
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("CheckTree() vailed calling treeChunk.data2children() zkr=" + zkresult2string(zkr) + " key=" + key + " level=" + to_string(level));
        return zkr;
    }

    for (uint64_t i=0; i<TREE_CHUNK_WIDTH; i++)
    {
        if (treeChunk.children[i].type == ZERO) // Zero
        {
            continue;
        }
        else if (treeChunk.children[i].type == INTERMEDIATE) // Intermediate node
        {
            checkTreeCounters.intermediateNodes++;
            result = CheckTree64(db, fea2string(db.fr, treeChunk.children[i].intermediate.hash), level+1, checkTreeCounters);
            if (zkr != ZKR_SUCCESS)
            {
                return zkr;
            }
        }
        else if (treeChunk.children[i].type == LEAF) // Leaf node
        {
            checkTreeCounters.leafNodes++;
            checkTreeCounters.maxLevel = zkmax(checkTreeCounters.maxLevel, level);
            checkTreeCounters.values++;
            if (checkTreeCounters.values % 1000 == 0)
            {
                zklog.info("CheckTree() intermediateNodes=" + to_string(checkTreeCounters.intermediateNodes) + " leafNodes=" + to_string(checkTreeCounters.leafNodes) + " values=" + to_string(checkTreeCounters.values) + " maxLevel=" + to_string(checkTreeCounters.maxLevel));
            }
            return ZKR_SUCCESS;
        }
        else
        {
            zklog.error("CheckTree() failed key=" + key + " level=" + to_string(level) + " invalid type=" + to_string(treeChunk.children[i].type));
            exitProcess();
        }
    }

    return ZKR_SUCCESS;
}