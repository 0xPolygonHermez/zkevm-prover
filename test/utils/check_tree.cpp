#include "check_tree.hpp"
#include "zkmax.hpp"
#include "scalar.hpp"

zkresult CheckTree (Database &db, const string &key, uint64_t level, CheckTreeCounters &checkTreeCounters, const string &prefix)
{
    zklog.info(prefix + "CheckTree() hash=" + key + " level=" + to_string(level));

    checkTreeCounters.maxLevel = zkmax(checkTreeCounters.maxLevel, level);

    vector<Goldilocks::Element> value;
    Goldilocks::Element vKey[4];
    if(db.useAssociativeCache) string2fea(db.fr, key, vKey);
    zkresult result = db.read(key, vKey, value, NULL, false);
    if (result != ZKR_SUCCESS)
    {
        zklog.error(prefix + "CheckTree() failed key=" + key + " level=" + to_string(level));
        return result;
    }
    if (value.size() != 12)
    {
        zklog.error(prefix + "CheckTree() invalid value.size=" + to_string(value.size()) + " key=" + key + " level=" + to_string(level));
        return ZKR_UNSPECIFIED;
    }

    if (!db.fr.isZero(value[9]))
    {
        zklog.error(prefix + "CheckTree() fe9 not zero value.size=" + to_string(value.size()) + " key=" + key + " level=" + to_string(level) + " fe9=" + db.fr.toString(value[9],16));
        return ZKR_UNSPECIFIED;
    }

    if (!db.fr.isZero(value[10]))
    {
        zklog.error("prefix + CheckTree() fe10 not zero value.size=" + to_string(value.size()) + " key=" + key + " level=" + to_string(level) + " fe10=" + db.fr.toString(value[10],16));
        return ZKR_UNSPECIFIED;
    }

    if (!db.fr.isZero(value[11]))
    {
        zklog.error(prefix + "CheckTree() fe11 not zero value.size=" + to_string(value.size()) + " key=" + key + " level=" + to_string(level) + " fe11=" + db.fr.toString(value[11],16));
        return ZKR_UNSPECIFIED;
    }

    uint64_t fe8 = db.fr.toU64(value[8]);

    if (fe8 == 0) // Intermediate node
    {
        checkTreeCounters.intermediateNodes++;

        if (!feaIsZero(value[0], value[1], value[2], value[3]))
        {
            string hashLeft = fea2string(db.fr, value[0], value[1], value[2], value[3]);
            zklog.info(prefix + "CheckTree() hashLeft=" + hashLeft + " level=" + to_string(level));
            result = CheckTree(db, hashLeft, level+1, checkTreeCounters, prefix + " ");
            if (result != ZKR_SUCCESS)
            {
                return result;
            }
        }

        if (!feaIsZero(value[4], value[5], value[6], value[7]))
        {
            string hashRight = fea2string(db.fr, value[4], value[5], value[6], value[7]);
            zklog.info(prefix + "CheckTree() hashRight=" + hashRight + " level=" + to_string(level));
            result = CheckTree(db, hashRight, level+1, checkTreeCounters, prefix + " ");
            if (result != ZKR_SUCCESS)
            {
                return result;
            }
        }

        return ZKR_SUCCESS;
    }
    else if (fe8 == 1) // Leaf node
    {
        checkTreeCounters.leafNodes++;

        //level++;
        string valueHash = fea2string(db.fr, value[4], value[5], value[6], value[7]);
        Goldilocks::Element vHash[4]={value[4], value[5], value[6], value[7]};
        zklog.info(prefix + "CheckTree() rkey=" + fea2string(db.fr, value[0], value[1], value[2], value[3]));
        zklog.info(prefix + "CheckTree() valueHash=" + valueHash);
        value.clear();
        zkresult result = db.read(valueHash, vHash, value, NULL, false);
        if (result != ZKR_SUCCESS)
        {
            zklog.error(prefix + "CheckTree() failed key=" + valueHash + " level=" + to_string(level));
            return result;
        }
        if (value.size() != 12)
        {
            zklog.error(prefix + "CheckTree() found value for key=" + valueHash + " at level=" + to_string(level) + " with incorrect size=" + to_string(value.size()));
            return ZKR_DB_ERROR;
        }
        string valueString;
        for (uint64_t i=0; i<value.size(); i++)
        {
            if (i != 0)
            {
                valueString += ':';
            }
            valueString += fr.toString(value[i], 16);
        }
        zklog.info(prefix + "CheckTree() value=" + valueString + " level=" + to_string(level));

        checkTreeCounters.maxLevel = zkmax(checkTreeCounters.maxLevel, level);
        checkTreeCounters.values++;
        if (checkTreeCounters.values % 1000 == 0)
        {
            zklog.info(prefix + "CheckTree() intermediateNodes=" + to_string(checkTreeCounters.intermediateNodes) + " leafNodes=" + to_string(checkTreeCounters.leafNodes) + " values=" + to_string(checkTreeCounters.values) + " maxLevel=" + to_string(checkTreeCounters.maxLevel));
        }
        return ZKR_SUCCESS;
    }
    else
    {
        zklog.error(prefix + "CheckTree() failed key=" + key + " level=" + to_string(level) + " invalid fe8=" + to_string(fe8));
        return ZKR_UNSPECIFIED;
    }
};