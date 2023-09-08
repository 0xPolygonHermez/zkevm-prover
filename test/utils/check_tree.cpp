#include "check_tree.hpp"
#include "zkmax.hpp"
#include "scalar.hpp"

zkresult CheckTree (Database &db, const string &key, uint64_t level, CheckTreeCounters &checkTreeCounters)
{
    checkTreeCounters.maxLevel = zkmax(checkTreeCounters.maxLevel, level);

    vector<Goldilocks::Element> value;
    Goldilocks::Element vKey[4];
    if(db.useAssociativeCache) string2fea(db.fr, key, vKey);
    zkresult result = db.read(key, vKey, value, NULL, false);
    if (result != ZKR_SUCCESS)
    {
        zklog.error("CheckTree() failed key=" + key + " level=" + to_string(level));
        return result;
    }
    if (value.size() != 12)
    {
        zklog.error("CheckTree() invalid value.size=" + to_string(value.size()) + " key=" + key + " level=" + to_string(level));
        return ZKR_UNSPECIFIED;
    }

    if (!db.fr.isZero(value[9]))
    {
        zklog.error("CheckTree() fe9 not zero value.size=" + to_string(value.size()) + " key=" + key + " level=" + to_string(level) + " fe9=" + db.fr.toString(value[9],16));
        return ZKR_UNSPECIFIED;
    }

    if (!db.fr.isZero(value[10]))
    {
        zklog.error("CheckTree() fe10 not zero value.size=" + to_string(value.size()) + " key=" + key + " level=" + to_string(level) + " fe10=" + db.fr.toString(value[10],16));
        return ZKR_UNSPECIFIED;
    }

    if (!db.fr.isZero(value[11]))
    {
        zklog.error("CheckTree() fe11 not zero value.size=" + to_string(value.size()) + " key=" + key + " level=" + to_string(level) + " fe11=" + db.fr.toString(value[11],16));
        return ZKR_UNSPECIFIED;
    }

    uint64_t fe8 = db.fr.toU64(value[8]);

    if (fe8 == 0) // Intermediate node
    {
        checkTreeCounters.intermediateNodes++;

        string hashLeft = fea2string(db.fr, value[0], value[1], value[2], value[3]);
        if (hashLeft == "0")
        {
            return ZKR_SUCCESS;
        }
        result = CheckTree(db, hashLeft, level+1, checkTreeCounters);
        if (result != ZKR_SUCCESS)
        {
            return result;
        }
        string hashRight = fea2string(db.fr, value[4], value[5], value[6], value[7]);
        if (hashRight == "0")
        {
            return ZKR_SUCCESS;
        }
        result = CheckTree(db, hashRight, level+1, checkTreeCounters);
        return result;
    }
    else if (fe8 == 1) // Leaf node
    {
        checkTreeCounters.leafNodes++;

        level++;
        string valueHash = fea2string(db.fr, value[4], value[5], value[6], value[7]);
        Goldilocks::Element vHash[4]={value[4], value[5], value[6], value[7]};
        value.clear();
        zkresult result = db.read(valueHash, vHash, value, NULL, false);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("CheckTree() failed key=" + valueHash + " level=" + to_string(level));
            return result;
        }
        if (value.size() != 12)
        {
            zklog.error("CheckTree() found value for key=" + valueHash + " at level=" + to_string(level) + " with incorrect size=" + to_string(value.size()));
            /*zklog.error("valueL=" + fea2string(db.fr, value[0], value[1], value[2], value[3]));
            zklog.error("valueH=" + fea2string(db.fr, value[4], value[5], value[6], value[7]));
            PoseidonGoldilocks poseidon;
            Goldilocks::Element valueFea[12];
            valueFea[0] = value[0];
            valueFea[1] = value[1];
            valueFea[2] = value[2];
            valueFea[3] = value[3];
            valueFea[4] = value[4];
            valueFea[5] = value[5];
            valueFea[6] = value[6];
            valueFea[7] = value[7];
            valueFea[8] = db.fr.zero();
            valueFea[9] = db.fr.zero();
            valueFea[10] = db.fr.zero();
            valueFea[11] = db.fr.zero();
            Goldilocks::Element hashFea[4];
            poseidon.hash(hashFea, valueFea);
            zklog.info("poseidon=" + fea2string(db.fr, hashFea));*/
            //return ZKR_UNSPECIFIED;
        }
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
        zklog.error("CheckTree() failed key=" + key + " level=" + to_string(level) + " invalid fe8=" + to_string(fe8));
        return ZKR_UNSPECIFIED;
    }
}