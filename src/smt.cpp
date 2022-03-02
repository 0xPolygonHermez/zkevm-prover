#include "smt.hpp"
#include "scalar.hpp"
#include "utils.hpp"

void Smt::set (RawFr &fr, Database &db, RawFr::Element &oldRoot, RawFr::Element &key, mpz_class &value, SmtSetResult &result)
{
    RawFr::Element r(oldRoot);
    vector <uint64_t> keys;
    splitKey(fr, key, keys);
    int64_t level = 0;
    mpz_class accKey = 0;
    mpz_class lastAccKey = 0;
    RawFr::Element foundKey(fr.zero());
    map< uint64_t, vector<RawFr::Element> > siblings;

    RawFr::Element insKey(fr.zero());
    mpz_class insValue = 0;
    mpz_class oldValue = 0;
    string mode;
    RawFr::Element newRoot(oldRoot);
    bool isOld0 = true;

    while ( (!fr.isZero(r)) && (fr.isZero(foundKey)) )
    {
        vector<RawFr::Element> dbValue;
        db.read(r, dbValue);
        if (dbValue.size()==0)
        {
            cerr << "Error: Smt::set() could not find key in database: " << fr.toString(r, 16) << endl;
            exit(-1);
        }
        siblings[level] = dbValue;

        if (!siblings[level].empty() && fr.eq(siblings[level][0], fr.one())) {
            mpz_class auxMpz;
            auxMpz = 1;
            auxMpz = auxMpz << level*arity;
            RawFr::Element shiftFe;
            scalar2fe(fr, auxMpz, shiftFe);
            RawFr::Element mulFe;
            fr.mul(mulFe, siblings[level][1], shiftFe);
            RawFr::Element accKeyFe;
            scalar2fe(fr, accKey, accKeyFe);
            fr.add(foundKey, accKeyFe, mulFe);
        } else {
            r = siblings[level][keys[level]];
            lastAccKey = accKey;
            mpz_class auxScalar;
            auxScalar = keys[level];
            accKey = accKey + (auxScalar << level*arity);
            level++;
        }
    }

    level--;
    accKey = lastAccKey;

    if (value != 0)
    {
        RawFr::Element v0, v1, v2, v3;
        scalar2fea(fr, value, v0, v1, v2, v3);
        if (!fr.isZero(foundKey))
        {
            if (fr.eq(key, foundKey)) // Update
            {
                mode = "update";

                /* Prepare the vector of field elements */
                vector<RawFr::Element> newLeaf;
                newLeaf.push_back(fr.one());
                newLeaf.push_back(siblings[level+1][1]);
                fea2scalar(fr, oldValue, siblings[level+1][2], siblings[level+1][3], siblings[level+1][4], siblings[level+1][5]);
                newLeaf.push_back(v0);
                newLeaf.push_back(v1);
                newLeaf.push_back(v2);
                newLeaf.push_back(v3);
                while (newLeaf.size() < (uint64_t(1)<<arity)) newLeaf.push_back(fr.zero());

                /* Call Poseidon hash function */
                RawFr::Element newLeafHash;
                hashSave(fr, db, newLeaf, newLeafHash);

                /* Process the resulting hash */
                if ( level >= 0 ){
                    siblings[level][keys[level]] = newLeafHash;
                } else {
                    newRoot = newLeafHash;
                }
            }
            else // insert with foundKey
            {
                mode = "insertFound";
                vector<RawFr::Element> node;
                int64_t level2 = level + 1;
                vector <uint64_t> foundKeys;
                splitKey(fr, foundKey, foundKeys);
                while (keys[level2] == foundKeys[level2]) level2++;

                vector<RawFr::Element> oldLeaf;
                oldLeaf.push_back(fr.one());
                mpz_class auxScalar;
                fe2scalar(fr, auxScalar, foundKey);
                auxScalar = auxScalar >> ((level2+1)*arity);
                RawFr::Element auxFe;
                scalar2fe(fr, auxScalar, auxFe);

                oldLeaf.push_back(auxFe);
                oldLeaf.push_back(siblings[level+1][2]);
                oldLeaf.push_back(siblings[level+1][3]);
                oldLeaf.push_back(siblings[level+1][4]);
                oldLeaf.push_back(siblings[level+1][5]);

                insKey = foundKey;
                fea2scalar(fr, insValue, siblings[level+1][2], siblings[level+1][3], siblings[level+1][4], siblings[level+1][5]);
                isOld0 = false;
                while (oldLeaf.size() < (uint64_t(1)<<arity)) oldLeaf.push_back(fr.zero());
                RawFr::Element oldLeafHash;
                hashSave(fr, db, oldLeaf, oldLeafHash);

                vector<RawFr::Element> newLeaf;
                newLeaf.push_back(fr.one());
                fe2scalar(fr, auxScalar, key);
                auxScalar = auxScalar >> ((level2+1)*arity);
                scalar2fe(fr, auxScalar, auxFe);

                newLeaf.push_back(auxFe);
                newLeaf.push_back(v0);
                newLeaf.push_back(v1);
                newLeaf.push_back(v2);
                newLeaf.push_back(v3);
                while(newLeaf.size() < (uint64_t(1)<<arity)) newLeaf.push_back(fr.zero());
                RawFr::Element newLeafHash;
                hashSave(fr, db, newLeaf, newLeafHash);

                while (node.size() < (uint64_t(1)<<arity)) node.push_back(fr.zero());
                node[keys[level2]] = newLeafHash;
                node[foundKeys[level2]] = oldLeafHash;

                RawFr::Element r2;
                hashSave(fr, db, node, r2);
                level2--;

                while (level2 != level)
                {
                    for (uint64_t i=0; i<(uint64_t(1)<<arity); i++) node[i] = fr.zero();
                    node[keys[level2]] = r2;

                    hashSave(fr, db, node, r2);
                    level2--;
                }

                if (level>=0) {
                    siblings[level][keys[level]] = r2;
                } else {
                    newRoot = r2;
                }
            }
        }
        else // insert without foundKey
        {
            mode = "insertNotFound";

            RawFr::Element auxFe;

            vector<RawFr::Element> newLeaf;
            newLeaf.push_back(fr.one());
            mpz_class auxScalar;
            fe2scalar(fr, auxScalar, key);
            auxScalar = auxScalar >> ((level+1)*arity);
            scalar2fe(fr, auxScalar, auxFe);

            newLeaf.push_back(auxFe);
            newLeaf.push_back(v0);
            newLeaf.push_back(v1);
            newLeaf.push_back(v2);
            newLeaf.push_back(v3);
            while(newLeaf.size() < (uint64_t(1)<<arity)) newLeaf.push_back(fr.zero());
            RawFr::Element newLeafHash;
            hashSave(fr, db, newLeaf, newLeafHash);

            if (level>=0) {
                siblings[level][keys[level]] = newLeafHash;
            } else {
                newRoot = newLeafHash;
            }
        }
    }
    else
    {
        if ( !fr.isZero(foundKey) && fr.eq(key, foundKey) ) // Delete
        {
            fea2scalar(fr, oldValue, siblings[level+1][2], siblings[level+1][3], siblings[level+1][4], siblings[level+1][5]);
            if ( level >= 0)
            {
                siblings[level][keys[level]] = fr.zero();

                int64_t uKey = getUniqueSibling(fr, siblings[level]);

                if (uKey >= 0)
                {
                    mode = "deleteFound";

                    vector<RawFr::Element> dbValue;
                    db.read(siblings[level][uKey], dbValue);
                    if (dbValue.size()==0)
                    {
                        cerr << "Error: Smt::set() could not find key in database: " << fr.toString(siblings[level][uKey], 16) << endl;
                        exit(-1);
                    }
                    siblings[level+1] = dbValue;

                    /* insKey = (addKey + ukey<<(level*arity)) + siblings[level+1][1]*(1<<((level+1)*arity)) */

                    RawFr::Element add1, add2, mul;
                    mpz_class auxScalar = accKey + uKey<<(level*arity);
                    scalar2fe(fr, auxScalar, add1);

                    auxScalar = 1;
                    auxScalar = auxScalar <<((level+1)*arity);
                    scalar2fe(fr, auxScalar, add2);

                    fr.mul(mul, siblings[level+1][1], add2);
                    
                    fr.add(insKey, add1, mul);

                    RawFr::Element insV0 = siblings[level+1][2];
                    RawFr::Element insV1 = siblings[level+1][3];
                    RawFr::Element insV2 = siblings[level+1][4];
                    RawFr::Element insV3 = siblings[level+1][5];
                    fea2scalar(fr, insValue, insV0, insV1, insV2, insV3);
                    isOld0 = false;

                    while ( (uKey>=0) && (level>=0) )
                    {
                        level--;
                        if (level>=0) {
                            uKey = getUniqueSibling(fr, siblings[level]);
                        }
                    }

                    vector<RawFr::Element> oldLeaf;
                    oldLeaf.push_back(fr.one());

                    fe2scalar(fr, auxScalar, insKey);
                    auxScalar = auxScalar >> ((level+1)*arity);
                    scalar2fe(fr, auxScalar, add1);
                    oldLeaf.push_back(add1);
                        
                    oldLeaf.push_back(insV0);
                    oldLeaf.push_back(insV1);
                    oldLeaf.push_back(insV2);
                    oldLeaf.push_back(insV3);
                    while (oldLeaf.size() < (uint64_t(1)<<arity)) oldLeaf.push_back(fr.zero());

                    RawFr::Element oldLeafHash;
                    hashSave(fr, db, oldLeaf, oldLeafHash);

                    if (level >= 0) {
                        siblings[level][keys[level]] = oldLeafHash;
                    } else {
                        newRoot = oldLeafHash;
                    }
                }
                else
                {
                    mode = "deleteNotFound";
                }
            }
            else
            {
                mode = "deleteLast";
                newRoot = fr.zero();
            }
        }
        else
        {
            mode = "zeroToZero";
        }
    }

    map< uint64_t, vector<RawFr::Element> >::iterator it;
    it = siblings.find(level+1);
    siblings.erase(it, siblings.end());

    while (level >= 0) {
        hashSave(fr, db, siblings[level], newRoot);
        level--;
        if (level >= 0)
            siblings[level][keys[level]] = newRoot;
    }

    result.oldRoot  = oldRoot;
    result.newRoot  = newRoot;
    result.key      = key;
    result.siblings = siblings;
    result.insKey   = insKey;
    result.insValue = insValue;
    result.isOld0   = isOld0;
    result.oldValue = oldValue;
    result.newValue = value;
    result.mode     = mode;     
}

void Smt::get (RawFr &fr, Database &db, RawFr::Element &root, RawFr::Element &key, SmtGetResult &result)
{
    RawFr::Element r(root);
    vector <uint64_t> keys;
    splitKey(fr, key, keys);
    uint64_t level = 0;
    mpz_class accKey = 0;
    mpz_class lastAccKey = 0;
    RawFr::Element foundKey(fr.zero());
    map< uint64_t, vector<RawFr::Element> > siblings;
    RawFr::Element insKey(fr.zero());
    mpz_class insValue = 0;
    mpz_class value = 0;
    bool isOld0 = true;

    // while root!=0 and !foundKey
    while ( (!fr.isZero(r)) && (fr.isZero(foundKey)) )
    {

        // siblings[level] = db.read(root)
        vector<RawFr::Element> dbValue;
        db.read(r, dbValue);
        if (dbValue.size()==0)
        {
            cerr << "Error: Smt::get() could not find key in database: " << fr.toString(r, 16) << endl;
            exit(-1);
        }
        siblings[level] = dbValue;

        // if siblings[level][0]=1 then this is a leaf
        if (fr.eq(siblings[level][0], fr.one())) {
            mpz_class auxMpz;
            auxMpz = 1;
            auxMpz = auxMpz << level*arity;
            RawFr::Element shiftFe;
            scalar2fe(fr, auxMpz, shiftFe);
            RawFr::Element mulFe;
            fr.mul(mulFe, siblings[level][1], shiftFe);
            RawFr::Element accKeyFe;
            scalar2fe(fr, accKey, accKeyFe);
            fr.add(foundKey, accKeyFe, mulFe);
        }
        // This is an intermediate node
        else
        {
            r = siblings[level][keys[level]];
            lastAccKey = accKey;
            mpz_class auxScalar;
            auxScalar = keys[level];
            accKey = accKey + (auxScalar << level*arity);
            level++;
        }
    }

    level--;
    accKey = lastAccKey;

    // if foundKey!=0
    if (!fr.isZero(foundKey))
    {
        // if foundKey==key, value=siblings[level+1][2...5]
        if (fr.eq(key, foundKey))
        {
            fea2scalar(fr, value, siblings[level+1][2], siblings[level+1][3], siblings[level+1][4], siblings[level+1][5]);
        }
        // else, insValue=siblings[level+1][2...5]
        else
        {
            insKey = foundKey;
            fea2scalar(fr, insValue, siblings[level+1][2], siblings[level+1][3], siblings[level+1][4], siblings[level+1][5]);
            isOld0 = false;
        }
    }

    map< uint64_t, vector<RawFr::Element> >::iterator it;
    it = siblings.find(level+1);
    siblings.erase(it, siblings.end());

    result.root     = root;
    result.key      = key;
    result.value    = value;
    result.siblings = siblings;
    result.insKey   = insKey;
    result.insValue = insValue;
    result.isOld0   = isOld0;
}

// Split the fe key into 4-bits chuncks, e.g. 0x123456EF -> { 1, 2, 3, 4, 5, 6, E, F }
void Smt::splitKey (RawFr &fr, RawFr::Element &key, vector<uint64_t> &result)
{
    // Convert the key field element into a scalar
    mpz_class auxk;
    fe2scalar(fr, auxk, key);

    // Split the field element in chunks of 1<<arity size, and store them in the result vector
    for (uint64_t i=0; i<maxLevels; i++)
    {
        mpz_class aux;
        aux = auxk & mask;
        result.push_back(aux.get_ui());
        auxk = auxk >> arity;
    }
}

void Smt::hashSave (RawFr &fr, Database &db, vector<RawFr::Element> &a, RawFr::Element &hash)
{
    // Calculate the poseidon hash of the vector of field elements
    poseidon.hash(a, &hash);

    // Fill a database value with the field elements
    vector<RawFr::Element> dbValue;
    for (uint64_t i=0; i<a.size(); i++)
        dbValue.push_back(a[i]);

    // Add the key:value pair to the database, using the hash as a key
    db.create(hash, dbValue);
}

int64_t Smt::getUniqueSibling(RawFr &fr, vector<RawFr::Element> &a)
{
    // Search for a unique, zero field element in a
    uint64_t nFound = 0;
    uint64_t fnd = 0;
    for (uint64_t i=0; i<a.size(); i++)
    {
        if (fr.isZero(a[i]))
        {
            nFound++;
            fnd = i;
        }
    }
    if (nFound == 1) return fnd;
    return -1;
}
