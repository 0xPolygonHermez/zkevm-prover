#include "smt.hpp"
#include "scalar.hpp"

void Smt::set (Context &ctx, RawFr::Element &oldRoot, RawFr::Element &key, mpz_class &value, SmtSetResult &result)
{
    RawFr::Element r(oldRoot);
    vector <uint64_t> keys;
    splitKey(ctx, key, keys);
    uint64_t level = 0;
    mpz_class accKey = 0;
    mpz_class lastAccKey = 0;
    RawFr::Element foundKey(ctx.pFr->zero());
    map< uint64_t, vector<RawFr::Element> > siblings;

    RawFr::Element insKey(ctx.pFr->zero());
    mpz_class insValue = 0;
    mpz_class oldValue = 0;
    string mode;
    RawFr::Element newRoot(oldRoot);
    bool isOld0 = true;

    while ( (!ctx.pFr->isZero(r)) && (!ctx.pFr->isZero(foundKey)) ) // TODO: review, since it was !foundKey
    {
        siblings[level] = ctx.db[r];
        if (ctx.pFr->eq(siblings[level][0], ctx.pFr->one())) {
            mpz_class auxMpz;
            auxMpz = 1;
            auxMpz = auxMpz << level*ARITY;
            RawFr::Element shiftFe;
            scalar2fe(*ctx.pFr, auxMpz, shiftFe);
            RawFr::Element mulFe;
            ctx.pFr->mul(mulFe, siblings[level][1], shiftFe);
            RawFr::Element accKeyFe;
            scalar2fe(*ctx.pFr, accKey, accKeyFe);
            ctx.pFr->add(foundKey, accKeyFe, mulFe);
        } else {
            r = siblings[level][keys[level]];
            lastAccKey = accKey;
            mpz_class auxScalar;
            auxScalar = keys[level];
            accKey = accKey + (auxScalar << level*ARITY);
            level++;
        }
    }

    level--;
    accKey = lastAccKey;

    if (value != 0)
    {
        RawFr::Element v0, v1, v2, v3;
        scalar2fea(*ctx.pFr, value, v0, v1, v2, v3);
        if (!ctx.pFr->isZero(foundKey)) // TODO: review, since it was !foundKey
        {
            if (ctx.pFr->eq(key, foundKey)) // Update
            {
                mode = "update";

                /* Prepare the vector of field elements */
                vector<RawFr::Element> newLeaf;
                RawFr::Element aux;
                newLeaf[0] = ctx.pFr->one();
                newLeaf[1] = siblings[level+1][1];
                fea2scalar(*ctx.pFr, oldValue, siblings[level+1][2], siblings[level+1][3], siblings[level+1][4], siblings[level+1][5]);
                newLeaf[2] = v0;
                newLeaf[3] = v1;
                newLeaf[4] = v2;
                newLeaf[5] = v3;
                while (newLeaf.size() < (1<<ARITY))
                {
                    newLeaf.push_back(ctx.pFr->zero());
                }

                /* Call Poseidon hash function */
                RawFr::Element newLeafHash;
                hashSave(ctx, newLeaf, newLeafHash);

                /* Process the resulting hash */
                if ( level > 0 ){
                    siblings[level][keys[level]] = newLeafHash;
                } else {
                    newRoot = newLeafHash;
                }
            }
            else // insert with foundKey
            {
                mode = "insertFound";
                vector<RawFr::Element> node;
                uint64_t level2 = level + 1;
                vector <uint64_t> foundKeys;
                splitKey(ctx, foundKey, foundKeys);
                while (keys[level2] == foundKeys[level2]) level2++;

                vector<RawFr::Element> oldLeaf;
                oldLeaf[0] = ctx.pFr->one();
                mpz_class auxScalar;
                fe2scalar(*ctx.pFr, auxScalar, foundKey);
                auxScalar = auxScalar >> ((level2+1)*ARITY);
                RawFr::Element auxFe;
                scalar2fe(*ctx.pFr, auxScalar, auxFe);

                oldLeaf[1] = auxFe;
                oldLeaf[2] = siblings[level+1][2];
                oldLeaf[3] = siblings[level+1][3];
                oldLeaf[4] = siblings[level+1][4];
                oldLeaf[5] = siblings[level+1][5];

                insKey = foundKey;
                fea2scalar(*ctx.pFr, insValue, siblings[level+1][2], siblings[level+1][3], siblings[level+1][4], siblings[level+1][5]);
                isOld0 = false;
                while (oldLeaf.size() < (1<<ARITY)) oldLeaf.push_back(ctx.pFr->zero());
                RawFr::Element oldLeafHash;
                hashSave(ctx, oldLeaf, oldLeafHash);

                vector<RawFr::Element> newLeaf;
                newLeaf[0] = ctx.pFr->one();
                fe2scalar(*ctx.pFr, auxScalar, key);
                auxScalar = auxScalar >> ((level2+1)*ARITY);
                scalar2fe(*ctx.pFr, auxScalar, auxFe);

                newLeaf[1] = auxFe;
                newLeaf[2] = v0;
                newLeaf[3] = v1;
                newLeaf[4] = v2;
                newLeaf[5] = v3;
                while(newLeaf.size() < (1<<ARITY)) newLeaf.push_back(ctx.pFr->zero());
                RawFr::Element newLeafHash;
                hashSave(ctx, newLeaf, newLeafHash);

                for (uint64_t i=0; i<(1<<ARITY); i++) node[i] = ctx.pFr->zero();
                node[keys[level2]] = newLeafHash;
                node[foundKeys[level2]] = oldLeafHash;

                RawFr::Element r2;
                hashSave(ctx, node, r2);
                level2--;

                while (level2 != level)
                {
                    for (uint64_t i=0; i<(1<<ARITY); i++) node[i] = ctx.pFr->zero();
                    node[keys[level2]] = r2;

                    hashSave(ctx, node, r2);
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

            vector<RawFr::Element> newLeaf;
            newLeaf[0] = ctx.pFr->one();
            mpz_class auxScalar;
            fe2scalar(*ctx.pFr, auxScalar, key);
            auxScalar = auxScalar >> ((level+1)*ARITY);
            RawFr::Element auxFe;
            scalar2fe(*ctx.pFr, auxScalar, auxFe);

            newLeaf[1] = auxFe;
            newLeaf[2] = v0;
            newLeaf[3] = v1;
            newLeaf[4] = v2;
            newLeaf[5] = v3;
            while(newLeaf.size() < (1<<ARITY)) newLeaf.push_back(ctx.pFr->zero());
            RawFr::Element newLeafHash;
            hashSave(ctx, newLeaf, newLeafHash);

            if (level>=0) {
                siblings[level][keys[level]] = newLeafHash;
            } else {
                newRoot = newLeafHash;
            }
        }
    }
    else
    {
        if ( !ctx.pFr->isZero(foundKey) && ctx.pFr->eq(key, foundKey) ) // Delete
        {
            fea2scalar(*ctx.pFr, oldValue, siblings[level+1][2], siblings[level+1][3], siblings[level+1][4], siblings[level+1][5]);
            if ( level >= 0)
            {
                siblings[level][keys[level]] = ctx.pFr->zero();

                int64_t uKey = getUniqueSibling(ctx, siblings[level]);

                if (uKey >= 0)
                {
                    mode = "deleteFound";
                    siblings[level+1] = ctx.db[siblings[level][uKey]];

                    /* insKey = (addKey + ukey<<(level*ARITY)) + siblings[level+1][1]*(1<<((level+1)*ARITY)) */

                    RawFr::Element add1, add2, mul;
                    mpz_class auxScalar = accKey + uKey<<(level*ARITY);
                    scalar2fe(*ctx.pFr, auxScalar, add1);

                    auxScalar = 1;
                    auxScalar = auxScalar <<((level+1)*ARITY);
                    scalar2fe(*ctx.pFr, auxScalar, add2);

                    ctx.pFr->mul(mul, siblings[level+1][1], add2);
                    
                    ctx.pFr->add(insKey, add1, mul);

                    RawFr::Element insV0 = siblings[level+1][2];
                    RawFr::Element insV1 = siblings[level+1][3];
                    RawFr::Element insV2 = siblings[level+1][4];
                    RawFr::Element insV3 = siblings[level+1][5];
                    fea2scalar(*ctx.pFr, insValue, insV0, insV1, insV2, insV3);
                    isOld0 = false;

                    while ( (uKey>=0) && (level>=0) )
                    {
                        level--;
                        if (level>=0) {
                            uKey = getUniqueSibling(ctx, siblings[level]);
                        }
                    }

                    vector<RawFr::Element> oldLeaf;
                    oldLeaf[0] = ctx.pFr->one();

                    fe2scalar(*ctx.pFr, auxScalar, insKey);
                    auxScalar = auxScalar >> ((level+1)*ARITY);
                    scalar2fe(*ctx.pFr, auxScalar, add1);
                    oldLeaf[1] = add1;
                        
                    oldLeaf[2] = insV0;
                    oldLeaf[3] = insV1;
                    oldLeaf[4] = insV2;
                    oldLeaf[5] = insV3;
                    while (oldLeaf.size() < (1<<ARITY)) oldLeaf.push_back(ctx.pFr->zero());

                    RawFr::Element oldLeafHash;
                    hashSave(ctx, oldLeaf, oldLeafHash);

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
                newRoot = ctx.pFr->zero();
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
        hashSave(ctx, siblings[level], newRoot);
        level--;
        if (level >= 0) siblings[level][keys[level]] = newRoot; 
    }

    // TODO: Use result members directly to avoid this copy.
    result.oldRoot = oldRoot;
    result.newRoot = newRoot;
    result.key = key;
    result.siblings = siblings;
    result.insKey = insKey;
    result.insValue = insValue;
    result.isOld0 = isOld0;
    result.oldValue = oldValue;
    result.newValue = value;
    result.mode = mode;     
}

void Smt::get (Context &ctx, RawFr::Element &root, RawFr::Element &key, SmtGetResult &result)
{
    RawFr::Element r(root);
    vector <uint64_t> keys;
    splitKey(ctx, key, keys);
    uint64_t level = 0;
    mpz_class accKey = 0;
    mpz_class lastAccKey = 0;
    RawFr::Element foundKey(ctx.pFr->zero());
    map< uint64_t, vector<RawFr::Element> > siblings;
    RawFr::Element insKey(ctx.pFr->zero());
    mpz_class insValue = 0;
    mpz_class value = 0;
    bool isOld0 = true;

    while ( (!ctx.pFr->isZero(r)) && (!ctx.pFr->isZero(foundKey)) ) // TODO: review, since it was !foundKey
    {
        siblings[level] = ctx.db[r];
        if (ctx.pFr->eq(siblings[level][0], ctx.pFr->one())) {
            mpz_class auxMpz;
            auxMpz = 1;
            auxMpz = auxMpz << level*ARITY;
            RawFr::Element shiftFe;
            scalar2fe(*ctx.pFr, auxMpz, shiftFe);
            RawFr::Element mulFe;
            ctx.pFr->mul(mulFe, siblings[level][1], shiftFe);
            RawFr::Element accKeyFe;
            scalar2fe(*ctx.pFr, accKey, accKeyFe);
            ctx.pFr->add(foundKey, accKeyFe, mulFe);
        } else {
            r = siblings[level][keys[level]];
            lastAccKey = accKey;
            mpz_class auxScalar;
            auxScalar = keys[level];
            accKey = accKey + (auxScalar << level*ARITY);
            level++;
        }
    }

    level--;
    accKey = lastAccKey;

    if (!ctx.pFr->isZero(foundKey))
    {
        if (!ctx.pFr->eq(key, foundKey))
        {
            fea2scalar(*ctx.pFr, value, siblings[level+1][2], siblings[level+1][3], siblings[level+1][4], siblings[level+1][5]);
        }
        else
        {
            insKey = foundKey;
            fea2scalar(*ctx.pFr, insValue, siblings[level+1][2], siblings[level+1][3], siblings[level+1][4], siblings[level+1][5]);
            isOld0 = false;
        }
    }

    map< uint64_t, vector<RawFr::Element> >::iterator it;
    it = siblings.find(level+1);
    siblings.erase(it, siblings.end());

    // TODO: Use result members directly to avoid this copy.
    result.root = root;
    result.key = key;
    result.value = value;
    result.siblings = siblings;
    result.insKey = insKey;
    result.insValue = insValue;
    result.isOld0 = isOld0;
}

void Smt::splitKey (Context &ctx, RawFr::Element &key, vector<uint64_t> &result)
{
    mpz_class auxk;
    fe2scalar(*ctx.pFr, auxk, key);
    for (uint64_t i=0; i<maxLevels; i++)
    {
        mpz_class aux;
        aux = auxk & mask;
        result.push_back(aux.get_ui());
        auxk = auxk >> ARITY;
    }
}

void Smt::hashSave (Context &ctx, vector<RawFr::Element> &a, RawFr::Element &hash)
{
    poseidon.hash(a, &hash);
    ctx.db[hash] = a;
}

int64_t Smt::getUniqueSibling(Context &ctx, vector<RawFr::Element> &a)
{
    uint64_t nFound = 0;
    uint64_t fnd = 0;
    for (uint64_t i=0; i<a.size(); i++)
    {
        if (!ctx.pFr->isZero(a[i]))
        {
            nFound++;
            fnd = i;
        }
    }
    if (nFound == 1) return fnd;
    return -1;
}