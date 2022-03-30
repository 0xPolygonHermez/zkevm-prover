#include "smt.hpp"
#include "scalar.hpp"
#include "utils.hpp"

void Smt::set ( Database &db, FieldElement (&oldRoot)[4], FieldElement (&key)[4], mpz_class &value, SmtSetResult &result )
{
    FieldElement r[4];
    for (uint64_t i=0; i<4; i++) r[i] = oldRoot[i];
    FieldElement newRoot[4];
    for (uint64_t i=0; i<4; i++) newRoot[i] = oldRoot[i];

    vector <uint64_t> keys;
    splitKey(key, keys);

    int64_t level = 0;

    vector<uint64_t> accKey;
    mpz_class lastAccKey = 0;
    FieldElement foundKey[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};
    FieldElement foundRKey[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};
    FieldElement insKey[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};
    
    map< uint64_t, vector<FieldElement> > siblings;

    mpz_class insValue = 0;
    mpz_class oldValue = 0;
    mpz_class foundVal = 0;
    FieldElement foundOldValH[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};

    string mode;

    bool isOld0 = true;

    // while root!=0 and !foundKey
    while ( (!fr.isZero(r)) && (fr.isZero(foundKey)) )
    {
        // siblings[level] = db.read(root)
        mpz_class rootScalar;
        fea2scalar(fr, rootScalar, r);
        string rootString;
        rootString = rootScalar.get_str(16);
        vector<FieldElement> dbValue;
        db.read(rootString, dbValue);
        if (dbValue.size()==0)
        {
            cerr << "Error: Smt::set() could not find key in database: " << rootString << endl;
            exit(-1);
        }
        siblings[level] = dbValue;

        // if siblings[level][8]=1 then this is a leaf
        if (siblings[level].size()>8 && fr.eq(siblings[level][8], fr.one())) {
            foundOldValH[0] = siblings[level][4];
            foundOldValH[1] = siblings[level][5];
            foundOldValH[2] = siblings[level][6];
            foundOldValH[3] = siblings[level][7];
            mpz_class valueHashScalar;
            fea2scalar(fr, valueHashScalar, foundOldValH);
            string valueHashString;
            valueHashString = valueHashScalar.get_str(16);
            vector<FieldElement> dbValue;
            db.read(valueHashString, dbValue);
            if (dbValue.size()==0)
            {
                cerr << "Error: Smt::get() could not find value key in database: " << valueHashString << endl;
                exit(-1);
            }
            foundRKey[0] = siblings[level][0];
            foundRKey[1] = siblings[level][1];
            foundRKey[2] = siblings[level][2];
            foundRKey[3] = siblings[level][3];
            FieldElement valueFea[8];
            for (uint64_t i=0; i<8; i++) valueFea[i] = dbValue[i];
            fea2scalar(fr, foundVal, valueFea);
            joinKey(accKey, foundRKey, foundKey);
        }
        // This is an intermediate node
        else
        {
            r[0] = siblings[level][keys[level]*4];
            r[1] = siblings[level][keys[level]*4 + 1];
            r[2] = siblings[level][keys[level]*4 + 2];
            r[3] = siblings[level][keys[level]*4 + 3];
            accKey.push_back(keys[level]);
            level++;
        }
    }

    level--;
    accKey.pop_back();

    if (value != 0)
    {
        if (!fr.isZero(foundKey))
        {
            if (fr.eq(key, foundKey)) // Update
            {
                mode = "update";

                /* Prepare the vector of field elements */
                FieldElement v0, v1, v2, v3, v4, v5, v6, v7;
                scalar2fea(fr, value, v0, v1, v2, v3, v4, v5, v6, v7);
                vector<FieldElement> v;
                v.push_back(v0);
                v.push_back(v1);
                v.push_back(v2);
                v.push_back(v3);
                v.push_back(v4);
                v.push_back(v5);
                v.push_back(v6);
                v.push_back(v7);

                // Prepare the capacity = 0, 0, 0, 0
                vector<FieldElement> c;
                c.push_back(0);
                c.push_back(0);
                c.push_back(0);
                c.push_back(0);

                // Save and get the hash
                FieldElement newValH[4];
                hashSave(db, v, c, newValH);
                
                // Prepare the vector of field elements
                v.clear();
                v.push_back(foundRKey[0]);
                v.push_back(foundRKey[1]);
                v.push_back(foundRKey[2]);
                v.push_back(foundRKey[3]);
                v.push_back(newValH[0]);
                v.push_back(newValH[1]);
                v.push_back(newValH[2]);
                v.push_back(newValH[3]);

                // Prepare the capacity = 1, 0, 0, 0
                c[0] = 1;

                // Save and get the hash
                FieldElement newLeafHash[4];
                hashSave(db, v, c, newLeafHash);

                /* Process the resulting hash */
                if ( level >= 0 )
                {
                    siblings[level][keys[level]*4] = newLeafHash[0];
                    siblings[level][keys[level]*4 + 1] = newLeafHash[1];
                    siblings[level][keys[level]*4 + 2] = newLeafHash[2];
                    siblings[level][keys[level]*4 + 3] = newLeafHash[3];
                }
                else
                {
                    newRoot[0] = newLeafHash[0];
                    newRoot[1] = newLeafHash[1];
                    newRoot[2] = newLeafHash[2];
                    newRoot[3] = newLeafHash[3];
                }
            }
            else // insert with foundKey
            {
                mode = "insertFound";

                int64_t level2 = level + 1;
                vector <uint64_t> foundKeys;
                splitKey(foundKey, foundKeys);
                while (keys[level2] == foundKeys[level2]) level2++;

                FieldElement oldKey[4];
                removeKeyBits(foundKey, level2+1, oldKey);

                /* Prepare the vector of field elements */
                vector<FieldElement> v;
                v.push_back(oldKey[0]);
                v.push_back(oldKey[1]);
                v.push_back(oldKey[2]);
                v.push_back(oldKey[3]);
                v.push_back(foundOldValH[0]);
                v.push_back(foundOldValH[1]);
                v.push_back(foundOldValH[2]);
                v.push_back(foundOldValH[3]);

                // Prepare the capacity = 1, 0, 0, 0
                vector<FieldElement> c;
                c.push_back(1);
                c.push_back(0);
                c.push_back(0);
                c.push_back(0);

                // Save and get the hash
                FieldElement oldLeafHash[4];
                hashSave(db, v, c, oldLeafHash);

                insKey[0] = foundKey[0];
                insKey[1] = foundKey[1];
                insKey[2] = foundKey[2];
                insKey[3] = foundKey[3];
                insValue = foundVal;
                isOld0 = false;

                FieldElement newKey[4];
                removeKeyBits(key, level2 + 1, newKey);

                FieldElement valueFea[8];
                scalar2fea(fr, value, valueFea);
                v.clear();
                v.push_back(valueFea[0]);
                v.push_back(valueFea[1]);
                v.push_back(valueFea[2]);
                v.push_back(valueFea[3]);
                v.push_back(valueFea[4]);
                v.push_back(valueFea[5]);
                v.push_back(valueFea[6]);
                v.push_back(valueFea[7]);

                c[0]=0;
                FieldElement newValH[4];
                hashSave(db, v, c, newValH);
                
                v.clear();
                v.push_back(newKey[0]);
                v.push_back(newKey[1]);
                v.push_back(newKey[2]);
                v.push_back(newKey[3]);
                v.push_back(newValH[0]);
                v.push_back(newValH[1]);
                v.push_back(newValH[2]);
                v.push_back(newValH[3]);

                c[0] = 1;
                
                FieldElement newLeafHash[4];
                hashSave(db, v, c, newLeafHash);

                vector<FieldElement> node;
                for (uint64_t i=0; i<8; i++)
                {
                    node.push_back(fr.zero());
                }
                for (uint64_t j=0; j<4; j++)
                {
                    node[keys[level2] * 4 + j] = newLeafHash[j];
                    node[foundKeys[level2] * 4 + j] = oldLeafHash[j];
                }

                FieldElement r2[4];
                c[0] = fr.zero();
                hashSave(db, node, c, r2);
                level2--;

                while (level2!=level)
                {
                    for (uint64_t i = 0; i < 8; i++)
                    {
                        node[i] = fr.zero();
                    }
                    for (uint64_t j = 0; j < 4; j++)
                    {
                        node[keys[level2] * 4 + j] = r2[j];
                    }
                    c[0] = fr.zero();
                    hashSave(db, node, c, r2);
                    level2--;
                }

                if (level >= 0)
                {
                    for (uint64_t j = 0; j < 4; j++)
                    {
                        siblings[level][keys[level] * 4 + j] = r2[j];
                    }
                }
                else
                {
                    newRoot[0] = r2[0];
                    newRoot[1] = r2[1];
                    newRoot[2] = r2[2];
                    newRoot[3] = r2[3];
                }
            }
        }
        else // insert without foundKey
        {
            mode = "insertNotFound";

            FieldElement newKey[4];
            removeKeyBits(key, level+1, newKey);

            FieldElement valueFea[8];
            scalar2fea(fr, value, valueFea);
            vector<FieldElement> valueVector;
            for (uint64_t i=0; i<8; i++) valueVector.push_back(valueFea[i]);
            vector<FieldElement> c;
            for (uint64_t i=0; i<4; i++) c.push_back(fr.zero());
            FieldElement newValH[4];
            hashSave(db, valueVector, c, newValH);

            vector<FieldElement> keyvalVector;
            for (uint64_t i=0; i<4; i++) keyvalVector.push_back(newKey[i]);
            for (uint64_t i=0; i<4; i++) keyvalVector.push_back(newValH[i]);
            c[0] = 1;
            FieldElement newLeafHash[4];
            hashSave(db, keyvalVector, c, newLeafHash);

            if (level>=0)
            {
                for (uint64_t j=0; j<4; j++)
                {
                    siblings[level][keys[level]*4 + j] = newLeafHash[j];
                }
            }
            else
            {
                newRoot[0] = newLeafHash[0];
                newRoot[1] = newLeafHash[1];
                newRoot[2] = newLeafHash[2];
                newRoot[3] = newLeafHash[3];
            }
        }
    }
    else
    {
        if ( !fr.isZero(foundKey) && fr.eq(key, foundKey) ) // Delete
        {
            if ( level >= 0)
            {
                for (uint64_t j=0; j<4; j++)
                {
                    siblings[level][keys[level]*4 + j] = fr.zero();
                }

                int64_t uKey = getUniqueSibling(siblings[level]);

                if (uKey >= 0)
                {
                    mode = "deleteFound";

                    FieldElement auxFea[4];
                    for (uint64_t i=0; i<4; i++) auxFea[i] = siblings[level][uKey*4+i];
                    mpz_class auxScalar;
                    fea2scalar(fr, auxScalar, auxFea);
                    string auxString;
                    auxString = auxScalar.get_str(16);

                    vector<FieldElement> dbValue;
                    db.read(auxString, dbValue);
                    if (dbValue.size()==0)
                    {
                        cerr << "Error: Smt::set() could not find key in database: " << auxString << endl;
                        exit(-1);
                    }
                    siblings[level+1] = dbValue;

                    // If it is a leaf node
                    if ( siblings[level+1].size()>8 && siblings[level+1][8]==fr.one())
                    {
                        FieldElement valH[4];
                        for (uint64_t i=0; i<4; i++) valH[i] = siblings[level+1][4+i];
                        mpz_class valHScalar;
                        fea2scalar(fr, valHScalar, valH);
                        string valHString;
                        valHString = valHScalar.get_str(16);
                        vector<FieldElement> dbValue;
                        db.read(valHString, dbValue);
                        if (dbValue.size()<8)
                        {
                            cerr << "Error: Smt::set() could not find key in database: " << valHString << endl;
                            exit(-1);
                        }
                        FieldElement valA[8];
                        for (uint64_t i=0; i<8; i++) valA[i] = dbValue[i];
                        FieldElement rKey[4];
                        for (uint64_t i=0; i<4; i++) rKey[i] = siblings[level+1][i];
                        mpz_class val;
                        fea2scalar(fr, val, valA);

                        vector<uint64_t> auxBits;
                        auxBits = accKey;
                        auxBits.push_back(uKey);
                        joinKey(auxBits, rKey, insKey );

                        insValue = val;
                        isOld0 = false;

                        while (uKey>=0 && level>=0)
                        {
                            level--;
                            if (level >= 0)
                            {
                                uKey = getUniqueSibling(siblings[level]);
                            }
                        }

                        FieldElement oldKey[4];
                        removeKeyBits(insKey, level+1, oldKey);

                        vector<FieldElement> a;
                        for (uint64_t i=0; i<4; i++) a.push_back(oldKey[i]);
                        for (uint64_t i=0; i<4; i++) a.push_back(valH[i]);
                        vector<FieldElement> c;
                        c.push_back(fr.one());
                        c.push_back(fr.zero());
                        c.push_back(fr.zero());
                        c.push_back(fr.zero());

                        FieldElement oldLeafHash[4];
                        hashSave(db, a, c, oldLeafHash);

                        if (level >= 0)
                        {
                            for (uint64_t j=0; j< 4; j++)
                            {
                                siblings[level][keys[level]*4 + j] = oldLeafHash[j];
                            }
                        }
                        else
                        {
                            newRoot[0] = oldLeafHash[0];
                            newRoot[1] = oldLeafHash[1];
                            newRoot[2] = oldLeafHash[2];
                            newRoot[3] = oldLeafHash[3];
                        }
                    }
                    else
                    {
                        mode = "deleteNotFound";
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
                newRoot[0] = fr.zero();
                newRoot[1] = fr.zero();
                newRoot[2] = fr.zero();
                newRoot[3] = fr.zero();
            }
        }
        else
        {
            mode = "zeroToZero";
        }
    }

    map< uint64_t, vector<FieldElement> >::iterator it;
    it = siblings.find(level+1);
    siblings.erase(it, siblings.end());

    while (level >= 0)
    {
        vector<FieldElement> a, c;
        for (uint64_t i=0; i<8; i++)
        {
            a.push_back(siblings[level][i]);
        }
        for (uint64_t i=8; i<12; i++)
        {
            c.push_back(siblings[level][i]);
        }
        
        hashSave(db, a, c, newRoot);
        level--;
        if (level >= 0)
        {
            for (uint64_t j=0; j<4; j++)
            {
                siblings[level][keys[level]*4 + j] = newRoot[j];
            }
        }
    }

    result.oldRoot[0] = oldRoot[0];
    result.oldRoot[1] = oldRoot[1];
    result.oldRoot[2] = oldRoot[2];
    result.oldRoot[3] = oldRoot[3];
    result.newRoot[0] = newRoot[0];
    result.newRoot[1] = newRoot[1];
    result.newRoot[2] = newRoot[2];
    result.newRoot[3] = newRoot[3];
    result.key[0]     = key[0];
    result.key[1]     = key[1];
    result.key[2]     = key[2];
    result.key[3]     = key[3];
    result.siblings   = siblings;
    result.insKey[0]  = insKey[0];
    result.insKey[1]  = insKey[1];
    result.insKey[2]  = insKey[2];
    result.insKey[3]  = insKey[3];
    result.insValue   = insValue;
    result.isOld0     = isOld0;
    result.oldValue   = oldValue;
    result.newValue   = value;
    result.mode       = mode;     
}

void Smt::get ( Database &db, FieldElement (&root)[4], FieldElement (&key)[4], SmtGetResult &result )
{
    FieldElement r[4];
    for (uint64_t i=0; i<4; i++)
    {
        r[i] = root[i];
    }
    
    vector <uint64_t> keys;
    splitKey(key, keys);

    uint64_t level = 0;

    vector<uint64_t> accKey;
    mpz_class lastAccKey = 0;
    FieldElement foundKey[4] = {0, 0, 0, 0};
    FieldElement insKey[4] = {0, 0, 0, 0};
    
    map< uint64_t, vector<FieldElement> > siblings;
    
    mpz_class insValue = 0;
    mpz_class value = 0;
    mpz_class foundVal = 0;

    bool isOld0 = true;

    // while root!=0 and !foundKey
    while ( (!fr.isZero(r)) && fr.isZero(foundKey) )
    {
        // siblings[level] = db.read(root)
        mpz_class rootScalar;
        fea2scalar(fr, rootScalar, r);
        string rootString;
        rootString = rootScalar.get_str(16);
        vector<FieldElement> dbValue;
        db.read(rootString, dbValue);
        if (dbValue.size()==0)
        {
            cerr << "Error: Smt::get() could not find key in database: " << rootString << endl;
            exit(-1);
        }
        siblings[level] = dbValue;

        // if siblings[level][8]=1 then this is a leaf
        if (siblings[level].size()>8 && fr.eq(siblings[level][8], fr.one()))
        {
            FieldElement valueHashFea[4];
            valueHashFea[0] = siblings[level][4];
            valueHashFea[1] = siblings[level][5];
            valueHashFea[2] = siblings[level][6];
            valueHashFea[3] = siblings[level][7];
            mpz_class valueHashScalar;
            fea2scalar(fr, valueHashScalar, valueHashFea);
            string valueHashString;
            valueHashString = valueHashScalar.get_str(16);
            vector<FieldElement> dbValue;
            db.read(valueHashString, dbValue);
            if (dbValue.size()==0)
            {
                cerr << "Error: Smt::get() could not find value key in database: " << valueHashString << endl;
                exit(-1);
            }
            FieldElement foundRKey[4];
            foundRKey[0] = siblings[level][0];
            foundRKey[1] = siblings[level][1];
            foundRKey[2] = siblings[level][2];
            foundRKey[3] = siblings[level][3];
            FieldElement fea[8];
            for (uint64_t i=0; i<8; i++)
            {
                fea[i] = dbValue[i];
            }
            fea2scalar(fr, foundVal, fea);
            joinKey(accKey, foundRKey, foundKey);
        }
        // This is an intermediate node
        else
        {
            r[0] = siblings[level][keys[level]*4];
            r[1] = siblings[level][keys[level]*4 + 1];
            r[2] = siblings[level][keys[level]*4 + 2];
            r[3] = siblings[level][keys[level]*4 + 3];
            accKey.push_back(keys[level]);
            level++;
        }
    }

    level--;
    accKey.pop_back();

    // if foundKey!=0
    if (!fr.isZero(foundKey))
    {
        // if foundKey==key, then foundVal is what we were looking for
        if ( fr.eq(key, foundKey) )
        {
            value = foundVal;
        }
        else
        {
            insKey[0] = foundKey[0];
            insKey[1] = foundKey[1];
            insKey[2] = foundKey[2];
            insKey[3] = foundKey[3];
            insValue = foundVal;
            isOld0 = false;
        }
    }

    map< uint64_t, vector<FieldElement> >::iterator it;
    it = siblings.find(level+1);
    siblings.erase(it, siblings.end());

    result.root[0]   = root[0];
    result.root[1]   = root[1];
    result.root[2]   = root[2];
    result.root[3]   = root[3];
    result.key[0]    = key[0];
    result.key[1]    = key[1];
    result.key[2]    = key[2];
    result.key[3]    = key[3];
    result.value     = value;
    result.siblings  = siblings;
    result.insKey[0] = insKey[0];
    result.insKey[1] = insKey[1];
    result.insKey[2] = insKey[2];
    result.insKey[3] = insKey[3];
    result.insValue  = insValue;
    result.isOld0    = isOld0;
}

// Split the fe key into 4-bits chuncks, e.g. 0x123456EF -> { 1, 2, 3, 4, 5, 6, E, F }
void Smt::splitKey ( FieldElement (&key)[4], vector<uint64_t> &result )
{
    // Copy the key to local variables
    mpz_class auxk[4];
    for (uint64_t i=0; i<4; i++)
    {
        auxk[i] = key[i];
    }

    // Split the key in bits, taking one bit from a different scalar every time
    for (uint64_t i=0; i<64; i++)
    {
        for (uint64_t j=0; j<4; j++)
        {
            mpz_class aux;
            aux = auxk[j] & 1;
            result.push_back(aux.get_ui());
            auxk[j] = auxk[j] >> 1;
        }
    }
}

// Joins full key from remaining key and path already used
// bits = key path used
// rkey = remaining key
// key = full key (returned)
void Smt::joinKey ( vector<uint64_t> &bits, FieldElement (&rkey)[4], FieldElement (&key)[4] )
{
    uint64_t n[4] = {0, 0, 0, 0};
    mpz_class accs[4] = {0, 0, 0, 0};
    for (uint64_t i=0; i<bits.size(); i++)
    {
        if (bits[i])
        {
            accs[i%4] = (accs[i%4] | (mpz_class(1)<<n[i%4]))%fr.prime();
        }
        n[i%4] += 1;
    }
    FieldElement auxk[4];
    for (uint64_t i=0; i<4; i++)
    {
        auxk[i] = rkey[i];
    }
    for (uint64_t i=0; i<4; i++)
    {
        mpz_class aux = auxk[i];
        aux = ((aux<<n[i]) | accs[i])%mpz_class(fr.prime());
        auxk[i] = aux.get_ui();
    }
    for (uint64_t i=0; i<4; i++)
    {
        key[i] = auxk[i];
    }
}

/**
 * Removes bits from the key depending on the smt level
 * key -key
 * nBits - bits to remove
 * returns rkey - remaining key bits to store
 */
void Smt::removeKeyBits ( FieldElement (&key)[4], uint64_t nBits, FieldElement (&rkey)[4] )
{
    uint64_t fullLevels = nBits / 4;
    mpz_class auxk[4];

    for (uint64_t i=0; i<4; i++)
    {
        auxk[i] = key[i];
    }

    for (uint64_t i = 0; i < 4; i++)
    {
        uint64_t n = fullLevels;
        if (fullLevels * 4 + i < nBits) n += 1;
        auxk[i] = auxk[i] >> n;
    }

    for (uint64_t i=0; i<4; i++)
    {
        scalar2fe(fr, auxk[i], rkey[i]);
    }
}

void Smt::hashSave ( Database &db,
                     vector<FieldElement> &a,
                     vector<FieldElement> &c,
                     FieldElement (&hash)[4] )
{
    // Check that the addition of both vectors matches a size of 12
    uint64_t aSize = a.size();
    uint64_t cSize = c.size();
    if (aSize + cSize != 12)
    {
        cerr << "Error: Smt::hashSave() found invalid vector sizes.  a:" << aSize << " c:" << cSize << endl;
        exit(-1);
    }

    // Calculate the poseidon hash of the vector of field elements: v = a | c
    FieldElement v[12];
    for (uint64_t i=0; i<aSize; i++)
    {
        v[i] = a[i];
    }
    for (uint64_t i=0; i<cSize; i++)
    {
        v[aSize+i] = c[i];
    }
    poseidon.hash(v);

    // Fill a database value with the field elements
    FieldElement v2[4];
    for (uint64_t i=0; i<4; i++)
    {
        v2[i] = v[i];
    }
    mpz_class hashScalar;
    fea2scalar(fr, hashScalar, v2);
    string hashString = hashScalar.get_str(16);

    // Add the key:value pair to the database, using the hash as a key
    vector<FieldElement> dbValue;
    for (uint64_t i=0; i<aSize; i++) dbValue.push_back(a[i]);
    for (uint64_t i=0; i<cSize; i++) dbValue.push_back(c[i]);
    db.create(hashString, dbValue);

    // Return the hash
    for (uint64_t i=0; i<4; i++)
    {
        hash[i] = v[i];
    }
}

int64_t Smt::getUniqueSibling(vector<FieldElement> &a)
{
    // Search for a unique, zero field element in a
    uint64_t nFound = 0;
    uint64_t fnd = 0;
    for (uint64_t i=0; i<a.size(); i+=4)
    {
        if (!fr.isZero(a))
        {
            nFound++;
            fnd = i / 4;
        }
    }
    if (nFound == 1) return fnd;
    return -1;
}