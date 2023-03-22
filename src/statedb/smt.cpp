#include "smt.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "zkresult.hpp"
#include "zkmax.hpp"

zkresult Smt::set(Database &db, const Goldilocks::Element (&oldRoot)[4], const Goldilocks::Element (&key)[4], const mpz_class &value, const bool persistent, SmtSetResult &result, DatabaseMap *dbReadLog)
{
#ifdef LOG_SMT
    cout << "Smt::set() called with oldRoot=" << fea2string(fr,oldRoot) << " key=" << fea2string(fr,key) << " value=" << value.get_str(16) << " persistent=" << persistent << endl;
#endif
    Goldilocks::Element r[4];
    for (uint64_t i=0; i<4; i++) r[i] = oldRoot[i];
    Goldilocks::Element newRoot[4];
    for (uint64_t i=0; i<4; i++) newRoot[i] = oldRoot[i];

    // Get a list of the bits of the key to navigate top-down through the tree
    vector <uint64_t> keys;
    splitKey(key, keys);

    int64_t level = 0;
    uint64_t proofHashCounter = 0;

    vector<uint64_t> accKey;
    mpz_class lastAccKey = 0;
    bool bFoundKey = false;
    Goldilocks::Element foundKey[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};
    Goldilocks::Element foundRKey[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};
    Goldilocks::Element insKey[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};

    map< uint64_t, vector<Goldilocks::Element> > siblings;

    mpz_class insValue = 0;
    mpz_class oldValue = 0;
    mpz_class foundVal = 0;
    Goldilocks::Element foundOldValH[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};

    string mode;

    bool isOld0 = true;
    zkresult dbres;

    // Start natigating the tree from the top: r = root
    // Go down while r!=0 (while there is branch) until we find the key
    while ( (!fr.isZero(r[0]) || !fr.isZero(r[1]) || !fr.isZero(r[2]) || !fr.isZero(r[3])) && !bFoundKey )
    {
        // Read the content of db for entry r: siblings[level] = db.read(r)
        string rootString = fea2string(fr, r);
        vector<Goldilocks::Element> dbValue;

        dbres = db.read(rootString, dbValue, dbReadLog);
        if (dbres != ZKR_SUCCESS)
        {
            cerr << "Error: Smt::set() db.read error: " << dbres << " (" << zkresult2string(dbres) << ") root:" << rootString << endl;
            return dbres;
        }

        // Get a copy of the content of this database entry, at the corresponding level: 0, 1...
        siblings[level] = dbValue;

        // if siblings[level][8]=1 then this is a leaf
        if ( siblings[level].size()>8 && fr.equal(siblings[level][8], fr.one()) )
        {
            // Second 4 elements are the hash of the old value, so we can get old value=db(valueHash)
            foundOldValH[0] = siblings[level][4];
            foundOldValH[1] = siblings[level][5];
            foundOldValH[2] = siblings[level][6];
            foundOldValH[3] = siblings[level][7];
            string valueHashString = fea2string(fr, foundOldValH);
            vector<Goldilocks::Element> dbValue;
            dbres = db.read(valueHashString, dbValue, dbReadLog);
            if (dbres != ZKR_SUCCESS)
            {
                cerr << "Error: Smt::set() db.read error: " << dbres << " (" << zkresult2string(dbres) << ") key:" << valueHashString << endl;
                return dbres;
            }

            // Convert the 8 found value fields to a foundVal scalar
            Goldilocks::Element valueFea[8];
            for (uint64_t i=0; i<8; i++) valueFea[i] = dbValue[i];
            fea2scalar(fr, foundVal, valueFea);

            // First 4 elements are the remaining key of the old value
            foundRKey[0] = siblings[level][0];
            foundRKey[1] = siblings[level][1];
            foundRKey[2] = siblings[level][2];
            foundRKey[3] = siblings[level][3];

            // Joining the consumed key bits, we have the complete found key of the old value
            joinKey(accKey, foundRKey, foundKey);
            bFoundKey = true;

#ifdef LOG_SMT
            cout << "Smt::set() found at level=" << level << " foundVal=" << foundVal.get_str(16) << " foundKey=" << fea2string(fr,foundKey) << " foundRKey=" << fea2string(fr,foundRKey) << endl;
#endif
        }
        // This is an intermediate node
        else
        {
            // Take either the first 4 (keys[level]=0) or the second 4 (keys[level]=1) siblings as the hash of the next level
            r[0] = siblings[level][keys[level]*4];
            r[1] = siblings[level][keys[level]*4 + 1];
            r[2] = siblings[level][keys[level]*4 + 2];
            r[3] = siblings[level][keys[level]*4 + 3];

            // Store the used key bit in accKey
            accKey.push_back(keys[level]);

#ifdef LOG_SMT
            cout << "Smt::set() down 1 level=" << level << " keys[level]=" << keys[level] << " root/hash=" << fea2string(fr,r) << endl;
#endif
            // Increase the level
            level++;
        }
    }

    // One step back
    level--;
    accKey.pop_back();

    if (!fr.isZero(oldRoot[0]) || !fr.isZero(oldRoot[1]) || !fr.isZero(oldRoot[2]) || !fr.isZero(oldRoot[3]))
    {
        proofHashCounter = zkmin(siblings.size(), uint64_t(level+1));
        if (foundVal)
        {
            proofHashCounter += 2;
        }
    }

    // If value!=0, it means we want to update an existing leaf node value, or create a new leaf node with the new value, in case keys are different
    if (value != 0)
    {
        // If we found a leaf node going down the tree
        if (bFoundKey)
        {
            // In case the found key is the same as the key we want to se, this is an update of the value of the existing leaf node
            if (fr.equal(key[0], foundKey[0]) && fr.equal(key[1], foundKey[1]) && fr.equal(key[2], foundKey[2]) && fr.equal(key[3], foundKey[3])) // Update
            {
                mode = "update";
#ifdef LOG_SMT
                cout << "Smt::set() mode=" << mode << endl;
#endif
                oldValue = foundVal;

                // First, we create the db entry for the new VALUE, and store the calculated hash in newValH
                Goldilocks::Element v[8];
                scalar2fea(fr, value, v);

                // Prepare the capacity = 0, 0, 0, 0
                Goldilocks::Element c[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};

                // Save and get the new value hash
                Goldilocks::Element newValH[4];
                dbres = hashSave(db, v, c, persistent, newValH);
                if (dbres != ZKR_SUCCESS)
                {
                    return dbres;
                }

                // Second, we create the db entry for the new leaf node = RKEY + HASH, and store the calculated hash in newLeafHash
                for (uint64_t i=0; i<4; i++) v[i] = foundRKey[i];
                for (uint64_t i=0; i<4; i++) v[4+i] = newValH[i];

                // Prepare the capacity = 1, 0, 0, 0
                c[0] = fr.one();

                // Save and get the hash
                Goldilocks::Element newLeafHash[4];
                dbres = hashSave(db, v, c, persistent, newLeafHash);
                if (dbres != ZKR_SUCCESS)
                {
                    return dbres;
                }

                // Increment the counter
                proofHashCounter += 2;

                // If we are not at the top, the new leaf hash will become part of the higher level content, based on the keys[level] bit
                if ( level >= 0 )
                {
                    siblings[level][keys[level]*4] = newLeafHash[0];
                    siblings[level][keys[level]*4 + 1] = newLeafHash[1];
                    siblings[level][keys[level]*4 + 2] = newLeafHash[2];
                    siblings[level][keys[level]*4 + 3] = newLeafHash[3];
                }
                // If this is the top, then this is the new root
                else
                {
                    newRoot[0] = newLeafHash[0];
                    newRoot[1] = newLeafHash[1];
                    newRoot[2] = newLeafHash[2];
                    newRoot[3] = newLeafHash[3];
                }
#ifdef LOG_SMT
                cout << "Smt::set() updated an existing node at level=" << level << " leaf node hash=" << fea2string(fr,newLeafHash) << " value hash=" << fea2string(fr,newValH) << endl;
#endif
            }
            else // keys are not equal, so insert with foundKey
            {
                mode = "insertFound";
#ifdef LOG_SMT
                cout << "Smt::set() mode=" << mode << endl;
#endif

                // Increase the level since we need to create a new leaf node
                int64_t level2 = level + 1;

                // Split the found key in bits
                vector <uint64_t> foundKeys;
                splitKey(foundKey, foundKeys);

                // While the key bits are the same, increase the level; we want to find the first bit when the keys differ
                while (keys[level2] == foundKeys[level2]) level2++;

                // Store the key of the old value at the new level
                Goldilocks::Element oldKey[4];
                removeKeyBits(foundKey, level2+1, oldKey);

                // Insert a new leaf node for the old value, and store the hash in oldLeafHash

                // Prepare the vector of field elements
                Goldilocks::Element v[8];
                for (uint64_t i=0; i<4; i++) v[i] = oldKey[i];
                for (uint64_t i=0; i<4; i++) v[4+i] = foundOldValH[i];

                // Prepare the capacity = 1, 0, 0, 0
                Goldilocks::Element c[4] = {fr.one(), fr.zero(), fr.zero(), fr.zero()};

                // Save and get the hash
                Goldilocks::Element oldLeafHash[4];
                dbres = hashSave(db, v, c, persistent, oldLeafHash);
                if (dbres != ZKR_SUCCESS)
                {
                    return dbres;
                }

                // Record the inserted key for the reallocated old value
                insKey[0] = foundKey[0];
                insKey[1] = foundKey[1];
                insKey[2] = foundKey[2];
                insKey[3] = foundKey[3];
                insValue = foundVal;
                isOld0 = false;

#ifdef LOG_SMT
                cout << "Smt::set() stored leaf node insValue=" << insValue.get_str(16) << " insKey=" << fea2string(fr,insKey) << endl;
#endif

                // Insert a new value node for the new value, and store the calculated hash in newValH

                // Calculate the key of the new leaf node of the new value
                Goldilocks::Element newKey[4];
                removeKeyBits(key, level2 + 1, newKey);

                // Convert the value scalar to an array of field elements
                Goldilocks::Element valueFea[8];
                scalar2fea(fr, value, valueFea);

                // Capacity is marking the node as intermediate
                c[0] = fr.zero();

                // Create the intermediate node
                Goldilocks::Element newValH[4];
                dbres = hashSave(db, valueFea, c, persistent, newValH);
                if (dbres != ZKR_SUCCESS)
                {
                    return dbres;
                }

                // Insert a new leaf node for the new key-value hash pair

                // Calculate the key-value hash content
                for (uint64_t i=0; i<4; i++) v[i] = newKey[i];
                for (uint64_t i=0; i<4; i++) v[4+i] = newValH[i];

                // Capacity is marking the node as leaf
                c[0] = fr.one();

                // Create the node and store the hash in newLeafHash
                Goldilocks::Element newLeafHash[4];
                dbres = hashSave(db, v, c, persistent, newLeafHash);
                if (dbres != ZKR_SUCCESS)
                {
                    return dbres;
                }

                // Insert a new bifurcation intermediate node with both hashes (old and new) in the right position based on the bit

                // Prepare the 2 hashes: new|old or old|new, based on the bit
                Goldilocks::Element node[8];
                for (uint64_t j=0; j<4; j++)
                {
                    node[keys[level2] * 4 + j] = newLeafHash[j];
                    node[foundKeys[level2] * 4 + j] = oldLeafHash[j];
                }

                // Capacity is marking the node as intermediate
                c[0] = fr.zero();

                // Create the node and store the calculated hash in r2
                Goldilocks::Element r2[4];
                dbres = hashSave(db, node, c, persistent, r2);
                if (dbres != ZKR_SUCCESS)
                {
                    return dbres;
                }
                proofHashCounter += 4;
                level2--;
#ifdef LOG_SMT
                cout << "Smt::set() inserted a new intermediate node level=" << level2 << " leaf node hash=" << fea2string(fr,r2) << endl;
#endif
                // Climb the branch up to the level where the key bits were common
                while (level2!=level)
                {
                    // Create all intermediate nodes, one per bit of the incremental remaining key: zero|r2 or r2|zero, based on the bit
                    for (uint64_t i = 0; i < 8; i++)
                    {
                        node[i] = fr.zero();
                    }
                    for (uint64_t j = 0; j < 4; j++)
                    {
                        node[keys[level2] * 4 + j] = r2[j];
                    }

                    // Capacity is marking the node as intermediate
                    c[0] = fr.zero();

                    // Create the intermediate node and store the calculated hash in r2
                    dbres = hashSave(db, node, c, persistent, r2);
                    if (dbres != ZKR_SUCCESS)
                    {
                        return dbres;
                    }

                    proofHashCounter += 1;

#ifdef LOG_SMT
                    cout << "Smt::set() inserted a new intermediate level=" << level2 << " leaf node hash=" << fea2string(fr,r2) << endl;
#endif

                    // Climb the branch
                    level2--;
                }

                // If not at the top of the tree, update the stored siblings for the root of the branch
                if (level >= 0)
                {
                    for (uint64_t j = 0; j < 4; j++)
                    {
                        siblings[level][keys[level] * 4 + j] = r2[j];
                    }
                }
                // If at the top of the tree, update newRoot
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
#ifdef LOG_SMT
            cout << "Smt::set() mode=" << mode << endl;
#endif
            // We could not find any key with any bit in common, so we need to create a new intermediate node, and a new leaf node

            // Value node creation

            // Build the new remaining key
            Goldilocks::Element newKey[4];
            removeKeyBits(key, level+1, newKey);

            // Convert the scalar value to an array of 8 field elements
            Goldilocks::Element valueFea[8];
            scalar2fea(fr, value, valueFea);

            // Capacity mars the node as intermediate/value
            Goldilocks::Element c[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};

            // Create the node and store the calculated hash in newValH
            Goldilocks::Element newValH[4];
            dbres = hashSave(db, valueFea, c, persistent, newValH);
            if (dbres != ZKR_SUCCESS)
            {
                return dbres;
            }

            // Insert the new key-value hash leaf node

            // Calculate the node content: key|hash
            Goldilocks::Element keyvalVector[8];
            for (uint64_t i=0; i<4; i++) keyvalVector[i] = newKey[i];
            for (uint64_t i=0; i<4; i++) keyvalVector[4+i] = newValH[i];

            // Capacity marks the node as leaf
            c[0] = fr.one();

            // Create the new leaf node and store the calculated hash in newLeafHash
            Goldilocks::Element newLeafHash[4];
            dbres = hashSave(db, keyvalVector, c, persistent, newLeafHash);
            if (dbres != ZKR_SUCCESS)
            {
                return dbres;
            }

            proofHashCounter += 2;

            // If not at the top of the tree, update siblings with the new leaf node hash
            if (level>=0)
            {
                for (uint64_t j=0; j<4; j++)
                {
                    siblings[level][keys[level]*4 + j] = newLeafHash[j];
                }
            }
            // If at the top of the tree, update the new root
            else
            {
                newRoot[0] = newLeafHash[0];
                newRoot[1] = newLeafHash[1];
                newRoot[2] = newLeafHash[2];
                newRoot[3] = newLeafHash[3];
            }
        }
    }
    // If value=0, we are possibly going to delete an existing node
    else
    {
        // Setting a value=0 in an existing key, i.e. deleting
        if ( bFoundKey && fr.equal(key[0], foundKey[0]) && fr.equal(key[1], foundKey[1]) && fr.equal(key[2], foundKey[2]) && fr.equal(key[3], foundKey[3]) ) // Delete
        {
            oldValue = foundVal;

            // If level > 0, we are going to delete and existing node (not the root node)
            if ( level >= 0)
            {
                // Set the hash of the deleted node to zero
                for (uint64_t j=0; j<4; j++)
                {
                    siblings[level][keys[level]*4 + j] = fr.zero();
                }

                // Find if there is only one non-zero hash in the siblings list for this level
                int64_t uKey = getUniqueSibling(siblings[level]);

                // If there is only one, it is the new deleted one
                if (uKey >= 0)
                {
                    mode = "deleteFound";
#ifdef LOG_SMT
                    cout << "Smt::set() mode=" << mode << endl;
#endif
                    // Calculate the key of the deleted element
                    Goldilocks::Element auxFea[4];
                    for (uint64_t i=0; i<4; i++) auxFea[i] = siblings[level][uKey*4+i];
                    string auxString = fea2string(fr, auxFea);

                    // Read its 2 siblings
                    vector<Goldilocks::Element> dbValue;
                    dbres = db.read(auxString, dbValue, dbReadLog);
                    if ( dbres != ZKR_SUCCESS)
                    {
                        cerr << "Error: Smt::set() db.read error: " << dbres << " (" << zkresult2string(dbres) << ") root:" << auxString << endl;
                        return dbres;
                    }

                    // Store them in siblings
                    siblings[level+1] = dbValue;

                    // If it is a leaf node
                    if ( siblings[level+1].size()>8 && fr.equal( siblings[level+1][8], fr.one() ) )
                    {
                        // Calculate the value hash
                        Goldilocks::Element valH[4];
                        for (uint64_t i=0; i<4; i++) valH[i] = siblings[level+1][4+i];
                        string valHString = fea2string(fr, valH);

                        // Read its siblings
                        vector<Goldilocks::Element> dbValue;
                        dbres = db.read(valHString, dbValue, dbReadLog);
                        if (dbres != ZKR_SUCCESS)
                        {
                            cerr << "Error: Smt::set() db.read error: " << dbres << " (" << zkresult2string(dbres) << ") root:" << valHString << endl;
                            return dbres;
                        }
                        else if (dbValue.size()<8)
                        {
                            cerr << "Error: Smt::set() dbValue.size()<8 root:" << valHString << endl;
                            return ZKR_SMT_INVALID_DATA_SIZE;
                        }

                        // Store the value as a scalar in val
                        Goldilocks::Element valA[8];
                        for (uint64_t i=0; i<8; i++) valA[i] = dbValue[i];
                        mpz_class val;
                        fea2scalar(fr, val, valA);

                        // Increment the counter
                        proofHashCounter += 2;

                        // Store the key in rKey
                        Goldilocks::Element rKey[4];
                        for (uint64_t i=0; i<4; i++) rKey[i] = siblings[level+1][i];

                        // Calculate the insKey
                        vector<uint64_t> auxBits;
                        auxBits = accKey;
                        auxBits.push_back(uKey);
                        joinKey(auxBits, rKey, insKey );

                        insValue = val;
                        isOld0 = false;

                        // Climb the branch until there are two siblings
                        while (uKey>=0 && level>=0)
                        {
                            level--;
                            if (level >= 0)
                            {
                                uKey = getUniqueSibling(siblings[level]);
                            }
                        }

                        // Calculate the old remaining key
                        Goldilocks::Element oldKey[4];
                        removeKeyBits(insKey, level+1, oldKey);

                        // Create the old leaf node
                        Goldilocks::Element a[8];
                        for (uint64_t i=0; i<4; i++) a[i] = oldKey[i];
                        for (uint64_t i=0; i<4; i++) a[4+i] = valH[i];

                        // Capacity marks the node as a leaf
                        Goldilocks::Element c[4] = {fr.one(), fr.zero(), fr.zero(), fr.zero()};

                        // Create node and store computed hash in oldLeafHash
                        Goldilocks::Element oldLeafHash[4];
                        dbres = hashSave(db, a, c, persistent, oldLeafHash);
                        if (dbres != ZKR_SUCCESS)
                        {
                            return dbres;
                        }

                        // Increment the counter
                        proofHashCounter += 1;

                        // If not root node, store the oldLeafHash in the sibling based on key bit
                        if (level >= 0)
                        {
                            for (uint64_t j=0; j< 4; j++)
                            {
                                siblings[level][keys[level]*4 + j] = oldLeafHash[j];
                            }
                        }
                        // If we are at the top of the tree, then update new root
                        else
                        {
                            newRoot[0] = oldLeafHash[0];
                            newRoot[1] = oldLeafHash[1];
                            newRoot[2] = oldLeafHash[2];
                            newRoot[3] = oldLeafHash[3];
                        }
                    }
                    // Not a leaf node
                    else
                    {
                        mode = "deleteNotFound";
#ifdef LOG_SMT
                        cout << "Smt::set() mode=" << mode << endl;
#endif
                    }
                }
                // 2 siblings found
                else
                {
                    mode = "deleteNotFound";
#ifdef LOG_SMT
                    cout << "Smt::set() mode=" << mode << endl;
#endif
                }
            }
            // If level=0, this means we are deleting the root node
            else
            {
                mode = "deleteLast";
#ifdef LOG_SMT
                cout << "Smt::set() mode=" << mode << endl;
#endif
                newRoot[0] = fr.zero();
                newRoot[1] = fr.zero();
                newRoot[2] = fr.zero();
                newRoot[3] = fr.zero();
            }
        }
        // Setting to zero a node that does not exist, so nothing to do
        else
        {
            mode = "zeroToZero";
            if (bFoundKey)
            {
                for (uint64_t i=0; i<4; i++) insKey[i] = foundKey[i];
                insValue = foundVal;
                isOld0 = false;
            }
#ifdef LOG_SMT
            cout << "Smt::set() mode=" << mode << endl;
#endif
        }
    }

    // Delete the extra siblings
    map< uint64_t, vector<Goldilocks::Element> >::iterator it;
    it = siblings.find(level+1);
    siblings.erase(it, siblings.end());

    // Go up the tree creating all intermediate nodes up to the new root
    while (level >= 0)
    {
        // Write the siblings and get the calculated db entry hash in newRoot
        Goldilocks::Element a[8], c[4];
        for (uint64_t i=0; i<8; i++) a[i] = siblings[level][i];
        for (uint64_t i=0; i<4; i++) c[i] = siblings[level][8+i];
        dbres = hashSave(db, a, c, persistent, newRoot);
        if (dbres != ZKR_SUCCESS)
        {
            return dbres;
        }

        // Increment the counter
        proofHashCounter += 1;

        // Go up 1 level
        level--;
        if (level >= 0)
        {
            // Overwrite the first or second 4 elements (based on keys[level] bit) with the new root hash from the lower level
            for (uint64_t j=0; j<4; j++)
            {
                siblings[level][keys[level]*4 + j] = newRoot[j];
            }
        }
    }

    if ( persistent &&
         (
            !fr.equal(oldRoot[0], newRoot[0]) ||
            !fr.equal(oldRoot[1], newRoot[1]) ||
            !fr.equal(oldRoot[2], newRoot[2]) ||
            !fr.equal(oldRoot[3], newRoot[3])
        ) )
    {
        dbres = saveStateRoot(db, newRoot);
        if (dbres != ZKR_SUCCESS)
        {
            return dbres;
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
    result.proofHashCounter = proofHashCounter;

#ifdef LOG_SMT
    cout << "Smt::set() returns isOld0=" << result.isOld0 << " insKey=" << fea2string(fr,result.insKey) << " oldValue=" << result.oldValue.get_str(16) << " newRoot=" << fea2string(fr,result.newRoot) << " mode=" << result.mode << endl << endl;
#endif
#ifdef LOG_SMT_SET_PRINT_TREE
    db.printTree(fea2string(fr,result.newRoot));
#endif

    return ZKR_SUCCESS;
}

zkresult Smt::get(Database &db, const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], SmtGetResult &result, DatabaseMap *dbReadLog)
{
#ifdef LOG_SMT
    cout << "Smt::get() called with root=" << fea2string(fr,root) << " and key=" << fea2string(fr,key) << endl;
#endif
    Goldilocks::Element r[4];
    for (uint64_t i=0; i<4; i++)
    {
        r[i] = root[i];
    }

    // Get a list of the bits of the key to navigate top-down through the tree
    vector <uint64_t> keys;
    splitKey(key, keys);

    uint64_t level = 0;

    vector<uint64_t> accKey;
    mpz_class lastAccKey = 0;
    bool bFoundKey = false;
    Goldilocks::Element foundKey[4] = {0, 0, 0, 0};
    Goldilocks::Element insKey[4] = {0, 0, 0, 0};

    map< uint64_t, vector<Goldilocks::Element> > siblings;

    mpz_class insValue = 0;
    mpz_class value = 0;
    mpz_class foundVal = 0;

    bool isOld0 = true;
    zkresult dbres;

#ifdef LOG_SMT
    //cout << "Smt::get() found database content:" << endl;
    //db.print();
#endif

    // Start natigating the tree from the top: r = root
    // Go down while r!=0 (while there is branch) until we find the key
    while ( ( !fr.isZero(r[0]) || !fr.isZero(r[1]) || !fr.isZero(r[2]) || !fr.isZero(r[3]) ) && !bFoundKey )
    {
        // Read the content of db for entry r: siblings[level] = db.read(r)
        string rString = fea2string(fr, r);
        vector<Goldilocks::Element> dbValue;
        dbres = db.read(rString, dbValue, dbReadLog);
        if (dbres != ZKR_SUCCESS)
        {
            cerr << "Error: Smt::get() db.read error: " << dbres << " (" << zkresult2string(dbres) << ") root:" << rString << endl;
            return dbres;
        }

        // Get a copy of the content of this database entry, at the corresponding level: 0, 1...
        siblings[level] = dbValue;

        // if siblings[level][8]=1 then this is a leaf
        if (siblings[level].size()>8 && fr.equal(siblings[level][8], fr.one()))
        {
            // Second 4 elements are the hash of the value, so we can get value=db(valueHash)
            Goldilocks::Element valueHashFea[4];
            valueHashFea[0] = siblings[level][4];
            valueHashFea[1] = siblings[level][5];
            valueHashFea[2] = siblings[level][6];
            valueHashFea[3] = siblings[level][7];
            string valueHashString = fea2string(fr, valueHashFea);
            vector<Goldilocks::Element> dbValue;
            dbres = db.read(valueHashString, dbValue, dbReadLog);
            if (dbres != ZKR_SUCCESS)
            {
                cerr << "Error: Smt::get() db.read error: " << dbres << " (" << zkresult2string(dbres) << ") root:" << valueHashString << endl;
                return dbres;
            }

            // First 4 elements are the remaining key
            Goldilocks::Element foundRKey[4];
            foundRKey[0] = siblings[level][0];
            foundRKey[1] = siblings[level][1];
            foundRKey[2] = siblings[level][2];
            foundRKey[3] = siblings[level][3];

            // We convert the 8 found value elements to a scalar called foundVal
            Goldilocks::Element fea[8];
            for (uint64_t i=0; i<8; i++)
            {
                fea[i] = dbValue[i];
            }
            fea2scalar(fr, foundVal, fea);

            // We construct the whole key of that value in the database, and we call it foundKey
            joinKey(accKey, foundRKey, foundKey);
            bFoundKey = true;
#ifdef LOG_SMT
            cout << "Smt::get() found at level=" << level << " value/hash=" << fea2string(fr,valueHashFea) << " foundKey=" << fea2string(fr, foundKey) << " value=" << foundVal.get_str(16) << endl;
#endif
        }
        // If this is an intermediate node
        else
        {
            // Take either the first 4 (keys[level]=0) or the second 4 (keys[level]=1) siblings as the hash of the next level
            r[0] = siblings[level][keys[level]*4];
            r[1] = siblings[level][keys[level]*4 + 1];
            r[2] = siblings[level][keys[level]*4 + 2];
            r[3] = siblings[level][keys[level]*4 + 3];

            // Store the used key bit in accKey
            accKey.push_back(keys[level]);

#ifdef LOG_SMT
            cout << "Smt::get() down 1 level=" << level << " keys[level]=" << keys[level] << " root/hash=" << fea2string(fr,r) << endl;
#endif
            // Increase the level
            level++;
        }
    }

    // One step back
    level--;
    accKey.pop_back();

    // if we found the key, then we reached a leaf node while going down the tree
    if (bFoundKey)
    {
        // if foundKey==key, then foundVal is what we were looking for
        if ( fr.equal(key[0], foundKey[0]) && fr.equal(key[1], foundKey[1]) && fr.equal(key[2], foundKey[2]) && fr.equal(key[3], foundKey[3]) )
        {
            value = foundVal;
        }
        // if foundKey!=key, then the requested value was not found
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

    // We leave the siblings only up to the leaf node level
    map< uint64_t, vector<Goldilocks::Element> >::iterator it;
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
    if (!fr.isZero(root[0]) || !fr.isZero(root[1]) || !fr.isZero(root[2]) || !fr.isZero(root[3]))
    {
        result.proofHashCounter = siblings.size();
        if ((value != 0) || !isOld0 )
        {
            result.proofHashCounter += 2;
        }
    }
    else
    {
        result.proofHashCounter = 0;
    }

#ifdef LOG_SMT
    cout << "Smt::get() returns isOld0=" << result.isOld0 << " insKey=" << fea2string(fr,result.insKey) << " and value=" << result.value.get_str(16) << endl << endl;
#endif

    return ZKR_SUCCESS;
}

// Split the fe key into 4-bits chuncks, e.g. 0x123456EF -> { 1, 2, 3, 4, 5, 6, E, F }
void Smt::splitKey ( const Goldilocks::Element (&key)[4], vector<uint64_t> &result )
{
    // Copy the key to local variables
    mpz_class auxk[4];
    for (uint64_t i=0; i<4; i++)
    {
        auxk[i] = fr.toU64(key[i]);
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
void Smt::joinKey ( const vector<uint64_t> &bits, const Goldilocks::Element (&rkey)[4], Goldilocks::Element (&key)[4] )
{
    uint64_t n[4] = {0, 0, 0, 0};
    mpz_class accs[4] = {0, 0, 0, 0};
    for (uint64_t i=0; i<bits.size(); i++)
    {
        if (bits[i])
        {
            accs[i%4] = (accs[i%4] | (mpz_class(1)<<n[i%4]))/*%fr.prime()*/;
        }
        n[i%4] += 1;
    }
    Goldilocks::Element auxk[4];
    for (uint64_t i=0; i<4; i++) auxk[i] = rkey[i];
    for (uint64_t i=0; i<4; i++)
    {
        mpz_class aux = fr.toU64(auxk[i]);
        aux = ((aux<<n[i]) | accs[i])/*%mpz_class(fr.prime())*/;
        auxk[i] = fr.fromU64(aux.get_ui());
    }
    for (uint64_t i=0; i<4; i++) key[i] = auxk[i];
}

/**
 * Removes bits from the key depending on the smt level
 * key -key
 * nBits - bits to remove
 * returns rkey - remaining key bits to store
 */
void Smt::removeKeyBits ( const Goldilocks::Element (&key)[4], uint64_t nBits, Goldilocks::Element (&rkey)[4] )
{
    uint64_t fullLevels = nBits / 4;
    mpz_class auxk[4];

    for (uint64_t i=0; i<4; i++)
    {
        auxk[i] = fr.toU64(key[i]);
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

zkresult Smt::hashSave ( Database &db, const Goldilocks::Element (&a)[8], const Goldilocks::Element (&c)[4], const bool persistent, Goldilocks::Element (&hash)[4])
{
    // Calculate the poseidon hash of the vector of field elements: v = a | c
    Goldilocks::Element v[12];
    for (uint64_t i=0; i<8; i++) v[i] = a[i];
    for (uint64_t i=0; i<4; i++) v[8+i] = c[i];
    poseidon.hash(hash, v);

    // Fill a database value with the field elements
    string hashString = fea2string(fr, hash);

    // Add the key:value pair to the database, using the hash as a key
    vector<Goldilocks::Element> dbValue;
    for (uint64_t i=0; i<8; i++) dbValue.push_back(a[i]);
    for (uint64_t i=0; i<4; i++) dbValue.push_back(c[i]);
    zkresult zkr;
    zkr = db.write(hashString, dbValue, persistent);
    if (zkr != ZKR_SUCCESS)
    {
        cerr << "Error: Smt::hashSave() failed calling db.write() key=" << hashString << " result=" << zkr << "=" << zkresult2string(zkr) << endl;
    }

#ifdef LOG_SMT
    cout << "Smt::hashSave() key=" << hashString << " value=";
    for (uint64_t i=0; i<12; i++) cout << fr.toString(dbValue[i],16) << ":";
    cout << " zkr=" << zkr;
    cout << endl;
#endif
    return zkr;
}

zkresult Smt::saveStateRoot(Database &db, const Goldilocks::Element (&stateRoot)[4])
{
    // Copy the state root in the first 4 elements of dbValue
    vector<Goldilocks::Element> dbValue;
    for (uint64_t i=0; i<4; i++) dbValue.push_back(stateRoot[i]);
    for (uint64_t i=0; i<8; i++) dbValue.push_back(fr.zero());

    // Write to db using the dbStateRootKey
    zkresult zkr;
    zkr = db.write(Database::dbStateRootKey, dbValue, true, true);
    if (zkr != ZKR_SUCCESS)
    {
        cerr << "Error: Smt::saveStateRoot() failed calling db.write() result=" << zkr << "=" << zkresult2string(zkr) << endl;
    }

#ifdef LOG_SMT
    cout << "Smt::saveStateRoot() key=" << Database::dbStateRootKey << " value=";
    for (uint64_t i=0; i<12; i++) cout << fr.toString(dbValue[i],16) << ":";
    cout << " zkr=" << zkr;
    cout << endl;
#endif
    return zkr;
}

int64_t Smt::getUniqueSibling(vector<Goldilocks::Element> &a)
{
    // Search for a unique, zero field element in vector a
    uint64_t nFound = 0;
    uint64_t fnd = 0;
    for (uint64_t i=0; i<a.size(); i+=4)
    {
        Goldilocks::Element fea[4] = {a[i], a[i+1], a[i+2], a[i+3]};
        if ( !fr.isZero(fea[0]) || !fr.isZero(fea[1]) || !fr.isZero(fea[2]) || !fr.isZero(fea[3]) )
        {
            nFound++;
            fnd = i / 4;
        }
    }
    if (nFound == 1) return fnd;
    return -1;
}

string SmtSetResult::toString (Goldilocks &fr)
{
    string result;
    result += "mode=" + mode + "\n";
    result += "oldRoot=" + fea2string(fr, oldRoot) + "\n";
    result += "newRoot=" + fea2string(fr, newRoot) + "\n";
    result += "key=" + fea2string(fr, key) + "\n";
    result += "insKey=" + fea2string(fr, insKey) + "\n";
    result += "oldValue=" + oldValue.get_str(16) + "\n";
    result += "newValue=" + newValue.get_str(16) + "\n";
    result += "insValue=" + insValue.get_str(16) + "\n";
    result += "isOld0=" + to_string(isOld0) + "\n";
    result += "proofHashCounter=" + to_string(proofHashCounter) + "\n";
    map< uint64_t, vector<Goldilocks::Element> >::const_iterator it;
    for (it=siblings.begin(); it!=siblings.end(); it++)
    {
        result += "siblings[" + to_string(it->first) + "]=";
        for (uint64_t i=0; i<it->second.size(); i++)
        {
            result += fr.toString(it->second[i], 16) + ":";
        }
        result += "\n";
    }
    return result;
}

string SmtGetResult::toString (Goldilocks &fr)
{
    string result;
    result += "root=" + fea2string(fr, root) + "\n";
    result += "key=" + fea2string(fr, key) + "\n";
    result += "insKey=" + fea2string(fr, insKey) + "\n";
    result += "value=" + value.get_str(16) + "\n";
    result += "insValue=" + insValue.get_str(16) + "\n";
    result += "isOld0=" + to_string(isOld0) + "\n";
    result += "proofHashCounter=" + to_string(proofHashCounter) + "\n";
    map< uint64_t, vector<Goldilocks::Element> >::const_iterator it;
    for (it=siblings.begin(); it!=siblings.end(); it++)
    {
        result += "siblings[" + to_string(it->first) + "]=";
        for (uint64_t i=0; i<it->second.size(); i++)
        {
            result += fr.toString(it->second[i], 16) + ":";
        }
        result += "\n";
    }
    return result;
}