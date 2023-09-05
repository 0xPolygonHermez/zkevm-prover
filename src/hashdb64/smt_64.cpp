#include "smt_64.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "zkresult.hpp"
#include "zkmax.hpp"
#include "zklog.hpp"
#include <bitset>
#include "state_manager_64.hpp"
#include "key.hpp"
#include "key_utils.hpp"
#include "tree_chunk.hpp"

//#define SMT64_PRINT_TREE_CHUNKS

zkresult Smt64::writeTree (Database64 &db, const Goldilocks::Element (&oldRoot)[4], const vector<KeyValue> &_keyValues, Goldilocks::Element (&newRoot)[4])
{
    zkresult zkr;

    vector<KeyValue> keyValues(_keyValues);

    vector<TreeChunk *> chunks;
    vector<DB64Query> dbQueries;

    // Tree level; we start at level 0, then we increase it 6 by 6
    uint64_t level = 0;

    // Create the first tree chunk (the root one), and store it in chunks[0]
    TreeChunk *c = new TreeChunk(db, poseidon);
    if (c == NULL)
    {
        zklog.error("Smt64::writeTree() failed calling new TreeChunk()");
        exitProcess();
    }
    chunks.push_back(c);

    uint64_t chunksProcessed = 0;

    // Get the old root as a string
    string oldRootString = fea2string(fr, oldRoot);

    // If old root is zero, init chunks[0] as an empty tree chunk
    if (oldRootString == "0")
    {
        chunks[0]->resetToZero(level);
    }
    else
    {
        DB64Query dbQuery(oldRootString, oldRoot, chunks[0]->data);
        dbQueries.push_back(dbQuery);
    }

    // Copy the key values list into the root tree chunk
    uint64_t keyValuesSize = keyValues.size();
    c->list.reserve(keyValuesSize);
    for (uint64_t i=0; i<keyValuesSize; i++)
    {
        c->list.emplace_back(i);
    }

    while (chunksProcessed < chunks.size())
    {
        zkr = db.read(dbQueries);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Smt64::writeTree() failed calling db.multiRead() result=" + zkresult2string(zkr));
            for (uint c = 0; c < chunks.size(); c++) delete chunks[c];
            return zkr;
        }
        dbQueries.clear();

        int chunksToProcess = chunks.size();

        for (int i=chunksProcessed; i<chunksToProcess; i++)
        {
            chunks[i]->setLevel(level);
            if (chunks[i]->data.size() > 0)
            {
                zkr = chunks[i]->data2children();
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("Smt64::writeTree() failed calling chunks[i]->data2children() result=" + zkresult2string(zkr));
                    return zkr;
                }
            }
            for (uint64_t j=0; j<chunks[i]->list.size(); j++)
            {
                bool keyBits[256];
                splitKey(fr, keyValues[chunks[i]->list[j]].key.fe, keyBits);
                uint64_t k = getKeyChildren64Position(keyBits, level);
                switch (chunks[i]->getChild(k).type)
                {
                    case ZERO:
                    {
                        if (keyValues[chunks[i]->list[j]].value != 0)
                        {
                            chunks[i]->setLeafChild(k, keyValues[chunks[i]->list[j]].key.fe, keyValues[chunks[i]->list[j]].value);                  
                        }
                        break;
                    }
                    case LEAF:
                    {
                        // If the key is the same, then check the value
                        if (fr.equal(chunks[i]->getChild(k).leaf.key[0], keyValues[chunks[i]->list[j]].key.fe[0]) &&
                            fr.equal(chunks[i]->getChild(k).leaf.key[1], keyValues[chunks[i]->list[j]].key.fe[1]) &&
                            fr.equal(chunks[i]->getChild(k).leaf.key[2], keyValues[chunks[i]->list[j]].key.fe[2]) &&
                            fr.equal(chunks[i]->getChild(k).leaf.key[3], keyValues[chunks[i]->list[j]].key.fe[3]))
                        {
                            // If value is different, copy it
                            if (chunks[i]->getChild(k).leaf.value != keyValues[chunks[i]->list[j]].value)
                            {
                                if (keyValues[chunks[i]->list[j]].value == 0)
                                {
                                    chunks[i]->setZeroChild(k);
                                }
                                else
                                {
                                    chunks[i]->setLeafChild(k, keyValues[chunks[i]->list[j]].key.fe, keyValues[chunks[i]->list[j]].value);
                                }
                            }
                        }
                        else
                        {
                            // We create a new trunk
                            TreeChunk *c = new TreeChunk(db, poseidon);
                            if (c == NULL)
                            {
                                zklog.error("Smt64::writeTree() failed calling new TreeChunk()");
                                exitProcess();
                            }

                            // Reset to zero
                            c->resetToZero(level + 6);

                            // We create a KeyValue from the original leaf node
                            KeyValue kv;
                            kv.key.fe[0] = chunks[i]->getChild(k).leaf.key[0];
                            kv.key.fe[1] = chunks[i]->getChild(k).leaf.key[1];
                            kv.key.fe[2] = chunks[i]->getChild(k).leaf.key[2];
                            kv.key.fe[3] = chunks[i]->getChild(k).leaf.key[3];
                            kv.value = chunks[i]->getChild(k).leaf.value;

                            // We add to the list the original leaf node
                            keyValues.emplace_back(kv);
                            c->list.emplace_back(keyValues.size()-1);

                            // We add to the list the new key-value
                            c->list.emplace_back(chunks[i]->list[j]);

                            int cId = chunks.size();
                            chunks.push_back(c);
                            chunks[i]->setTreeChunkChild(k, cId);
                        }
                        break;
                    }
                    case TREE_CHUNK:
                    {
                        // Simply add it to the list of the descendant tree chunk
                        chunks[chunks[i]->getChild(k).treeChunkId]->list.push_back(chunks[i]->list[j]);

                        break;
                    }
                    // If this is an intermediate node, then create the corresponding tree chunk
                    case INTERMEDIATE:
                    {
                        // We create a new trunk
                        TreeChunk *c = new TreeChunk(db, poseidon);
                        if (c == NULL)
                        {
                            zklog.error("Smt64::writeTree() failed calling new TreeChunk()");
                            exitProcess();
                        }
                        c->setLevel(level + 6);
                        
                        // Create a new query to populate this tree chunk from database
                        DB64Query dbQuery(fea2string(fr, chunks[i]->getChild(k).intermediate.hash),
                                        chunks[i]->getChild(k).intermediate.hash,
                                        c->data);
                        dbQueries.push_back(dbQuery);

                        // Add the requested key-value to the new tree chunk list
                        c->list.push_back(chunks[i]->list[j]);
                        int cId = chunks.size();
                        chunks.push_back(c);
                        chunks[i]->setTreeChunkChild(k, cId);

                        break;
                    }
                    default:
                    {
                        zklog.error("Smt64::writeTree() found invalid chunks[i]->getChild(k).type=" + to_string(chunks[i]->getChild(k).type));
                        exitProcess();
                    }
                }
            }
        }

        chunksProcessed = chunksToProcess;
        level += 6;
    }

    dbQueries.clear();

    // Calculate the new root hash of the whole tree
    Child result;
    zkr = calculateHash(result, chunks, dbQueries, 0, 0);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Smt64::writeTree() failed calling calculateHash() result=" + zkresult2string(zkr));
        for (uint c = 0; c < chunks.size(); c++) delete chunks[c];
        return zkr;
    }

    // Based on the result, calculate the new root hash
    if (result.type == LEAF)
    {
        newRoot[0] = result.leaf.hash[0];
        newRoot[1] = result.leaf.hash[1];
        newRoot[2] = result.leaf.hash[2];
        newRoot[3] = result.leaf.hash[3];
        string newRootString = fea2string(fr, newRoot);

        if (!chunks[0]->getDataValid())
        {
            zkr = chunks[0]->children2data();
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("Smt64::writeTree() failed calling chunks[0]->children2data() result=" + zkresult2string(zkr));
                for (uint c = 0; c < chunks.size(); c++) delete chunks[c];
                return zkr;
            }
            DB64Query dbQuery(newRootString, newRoot, chunks[0]->data);
            dbQueries.push_back(dbQuery);
        }
    }
    else if (result.type == INTERMEDIATE)
    {
        newRoot[0] = result.intermediate.hash[0];
        newRoot[1] = result.intermediate.hash[1];
        newRoot[2] = result.intermediate.hash[2];
        newRoot[3] = result.intermediate.hash[3];
    }
    else if (result.type == ZERO)
    { 
        newRoot[0] = fr.zero();
        newRoot[1] = fr.zero();
        newRoot[2] = fr.zero();
        newRoot[3] = fr.zero();
    }
    else
    {
        zklog.error("Smt64::writeTree() found invalid result.type=" + to_string(result.type));
        for (uint c = 0; c < chunks.size(); c++) delete chunks[c];
        return zkr;
    }

    // Save chunks data to database
    zkr = db.write(dbQueries, true);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Smt64::writeTree() failed calling db.write() result=" + zkresult2string(zkr));
        for (uint c = 0; c < chunks.size(); c++) delete chunks[c];
        return zkr;
    }

#ifdef SMT64_PRINT_TREE_CHUNKS
    // Print chunks
    for (uint c = 0; c < chunks.size(); c++)
    {
        zklog.info("Smt64::writeTree() chunk " + to_string(c));
        chunks[c]->print();
    }
#endif

    // Free memory
    for (uint c = 0; c < chunks.size(); c++) delete chunks[c];

    return ZKR_SUCCESS;
}

zkresult Smt64::calculateHash (Child &result, vector<TreeChunk *> &chunks, vector<DB64Query> &dbQueries, int chunkId, int level)
{
    zkresult zkr;
    vector<Child> results(64);

    // Convert all TREE_CHUNK children into something else, typically INTERMEDIATE children,
    // but they could also be LEAF (only one child below this level) or ZERO 9no children below this level)
    for (uint64_t i=0; i<64; i++)
    {
        if (chunks[chunkId]->getChild(i).type == TREE_CHUNK)
        {
            calculateHash(result, chunks, dbQueries, chunks[chunkId]->getChild(i).treeChunkId, level + 6);
            chunks[chunkId]->setChild(i, result);
        }
    }

    // Calculate the hash of this chunk
    zkr = chunks[chunkId]->calculateHash();
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Smt64::recalculateHash() failed calling chunks[chunkId]->calculateHash() result=" + zkresult2string(zkr));
        return zkr;
    }

    // Copy the result child
    result = chunks[chunkId]->getChild1();

    // Add to the database queries
    if (result.type != ZERO)
    {
        // Encode the 64 children into database format
        zkr = chunks[chunkId]->children2data();
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Smt64::recalculateHash() failed calling chunks[chunkId]->children2data() result=" + zkresult2string(zkr));
            return zkr;
        }

        Goldilocks::Element hash[4];
        chunks[chunkId]->getHash(hash);
        DB64Query dbQuery(fea2string(fr, hash), hash, chunks[chunkId]->data);
        dbQueries.emplace_back(dbQuery);
    }

    return ZKR_SUCCESS;
}

zkresult Smt64::readTree (Database64 &db, const Goldilocks::Element (&root)[4], vector<KeyValue> &keyValues)
{
    zkresult zkr;

    vector<TreeChunk *> chunks;
    vector<DB64Query> dbQueries;

    // Tree level; we start at level 0, then we increase it 6 by 6
    uint64_t level = 0;

    // Create the first tree chunk (the root one), and store it in chunks[0]
    TreeChunk *c = new TreeChunk(db, poseidon);
    if (c == NULL)
    {
        zklog.error("Smt64::readTree() failed calling new TreeChunk()");
        exitProcess();
    }
    chunks.push_back(c);

    uint64_t chunksProcessed = 0;

    // Get the old root as a string
    string rootString = fea2string(fr, root);

    // If root is zero, return all values as zero
    if (rootString == "0")
    {
        delete c;
        for (uint64_t i=0; i<keyValues.size(); i++)
        {
            keyValues[i].value = 0;
        }
        return ZKR_SUCCESS;
    }
    else
    {
        DB64Query dbQuery(rootString, root, chunks[0]->data);
        dbQueries.push_back(dbQuery);
    }

    // Copy the key values list into the root tree chunk
    uint64_t keyValuesSize = keyValues.size();
    c->list.reserve(keyValuesSize);
    for (uint64_t i=0; i<keyValuesSize; i++)
    {
        c->list.emplace_back(i);
    }

    while (chunksProcessed < chunks.size())
    {
        zkr = db.read(dbQueries);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Smt64::readTree() failed calling db.multiRead() result=" + zkresult2string(zkr));
            for (uint c = 0; c < chunks.size(); c++) delete chunks[c];
            return zkr;
        }
        dbQueries.clear();

        int chunksToProcess = chunks.size();

        for (int i=chunksProcessed; i<chunksToProcess; i++)
        {
            chunks[i]->setLevel(level);
            if (chunks[i]->data.size() > 0)
            {
                zkr = chunks[i]->data2children();
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("Smt64::readTree() failed calling chunks[i]->data2children() result=" + zkresult2string(zkr));
                    return zkr;
                }
            }
            for (uint64_t j=0; j<chunks[i]->list.size(); j++)
            {
                bool keyBits[256];
                splitKey(fr, keyValues[chunks[i]->list[j]].key.fe, keyBits);
                uint64_t k = getKeyChildren64Position(keyBits, level);
                switch (chunks[i]->getChild(k).type)
                {
                    case ZERO:
                    {
                        for (uint64_t kv=0; kv<keyValues.size(); kv++)
                        {
                            if (fr.equal(keyValues[kv].key.fe[0], keyValues[chunks[i]->list[j]].key.fe[0]) &&
                                fr.equal(keyValues[kv].key.fe[1], keyValues[chunks[i]->list[j]].key.fe[1]) &&
                                fr.equal(keyValues[kv].key.fe[2], keyValues[chunks[i]->list[j]].key.fe[2]) &&
                                fr.equal(keyValues[kv].key.fe[3], keyValues[chunks[i]->list[j]].key.fe[3]))
                            {
                                keyValues[kv].value = 0;
                            }
                        }
                        break;
                    }
                    case LEAF:
                    {
                        // If the key is the same, then check the value
                        if (fr.equal(chunks[i]->getChild(k).leaf.key[0], keyValues[chunks[i]->list[j]].key.fe[0]) &&
                            fr.equal(chunks[i]->getChild(k).leaf.key[1], keyValues[chunks[i]->list[j]].key.fe[1]) &&
                            fr.equal(chunks[i]->getChild(k).leaf.key[2], keyValues[chunks[i]->list[j]].key.fe[2]) &&
                            fr.equal(chunks[i]->getChild(k).leaf.key[3], keyValues[chunks[i]->list[j]].key.fe[3]))
                        {
                            keyValues[chunks[i]->list[j]].value = chunks[i]->getChild(k).leaf.value;
                        }
                        else
                        {
                            keyValues[chunks[i]->list[j]].value = 0;
                        }
                        break;
                    }
                    case TREE_CHUNK:
                    {
                        // Simply add it to the list of the descendant tree chunk
                        chunks[chunks[i]->getChild(k).treeChunkId]->list.push_back(chunks[i]->list[j]);

                        break;
                    }
                    // If this is an intermediate node, then create the corresponding tree chunk
                    case INTERMEDIATE:
                    {
                        // We create a new trunk
                        TreeChunk *c = new TreeChunk(db, poseidon);
                        if (c == NULL)
                        {
                            zklog.error("Smt64::readTree() failed calling new TreeChunk()");
                            exitProcess();
                        }
                        c->setLevel(level + 6);
                        
                        // Create a new query to populate this tree chunk from database
                        DB64Query dbQuery(fea2string(fr, chunks[i]->getChild(k).intermediate.hash),
                                        chunks[i]->getChild(k).intermediate.hash,
                                        c->data);
                        dbQueries.push_back(dbQuery);

                        // Add the requested key-value to the new tree chunk list
                        c->list.push_back(chunks[i]->list[j]);
                        int cId = chunks.size();
                        chunks.push_back(c);
                        chunks[i]->setTreeChunkChild(k, cId);

                        break;
                    }
                    default:
                    {
                        zklog.error("Smt64::readTree() found invalid chunks[i]->getChild(k).type=" + to_string(chunks[i]->getChild(k).type));
                        exitProcess();
                    }
                }
            }
        }

        chunksProcessed = chunksToProcess;
        level += 6;
    }

    dbQueries.clear();

#ifdef SMT64_PRINT_TREE_CHUNKS
    // Print chunks
    for (uint c = 0; c < chunks.size(); c++)
    {
        zklog.info("Smt64::readTree() chunk " + to_string(c));
        chunks[c]->print();
    }
#endif

    // Free memory
    for (uint c = 0; c < chunks.size(); c++) delete chunks[c];

    return ZKR_SUCCESS;
}

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
        bool bIsVirtual = stateManager64.isVirtualStateRoot(oldRootString);
        stateManager64.setOldStateRoot(batchUUID, tx, oldRootString, bIsVirtual, persistence);

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
        stateManager64.setNewStateRoot(batchUUID, tx, newRootString, true, persistence);
    }
    else
    {
        // TODO: implementation
    }

#if 0
#ifdef LOG_SMT
    zklog.info("Smt64::set() called with oldRoot=" + fea2string(fr,oldRoot) + " key=" + fea2string(fr,key) + " value=" + value.get_str(16) + " persistent=" + to_string(persistent));
#endif

    zkresult zkr;

    bool bUseStateManager = db.config.stateManager && (batchUUID.size() > 0);

    SmtContext64 ctx(db, bUseStateManager, batchUUID, tx, persistence);

    if (bUseStateManager)
    {
        stateManager64.setOldStateRoot(batchUUID, tx, fea2string(fr, oldRoot), persistence);
    }

    Goldilocks::Element r[4];
    for (uint64_t i=0; i<4; i++) r[i] = oldRoot[i];
    Goldilocks::Element newRoot[4];
    for (uint64_t i=0; i<4; i++) newRoot[i] = oldRoot[i];

    // Get a list of the bits of the key to navigate top-down through the tree
    bool keys[256];
    splitKey(fr, key, keys);

    int64_t level = 0;
    uint64_t proofHashCounter = 0;

    vector<uint64_t> accKey;
    mpz_class lastAccKey = 0;
    bool bFoundKey = false;
    Goldilocks::Element foundKey[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};
    //Goldilocks::Element foundRKey[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};
    Goldilocks::Element insKey[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};

    map< uint64_t, vector<Goldilocks::Element> > siblings;

    vector<TreeChunk *> chunks;

    vector<string> nodesToDelete; // vector to store all nodes keys to delete because they are no longer part of the tree
    //Goldilocks::Element nodeToDelete[4]; // key, in field element format, of a node to delete
    string nodeToDeleteString; // key, in string format, of a node to delete

    mpz_class insValue = 0;
    mpz_class oldValue = 0;
    mpz_class foundValue = 0;
    //Goldilocks::Element foundValueHash[4] = {fr.zero(), fr.zero(), fr.zero(), fr.zero()};
    string foundValueHashString;

    string mode;

    bool isOld0 = true;
    zkresult dbres;
    string dbValue; // used to call db.read()

    // Start natigating the tree from the top: r = root
    // Go down while r!=0 (while there is branch) until we find the key
    while ( (!fr.isZero(r[0]) || !fr.isZero(r[1]) || !fr.isZero(r[2]) || !fr.isZero(r[3])) && !bFoundKey )
    {
        // Read the content of db for entry r: siblings[level] = db.read(r)
        string rootString = fea2string(fr, r);

        dbres = ZKR_UNSPECIFIED;
        if (bUseStateManager)
        {
            dbres = stateManager64.read(batchUUID, rootString, dbValue, dbReadLog);
        }
        if (dbres != ZKR_SUCCESS)
        {
            dbres = db.read(rootString, r, dbValue, dbReadLog, false, keys, level);
        }
        if (dbres != ZKR_SUCCESS)
        {
            zklog.error("Smt64::set() db.read error: " + to_string(dbres) + " (" + zkresult2string(dbres) + ") root:" + rootString);
            return dbres;
        }

        TreeChunk * pTreeChunk = new TreeChunk(db, poseidon);
        chunks.emplace_back(pTreeChunk);
        pTreeChunk->level = level;
        pTreeChunk->data = dbValue;
        pTreeChunk->bDataValid = true;
        zkr = pTreeChunk->data2children();
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Smt64::set() failed calling treeChunk.data2children() result=" + zkresult2string(zkr));
            return zkr;
        }
        uint64_t children64Position = getKeyChildren64Position(keys, level);


        // Get a copy of the content of this database entry, at the corresponding level: 0, 1...
        /*siblings[level].resize(12);
        siblings[level][0].fe = dbValue[0].fe;
        siblings[level][1].fe = dbValue[1].fe;
        siblings[level][2].fe = dbValue[2].fe;
        siblings[level][3].fe = dbValue[3].fe;
        siblings[level][4].fe = dbValue[4].fe;
        siblings[level][5].fe = dbValue[5].fe;
        siblings[level][6].fe = dbValue[6].fe;
        siblings[level][7].fe = dbValue[7].fe;
        siblings[level][8].fe = dbValue[8].fe;
        siblings[level][9].fe = dbValue[9].fe;
        siblings[level][10].fe = dbValue[10].fe;
        siblings[level][11].fe = dbValue[11].fe;*/
        
        // if siblings[level][8]=1 then this is a leaf node
        //if ( siblings[level].size()>8 && fr.equal(siblings[level][8], fr.one()) )
        if (pTreeChunk->children64[children64Position].type == LEAF)
        {
            // Second 4 elements are the hash of the old value, so we can get old value=db(valueHash)
            /*foundValueHash[0] = siblings[level][4];
            foundValueHash[1] = siblings[level][5];
            foundValueHash[2] = siblings[level][6];
            foundValueHash[3] = siblings[level][7];
            foundValueHashString = fea2string(fr, foundValueHash);
            dbres = ZKR_UNSPECIFIED;
            if (bUseStateManager)
            {
                dbres = stateManager64.read(batchUUID, foundValueHashString, dbValue, dbReadLog);
            }
            if (dbres != ZKR_SUCCESS)
            {
                dbres = db.read(foundValueHashString, foundValueHash, dbValue, dbReadLog);
            }
            if (dbres != ZKR_SUCCESS)
            {
                zklog.error("Smt64::set() db.read error: " + to_string(dbres) + " (" + zkresult2string(dbres) + ") key:" + foundValueHashString);
                return dbres;
            }*/

            // Convert the 8 found value fields to a foundValue scalar
            //Goldilocks::Element valueFea[8];
            //for (uint64_t i=0; i<8; i++) valueFea[i] = dbValue[i];
            //fea2scalar(fr, foundValue, valueFea);
            foundValue = pTreeChunk->children64[children64Position].leaf.value;

            // First 4 elements are the remaining key of the old value
            //foundRKey[0] = siblings[level][0];
            //foundRKey[1] = siblings[level][1];
            //foundRKey[2] = siblings[level][2];
            //foundRKey[3] = siblings[level][3];

            // Joining the consumed key bits, we have the complete found key of the old value
            //joinKey(fr, accKey, foundRKey, foundKey);
            foundKey[0] = pTreeChunk->children64[children64Position].leaf.key[0];
            foundKey[1] = pTreeChunk->children64[children64Position].leaf.key[1];
            foundKey[2] = pTreeChunk->children64[children64Position].leaf.key[2];
            foundKey[3] = pTreeChunk->children64[children64Position].leaf.key[3];
            bFoundKey = true;

#ifdef LOG_SMT
            zklog.info("Smt64::set() found at level=" + to_string(level) + " foundValue=" + foundValue.get_str(16) + " foundKey=" + fea2string(fr,foundKey) + " foundRKey=" + fea2string(fr,foundRKey));
#endif
        }
        // This is a zero node
        else if (pTreeChunk->children64[children64Position].type == ZERO)
        {
            bFoundKey = true;
        }
        // This is an intermediate node
        else
        {
            // Take either the first 4 (keys[level]=0) or the second 4 (keys[level]=1) siblings as the hash of the next level
            //r[0] = siblings[level][keys[level]*4];
            //r[1] = siblings[level][keys[level]*4 + 1];
            //r[2] = siblings[level][keys[level]*4 + 2];
            //r[3] = siblings[level][keys[level]*4 + 3];

            // Store the used key bit in accKey
            //accKey.push_back(keys[level]);

#ifdef LOG_SMT
            zklog.info("Smt64::set() down 1 level=" + to_string(level) + " keys[level]=" + to_string(keys[level]) + " root/hash=" + fea2string(fr,r));
#endif
            // Increase the level
            level += 6;
        }
    }

    // One step back
    //level--;
    //accKey.pop_back();

    // Calculate the number of hashes needed so far
    if (!fr.isZero(oldRoot[0]) || !fr.isZero(oldRoot[1]) || !fr.isZero(oldRoot[2]) || !fr.isZero(oldRoot[3]))
    {
        proofHashCounter = zkmin(siblings.size(), uint64_t(level+1));
        if (foundValue)
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
                zklog.info("Smt64::set() mode=" + mode);
#endif
                oldValue = foundValue;

                // First, we create the db entry for the new VALUE, and store the calculated hash in newValH
                /*Goldilocks::Element v[8];
                scalar2fea(fr, value, v);

                // Save and get the new value hash
                Goldilocks::Element newValH[4];
                //dbres = hashSaveZero(ctx, v, newValH);
                if (dbres != ZKR_SUCCESS)
                {
                    return dbres;
                }

                // Second, we create the db entry for the new leaf node = RKEY + HASH, and store the calculated hash in newLeafHash
                for (uint64_t i=0; i<4; i++) v[i] = foundRKey[i];
                for (uint64_t i=0; i<4; i++) v[4+i] = newValH[i];

                // Save and get the new leaf node hash
                Goldilocks::Element newLeafHash[4];
                //dbres = hashSaveOne(ctx, v, newLeafHash);
                if (dbres != ZKR_SUCCESS)
                {
                    return dbres;
                }

                // Increment the hash counter
                proofHashCounter += 2;

                // If we are not at the top, the new leaf hash will become part of the higher level content, based on the keys[level] bit
                if ( level >= 0 )
                {
                    if (bUseStateManager && (foundValue != value))
                    {
                        for (uint64_t j=0; j<4; j++)
                        {
                            nodeToDelete[j] = siblings[level][keys[level]*4 + j];
                            siblings[level][keys[level]*4 + j] = newLeafHash[j];
                        }
                        if (!fr.equal(nodeToDelete[0], newLeafHash[0]) || !fr.equal(nodeToDelete[1], newLeafHash[1]) || !fr.equal(nodeToDelete[2], newLeafHash[2]) || !fr.equal(nodeToDelete[3], newLeafHash[3]))
                        {
                            nodeToDeleteString = fea2string(fr, nodeToDelete);
                            if (nodeToDeleteString != "0")
                            {
                                stateManager64.deleteNode(batchUUID, tx, nodeToDeleteString, persistence);
                            }
                        }
                    }
                    else
                    {
                        for (uint64_t j=0; j<4; j++)
                        {
                            siblings[level][keys[level]*4 + j] = newLeafHash[j];
                        }
                    }
                }
                // If this is the top, then this is the new root
                else
                {
                    newRoot[0] = newLeafHash[0];
                    newRoot[1] = newLeafHash[1];
                    newRoot[2] = newLeafHash[2];
                    newRoot[3] = newLeafHash[3];
                }*/

                uint64_t children64Position = getKeyChildren64Position(keys, level);

                chunks[chunks.size()-1]->children64[children64Position].leaf.value = value;
                chunks[chunks.size()-1]->bChildrenRestValid = false;
                chunks[chunks.size()-1]->bHashValid = false;
                chunks[chunks.size()-1]->bDataValid = false;
                zkr = chunks[chunks.size()-1]->calculateHash();
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("Smt64::set() failed calling chunks[chunks.size()-1]->calculateHash() zkr=" + zkresult2string(zkr));
                    // TODO: delete chunks
                    return zkr;
                }
                zkr = chunks[chunks.size()-1]->children2data();
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("Smt64::set() failed calling chunks[chunks.size()-1]->children2data() zkr=" + zkresult2string(zkr));
                    // TODO: delete chunks
                    return zkr;
                }

                zkr = hashSave(ctx, *chunks[chunks.size()-1]);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("SMT64::Set() failed calling hashSave() result=" + zkresult2string(zkr));
                    return zkr;
                }
                newRoot[0] = chunks[chunks.size()-1]->hash[0];
                newRoot[1] = chunks[chunks.size()-1]->hash[1];
                newRoot[2] = chunks[chunks.size()-1]->hash[2];
                newRoot[3] = chunks[chunks.size()-1]->hash[3];

#ifdef LOG_SMT
                zklog.info("Smt64::set() updated an existing node at level=" + to_string(level) + " leaf node hash=" + fea2string(fr,newLeafHash) + " value hash=" + fea2string(fr,newValH));
#endif
            }
            else // keys are not equal, so insert with foundKey
            {
                mode = "insertFound";
#ifdef LOG_SMT
                zklog.info("Smt64::set() mode=" + mode);
#endif
                uint64_t children64Position = getKeyChildren64Position(keys, level);

                chunks[chunks.size()-1]->children64[children64Position].type = LEAF;
                chunks[chunks.size()-1]->children64[children64Position].leaf.key[0] = key[0];
                chunks[chunks.size()-1]->children64[children64Position].leaf.key[1] = key[1];
                chunks[chunks.size()-1]->children64[children64Position].leaf.key[2] = key[2];
                chunks[chunks.size()-1]->children64[children64Position].leaf.key[3] = key[3];
                chunks[chunks.size()-1]->children64[children64Position].leaf.value = value;
                chunks[chunks.size()-1]->bChildrenRestValid = false;
                chunks[chunks.size()-1]->bHashValid = false;
                chunks[chunks.size()-1]->bDataValid = false;
                zkr = chunks[chunks.size()-1]->calculateHash();
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("Smt64::set() failed calling chunks[chunks.size()-1]->calculateHash() zkr=" + zkresult2string(zkr));
                    // TODO: delete chunks
                    return zkr;
                }
                zkr = chunks[chunks.size()-1]->children2data();
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("Smt64::set() failed calling chunks[chunks.size()-1]->children2data() zkr=" + zkresult2string(zkr));
                    // TODO: delete chunks
                    return zkr;
                }

                zkr = hashSave(ctx, *chunks[chunks.size()-1]);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("SMT64::Set() failed calling hashSave() result=" + zkresult2string(zkr));
                    return zkr;
                }
                newRoot[0] = chunks[chunks.size()-1]->hash[0];
                newRoot[1] = chunks[chunks.size()-1]->hash[1];
                newRoot[2] = chunks[chunks.size()-1]->hash[2];
                newRoot[3] = chunks[chunks.size()-1]->hash[3];

                // Increase the level since we need to create a new leaf node
                /*int64_t level2 = level + 1;

                // Split the found key in bits
                bool foundKeys[256];
                splitKey(fr, foundKey, foundKeys);

                // While the key bits are the same, increase the level; we want to find the first bit when the keys differ
                while (keys[level2] == foundKeys[level2]) level2++;

                // Store the key of the old value at the new level
                Goldilocks::Element oldKey[4];
                removeKeyBits(fr, foundKey, level2+1, oldKey);

                // Insert a new leaf node for the old value, and store the hash in oldLeafHash

                // Prepare the vector of field elements
                Goldilocks::Element v[8];
                for (uint64_t i=0; i<4; i++) v[i] = oldKey[i];
                for (uint64_t i=0; i<4; i++) v[4+i] = foundValueHash[i];

                // Save and get the hash
                Goldilocks::Element oldLeafHash[4];
                //dbres = hashSaveOne(ctx, v, oldLeafHash);
                if (dbres != ZKR_SUCCESS)
                {
                    return dbres;
                }

                // Record the inserted key for the reallocated old value
                insKey[0] = foundKey[0];
                insKey[1] = foundKey[1];
                insKey[2] = foundKey[2];
                insKey[3] = foundKey[3];
                insValue = foundValue;
                isOld0 = false;

#ifdef LOG_SMT
                zklog.info("Smt64::set() stored leaf node insValue=" + insValue.get_str(16) + " insKey=" + fea2string(fr,insKey));
#endif

                // Insert a new value node for the new value, and store the calculated hash in newValH

                // Calculate the key of the new leaf node of the new value
                Goldilocks::Element newKey[4];
                removeKeyBits(fr, key, level2 + 1, newKey);

                // Convert the value scalar to an array of field elements
                Goldilocks::Element valueFea[8];
                scalar2fea(fr, value, valueFea);

                // Create the value node
                Goldilocks::Element newValH[4];
                //dbres = hashSaveZero(ctx, valueFea, newValH);
                if (dbres != ZKR_SUCCESS)
                {
                    return dbres;
                }

                // Insert a new leaf node for the new key-value hash pair

                // Calculate the key-value hash content
                for (uint64_t i=0; i<4; i++) v[i] = newKey[i];
                for (uint64_t i=0; i<4; i++) v[4+i] = newValH[i];

                // Create the leaf node and store the hash in newLeafHash
                Goldilocks::Element newLeafHash[4];
                //dbres = hashSaveOne(ctx, v, newLeafHash);
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

                // Create the intermediate node and store the calculated hash in r2
                Goldilocks::Element r2[4];
                //dbres = hashSaveZero(ctx, node, r2);
                if (dbres != ZKR_SUCCESS)
                {
                    return dbres;
                }
                proofHashCounter += 4;
                level2--;
#ifdef LOG_SMT
                zklog.info("Smt64::set() inserted a new intermediate node level=" + to_string(level2) + " leaf node hash=" + fea2string(fr,r2));
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

                    // Create the intermediate node and store the calculated hash in r2
                    //dbres = hashSaveZero(ctx, node, r2);
                    if (dbres != ZKR_SUCCESS)
                    {
                        return dbres;
                    }

                    proofHashCounter += 1;

#ifdef LOG_SMT
                    zklog.info("Smt64::set() inserted a new intermediate level=" + to_string(level2) + " leaf node hash=" + fea2string(fr,r2));
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
                }*/
            }
        }
        else // insert without foundKey
        {
            mode = "insertNotFound";
#ifdef LOG_SMT
            zklog.info("Smt64::set() mode=" + mode);
#endif
            // We could not find any key with any bit in common, so we need to create a new intermediate node, and a new leaf node

            // Value node creation
/*
            // Build the new remaining key
            Goldilocks::Element newKey[4];
            removeKeyBits(fr, key, level+1, newKey);

            // Convert the scalar value to an array of 8 field elements
            Goldilocks::Element valueFea[8];
            scalar2fea(fr, value, valueFea);

            // Create the value node and store the calculated hash in newValH
            Goldilocks::Element newValH[4];
            dbres = hashSaveZero(ctx, valueFea, newValH);
            if (dbres != ZKR_SUCCESS)
            {
                return dbres;
            }

            // Insert the new key-value hash leaf node

            // Calculate the node content: key|hash
            Goldilocks::Element keyvalVector[8];
            for (uint64_t i=0; i<4; i++) keyvalVector[i] = newKey[i];
            for (uint64_t i=0; i<4; i++) keyvalVector[4+i] = newValH[i];

            // Create the new leaf node and store the calculated hash in newLeafHash
            Goldilocks::Element newLeafHash[4];
            dbres = hashSaveOne(ctx, keyvalVector, newLeafHash);
            if (dbres != ZKR_SUCCESS)
            {
                return dbres;
            }

            proofHashCounter += 2;
            */

            // If not at the top of the tree, update siblings with the new leaf node hash
            //if (level>=0)
            if (level > 0)
            {
                /*
                if (bUseStateManager)
                {
                    for (uint64_t j=0; j<4; j++)
                    {
                        nodeToDelete[j] = siblings[level][keys[level]*4 + j];
                        siblings[level][keys[level]*4 + j] = newLeafHash[j];
                    }
                    if (!fr.equal(nodeToDelete[0], newLeafHash[0]) || !fr.equal(nodeToDelete[1], newLeafHash[1]) || !fr.equal(nodeToDelete[2], newLeafHash[2]) || !fr.equal(nodeToDelete[3], newLeafHash[3]))
                    {
                        nodeToDeleteString = fea2string(fr, nodeToDelete);
                        if (nodeToDeleteString != "0")
                        {
                            stateManager64.deleteNode(batchUUID, tx, nodeToDeleteString, persistence);
                        }
                    }
                }
                else
                {
                    for (uint64_t j=0; j<4; j++)
                    {
                        siblings[level][keys[level]*4 + j] = newLeafHash[j];
                    }
                }
                */
            }
            // If at the top of the tree, update the new root
            else
            {
                TreeChunk treeChunk(db, poseidon);
                treeChunk.level = 0;
                for (uint64_t child=0; child<64; child++)
                {
                    treeChunk.children64[child].type = ZERO;
                }
                uint64_t children64Position = getKeyChildren64Position(keys, level);
                treeChunk.children64[children64Position].type = LEAF;
                treeChunk.children64[children64Position].leaf.key[0] = key[0];
                treeChunk.children64[children64Position].leaf.key[1] = key[1];
                treeChunk.children64[children64Position].leaf.key[2] = key[2];
                treeChunk.children64[children64Position].leaf.key[3] = key[3];
                treeChunk.children64[children64Position].leaf.level = 0;
                treeChunk.children64[children64Position].leaf.value = value;
                treeChunk.children64[children64Position].leaf.calculateHash(fr, poseidon);
                treeChunk.bChildren64Valid = true;
                zkr = treeChunk.calculateHash();
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("SMT64::Set() failed calling treeChunk.calculateHash() result=" + zkresult2string(zkr));
                    return zkr;
                }
                //treeChunk.print();

                zkr = treeChunk.children2data();
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("SMT64::Set() failed calling treeChunk.children2data() result=" + zkresult2string(zkr));
                    return zkr;
                }
                zkr = hashSave(ctx, treeChunk);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("SMT64::Set() failed calling hashSave() result=" + zkresult2string(zkr));
                    return zkr;
                }

                newRoot[0] = treeChunk.hash[0];
                newRoot[1] = treeChunk.hash[1];
                newRoot[2] = treeChunk.hash[2];
                newRoot[3] = treeChunk.hash[3];
            }
        }
    }
    // If value=0, we are possibly going to delete an existing node
    else
    {
        // Setting a value=0 in an existing key, i.e. deleting
        if ( bFoundKey && fr.equal(key[0], foundKey[0]) && fr.equal(key[1], foundKey[1]) && fr.equal(key[2], foundKey[2]) && fr.equal(key[3], foundKey[3]) ) // Delete
        {
            oldValue = foundValue;

            // If level > 0, we are going to delete an existing node (not the root node)
            //if ( level >= 0)
            if (chunks[chunks.size()-1]->numberOfNonZeroChildren() > 1)
            {
                uint64_t children64Positon = getKeyChildren64Position(keys, level);
                chunks[chunks.size()-1]->children64[children64Positon].type = ZERO;
                chunks[chunks.size()-1]->bDataValid = false;
                chunks[chunks.size()-1]->bChildrenRestValid = false;
                chunks[chunks.size()-1]->bHashValid = false;
                zkr = chunks[chunks.size()-1]->calculateHash();
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("SMT64::Set() failed calling treeChunk.calculateHash() result=" + zkresult2string(zkr));
                    return zkr;
                }
                //treeChunk.print();

                zkr = chunks[chunks.size()-1]->children2data();
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("SMT64::Set() failed calling treeChunk.children2data() result=" + zkresult2string(zkr));
                    return zkr;
                }
                zkr = hashSave(ctx, *chunks[chunks.size()-1]);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("SMT64::Set() failed calling hashSave() result=" + zkresult2string(zkr));
                    return zkr;
                }

                newRoot[0] = chunks[chunks.size()-1]->hash[0];
                newRoot[1] = chunks[chunks.size()-1]->hash[1];
                newRoot[2] = chunks[chunks.size()-1]->hash[2];
                newRoot[3] = chunks[chunks.size()-1]->hash[3];
                // Set the hash of the deleted node to zero
                /*if (bUseStateManager)
                {
                    for (uint64_t j=0; j<4; j++)
                    {
                        nodeToDelete[j] = siblings[level][keys[level]*4 + j];
                        siblings[level][keys[level]*4 + j] = fr.zero();
                    }
                    nodeToDeleteString = fea2string(fr, nodeToDelete);
                    if (nodeToDeleteString != "0")
                    {
                        stateManager64.deleteNode(batchUUID, tx, nodeToDeleteString, persistence);
                    }
                }
                else
                {
                    for (uint64_t j=0; j<4; j++)
                    {
                        siblings[level][keys[level]*4 + j] = fr.zero();
                    }
                }*/

                // Find if there is only one non-zero hash in the siblings list for this level
                //int64_t uKey = getUniqueSibling(siblings[level]);

                // If there is only one, it is the new deleted one
                //if (uKey >= 0)
                //if (true)
                //{
                    mode = "deleteFound";
#ifdef LOG_SMT
                    zklog.info("Smt64::set() mode=" + mode);
#endif
#if 0
                    // Calculate the key of the deleted element
                    Goldilocks::Element auxFea[4];
                    //for (uint64_t i=0; i<4; i++) auxFea[i] = siblings[level][uKey*4+i];
                    string auxString = fea2string(fr, auxFea);

                    // Read its 2 siblings
                    dbres = ZKR_UNSPECIFIED;
                    if (bUseStateManager)
                    {
                        dbres = stateManager64.read(batchUUID, auxString, dbValue, dbReadLog);
                    }
                    if (dbres != ZKR_SUCCESS)
                    {
                        dbres = db.read(auxString, auxFea, dbValue, dbReadLog, false, keys, level);
                    }
                    if ( dbres != ZKR_SUCCESS)
                    {
                        zklog.error("Smt64::set() db.read error: " + to_string(dbres) + " (" + zkresult2string(dbres) + ") root:" + auxString);
                        return dbres;
                    }

                    // Store them in siblings
                    //siblings[level+1] = dbValue;

                    // If it is a leaf node
                    if ( siblings[level+1].size()>8 && fr.equal( siblings[level+1][8], fr.one() ) )
                    {
                        // Calculate the value hash
                        Goldilocks::Element valH[4];
                        for (uint64_t i=0; i<4; i++) valH[i] = siblings[level+1][4+i];
                        string valHString = fea2string(fr, valH);

                        // Read its siblings
                        dbres = ZKR_UNSPECIFIED;
                        if (bUseStateManager)
                        {
                            dbres = stateManager64.read(batchUUID, valHString, dbValue, dbReadLog);
                        }
                        if (dbres != ZKR_SUCCESS)
                        {
                            dbres = db.read(valHString, valH, dbValue, dbReadLog);
                        }
                        if (dbres != ZKR_SUCCESS)
                        {
                            zklog.error("Smt64::set() db.read error: " + to_string(dbres) + " (" + zkresult2string(dbres) + ") root:" + valHString);
                            return dbres;
                        }
                        else if (dbValue.size()<8)
                        {
                            zklog.error("Smt64::set() dbValue.size()<8 root:" + valHString);
                            return ZKR_SMT_INVALID_DATA_SIZE;
                        }

                        // Store the value as a scalar in val
                        Goldilocks::Element valA[8];
                        //for (uint64_t i=0; i<8; i++) valA[i] = dbValue[i];
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
                        //auxBits.push_back(uKey);
                        joinKey(fr, auxBits, rKey, insKey );

                        insValue = val;
                        isOld0 = false;

                        // Climb the branch until there are two siblings
                        /*while (uKey>=0 && level>=0)
                        {
                            level--;
                            if (level >= 0)
                            {
                                uKey = getUniqueSibling(siblings[level]);
                            }
                        }*/

                        // Calculate the old remaining key
                        Goldilocks::Element oldKey[4];
                        removeKeyBits(fr, insKey, level+1, oldKey);

                        // Create the old leaf node
                        Goldilocks::Element a[8];
                        for (uint64_t i=0; i<4; i++) a[i] = oldKey[i];
                        for (uint64_t i=0; i<4; i++) a[4+i] = valH[i];

                        // Create leaf node and store computed hash in oldLeafHash
                        Goldilocks::Element oldLeafHash[4];
                        //dbres = hashSaveOne(ctx, a, oldLeafHash);
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
        #endif
                    /*}
                    // Not a leaf node
                    else
                    {
                        mode = "deleteNotFound";
#ifdef LOG_SMT
                        zklog.info("Smt64::set() mode=" + mode);
#endif
                    }*/
                /*}
                // 2 siblings found
                else
                {
                    mode = "deleteNotFound";
#ifdef LOG_SMT
                    zklog.info("Smt64::set() mode=" + mode);
#endif
                }*/
            }
            // If level=0, this means we are deleting the root node
            else
            {
                mode = "deleteLast";
#ifdef LOG_SMT
                zklog.info("Smt64::set() mode=" + mode);
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
                insValue = foundValue;
                isOld0 = false;
            }
#ifdef LOG_SMT
            zklog.info("Smt64::set() mode=" + mode);
#endif
        }
    }

    // Delete the extra siblings
    map< uint64_t, vector<Goldilocks::Element> >::iterator it;
    it = siblings.find(level+1);
    siblings.erase(it, siblings.end());

    // Go up the tree creating all intermediate nodes up to the new root
    /*while (level >= 0)
    {
        // Write the siblings and get the calculated db entry hash in newRoot
        Goldilocks::Element a[8], c[4];
        for (uint64_t i=0; i<8; i++) a[i] = siblings[level][i];
        for (uint64_t i=0; i<4; i++) c[i] = siblings[level][8+i];
        //dbres = hashSave(ctx, a, c, newRoot);
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
            if (bUseStateManager)
            {
                for (uint64_t j=0; j<4; j++)
                {
                    nodeToDelete[j] = siblings[level][keys[level]*4 + j];
                    siblings[level][keys[level]*4 + j] = newRoot[j];
                }
                if (!fr.equal(nodeToDelete[0], newRoot[0]) || !fr.equal(nodeToDelete[1], newRoot[1]) || !fr.equal(nodeToDelete[2], newRoot[2]) || !fr.equal(nodeToDelete[3], newRoot[3]))
                {
                    nodeToDeleteString = fea2string(fr, nodeToDelete);
                    if (nodeToDeleteString != "0")
                    {
                        stateManager64.deleteNode(batchUUID, tx, nodeToDeleteString, persistence);
                    }
                }
            }
            else
            {
                for (uint64_t j=0; j<4; j++)
                {
                    siblings[level][keys[level]*4 + j] = newRoot[j];
                }
            }
        }
    }*/

    if (bUseStateManager)
    {
        stateManager64.setNewStateRoot(batchUUID, tx, fea2string(fr, newRoot), persistence);
    }
    else if ( (persistence == PERSISTENCE_DATABASE) &&
         (
            !fr.equal(oldRoot[0], newRoot[0]) ||
            !fr.equal(oldRoot[1], newRoot[1]) ||
            !fr.equal(oldRoot[2], newRoot[2]) ||
            !fr.equal(oldRoot[3], newRoot[3])
        ) )
    {
        dbres = updateStateRoot(db, newRoot);
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

    for (uint64_t i=0; i<chunks.size(); i++)
    {
        delete chunks[i];
    }
#ifdef LOG_SMT
    zklog.info("Smt64::set() returns isOld0=" + to_string(result.isOld0) + " insKey=" + fea2string(fr,result.insKey) + " oldValue=" + result.oldValue.get_str(16) + " newRoot=" + fea2string(fr,result.newRoot) + " mode=" + result.mode);
#endif
#ifdef LOG_SMT_SET_PRINT_TREE
    db.printTree(fea2string(fr,result.newRoot));
#endif
#endif // 0
    return ZKR_SUCCESS;
}

zkresult Smt64::get (const string &batchUUID, Database64 &db, const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], SmtGetResult &result, DatabaseMap *dbReadLog)
{
#ifdef LOG_SMT
    zklog.info("Smt64::get() called with root=" + fea2string(fr,root) + " and key=" + fea2string(fr,key));
#endif

    bool bUseStateManager = db.config.stateManager && (batchUUID.size() > 0);

    // Read the content of db for entry r: siblings[level] = db.read(r)
    string rootString = fea2string(fr, root);
    mpz_class value;
    zkresult zkr = ZKR_UNSPECIFIED;
    if (bUseStateManager)
    {
        zkr = stateManager64.read(batchUUID, rootString, value, dbReadLog);
    }
    if (zkr != ZKR_SUCCESS)
    {
        zkr = db.readKV(root, key, value, dbReadLog);
    }
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Smt64::get() db.read error: " + to_string(zkr) + " (" + zkresult2string(zkr) + ") root:" + rootString);
        return zkr;
    }
    
    result.value = value;

    return ZKR_SUCCESS;

#if 0
#ifdef LOG_SMT
    zklog.info("Smt64::get() called with root=" + fea2string(fr,root) + " and key=" + fea2string(fr,key));
#endif

    bool bUseStateManager = db.config.stateManager && (batchUUID.size() > 0);

    zkresult zkr;

    Goldilocks::Element r[4];
    for (uint64_t i=0; i<4; i++)
    {
        r[i] = root[i];
    }

    // Get a list of the bits of the key to navigate top-down through the tree
    bool keyBits[256];
    splitKey(fr, key, keyBits);

    uint64_t level = 0;

    vector<uint64_t> accKey;
    mpz_class lastAccKey = 0;
    bool bFoundKey = false;
    Goldilocks::Element foundKey[4] = {0, 0, 0, 0};
    Goldilocks::Element insKey[4] = {0, 0, 0, 0};

    map< uint64_t, vector<Goldilocks::Element> > siblings;

    mpz_class insValue = 0;
    mpz_class value = 0;
    mpz_class foundValue = 0;

    bool isOld0 = true;
    zkresult dbres;
    string dbValue; // used to call db.read()

#ifdef LOG_SMT
    //zklog.info("Smt64::get() found database content:");
    //db.print();
#endif

    // Start natigating the tree from the top: r = root
    // Go down while r!=0 (while there is branch) until we find the key
    while ( ( !fr.isZero(r[0]) || !fr.isZero(r[1]) || !fr.isZero(r[2]) || !fr.isZero(r[3]) ) && !bFoundKey )
    {
        // Read the content of db for entry r: siblings[level] = db.read(r)
        string rString = fea2string(fr, r);
        dbres = ZKR_UNSPECIFIED;
        if (bUseStateManager)
        {
            dbres = stateManager64.read(batchUUID, rString, dbValue, dbReadLog);
        }
        if (dbres != ZKR_SUCCESS)
        {
            dbres = db.read(rString, r, dbValue, dbReadLog, false, keyBits, level);
        }
        if (dbres != ZKR_SUCCESS)
        {
            zklog.error("Smt64::get() db.read error: " + to_string(dbres) + " (" + zkresult2string(dbres) + ") root:" + rString);
            return dbres;
        }

        // TODO: store the processed tree chunks in a cache or similar, so that we don't have to recalculate hashes for this tree chunk
        TreeChunk treeChunk(db, poseidon);
        treeChunk.hash[0] = r[0];
        treeChunk.hash[1] = r[1];
        treeChunk.hash[2] = r[2];
        treeChunk.hash[3] = r[3];
        treeChunk.data = dbValue;
        treeChunk.bDataValid = true;
        treeChunk.level = level;
        zkr = treeChunk.data2children();
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Smt64::get() failed calling treeChunk.data2children() result=" + zkresult2string(zkr) + " hash=" + rString);
            return zkr;
        }
        zkr = treeChunk.calculateHash();
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Smt64::get() failed calling treeChunk.calculateHash() result=" + zkresult2string(zkr) + " hash=" + rString);
            return zkr;
        }

        // Check that the resulting hash is the same we used to get the data from the database
        if (!fr.equal(treeChunk.hash[0], r[0]) || !fr.equal(treeChunk.hash[1], r[1]) || !fr.equal(treeChunk.hash[2], r[2]) || !fr.equal(treeChunk.hash[3], r[3]))
        {
            //treeChunk.print();
            zklog.error("Smt64::get() found calculated hash=" + fea2string(fr, treeChunk.hash) + " different from expected hash=" + fea2string(fr, r) + " data.size=" + to_string(treeChunk.data.size()) + " data=" + ba2string(treeChunk.data));
            return ZKR_UNSPECIFIED;
        }

        // Calculate the children64 index corresponding to this level and key
        uint64_t children64Position;
        children64Position =    keyBits[level] +
                              2*keyBits[level+1] +
                              4*keyBits[level+2] +
                              8*keyBits[level+3] +
                             16*keyBits[level+4] +
                             32*keyBits[level+5];

        // Check what we find in this position
        switch (treeChunk.children64[children64Position].type)
        {
            case ZERO:
            {
                r[0] = fr.zero();
                r[1] = fr.zero();
                r[2] = fr.zero();
                r[3] = fr.zero();
                // TODO: get siblings?
            }
            case LEAF:
            {
                // Copy the found value
                foundValue = treeChunk.children64[children64Position].leaf.value;

                // Copy the found key
                foundKey[0] = treeChunk.children64[children64Position].leaf.key[0];
                foundKey[1] = treeChunk.children64[children64Position].leaf.key[1];
                foundKey[2] = treeChunk.children64[children64Position].leaf.key[2];
                foundKey[3] = treeChunk.children64[children64Position].leaf.key[3];

                // Record that we found a key
                bFoundKey = true;
                // TODO: get siblings
            }
            case INTERMEDIATE:
            {
                // Copy the hash of the next tree chunk
                r[0] = treeChunk.children64[children64Position].intermediate.hash[0];
                r[1] = treeChunk.children64[children64Position].intermediate.hash[1];
                r[2] = treeChunk.children64[children64Position].intermediate.hash[2];
                r[3] = treeChunk.children64[children64Position].intermediate.hash[3];

                // Increase level
                level += 6;

                // TODO: get siblings
                break;
            }
            default:
            {
                zklog.error("Forest::Read() found invalid type=" + to_string(treeChunk.children64[children64Position].type));
                return ZKR_UNSPECIFIED;
            }
        }

#if 0


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
            string foundValueHashString = fea2string(fr, valueHashFea);
            dbres = ZKR_UNSPECIFIED;
            if (bUseStateManager)
            {
                dbres = stateManager64.read(batchUUID, foundValueHashString, dbValue, dbReadLog);
            }
            if (dbres != ZKR_SUCCESS)
            {
                dbres = db.read(foundValueHashString, valueHashFea, dbValue, dbReadLog);
            }
            if (dbres != ZKR_SUCCESS)
            {
                zklog.error("Smt64::get() db.read error: " + to_string(dbres) + " (" + zkresult2string(dbres) + ") root:" + foundValueHashString);
                return dbres;
            }

            // First 4 elements are the remaining key
            Goldilocks::Element foundRKey[4];
            foundRKey[0] = siblings[level][0];
            foundRKey[1] = siblings[level][1];
            foundRKey[2] = siblings[level][2];
            foundRKey[3] = siblings[level][3];

            // We convert the 8 found value elements to a scalar called foundValue
            Goldilocks::Element fea[8];
            for (uint64_t i=0; i<8; i++)
            {
                fea[i] = dbValue[i];
            }
            fea2scalar(fr, foundValue, fea);

            // We construct the whole key of that value in the database, and we call it foundKey
            joinKey(fr, accKey, foundRKey, foundKey);
            bFoundKey = true;
#ifdef LOG_SMT
            zklog.info("Smt64::get() found at level=" + to_string(level) + " value/hash=" + fea2string(fr,valueHashFea) + " foundKey=" + fea2string(fr, foundKey) + " value=" + foundValue.get_str(16));
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
            zklog.info("Smt64::get() down 1 level=" + to_string(level) + " keys[level]=" + to_string(keys[level]) + " root/hash=" + fea2string(fr,r));
#endif
            // Increase the level
            level += 6;
        }
#endif
    }

    // One step back
    level--;
    accKey.pop_back();

    // if we found the key, then we reached a leaf node while going down the tree
    if (bFoundKey)
    {
        // if foundKey==key, then foundValue is what we were looking for
        if ( fr.equal(key[0], foundKey[0]) &&
             fr.equal(key[1], foundKey[1]) &&
             fr.equal(key[2], foundKey[2]) &&
             fr.equal(key[3], foundKey[3]) )
        {
            value = foundValue;
        }
        // if foundKey!=key, then the requested value was not found
        else
        {
            insKey[0] = foundKey[0];
            insKey[1] = foundKey[1];
            insKey[2] = foundKey[2];
            insKey[3] = foundKey[3];
            insValue = foundValue;
            isOld0 = false;
        }
    }

    // We leave the siblings only up to the leaf node level
    // TODO manage siblings
    //map< uint64_t, vector<Goldilocks::Element> >::iterator it;
    //it = siblings.find(level+1);
    //siblings.erase(it, siblings.end());

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
    zklog.info("Smt64::get() returns isOld0=" + to_string(result.isOld0) + " insKey=" + fea2string(fr,result.insKey) + " and value=" + result.value.get_str(16));
#endif
#endif // 0
    return ZKR_SUCCESS;
}

zkresult Smt64::hashSave ( const SmtContext64 &ctx, const TreeChunk &treeChunk)
{
#if 0
    if (!treeChunk.bDataValid)
    {
        zklog.error("Smt64::hashSave() found treeChunk.bDataValid=false");
        return ZKR_UNSPECIFIED;
    }

    if (!treeChunk.bHashValid)
    {
        zklog.error("Smt64::hashSave() found treeChunk.bHashValid=false");
        return ZKR_UNSPECIFIED;
    }

    /*
    // Calculate the poseidon hash of the vector of field elements: v = a | c
    poseidon.hash(hash, v);

    // Fill a database value with the field elements
    string hashString = fea2string(fr, hash);

    // Add the key:value pair to the database, using the hash as a key
    vector<Goldilocks::Element> dbValue;
    for (uint64_t i=0; i<12; i++) dbValue.push_back(v[i]);
    zkresult zkr;*/

    zkresult zkr;

    string hashString = NormalizeToNFormat(fea2string(fr, treeChunk.hash), 64);

    if (ctx.bUseStateManager)
    {
        zkr = stateManager64.write(ctx.batchUUID, ctx.tx, hashString, treeChunk.data, ctx.persistence);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Smt64::hashSave() failed calling stateManager.write() key=" + hashString + " result=" + to_string(zkr) + "=" + zkresult2string(zkr));
        }
        else
        {
            //treeChunk.print();
        }
    }
    else
    {
        zkr = ctx.db.write(hashString, NULL, treeChunk.data, ctx.persistence == PERSISTENCE_DATABASE ? 1 : 0);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Smt64::hashSave() failed calling db.write() key=" + hashString + " result=" + to_string(zkr) + "=" + zkresult2string(zkr));
        }
    }
    
//#ifdef LOG_SMT
    {
        zklog.info("Smt64::hashSave() key=" + hashString + " data.size=" + to_string(treeChunk.data.size()) + " data=" + ba2string(treeChunk.data));
    }
//#endif
    return zkr;
#endif
    return ZKR_SUCCESS;
}

zkresult Smt64::updateStateRoot(Database64 &db, const Goldilocks::Element (&stateRoot)[4])
{
    // Write to db using the dbStateRootKey
    zkresult zkr;
    zkr = db.updateStateRoot(stateRoot);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Smt64::updateeStateRoot() failed calling db.updateeStateRoot() result=" + to_string(zkr) + "=" + zkresult2string(zkr));
    }

#ifdef LOG_SMT
    {
        string s = "Smt64::updateeStateRoot() value=";
        for (uint64_t i=0; i<4; i++) s += fr.toString(stateRoot[i],16) + ":";
        s += " zkr=" + zkresult2string(zkr);
        zklog.info(s);
    }
#endif
    return zkr;
}

int64_t Smt64::getUniqueSibling(vector<Goldilocks::Element> &a)
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