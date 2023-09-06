#include "tree_64.hpp"
#include "key_utils.hpp"

Tree64 tree64;

zkresult Tree64::WriteTree (Database64 &db, const Goldilocks::Element (&oldRoot)[4], const vector<KeyValue> &_keyValues, Goldilocks::Element (&newRoot)[4], const bool persistent)
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
        zklog.error("Tree64::WriteTree() failed calling new TreeChunk()");
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
            zklog.error("Tree64::WriteTree() failed calling db.multiRead() result=" + zkresult2string(zkr));
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
                    zklog.error("Tree64::WriteTree() failed calling chunks[i]->data2children() result=" + zkresult2string(zkr));
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
                                zklog.error("Tree64::WriteTree() failed calling new TreeChunk()");
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
                            zklog.error("Tree64::WriteTree() failed calling new TreeChunk()");
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
                        zklog.error("Tree64::WriteTree() found invalid chunks[i]->getChild(k).type=" + to_string(chunks[i]->getChild(k).type));
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
    zkr = CalculateHash(result, chunks, dbQueries, 0, 0);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Tree64::WriteTree() failed calling calculateHash() result=" + zkresult2string(zkr));
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
                zklog.error("Tree64::WriteTree() failed calling chunks[0]->children2data() result=" + zkresult2string(zkr));
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
        zklog.error("Tree64::WriteTree() found invalid result.type=" + to_string(result.type));
        for (uint c = 0; c < chunks.size(); c++) delete chunks[c];
        return zkr;
    }

    // Save chunks data to database
    zkr = db.write(dbQueries, persistent);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Tree64::WriteTree() failed calling db.write() result=" + zkresult2string(zkr));
        for (uint c = 0; c < chunks.size(); c++) delete chunks[c];
        return zkr;
    }

#ifdef SMT64_PRINT_TREE_CHUNKS
    // Print chunks
    for (uint c = 0; c < chunks.size(); c++)
    {
        zklog.info("Tree64::WriteTree() chunk " + to_string(c));
        chunks[c]->print();
    }
#endif

    // Free memory
    for (uint c = 0; c < chunks.size(); c++) delete chunks[c];

    return ZKR_SUCCESS;
}

zkresult Tree64::CalculateHash (Child &result, vector<TreeChunk *> &chunks, vector<DB64Query> &dbQueries, int chunkId, int level)
{
    zkresult zkr;
    vector<Child> results(64);

    // Convert all TREE_CHUNK children into something else, typically INTERMEDIATE children,
    // but they could also be LEAF (only one child below this level) or ZERO 9no children below this level)
    for (uint64_t i=0; i<64; i++)
    {
        if (chunks[chunkId]->getChild(i).type == TREE_CHUNK)
        {
            CalculateHash(result, chunks, dbQueries, chunks[chunkId]->getChild(i).treeChunkId, level + 6);
            chunks[chunkId]->setChild(i, result);
        }
    }

    // Calculate the hash of this chunk
    zkr = chunks[chunkId]->calculateHash();
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Tree64::CalculateHash() failed calling chunks[chunkId]->calculateHash() result=" + zkresult2string(zkr));
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
            zklog.error("Tree64::CalculateHash() failed calling chunks[chunkId]->children2data() result=" + zkresult2string(zkr));
            return zkr;
        }

        Goldilocks::Element hash[4];
        chunks[chunkId]->getHash(hash);
        DB64Query dbQuery(fea2string(fr, hash), hash, chunks[chunkId]->data);
        dbQueries.emplace_back(dbQuery);
    }

    return ZKR_SUCCESS;
}

zkresult Tree64::ReadTree (Database64 &db, const Goldilocks::Element (&root)[4], vector<KeyValue> &keyValues)
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
        zklog.error("Tree64::ReadTree() failed calling new TreeChunk()");
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
            zklog.error("Tree64::ReadTree() failed calling db.multiRead() result=" + zkresult2string(zkr));
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
                    zklog.error("Tree64::ReadTree() failed calling chunks[i]->data2children() result=" + zkresult2string(zkr));
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
                            zklog.error("Tree64::ReadTree() failed calling new TreeChunk()");
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
                        zklog.error("Tree64::ReadTree() found invalid chunks[i]->getChild(k).type=" + to_string(chunks[i]->getChild(k).type));
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
        zklog.info("Tree64::ReadTree() chunk " + to_string(c));
        chunks[c]->print();
    }
#endif

    // Free memory
    for (uint c = 0; c < chunks.size(); c++) delete chunks[c];

    return ZKR_SUCCESS;
}