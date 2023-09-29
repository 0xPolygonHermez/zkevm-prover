#include <iostream>
#include <thread>
#include "database_64.hpp"
#include "config.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"
#include "definitions.hpp"
#include "zkresult.hpp"
#include "utils.hpp"
#include <unistd.h>
#include "timer.hpp"
#include "hashdb_singleton.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkmax.hpp"
#include "hashdb_remote.hpp"
#include "key_value.hpp"
#include "key_utils.hpp"
#include "page_manager.hpp"
#include "header_page.hpp"
#include "tree_chunk.hpp"

// Helper functions
string removeBSXIfExists64(string s) {return ((s.at(0) == '\\') && (s.at(1) == 'x')) ? s.substr(2) : s;}

Database64::Database64 (Goldilocks &fr, const Config &config) :
        fr(fr),
        config(config),
        headerPage(0)
{
    zkresult zkr;
    headerPage = pageManager.getFreeMemoryPage();
    zkr = HeaderPage::InitEmptyPage(headerPage);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::Database64() failed calling HeaderPage::InitEmptyPage() result=" + zkresult2string(zkr));
        exitProcess();
    }
    // Initialize semaphores
    //sem_init(&senderSem, 0, 0);
    //sem_init(&getFlushDataSem, 0, 0);
};

Database64::~Database64()
{
}

// Database64 class implementation
void Database64::init(void)
{
    // Check that it has not been initialized before
    if (bInitialized)
    {
        zklog.error("Database64::init() called when already initialized");
        exitProcess();
    }

    // Configure the server, if configuration is provided
    if (config.databaseURL != "local")
    {
        // Sender thread creation
        //pthread_create(&senderPthread, NULL, dbSenderThread64, this);

        // Cache synchronization thread creation
        /*if (config.dbCacheSynchURL.size() > 0)
        {
            pthread_create(&cacheSynchPthread, NULL, dbCacheSynchThread64, this);

        }*/

        useRemoteDB = true;
    }
    else
    {
        useRemoteDB = false;
    }

    // Mark the database as initialized
    bInitialized = true;
}

zkresult Database64::readKV(const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, uint64_t &level ,DatabaseMap *dbReadLog)
{
    level = 128;
    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database64::readKV() called uninitialized");
        exitProcess();
    }

    struct timeval t;
    if (dbReadLog != NULL) gettimeofday(&t, NULL);

    zkresult rout = ZKR_UNSPECIFIED;
    
#ifdef LOG_DB_READ
    {
        string s = "Database64::readKV()";
        if (rout != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(rout);
        s += " key=" + keyStr;
        s += " value=";
        s += value.get_str(16) + ";";
        zklog.info(s);
    }
#endif
    return rout;

}
/*
zkresult Database64::readKV(const Goldilocks::Element (&root)[4], vector<KeyValueLevel> &KVLs, DatabaseMap *dbReadLog){
    zkresult zkr;
    for (uint64_t i=0; i<KVLs.size(); i++)
    {
        zkr = readKV(root, KVLs[i].key, KVLs[i].value, KVLs[i].level, dbReadLog);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Database64::readKV(KBs) failed calling read() result=" + zkresult2string(zkr) + " key=" + fea2string(fr, KVLs[i].key) );
            return zkr;
        }
    }
    return ZKR_SUCCESS;
}*/

zkresult Database64::setProgram (const string &_key, const vector<uint8_t> &data, const bool persistent)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database64::setProgram() called uninitialized");
        exitProcess();
    }

    zkresult r = ZKR_UNSPECIFIED;

    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

#ifdef LOG_DB_WRITE
    {
        string s = "Database64::setProgram()";
        if (r != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(r);
        s += " key=" + key;
        s += " data=";
        for (uint64_t i = 0; (i < (data.size()) && (i < 100)); i++)
            s += byte2string(data[i]);
        if (data.size() > 100) s += "...";
        s += " persistent=" + to_string(persistent);
        zklog.info(s);
    }
#endif

    return r;
}

zkresult Database64::getProgram(const string &_key, vector<uint8_t> &data, DatabaseMap *dbReadLog)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database64::getProgram() called uninitialized");
        exitProcess();
    }

    zkresult zkr = ZKR_UNSPECIFIED;

    struct timeval t;
    if (dbReadLog != NULL) gettimeofday(&t, NULL);

    // Normalize key format
    string key = NormalizeToNFormat(_key, 64);
    key = stringToLower(key);

    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::getProgram() requested a key that does not exist: " + key);
        zkr = ZKR_DB_KEY_NOT_FOUND;
    }

#ifdef LOG_DB_READ
    {
        string s = "Database64::getProgram()";
        if (r != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(r);
        s += " key=" + key;
        s += " data=";
        for (uint64_t i = 0; (i < (data.size()) && (i < 100)); i++)
            s += byte2string(data[i]);
        if (data.size() > 100) s += "...";
        zklog.info(s);
    }
#endif

    return zkr;
}
    
zkresult Database64::flush(uint64_t &thisBatch, uint64_t &lastSentBatch)
{
    #if 0
    if (!config.dbMultiWrite)
    {
        return ZKR_SUCCESS;
    }

    // If we are connected to a read-only database, just free memory and pretend to have sent all the data
    if (config.dbReadOnly)
    {
        multiWrite.Lock();
        multiWrite.data[multiWrite.pendingToFlushDataIndex].Reset();
        multiWrite.Unlock();

        return ZKR_SUCCESS;
    }

    //TimerStart(DATABASE_FLUSH);

    multiWrite.Lock();

    // Accept all intray data
    multiWrite.data[multiWrite.pendingToFlushDataIndex].acceptIntray();

    // Increase the last processed batch id and return the last sent batch id
    multiWrite.lastFlushId++;
    thisBatch = multiWrite.lastFlushId;
    lastSentBatch = multiWrite.storedFlushId;

#ifdef LOG_DB_FLUSH
    zklog.info("Database64::flush() thisBatch=" + to_string(thisBatch) + " lastSentBatch=" + to_string(lastSentBatch) + " multiWrite=[" + multiWrite.print() + "]");
#endif

    // Notify the thread
    sem_post(&senderSem);

    multiWrite.Unlock();
#endif
    return ZKR_SUCCESS;
}

zkresult Database64::getFlushStatus(uint64_t &storedFlushId, uint64_t &storingFlushId, uint64_t &lastFlushId, uint64_t &pendingToFlushNodes, uint64_t &pendingToFlushProgram, uint64_t &storingNodes, uint64_t &storingProgram)
{
    /*multiWrite.Lock();
    storedFlushId = multiWrite.storedFlushId;
    storingFlushId = multiWrite.storingFlushId;
    lastFlushId = multiWrite.lastFlushId;
    pendingToFlushNodes = multiWrite.data[multiWrite.pendingToFlushDataIndex].nodes.size();
    pendingToFlushProgram = multiWrite.data[multiWrite.pendingToFlushDataIndex].program.size();
    storingNodes = multiWrite.data[multiWrite.storingDataIndex].nodes.size();
    storingProgram = multiWrite.data[multiWrite.storingDataIndex].program.size();
    multiWrite.Unlock();*/

    return ZKR_SUCCESS;
}


// Get flush data, written to database by dbSenderThread; it blocks
zkresult Database64::getFlushData(uint64_t flushId, uint64_t &storedFlushId, unordered_map<string, string> (&nodes), unordered_map<string, string> (&program), string &nodesStateRoot)
{
#if 0
    //zklog.info("--> getFlushData()");

    // Set the deadline to now + 60 seconds
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
	deadline.tv_sec += 60;

    // Try to get the semaphore
    int iResult;
    iResult = sem_timedwait(&getFlushDataSem, &deadline);
    if (iResult != 0)
    {
        zklog.info("Database64::getFlushData() timed out");
        return ZKR_SUCCESS;
    }

    multiWrite.Lock();
    MultiWriteData64 &data = multiWrite.data[multiWrite.synchronizingDataIndex];

    zklog.info("Database64::getFlushData woke up: pendingToFlushDataIndex=" + to_string(multiWrite.pendingToFlushDataIndex) +
        " storingDataIndex=" + to_string(multiWrite.storingDataIndex) +
        " synchronizingDataIndex=" + to_string(multiWrite.synchronizingDataIndex) +
        " nodes=" + to_string(data.nodes.size()) +
        " program=" + to_string(data.program.size()) +
        " nodesStateRoot=" + data.nodesStateRoot);

    if (data.nodes.size() > 0)
    {
        nodes = data.nodes;
    }

    if (data.program.size() > 0)
    {
        program = data.program;
    }

    if (data.nodesStateRoot.size() > 0)
    {
        nodesStateRoot = data.nodesStateRoot;
    }

    multiWrite.Unlock();

    //zklog.info("<-- getFlushData()");
#endif

    return ZKR_SUCCESS;
}

void Database64::clearCache (void)
{
}

#if 0
void *dbSenderThread64 (void *arg)
{
    Database64 *pDatabase = (Database64 *)arg;
    zklog.info("dbSenderThread64() started");
    MultiWrite64 &multiWrite = pDatabase->multiWrite;

    while (true)
    {
        // Wait for the sending semaphore to be released, if there is no more data to send
        struct timespec currentTime;
        int iResult = clock_gettime(CLOCK_REALTIME, &currentTime);
        if (iResult == -1)
        {
            zklog.error("dbSenderThread64() failed calling clock_gettime()");
            exitProcess();
        }

        currentTime.tv_sec += 5;
        sem_timedwait(&pDatabase->senderSem, &currentTime);

        multiWrite.Lock();

        bool bDataEmpty = false;

        // If sending data is not empty (it failed before) then try to send it again
        if (!multiWrite.data[multiWrite.storingDataIndex].multiQuery.isEmpty())
        {
            zklog.warning("dbSenderThread64() found sending data index not empty, probably because of a previous error; resuming...");
        }
        // If processing data is empty, then simply pretend to have sent data
        else if (multiWrite.data[multiWrite.pendingToFlushDataIndex].IsEmpty())
        {
            //zklog.warning("dbSenderThread() found pending to flush data empty");

            // Mark as if we sent all batches
            multiWrite.storedFlushId = multiWrite.lastFlushId;
#ifdef LOG_DB_SENDER_THREAD
            zklog.info("dbSenderThread64() found multi write processing data empty, so ignoring");
#endif
            multiWrite.Unlock();
            continue;
        }
        // Else, switch data indexes
        else
        {
            // Accept all intray data
            multiWrite.data[multiWrite.pendingToFlushDataIndex].acceptIntray(true);

            // Advance processing and sending indexes
            multiWrite.storingDataIndex = (multiWrite.storingDataIndex + 1) % 3;
            multiWrite.pendingToFlushDataIndex = (multiWrite.pendingToFlushDataIndex + 1) % 3;
            multiWrite.data[multiWrite.pendingToFlushDataIndex].Reset();
#ifdef LOG_DB_SENDER_THREAD
            zklog.info("dbSenderThread64() updated: multiWrite=[" + multiWrite.print() + "]");
#endif

            // Record the last processed batch included in this data set
            multiWrite.storingFlushId = multiWrite.lastFlushId;

            // If there is no data to send, just pretend to have sent it
            if (multiWrite.data[multiWrite.storingDataIndex].IsEmpty())
            {
                // Update stored flush ID
                multiWrite.storedFlushId = multiWrite.storingFlushId;

                // Advance synchronizing index
                multiWrite.synchronizingDataIndex = (multiWrite.synchronizingDataIndex + 1) % 3;
#ifdef LOG_DB_SENDER_THREAD
                zklog.info("dbSenderThread64() no data to send: multiWrite=[" + multiWrite.print() + "]");
#endif
                bDataEmpty = true;
            }

        }

        // Unlock to let more processing batch data in
        multiWrite.Unlock();

        if (!bDataEmpty)
        {
#ifdef LOG_DB_SENDER_THREAD
            zklog.info("dbSenderThread64() starting to send data, multiWrite=[" + multiWrite.print() + "]");
#endif
            zkresult zkr;
            zkr = pDatabase->sendData();
            if (zkr == ZKR_SUCCESS)
            {
                multiWrite.Lock();
                multiWrite.storedFlushId = multiWrite.storingFlushId;
#ifdef LOG_DB_SENDER_THREAD
                zklog.info("dbSenderThread64() successfully sent data, multiWrite=[]" + multiWrite.print() + "]");
#endif
                // Advance synchronizing index
                multiWrite.synchronizingDataIndex = (multiWrite.synchronizingDataIndex + 1) % 3;
#ifdef LOG_DB_SENDER_THREAD
                zklog.info("dbSenderThread64() updated: multiWrite=[" + multiWrite.print() + "]");
#endif
                sem_post(&pDatabase->getFlushDataSem);
#ifdef LOG_DB_SENDER_THREAD
                zklog.info("dbSenderThread64() successfully called sem_post(&pDatabase->getFlushDataSem)");
#endif
                multiWrite.Unlock();
            }
            else
            {
                zklog.error("dbSenderThread64() failed calling sendData() error=" + zkresult2string(zkr));
                usleep(1000000);
            }
        }
    }

    zklog.info("dbSenderThread64() done");
    return NULL;
}
#endif

zkresult Database64::consolidateBlock (uint64_t blockNumber)
{
    return ZKR_UNSPECIFIED;
}

zkresult Database64::revertBlock (uint64_t blockNumber)
{
    return ZKR_UNSPECIFIED;
}

zkresult Database64::WriteTree (const Goldilocks::Element (&oldRoot)[4], const vector<KeyValue> &_keyValues, Goldilocks::Element (&newRoot)[4], const bool persistent)
{
    zkresult zkr;

    vector<KeyValue> keyValues(_keyValues);

    vector<TreeChunk *> chunks;
    vector<DB64Query> dbQueries;

    // Tree level; we start at level 0, then we increase it 6 by 6
    uint64_t level = 0;

    // Create the first tree chunk (the root one), and store it in chunks[0]
    TreeChunk *c = new TreeChunk(*this, poseidon);
    if (c == NULL)
    {
        zklog.error("Database64::WriteTree() failed calling new TreeChunk()");
        exitProcess();
    }
    chunks.push_back(c);

    uint64_t chunksProcessed = 0;

    // Get the old root as a string
    string oldRootString = fea2string(fr, oldRoot);

    // If old root is zero, init chunks[0] as an empty tree chunk
    if (fr.isZero(oldRoot[0]) && fr.isZero(oldRoot[1]) && fr.isZero(oldRoot[2]) && fr.isZero(oldRoot[3]))
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
        //zkr = db.read(dbQueries);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Database64::WriteTree() failed calling db.multiRead() result=" + zkresult2string(zkr));
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
                    zklog.error("Database64::WriteTree() failed calling chunks[i]->data2children() result=" + zkresult2string(zkr));
                    return zkr;
                }
            }
            for (uint64_t j=0; j<chunks[i]->list.size(); j++)
            {
                bool keyBits[256];
                splitKey(fr, keyValues[chunks[i]->list[j]].key, keyBits);
                uint64_t k = getKeyChildren64Position(keyBits, level);
                switch (chunks[i]->getChild(k).type)
                {
                    case ZERO:
                    {
                        if (keyValues[chunks[i]->list[j]].value != 0)
                        {
                            chunks[i]->setLeafChild(k, keyValues[chunks[i]->list[j]].key, keyValues[chunks[i]->list[j]].value);                  
                        }
                        break;
                    }
                    case LEAF:
                    {
                        // If the key is the same, then check the value
                        if (fr.equal(chunks[i]->getChild(k).leaf.key[0], keyValues[chunks[i]->list[j]].key[0]) &&
                            fr.equal(chunks[i]->getChild(k).leaf.key[1], keyValues[chunks[i]->list[j]].key[1]) &&
                            fr.equal(chunks[i]->getChild(k).leaf.key[2], keyValues[chunks[i]->list[j]].key[2]) &&
                            fr.equal(chunks[i]->getChild(k).leaf.key[3], keyValues[chunks[i]->list[j]].key[3]))
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
                                    chunks[i]->setLeafChild(k, keyValues[chunks[i]->list[j]].key, keyValues[chunks[i]->list[j]].value);
                                }
                            }
                        }
                        else
                        {
                            // We create a new trunk
                            TreeChunk *c = new TreeChunk(*this, poseidon);
                            if (c == NULL)
                            {
                                zklog.error("Database64::WriteTree() failed calling new TreeChunk()");
                                exitProcess();
                            }

                            // Reset to zero
                            c->resetToZero(level + 6);

                            // We create a KeyValue from the original leaf node
                            KeyValue kv;
                            kv.key[0] = chunks[i]->getChild(k).leaf.key[0];
                            kv.key[1] = chunks[i]->getChild(k).leaf.key[1];
                            kv.key[2] = chunks[i]->getChild(k).leaf.key[2];
                            kv.key[3] = chunks[i]->getChild(k).leaf.key[3];
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
                        TreeChunk *c = new TreeChunk(*this, poseidon);
                        if (c == NULL)
                        {
                            zklog.error("Database64::WriteTree() failed calling new TreeChunk()");
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
                        zklog.error("Database64::WriteTree() found invalid chunks[i]->getChild(k).type=" + to_string(chunks[i]->getChild(k).type));
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
    zkr = CalculateHash(result, chunks, dbQueries, 0, 0, NULL);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::WriteTree() failed calling calculateHash() result=" + zkresult2string(zkr));
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
                zklog.error("Database64::WriteTree() failed calling chunks[0]->children2data() result=" + zkresult2string(zkr));
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
        zklog.error("Database64::WriteTree() found invalid result.type=" + to_string(result.type));
        for (uint c = 0; c < chunks.size(); c++) delete chunks[c];
        return zkr;
    }

    // Save chunks data to database
    //zkr = db.write(dbQueries, persistent);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::WriteTree() failed calling db.write() result=" + zkresult2string(zkr));
        for (uint c = 0; c < chunks.size(); c++) delete chunks[c];
        return zkr;
    }

#ifdef SMT64_PRINT_TREE_CHUNKS
    // Print chunks
    for (uint c = 0; c < chunks.size(); c++)
    {
        zklog.info("Database64::WriteTree() chunk " + to_string(c));
        chunks[c]->print();
    }
#endif

    // Free memory
    for (uint c = 0; c < chunks.size(); c++) delete chunks[c];

    return ZKR_SUCCESS;
}

zkresult Database64::CalculateHash (Child &result, vector<TreeChunk *> &chunks, vector<DB64Query> &dbQueries, int chunkId, int level, vector<HashValueGL> *hashValues)
{
    zkresult zkr;
    vector<Child> results(64);

    // Convert all TREE_CHUNK children into something else, typically INTERMEDIATE children,
    // but they could also be LEAF (only one child below this level) or ZERO 9no children below this level)
    for (uint64_t i=0; i<64; i++)
    {
        if (chunks[chunkId]->getChild(i).type == TREE_CHUNK)
        {
            CalculateHash(result, chunks, dbQueries, chunks[chunkId]->getChild(i).treeChunkId, level + 6, hashValues);
            chunks[chunkId]->setChild(i, result);
        }
    }

    // Calculate the hash of this chunk
    zkr = chunks[chunkId]->calculateHash(hashValues);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::CalculateHash() failed calling chunks[chunkId]->calculateHash() result=" + zkresult2string(zkr));
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
            zklog.error("Database64::CalculateHash() failed calling chunks[chunkId]->children2data() result=" + zkresult2string(zkr));
            return zkr;
        }

        Goldilocks::Element hash[4];
        chunks[chunkId]->getHash(hash);
        DB64Query dbQuery(fea2string(fr, hash), hash, chunks[chunkId]->data);
        dbQueries.emplace_back(dbQuery);
    }

    return ZKR_SUCCESS;
}

zkresult Database64::ReadTree (const Goldilocks::Element (&root)[4], vector<KeyValue> &keyValues, vector<HashValueGL> *hashValues)
{
    zkresult zkr;

    vector<TreeChunk *> chunks;
    vector<DB64Query> dbQueries;

    // Tree level; we start at level 0, then we increase it 6 by 6
    uint64_t level = 0;

    // Create the first tree chunk (the root one), and store it in chunks[0]
    TreeChunk *c = new TreeChunk(*this, poseidon);
    if (c == NULL)
    {
        zklog.error("Database64::ReadTree() failed calling new TreeChunk()");
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
        //zkr = db.read(dbQueries);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Database64::ReadTree() failed calling db.multiRead() result=" + zkresult2string(zkr));
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
                    zklog.error("Database64::ReadTree() failed calling chunks[i]->data2children() result=" + zkresult2string(zkr));
                    return zkr;
                }
            }
            for (uint64_t j=0; j<chunks[i]->list.size(); j++)
            {
                bool keyBits[256];
                splitKey(fr, keyValues[chunks[i]->list[j]].key, keyBits);
                uint64_t k = getKeyChildren64Position(keyBits, level);
                switch (chunks[i]->getChild(k).type)
                {
                    case ZERO:
                    {
                        for (uint64_t kv=0; kv<keyValues.size(); kv++)
                        {
                            if (fr.equal(keyValues[kv].key[0], keyValues[chunks[i]->list[j]].key[0]) &&
                                fr.equal(keyValues[kv].key[1], keyValues[chunks[i]->list[j]].key[1]) &&
                                fr.equal(keyValues[kv].key[2], keyValues[chunks[i]->list[j]].key[2]) &&
                                fr.equal(keyValues[kv].key[3], keyValues[chunks[i]->list[j]].key[3]))
                            {
                                keyValues[kv].value = 0;
                            }
                        }
                        break;
                    }
                    case LEAF:
                    {
                        // If the key is the same, then check the value
                        if (fr.equal(chunks[i]->getChild(k).leaf.key[0], keyValues[chunks[i]->list[j]].key[0]) &&
                            fr.equal(chunks[i]->getChild(k).leaf.key[1], keyValues[chunks[i]->list[j]].key[1]) &&
                            fr.equal(chunks[i]->getChild(k).leaf.key[2], keyValues[chunks[i]->list[j]].key[2]) &&
                            fr.equal(chunks[i]->getChild(k).leaf.key[3], keyValues[chunks[i]->list[j]].key[3]))
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
                        TreeChunk *c = new TreeChunk(*this, poseidon);
                        if (c == NULL)
                        {
                            zklog.error("Database64::ReadTree() failed calling new TreeChunk()");
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
                        zklog.error("Database64::ReadTree() found invalid chunks[i]->getChild(k).type=" + to_string(chunks[i]->getChild(k).type));
                        exitProcess();
                    }
                }
            }
        }

        chunksProcessed = chunksToProcess;
        level += 6;
    }

    dbQueries.clear();

    if (hashValues != NULL)
    {
        // Calculate the new root hash of the whole tree
        Child result;
        zkr = CalculateHash(result, chunks, dbQueries, 0, 0, hashValues);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Database64::WriteTree() failed calling calculateHash() result=" + zkresult2string(zkr));
            for (uint c = 0; c < chunks.size(); c++) delete chunks[c];
            return zkr;
        }

        // Based on the result, calculate the new root hash
        /*if (result.type == LEAF)
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
                    zklog.error("Database64::WriteTree() failed calling chunks[0]->children2data() result=" + zkresult2string(zkr));
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
            zklog.error("Database64::WriteTree() found invalid result.type=" + to_string(result.type));
            for (uint c = 0; c < chunks.size(); c++) delete chunks[c];
            return zkr;
        }*/
        
    }

#ifdef SMT64_PRINT_TREE_CHUNKS
    // Print chunks
    for (uint c = 0; c < chunks.size(); c++)
    {
        zklog.info("Database64::ReadTree() chunk " + to_string(c));
        chunks[c]->print();
    }
#endif

    // Free memory
    for (uint c = 0; c < chunks.size(); c++) delete chunks[c];

    return ZKR_SUCCESS;
}