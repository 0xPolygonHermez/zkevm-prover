#ifndef DATABASE_64_HPP
#define DATABASE_64_HPP

#include <vector>
#include <map>
#include <pqxx/pqxx>
#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include "compare_fe.hpp"
#include "config.hpp"
#include <semaphore.h>
#include "zkresult.hpp"
#include "database_map.hpp"
#include "database_connection.hpp"
#include "zkassert.hpp"
#include "key_value.hpp"
#include "key_value_level.hpp"
#include "version_value.hpp"
#include "child.hpp"

using namespace std;


/*

A Tree (state) is made of a set of TreeChunks:

      /\
     /__\
        /\
       /__\
          /\
         /__\
        /\ 
       /__\

When we want to read [key, value] for a given root:
    - we call db.read(treeChunk.hash, treeChunk.data) starting from the root until we reach the [key, value] leaf node

When we want to write a new leaf node [key, newValue] on a given root and get the resulting newStateRoot
    - we calculate the new position of [key, newValue], creating new chunks if needed
    - we recalculate the hashes of all the modified and new chunks
    - we call db.write(treeChunk.hash, treeChunk.data) of all the modified and new chunks

Every time we write a [key, newValue] we are potentially creating a new Tree = SUM(TreeChunks) if newValue != oldValue
Every new Tree represents a newer version of the state
Many Trees, or states, coexist in the same Forest, or state history; the forest is stored in the database
Every executor.processBatch() can potentially create several new Trees (states), one per SMT.set()
The Forest takes note of the latest Tree hash to keep track of the current state:

     SR1      SR2      SR3      SR4
     /\       /\       /\       /\
    /__\     /__\     /__\     /__\
                /\       /\       /\
               /__\     /__\     /__\
                           /\       /\
                          /__\     /__\
                                  /\ 
                                 /__\

*/

class DB64Query
{
public:
    string key;
    Goldilocks::Element keyFea[4];
    string &value; // value can be an input in multiWrite(), or an output in multiRead()
    DB64Query(const string &_key, const Goldilocks::Element (&_keyFea)[4], string &_value) : key(_key), value(_value)
    {
        keyFea[0] = _keyFea[0];
        keyFea[1] = _keyFea[1];
        keyFea[2] = _keyFea[2];
        keyFea[3] = _keyFea[3];
    }
};

class Database64
{
public:
    Goldilocks &fr;
    const Config &config;
    PoseidonGoldilocks poseidon;

    // Basic flags
    bool bInitialized = false;
    bool useRemoteDB = false;

    uint64_t headerPageNumber;

public:
    //sem_t senderSem; // Semaphore to wakeup database sender thread when flush() is called
    //sem_t getFlushDataSem; // Semaphore to unblock getFlushData() callers when new data is available
private:
    //pthread_t senderPthread; // Database sender thread
    //pthread_t cacheSynchPthread; // Cache synchronization thread

    // Tree64
    zkresult CalculateHash (Child &result, std::vector<TreeChunk *> &chunks, vector<DB64Query> &dbQueries, int idChunk, int level, vector<HashValueGL> *hashValues);

public:

    // Constructor and destructor
    Database64(Goldilocks &fr, const Config &config);
    ~Database64();

public:
    // Basic methods
    void init(void);
    
    // Program
    zkresult getProgram (const string &_key, vector<uint8_t> &value, DatabaseMap *dbReadLog);
    zkresult setProgram (const string &_key, const vector<uint8_t> &value, const bool persistent);
    
    // Tree64
    zkresult WriteTree (const Goldilocks::Element (&oldRoot)[4], const vector<KeyValue> &keyValues, Goldilocks::Element (&newRoot)[4], const bool persistent);
    zkresult ReadTree  (const Goldilocks::Element (&root)[4], vector<KeyValue> &keyValues, vector<HashValueGL> *hashValues);

    // Key - Value - Level
    zkresult readKV    (const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value,  uint64_t &level, DatabaseMap *dbReadLog);
    zkresult readLevel (const Goldilocks::Element (&key)[4], uint64_t &level){ level=128; return ZKR_SUCCESS;}

    // Block
    zkresult consolidateBlock (uint64_t blockNumber); // TODO: Who reports this block number?
    zkresult revertBlock      (uint64_t blockNumber);

public:
    // Flush data pending to be stored permamently
    zkresult flush(uint64_t &flushId, uint64_t &lastSentFlushId);
    zkresult getFlushStatus(uint64_t &storedFlushId, uint64_t &storingFlushId, uint64_t &lastFlushId, uint64_t &pendingToFlushNodes, uint64_t &pendingToFlushProgram, uint64_t &storingNodes, uint64_t &storingProgram);

    // Get flush data, written to database by dbSenderThread; it blocks
    zkresult getFlushData(uint64_t flushId, uint64_t &lastSentFlushId, unordered_map<string, string> (&nodes), unordered_map<string, string> (&program), string &nodesStateRoot);

    // Clear cache
    void clearCache(void);

   
};

// Thread to send data to database
//void *dbSenderThread64(void *arg);

// Thread to synchronize cache from master hash DB server
//void *dbCacheSynchThread64(void *arg);

#endif