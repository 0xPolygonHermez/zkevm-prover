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

A Tree (state) is made of a set of TreeChunks, each of them stored in one 4kB page:

      /\
     /__\
        /\
       /__\
          /\
         /__\
        /\ 
       /__\

When we want to read [key, value] for a given root:
    - we search for the right page starting from the root until we reach the [key, value] leaf node in the final page

When we want to write a new leaf node [key, newValue] on a given root and get the resulting newStateRoot
    - we calculate the new position of [key, newValue], creating new pages if needed
    - we recalculate the hashes of all the modified and new chunks
    - we write every resulting hash in the proper position of the proper page

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

class Database64
{
private:

    bool     bInitialized = false;
    uint64_t headerPageNumber;
    pthread_mutex_t mutex;
    uint64_t currentFlushId;

public:

    // Constructor and destructor
    Database64(Goldilocks &fr, const Config &config);
    ~Database64();

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

    // Flush data pending to be stored permamently
    zkresult flush(uint64_t &flushId, uint64_t &lastSentFlushId);
    zkresult getFlushStatus(uint64_t &storedFlushId, uint64_t &storingFlushId, uint64_t &lastFlushId, uint64_t &pendingToFlushNodes, uint64_t &pendingToFlushProgram, uint64_t &storingNodes, uint64_t &storingProgram);

    // Lock/Unlock
    void Lock(void) { pthread_mutex_lock(&mutex); };
    void Unlock(void) { pthread_mutex_unlock(&mutex); };
};

#endif