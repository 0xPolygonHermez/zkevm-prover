#include "database_cache.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "zklog.hpp"
#include "zkmax.hpp"
#include "timer.hpp"

// DatabaseCache class implementation

// Add a record in the head of the cache. Returns true if the cache is full (or no cache), false otherwise
bool DatabaseCache::addKeyValue(const string &key, const void * value, const bool update) 
{
    if (maxSize == 0)
    {
        return true;
    }

    DatabaseCacheRecord * record;
    // If key already exists in the cache return. The findKey also sets the record in the head of the cache
    if (findKey(key, record))
    {
        if (update)
        {
            updateRecord(record, value);
            return true;
        }
        else
        {
            return false;
        }
    }

    record = allocRecord(key, value);

    bool full = false;

    record->prev = NULL;
    if (head == NULL) 
    {
        record->next = NULL;
        last = record;
    } 
    else 
    {
        record->next = head;
        head->prev = record;
    }
    head = record;

    cacheMap[key] = record;

    currentSize += record->size;
    full = (currentSize > maxSize); 
    // remove lats records from the cache to be under maxSize

    while ((currentSize > maxSize) && (last->prev != NULL))
    {
        // Set new last record
        DatabaseCacheRecord* tmp = last;
        last->prev->next = NULL;
        last = last->prev;
        
        // Free old last record
        cacheMap.erase(tmp->key);
        
        // Update cache size
        zkassert(currentSize >= tmp->size);
        currentSize -= tmp->size;

        freeRecord(tmp);
    }

    //zklog.info("DatabaseCache::addRecord() key=" + key + " cacheCurrentSize=" + to_string(cacheCurrentSize) + " cacheMap.size()=" + to_string(cacheMap.size()) + " record.size()=" + to_string(record->size));
    //printMemoryInfo(true);
    
    return full;
}

bool DatabaseCache::findKey(const string &key, DatabaseCacheRecord* &record) 
{
    attempts++;

    if (attempts%1000000 == 0)
    {
        zklog.info("DatabaseCache::addKeyValue() name=" + name + " count=" + to_string(cacheMap.size()) + " maxSize=" + to_string(maxSize) + " currentSize=" + to_string(currentSize) + " attempts=" + to_string(attempts) + " hits=" + to_string(hits) + " hit ratio=" + to_string(double(hits)*100.0/double(zkmax(attempts,1))) + "%");
    }
    
    unordered_map<string, DatabaseCacheRecord*>::iterator it = cacheMap.find(key);

    if (it != cacheMap.end())
    {
        hits++;
        record = (DatabaseCacheRecord*)it->second;

        // Move cache record to the top/head (if it's not the current head)
        if (head != record) 
        {
            // Remove record from the current position
            record->prev->next = record->next;

            // If record is the last then set record->prev as the new last
            if (last == record) last = record->prev;
            else record->next->prev = record->prev;
            
            // Put record on top/head of the list
            head->prev = record;
            record->prev = NULL;
            record->next = head;
            head = record;
        }
        return true;
    }
    return false;
}

void DatabaseCache::print(bool printContent)
{
    zklog.info("DatabaseCache::print() printContent=" + to_string(printContent) + " name=" + name);
    zklog.info("Cache current size: " + to_string(currentSize));
    zklog.info("Cache max size: " + to_string(maxSize));
    zklog.info("Head: " + (head != NULL ? head->key : "NULL"));
    zklog.info("Last: " + (last != NULL ? last->key : "NULL"));
    
    DatabaseCacheRecord* record = head;
    uint64_t count = 0;
    uint64_t size = 0;
    while (record != NULL) 
    {
        if (printContent)
        {
            zklog.info("key:" + record->key + " size=" + to_string(record->size) + " prev=" + to_string((uint64_t)record->prev) + " next=" + to_string((uint64_t)record->next));
        }
        count++;
        size += record->size;
        record = record->next;
    }
    zklog.info("Cache count: " + to_string(count));
    zklog.info("Cache calculated size: " + to_string(size));
}

void DatabaseCache::clear(void)
{
    lock_guard<recursive_mutex> guard(mlock);

    DatabaseCacheRecord* record = head;
    DatabaseCacheRecord* tmp;
    // Free cache records
    while (record != NULL) 
    {
        tmp = record->next;
        freeRecord(record);
        record = tmp;
    }
    head = NULL;
    last = NULL;
    attempts = 0;
    hits = 0;
    cacheMap.clear();
    currentSize = 0;
}

DatabaseCache::~DatabaseCache()
{
    //TimerStart(DATABASE_CACHE_DESTRUCTOR);
    //clear();
    //TimerStopAndLog(DATABASE_CACHE_DESTRUCTOR);
}

// DatabaseMTCache class implementation


DatabaseMTCache::~DatabaseMTCache()
{
    //TimerStart(DATABASE_MT_CACHE_DESTRUCTOR);
    clear();
    //TimerStopAndLog(DATABASE_MT_CACHE_DESTRUCTOR);
}

// Add a record in the head of the MT cache. Returns true if the cache is full (or no cache), false otherwise
bool DatabaseMTCache::add(const string &key, const vector<Goldilocks::Element> &value, const bool update)
{
    lock_guard<recursive_mutex> guard(mlock);

    if (maxSize == 0) return true;

    return addKeyValue(key, (const void *)&value, update);
}

bool DatabaseMTCache::find(const string &key, vector<Goldilocks::Element> &value)
{
    lock_guard<recursive_mutex> guard(mlock);

    if (maxSize == 0) return false;

    DatabaseCacheRecord* record;
    bool found = findKey(key, record);
    if (found) 
    {
        value = *((vector<Goldilocks::Element>*) record->value);
    }

    return found;
}

DatabaseCacheRecord * DatabaseMTCache::allocRecord(const string key, const void * value)
{
    // Allocate memory
    DatabaseCacheRecord * pRecord = new(DatabaseCacheRecord);
    if (pRecord == NULL)
    {
        zklog.error("DatabaseMTCache::allocRecord() failed calling new(DatabaseCacheRecord)");
        exitProcess();
    }
    vector<Goldilocks::Element>* pValue = new(vector<Goldilocks::Element>);
    if (pValue == NULL)
    {
        zklog.error("DatabaseMTCache::allocRecord() failed calling new(vector<Goldilocks::Element>)");
        exitProcess();
    }

    // Copy vector
    *pValue = *(const vector<Goldilocks::Element> *)value;

    // Assign values to record
    pRecord->value = pValue;
    pRecord->key = key;
    pRecord->size = 2*(
        sizeof(DatabaseCacheRecord)+
        (pRecord->key.capacity()+1)+
        sizeof(vector<Goldilocks::Element>)+
        sizeof(Goldilocks::Element)*pValue->capacity() );
        
    return pRecord;
}

void DatabaseMTCache::freeRecord(DatabaseCacheRecord* record)
{
    delete (vector<Goldilocks::Element>*)(record->value);
    delete record;
}

void DatabaseMTCache::updateRecord(DatabaseCacheRecord* record, const void * value)
{
    *(vector<Goldilocks::Element>*)(record->value) = *(vector<Goldilocks::Element>*)(value);
}

uint32_t DatabaseMTCache::fillCache(){

    vector<Goldilocks::Element> value0(12);
    value0[0].fe = 0;
    vector<Goldilocks::Element> value1(12);
    value1[0].fe = 0;

    Goldilocks::Element key[4];
    for (uint64_t i=0; i<maxSize/256; i++)
    {
        key[0].fe = i;
        key[1].fe = i+1;
        key[2].fe = i+2;
        key[3].fe = i+3;
        
        string keyString = fea2string(fr,key); 
        add(keyString,value0, false);
        //get a random value from 0 to i
        uint64_t random = rand() % (i+1);
        key[0].fe = random;
        key[1].fe = random+1;
        key[2].fe = random+2;
        key[3].fe = random+3;
        find(fea2string(fr, key), value1);
        value0[0]= value0[0]+value1[0];
    }
    return 1;

}

uint32_t DatabaseMTCache::fillCacheCahotic(){

    vector<Goldilocks::Element> value0(12);
    value0[0].fe = 0;
    vector<Goldilocks::Element> value1(12);
    value1[0].fe = 0;

    uint32_t ndist = 1<<20;
    uint32_t mask = ndist-1;
    char ** memory_distorsion = (char**) malloc(ndist*sizeof(char*));
    if(memory_distorsion == NULL)
    {
        zklog.error("DatabaseMTCache::fillCacheCahotic() failed calling malloc()");
        exitProcess();
    }
    for (uint32_t i=0; i<ndist; i++)
    {
        memory_distorsion[i] = NULL;
    }
    uint32_t count = 0;
    
    Goldilocks::Element key[4];
    for (uint64_t i=0; i<maxSize/256; i++)
    {
        key[0].fe = i;
        key[1].fe = i+1;
        key[2].fe = i+2;
        key[3].fe = i+3;
        
        string keyString = fea2string(fr,key); 
        add(keyString,value0, false);
        //get a random value from 0 to i
        uint64_t random = rand() % (i+1);
        key[0].fe = random;
        key[1].fe = random+1;
        key[2].fe = random+2;
        key[3].fe = random+3;
        find(fea2string(fr, key), value1);
        value0[0]= value0[0]+value1[0];
        if(memory_distorsion[i & mask] != NULL)
        {
            free(memory_distorsion[i & mask]);
        }
        memory_distorsion[i & mask] = (char*) malloc((1<<20)*sizeof(char));
        if(memory_distorsion[i & mask] == NULL)
        {
            zklog.error("DatabaseMTCache::fillCacheCahotic() failed calling malloc()");
            exitProcess();
        }
        if(i%2==0) memory_distorsion[i & mask][3728] = 'a';
    }
    for(uint32_t i=0; i<ndist; i++)
    {
        if(memory_distorsion[i] != NULL)
        {
            if(memory_distorsion[i][3728] == 'a')
            {
                count++;
            }
            free(memory_distorsion[i]);
        }
    }
    free(memory_distorsion);
    return count;

}

// DatabaseProgramCache class implementation

DatabaseProgramCache::~DatabaseProgramCache()
{
    //TimerStart(DATABASE_PROGRAM_CACHE_DESTRUCTOR);
    clear();
    //TimerStopAndLog(DATABASE_PROGRAM_CACHE_DESTRUCTOR);
}

// Add a record in the head of the Program cache. Returns true if the cache is full (or no cache), false otherwise
bool DatabaseProgramCache::add(const string &key, const vector<uint8_t> &value, const bool update)
{
    lock_guard<recursive_mutex> guard(mlock);

    if (maxSize == 0) return true;

    return addKeyValue(key, (const void *)&value, update);
}

bool DatabaseProgramCache::find(const string &key, vector<uint8_t> &value)
{
    lock_guard<recursive_mutex> guard(mlock);

    if (maxSize == 0) return false;

    DatabaseCacheRecord* record;
    bool found = findKey(key, record);
    if (found) 
    {
        value = *((vector<uint8_t>*) record->value);
    }

    return found;
}

DatabaseCacheRecord * DatabaseProgramCache::allocRecord(const string key, const void * value)
{
    // Allocate memory
    DatabaseCacheRecord * pRecord = new(DatabaseCacheRecord);
    if (pRecord == NULL)
    {
        zklog.error("DatabaseProgramCache::allocRecord() failed calling new(DatabaseCacheRecord)");
        exitProcess();
    }
    vector<uint8_t>* pValue = new(vector<uint8_t>);
    if (pValue == NULL)
    {
        zklog.error("DatabaseProgramCache::allocRecord() failed calling new(vector<uint8_t>)");
        exitProcess();
    }

    // Copy vector
    *pValue = *(const vector<uint8_t> *)value;

    // Assign values to record
    pRecord->value = pValue;
    pRecord->key = key;
    pRecord->size = 2*(
        sizeof(DatabaseCacheRecord)+
        (pRecord->key.capacity()+1)+
        sizeof(vector<uint8_t>)+
        sizeof(uint8_t)*pValue->capacity() );
        
    return pRecord;
}

void DatabaseProgramCache::freeRecord(DatabaseCacheRecord* record)
{
    delete (vector<uint8_t>*)(record->value);
    delete record;
}

void DatabaseProgramCache::updateRecord(DatabaseCacheRecord* record, const void * value)
{
    *(vector<uint8_t>*)(record->value) = *(vector<uint8_t>*)(value);
}
