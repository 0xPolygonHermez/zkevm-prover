#include "database_cache_64.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "zklog.hpp"
#include "zkmax.hpp"
#include "timer.hpp"

// DatabaseCache class implementation

// Add a record in the head of the cache. Returns true if the cache is full (or no cache), false otherwise
bool DatabaseCache64::addKeyValue(const string &key, const void * value, const bool update) 
{
    if (maxSize == 0)
    {
        return true;
    }

    if (attempts%1000000 == 0)
    {
        zklog.info("DatabaseCache64::addKeyValue() name=" + name + " count=" + to_string(cacheMap.size()) + " maxSize=" + to_string(maxSize) + " currentSize=" + to_string(currentSize) + " attempts=" + to_string(attempts) + " hits=" + to_string(hits) + " hit ratio=" + to_string(double(hits)*100.0/double(zkmax(attempts,1))) + "%");
    }

    DatabaseCacheRecord64 * record;
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
        DatabaseCacheRecord64* tmp = last;
        last->prev->next = NULL;
        last = last->prev;
        
        // Free old last record
        cacheMap.erase(tmp->key);
        
        // Update cache size
        zkassert(currentSize >= tmp->size);
        currentSize -= tmp->size;

        freeRecord(tmp);
    }

    //zklog.info("DatabaseCache64::addRecord() key=" + key + " cacheCurrentSize=" + to_string(cacheCurrentSize) + " cacheMap.size()=" + to_string(cacheMap.size()) + " record.size()=" + to_string(record->size));
    //printMemoryInfo(true);
    
    return full;
}

bool DatabaseCache64::findKey(const string &key, DatabaseCacheRecord64* &record) 
{
    attempts++;
    unordered_map<string, DatabaseCacheRecord64*>::iterator it = cacheMap.find(key);

    if (it != cacheMap.end())
    {
        hits++;
        record = (DatabaseCacheRecord64*)it->second;

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

void DatabaseCache64::print(bool printContent)
{
    zklog.info("DatabaseCache64::print() printContent=" + to_string(printContent) + " name=" + name);
    zklog.info("Cache current size: " + to_string(currentSize));
    zklog.info("Cache max size: " + to_string(maxSize));
    zklog.info("Head: " + (head != NULL ? head->key : "NULL"));
    zklog.info("Last: " + (last != NULL ? last->key : "NULL"));
    
    DatabaseCacheRecord64* record = head;
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

void DatabaseCache64::clear(void)
{
    lock_guard<recursive_mutex> guard(mlock);

    DatabaseCacheRecord64* record = head;
    DatabaseCacheRecord64* tmp;
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

DatabaseCache64::~DatabaseCache64()
{
    TimerStart(DATABASE_CACHE_DESTRUCTOR);
    //clear();
    TimerStopAndLog(DATABASE_CACHE_DESTRUCTOR);
}

// DatabaseMTCache class implementation


DatabaseMTCache64::~DatabaseMTCache64()
{
    TimerStart(DATABASE_MT_CACHE_DESTRUCTOR);
    clear();
    TimerStopAndLog(DATABASE_MT_CACHE_DESTRUCTOR);
}

// Add a record in the head of the MT cache. Returns true if the cache is full (or no cache), false otherwise
bool DatabaseMTCache64::add(const string &key, const vector<Goldilocks::Element> &value, const bool update)
{
    lock_guard<recursive_mutex> guard(mlock);

    if (maxSize == 0) return true;

    return addKeyValue(key, (const void *)&value, update);
}

bool DatabaseMTCache64::find(const string &key, vector<Goldilocks::Element> &value)
{
    lock_guard<recursive_mutex> guard(mlock);

    if (maxSize == 0) return false;

    DatabaseCacheRecord64* record;
    bool found = findKey(key, record);
    if (found) 
    {
        value = *((vector<Goldilocks::Element>*) record->value);
    }

    return found;
}

DatabaseCacheRecord64 * DatabaseMTCache64::allocRecord(const string key, const void * value)
{
    // Allocate memory
    DatabaseCacheRecord64 * pRecord = new(DatabaseCacheRecord64);
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
        sizeof(DatabaseCacheRecord64)+
        (pRecord->key.capacity()+1)+
        sizeof(vector<Goldilocks::Element>)+
        sizeof(Goldilocks::Element)*pValue->capacity() );
        
    return pRecord;
}

void DatabaseMTCache64::freeRecord(DatabaseCacheRecord64* record)
{
    delete (vector<Goldilocks::Element>*)(record->value);
    delete record;
}

void DatabaseMTCache64::updateRecord(DatabaseCacheRecord64* record, const void * value)
{
    *(vector<Goldilocks::Element>*)(record->value) = *(vector<Goldilocks::Element>*)(value);
}

// DatabaseProgramCache class implementation

DatabaseProgramCache64::~DatabaseProgramCache64()
{
    TimerStart(DATABASE_PROGRAM_CACHE_DESTRUCTOR);
    clear();
    TimerStopAndLog(DATABASE_PROGRAM_CACHE_DESTRUCTOR);
}

// Add a record in the head of the Program cache. Returns true if the cache is full (or no cache), false otherwise
bool DatabaseProgramCache64::add(const string &key, const vector<uint8_t> &value, const bool update)
{
    lock_guard<recursive_mutex> guard(mlock);

    if (maxSize == 0) return true;

    return addKeyValue(key, (const void *)&value, update);
}

bool DatabaseProgramCache64::find(const string &key, vector<uint8_t> &value)
{
    lock_guard<recursive_mutex> guard(mlock);

    if (maxSize == 0) return false;

    DatabaseCacheRecord64* record;
    bool found = findKey(key, record);
    if (found) 
    {
        value = *((vector<uint8_t>*) record->value);
    }

    return found;
}

DatabaseCacheRecord64 * DatabaseProgramCache64::allocRecord(const string key, const void * value)
{
    // Allocate memory
    DatabaseCacheRecord64 * pRecord = new(DatabaseCacheRecord64);
    if (pRecord == NULL)
    {
        zklog.error("DatabaseProgramCache64::allocRecord() failed calling new(DatabaseCacheRecord)");
        exitProcess();
    }
    vector<uint8_t>* pValue = new(vector<uint8_t>);
    if (pValue == NULL)
    {
        zklog.error("DatabaseProgramCache64::allocRecord() failed calling new(vector<uint8_t>)");
        exitProcess();
    }

    // Copy vector
    *pValue = *(const vector<uint8_t> *)value;

    // Assign values to record
    pRecord->value = pValue;
    pRecord->key = key;
    pRecord->size = 2*(
        sizeof(DatabaseCacheRecord64)+
        (pRecord->key.capacity()+1)+
        sizeof(vector<uint8_t>)+
        sizeof(uint8_t)*pValue->capacity() );
        
    return pRecord;
}

void DatabaseProgramCache64::freeRecord(DatabaseCacheRecord64* record)
{
    delete (vector<uint8_t>*)(record->value);
    delete record;
}

void DatabaseProgramCache64::updateRecord(DatabaseCacheRecord64* record, const void * value)
{
    *(vector<uint8_t>*)(record->value) = *(vector<uint8_t>*)(value);
}
