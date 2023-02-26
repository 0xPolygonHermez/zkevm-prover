#include "database_cache.hpp"
#include "utils.hpp"
#include "scalar.hpp"

// DatabaseCache class implementation

// Add a record in the head of the cache. Returns true if the cache is full (or no cache), false otherwise
bool DatabaseCache::addRecord(const string key, DatabaseCacheRecord* record) 
{
    if (cacheSize == 0) return true;

    DatabaseCacheRecord* tmpRecord;
    // If key already exists in the cache return. The findKey also sets the record in the head of the cache
    if (findKey(key, tmpRecord)) return false;

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

    if (cacheSize == -1) return false; // no cache limit

    cacheCurrentSize += record->size;
    full = (cacheCurrentSize > cacheSize); 
    // remove lats records from the cache to be under cacheSize
    while (cacheCurrentSize > cacheSize) 
    {
        // Set new last record
        DatabaseCacheRecord* tmp = last;
        if (last->prev != NULL) last->prev->next = NULL;
        last = last->prev;
        // Free old last record
        cacheMap.erase(tmp->key);
        freeRecord(tmp);
        // Update cache size
        cacheCurrentSize -= tmp->size;      
    }

    return full;
}

bool DatabaseCache::findKey(const string key, DatabaseCacheRecord* &record) 
{
    unordered_map<string, DatabaseCacheRecord*>::iterator it = cacheMap.find(key);

    if (it != cacheMap.end())
    {
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
    cout << "Cache current size: " << cacheCurrentSize << endl;
    cout << "Cache max size: " << cacheSize << endl;
    cout << "Head: " << (head != NULL ? head->key : "NULL") << endl;
    cout << "Last: " << (last != NULL ? last->key : "NULL") << endl;
    
    DatabaseCacheRecord* record = head;
    uint64_t count = 0;
    uint64_t size = 0;
    while (record != NULL) 
    {
        if (printContent)
        {
            cout << "key:" << record->key << endl;
        }
        count++;
        size += record->size;
        record = record->next;
    }
    cout << "Cache count: " << count << endl;
    cout << "Cache calculated size: " << size << endl;
}

DatabaseCache::~DatabaseCache()
{
    DatabaseCacheRecord* record = head;
    DatabaseCacheRecord* tmp;
    // Free cache records
    while (record != NULL) 
    {
        tmp = record->next;
        freeRecord(record);
        record = tmp;
    }
}

// DatabaseMTCache class implementation

// Add a record in the head of the MT cache. Returns true if the cache is full (or no cache), false otherwise
bool DatabaseMTCache::add(const string key, vector<Goldilocks::Element> value)
{
    lock_guard<recursive_mutex> guard(mlock);

    if (cacheSize == 0) return true;

    DatabaseCacheRecord* record = new(DatabaseCacheRecord);
    vector<Goldilocks::Element>* pValue = new(vector<Goldilocks::Element>);
    *pValue = value;
    record->value = pValue;
    record->key = key;
    //TODO: Use hardcoded value for the mtValue size (8*Goldilocks::Element)?
    record->size = sizeof(DatabaseCacheRecord)+record->key.size()+sizeof(Goldilocks::Element)*pValue->size();

    return addRecord(key, record);
}

bool DatabaseMTCache::find(const string key, vector<Goldilocks::Element> &value)
{
    lock_guard<recursive_mutex> guard(mlock);

    if (cacheSize == 0) return false;

    DatabaseCacheRecord* record;
    bool found = findKey(key, record);
    if (found) 
    {
        value = *((vector<Goldilocks::Element>*) record->value);
    }

    return found;
}

void DatabaseMTCache::freeRecord(DatabaseCacheRecord* record)
{
    delete (vector<Goldilocks::Element>*)(record->value);
    delete record;
}

// DatabaseProgramCache class implementation
// Add a record in the head of the Program cache. Returns true if the cache is full (or no cache), false otherwise
bool DatabaseProgramCache::add(const string key, vector<uint8_t> value)
{
    lock_guard<recursive_mutex> guard(mlock);

    if (cacheSize == 0) return true;

    DatabaseCacheRecord* record = new(DatabaseCacheRecord);
    vector<uint8_t>* pValue = new(vector<uint8_t>);
    *pValue = value;
    record->value = pValue;
    record->key = key;
    record->size = sizeof(DatabaseCacheRecord)+record->key.size()+pValue->size();

    return addRecord(key, record);
}

bool DatabaseProgramCache::find(const string key, vector<uint8_t> &value)
{
    lock_guard<recursive_mutex> guard(mlock);

    if (cacheSize == 0) return false;

    DatabaseCacheRecord* record;
    bool found = findKey(key, record);
    if (found) 
    {
        value = *((vector<uint8_t>*) record->value);
    }

    return found;
}

void DatabaseProgramCache::freeRecord(DatabaseCacheRecord* record)
{
    delete (vector<uint8_t>*)(record->value);
    delete record;
}
