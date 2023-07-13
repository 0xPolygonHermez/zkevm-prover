#include "database_associative_cache.hpp"
#include "scalar.hpp"
#include "zklog.hpp"
#include "zkmax.hpp"
#include "timer.hpp"
#include "zkassert.hpp"

// DatabaseAssociativeCache class implementation
// Add a record in the head of the cache. Returns true if the cache is full (or no cache), false otherwise
bool DatabaseAssociativeCache::addKeyValue(const uint64_t index, const string &remainingKey, const void * value, const bool update, const string& leftChildkey,const string& rightChildKey){

    if (nKeyBits == 0)
    {
        return true;
    }

    attempts++;
    if (attempts<<44 == 0)
    {
        zklog.info("DatabaseAssociativeCache::addKeyValue() name=" + name + " cacheSize=" + to_string(size) + " attempts=" + to_string(attempts) + " hits=" + to_string(hits) + " hit ratio=" + to_string(double(hits)*100.0/double(zkmax(attempts,1))) + "%");
    }

    
    DatabaseAssociativeCacheRecord * record;
    // If key already exists in the cache return. The findKey also sets the record in the head of the cache
    if (findKey(index,remainingKey, record))
    {
        if (update)
        {
            updateRecord(record, value,leftChildkey,rightChildKey);
            return true;
        }
        else
        {
            hits++;
            return false;
        }
    }

    record = allocRecord(remainingKey, value,leftChildkey,rightChildKey);
    buffer[index] = record;
    /*vector<Goldilocks::Element> &auxvalue = *((vector<Goldilocks::Element>*) record->value);
    Goldilocks fr;
    string auxLeftChildKey = fea2string(fr, auxvalue[0], auxvalue[1], auxvalue[2], auxvalue[3]);
    string auxRighChildKey = fea2string(fr, auxvalue[4], auxvalue[5], auxvalue[6], auxvalue[7]);
    auxLeftChildKey = NormalizeToNFormat(auxLeftChildKey, 64);
    auxLeftChildKey = stringToLower(auxLeftChildKey);
    auxRighChildKey = NormalizeToNFormat(auxRighChildKey, 64);
    auxRighChildKey = stringToLower(auxRighChildKey);
    if(auxLeftChildKey != leftChildkey || auxRighChildKey != rightChildKey){
        
        zklog.error("DatabaseAssociativeCache::addKeyValue() leftChildKey=" + leftChildkey + " rightChildKey=" + rightChildKey);
        zklog.error("DatabaseAssociativeCache::addKeyValue() auxLeftChildKey=" + auxLeftChildKey + " auxRighChildKey=" + auxRighChildKey);
        //exitProcess();
    }*/

    return true;
} 
bool DatabaseAssociativeCache::findKey(const uint64_t index, const string &remainingKey, DatabaseAssociativeCacheRecord* &record){
    
    if(nKeyBits==0 || buffer[index]== NULL || buffer[index]->remainingKey != remainingKey){
        return false;
    }
    record = buffer[index];
    return true;
}

// DatabaseMTAssociativeCach class implementation
DatabaseAssociativeCacheRecord * DatabaseMTAssociativeCache::allocRecord(const string remainingKey, const void * value, const string& leftChildkey,const string& rightChildKey)
{
    // Allocate memory
    DatabaseAssociativeCacheRecord * pRecord = new(DatabaseAssociativeCacheRecord);
    if (pRecord == NULL)
    {
        zklog.error("DatabaseMTAssociativeCache::allocRecord() failed calling new(DatabaseAssociativeCacheRecord)");
        exitProcess();
    }
    vector<Goldilocks::Element>* pValue = new(vector<Goldilocks::Element>);
    if (pValue == NULL)
    {
        zklog.error("DatabaseMTAssociativeCache::allocRecord() failed calling new(vector<Goldilocks::Element>)");
        exitProcess();
    }

    // Copy vector
    *pValue = *(const vector<Goldilocks::Element> *)value;

    // Assign values to record
    pRecord->value = pValue;
    pRecord->remainingKey = remainingKey;
    pRecord->leftChildKey = leftChildkey;
    pRecord->rightChildKey = rightChildKey;
        
    return pRecord;
}

// DatabaseProgramAssociativeCache class implementation
DatabaseAssociativeCacheRecord * DatabaseProgramAssociativeCache::allocRecord(const string remainingKey, const void * value, const string& leftChildkey,const string& rightChildKey)
{
    // Allocate memory
    DatabaseAssociativeCacheRecord * pRecord = new(DatabaseAssociativeCacheRecord);
    if (pRecord == NULL)
    {
        zklog.error("DatabaseProgramAssociativeCache::allocRecord() failed calling new(DatabaseAssociativeCacheRecord)");
        exitProcess();
    }
    vector<uint8_t>* pValue = new(vector<uint8_t>);
    if (pValue == NULL)
    {
        zklog.error("DatabaseProgramAssociativeCache::allocRecord() failed calling new(vector<uint8_t>)");
        exitProcess();
    }

    // Copy vector
    *pValue = *(const vector<uint8_t> *)value;

    // Assign values to record
    pRecord->value = pValue;
    pRecord->remainingKey = remainingKey;
    pRecord->leftChildKey = leftChildkey;
    pRecord->rightChildKey = rightChildKey;
        
    return pRecord;
}

// TODO:
// 1.Use ramaining key whenever possible
// 2.Avoid using pointers in the cache, use templates instead
// 3.Currently desctructor is not correct because it does not call the free function
// 4.conflict: understand conflics in write orderings