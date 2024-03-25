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
#include "key_value_page.hpp"
#include "raw_data_page.hpp"
#include "zkglobals.hpp"

// Helper functions
string removeBSXIfExists64(string s) {return ((s.at(0) == '\\') && (s.at(1) == 'x')) ? s.substr(2) : s;}

Database64::Database64 (Goldilocks &fr, const Config &config) : headerPageNumber(0), currentFlushId(0)
{
    // Init mutex
    pthread_mutex_init(&mutex, NULL);

    zkresult zkr;
    headerPageNumber = 0;
    zkr = HeaderPage::InitEmptyPage(headerPageNumber);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::Database64() failed calling HeaderPage::InitEmptyPage() result=" + zkresult2string(zkr));
        exitProcess();
    }
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

    // Mark the database as initialized
    bInitialized = true;
}

zkresult Database64::readKV(const Goldilocks::Element (&root)[4], const Goldilocks::Element (&key)[4], mpz_class &value, uint64_t &level ,DatabaseMap *dbReadLog)
{
    zkresult zkr;

    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database64::readKV() called uninitialized");
        exitProcess();
    }

    // Convert root to a byte array
    string rootString = fea2string(fr, root);
    string rootBa =  string2ba(rootString);

    // Get the version associated to this root
    uint64_t version;
    zkr = HeaderPage::ReadRootVersion(headerPageNumber, rootBa, version);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::readKV() faile calling HeaderPage::ReadRootVersion() result=" + zkresult2string(zkr) + " root=" + rootString + " key=" + fea2string(fr, key));
        return zkr;
    }

    // Get the version data
    VersionDataStruct versionData;
    zkr = HeaderPage::ReadVersionData(headerPageNumber, version, versionData);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::readKV() faile calling HeaderPage::ReadVersionData() result=" + zkresult2string(zkr) + " root=" + rootString + " key=" + fea2string(fr, key));
        return zkr;
    }

    // Get the value
    string keyString = fea2string(fr, key);
    string keyBa = string2ba(keyString);
    zkr = HeaderPage::KeyValueHistoryRead(versionData.keyValueHistoryPage, keyBa, version, value, level);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::readKV() faile calling HeaderPage::KeyValueHistoryRead() result=" + zkresult2string(zkr) + " root=" + rootString + " key=" + fea2string(fr, key));
        return zkr;
    }
    
#ifdef LOG_DB_READ
    {
        string s = "Database64::readKV()";
        if (zkr != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(zkr);
        s += " key=" + keyStr;
        s += " value=";
        s += value.get_str(16) + ";";
        zklog.info(s);
    }
#endif

    return zkr;
}

zkresult Database64::setProgram (const string &key, const vector<uint8_t> &data, const bool persistent)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database64::setProgram() called uninitialized");
        exitProcess();
    }

    string program;
    ba2ba(data, program);
    zkresult zkr = HeaderPage::WriteProgram(headerPageNumber, string2ba(key), program);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::setProgram() failed calling HeaderPage::WriteProgram() result=" + zkresult2string(zkr));
    }

#ifdef LOG_DB_WRITE
    {
        string s = "Database64::setProgram()";
        if (zkr != ZKR_SUCCESS)
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

    return zkr;
}

zkresult Database64::getProgram(const string &key, vector<uint8_t> &data, DatabaseMap *dbReadLog)
{
    // Check that it has been initialized before
    if (!bInitialized)
    {
        zklog.error("Database64::getProgram() called uninitialized");
        exitProcess();
    }

    struct timeval t;
    if (dbReadLog != NULL) gettimeofday(&t, NULL);

    string program;
    zkresult zkr = HeaderPage::ReadProgram(headerPageNumber, string2ba(key), program);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::getProgram() failed calling HeaderPage::ReadProgram() result=" + zkresult2string(zkr));
    }
    else
    {
        ba2ba(program, data);
    }

#ifdef LOG_DB_READ
    {
        string s = "Database64::getProgram()";
        if (zkr != ZKR_SUCCESS)
            s += " ERROR=" + zkresult2string(zkr);
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
    Lock();
    currentFlushId++;
    thisBatch = currentFlushId;
    lastSentBatch = currentFlushId;

#ifdef LOG_DB_FLUSH
    zklog.info("Database64::flush() thisBatch=" + to_string(thisBatch) + " lastSentBatch=" + to_string(lastSentBatch) + " multiWrite=[" + multiWrite.print() + "]");
#endif

    Unlock();

    return ZKR_SUCCESS;
}

zkresult Database64::getFlushStatus(uint64_t &storedFlushId, uint64_t &storingFlushId, uint64_t &lastFlushId, uint64_t &pendingToFlushNodes, uint64_t &pendingToFlushProgram, uint64_t &storingNodes, uint64_t &storingProgram)
{
    Lock();
    storedFlushId = currentFlushId;
    storingFlushId = currentFlushId;
    lastFlushId = currentFlushId;
    pendingToFlushNodes = 0;
    pendingToFlushProgram = 0;
    storingNodes = 0;
    storingProgram = 0;
    Unlock();

    return ZKR_SUCCESS;
}

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

    //HeaderPage::Print(headerPage, true);

    vector<KeyValue> keyValues(_keyValues);

    if (keyValues.size() == 0)
    {
        zklog.error("Database64::WriteTree() called with keyValues.size=0");
        return ZKR_DB_ERROR;
    }

    //HeaderPage::Print(headerPageNumber, true);

    uint64_t version = 0;

    // Check if the root is zero
    if (fr.isZero(oldRoot[0]) && fr.isZero(oldRoot[1]) && fr.isZero(oldRoot[2]) && fr.isZero(oldRoot[3]))
    {
        uint64_t lastVersion = HeaderPage::GetLastVersion(headerPageNumber);
        if (lastVersion != 0)
        {
            zklog.error("Database64::WriteTree() called with a zero old state root, but last version=" + to_string(lastVersion) + " oldRoot=" + fea2string(fr, oldRoot));
            return ZKR_DB_ERROR;
        }
        version = 1;
    }
    else
    {
        // Get the old root as a string and byte array
        string oldRootString = fea2string(fr, oldRoot);
        string oldRootBa = string2ba(oldRootString);

        // Get the version corresponding to this old state root
        uint64_t oldRootVersion;
        zkr = HeaderPage::ReadRootVersion(headerPageNumber, oldRootBa, oldRootVersion);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Database64::WriteTree() failed calling HeaderPage::ReadRootVersion() result=" + zkresult2string(zkr) + " oldRootString=" + oldRootString);
            return zkr;
        }

        // Get the last version
        uint64_t lastVersion = HeaderPage::GetLastVersion(headerPageNumber);
        if (oldRootVersion != lastVersion)
        {
            zklog.error("Database64::WriteTree() found oldRootVersion=" + to_string(oldRootVersion) + " but lastVersion=" + to_string(lastVersion) + " oldRootString=" + oldRootString);
            return ZKR_DB_ERROR;
        }
        version = lastVersion + 1;
    }

    // Get an editable header page
    headerPageNumber = pageManager.editPage(headerPageNumber);
    HeaderStruct *headerPage = (HeaderStruct *)pageManager.getPageAddress(headerPageNumber);

    // Write all key-values
    string keyString;
    string key;
    for (uint64_t i=0; i<keyValues.size(); i++)
    {
        keyString = fea2string(fr, keyValues[i].key);
        key = string2ba(keyString);
        zkr = HeaderPage::KeyValueHistoryWrite(headerPageNumber, key, version, keyValues[i].value);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Database64::WriteTree() failed calling HeaderPage::KeyValueHistoryWrite() result=" + zkresult2string(zkr) + " oldRoot=" + fea2string(fr, oldRoot) + " version=" + to_string(version));
            return ZKR_DB_ERROR;
        }
        /*zklog.info("Database64::WriteTree() called HeaderPage::KeyValueHistoryWrite() oldRoot=" + fea2string(fr, oldRoot) + " version=" + to_string(version) + " key=" + keyString + " value=" + keyValues[i].value.get_str(16));
        
        mpz_class readValue;
        zkr = HeaderPage::KeyValueHistoryRead(headerPageNumber, key, version, readValue);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Database64::WriteTree() failed calling HeaderPage::KeyValueHistoryRead() result=" + zkresult2string(zkr) + " oldRoot=" + fea2string(fr, oldRoot) + " version=" + to_string(version));
            return ZKR_DB_ERROR;
        }
        if (readValue != keyValues[i].value)
        {
            zklog.error("Database64::WriteTree() called HeaderPage::KeyValueHistoryRead() readValue=" + readValue.get_str(16) + " expectedValue=" +  keyValues[i].value.get_str(16) + " oldRoot=" + fea2string(fr, oldRoot) + " version=" + to_string(version));
            return ZKR_DB_ERROR;
        }*/
    }

    //HeaderPage::Print(headerPageNumber, true);

    // Calculate new state root hash
    zkr = HeaderPage::KeyValueHistoryCalculateHash(headerPageNumber, newRoot);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::WriteTree() failed calling HeaderPage::KeyValueHistoryCalculateHash() result=" + zkresult2string(zkr) + " oldRoot=" + fea2string(fr, oldRoot));
        return ZKR_DB_ERROR;
    }

/*  struct VersionDataStruct
    {
        uint8_t  root[32];
        uint64_t keyValueHistoryPage;
        uint64_t freePagesList; TODO
        uint64_t createdPagesList; TODO
        uint64_t modifiedPagesList; TODO
        uint64_t rawDataPage;
        uint64_t rawDataOffset;
    };*/

    // Create version data
    VersionDataStruct versionData;
    string newRootBa = string2ba(fea2string(fr, newRoot));
    zkassert(newRootBa.size() == 32);
    memcpy(versionData.root, newRootBa.c_str(), 32);
    versionData.keyValueHistoryPage = headerPage->keyValueHistoryPage;
    versionData.rawDataPage = headerPage->rawDataPage;
    versionData.rawDataOffset = RawDataPage::GetOffset(versionData.rawDataPage);

    // Write version->versionData pair
    zkr = HeaderPage::WriteVersionData(headerPageNumber, version, versionData);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::WriteTree() failed calling HeaderPage::WriteVersionData() result=" + zkresult2string(zkr) + " oldRoot=" + fea2string(fr, oldRoot));
        return ZKR_DB_ERROR;
    }

    // Write root->version pair
    zkr = HeaderPage::WriteRootVersion(headerPageNumber, newRootBa, version);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::WriteTree() failed calling HeaderPage::WriteRootVersion() result=" + zkresult2string(zkr) + " oldRoot=" + fea2string(fr, oldRoot));
        return ZKR_DB_ERROR;
    }

    // Set last version
    HeaderPage::SetLastVersion(headerPageNumber, version);

    // Flush all pages to disk
    pageManager.flushPages();

    headerPageNumber = 0;
    //HeaderPage::Print(headerPageNumber, true);

    return ZKR_SUCCESS;
}

zkresult Database64::ReadTree (const Goldilocks::Element (&root)[4], vector<KeyValue> &keyValues, vector<HashValueGL> *hashValues)
{
    zkresult zkr;

    //HeaderPage::Print(headerPageNumber, true);

    if (keyValues.size() == 0)
    {
        zklog.error("Database64::ReadTree() called with keyValues.size=0");
        return ZKR_DB_ERROR;
    }

    //HeaderPage::Print(headerPageNumber, true);

    // Check if the root is zero, i.e. if all values should be zero
    if (fr.isZero(root[0]) && fr.isZero(root[1]) && fr.isZero(root[2]) && fr.isZero(root[3]))
    {
        for (uint64_t i=0; i<keyValues.size(); i++)
        {
            keyValues[i].value = 0;
        }
        return ZKR_SUCCESS;
    }
    
    // Get the old root as a string and byte array
    string rootString = fea2string(fr, root);
    string rootBa = string2ba(rootString);

    // Get the version corresponding to this state root
    uint64_t version = 0;
    zkr = HeaderPage::ReadRootVersion(headerPageNumber, rootBa, version);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::ReadTree() failed calling HeaderPage::ReadRootVersion() result=" + zkresult2string(zkr) + " rootString=" + rootString);
        return zkr;
    }

    // Get the version data
    VersionDataStruct versionData;
    zkr = HeaderPage::ReadVersionData(headerPageNumber, version, versionData);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("Database64::ReadTree() failed calling HeaderPage::ReadVersionData() result=" + zkresult2string(zkr) + " rootString=" + rootString);
        return zkr;
    }

    // Read all key-values
    string keyString;
    string key;
    uint64_t level;
    for (uint64_t i=0; i<keyValues.size(); i++)
    {
        keyString = fea2string(fr, keyValues[i].key);
        key = string2ba(keyString);
        zkr = HeaderPage::KeyValueHistoryRead(versionData.keyValueHistoryPage, key, version, keyValues[i].value, level);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("Database64::ReadTree() failed calling HeaderPage::KeyValueHistoryRead() result=" + zkresult2string(zkr) + " rootString=" + rootString + " version=" + to_string(version));
            return zkr;
        }
        //zklog.info("Database64::ReadTree() called HeaderPage::KeyValueHistoryRead() rootString=" + rootString + " version=" + to_string(version) + " key=" + keyString + " value=" + keyValues[i].value.get_str(16));
    }

    return ZKR_SUCCESS;
}