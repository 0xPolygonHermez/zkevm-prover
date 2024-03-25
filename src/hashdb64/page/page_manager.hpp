#ifndef PAGE_MANAGER_HPP
#define PAGE_MANAGER_HPP

#include <cstdint>
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkresult.hpp"
#include <vector>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include "zkassert.hpp"
#include <cassert>
#include <unistd.h>

#define MULTIPLE_WRITES 0

class PageManager
{
public:

    PageManager();
    PageManager(const uint64_t nPages_);
    PageManager(const string fileName_, const uint64_t fileSize_= 1ULL<<37, const uint64_t nFiles_=1, const string folderName_="db");
    ~PageManager();
    

    uint64_t getFreePage();
    void releasePage(const uint64_t pageNumber);
    uint64_t editPage(const uint64_t pageNumber);
    void flushPages();

    inline uint64_t getNumFreePages(){
#if MULTIPLE_WRITES
        lock_guard<mutex> guard(freePagesLock);
        shared_lock<shared_mutex> guard2(pagesLock);
#endif
        return numFreePages+nPages-firstUnusedPage;
    };
    inline char *getPageAddress(const uint64_t pageNumber)
    {
        shared_lock<shared_mutex> guard(pagesLock);
        assert(pageNumber < nPages);
        uint64_t fileId = pageNumber/pagesPerFile;
        uint64_t pageInFile = pageNumber % pagesPerFile;
        return pages[fileId] + pageInFile * (uint64_t)4096;
    };
    void getPageAddressFile(const uint64_t pageNumber, char *out)
    {
        shared_lock<shared_mutex> guard(pagesLock);
        assert(pageNumber < nPages);
        uint64_t fileId = pageNumber/pagesPerFile;
        uint64_t pageInFile = pageNumber % pagesPerFile;
        pread(fileDescriptors[fileId], out, 4096, pageInFile * (uint64_t)4096);
    };

    inline void readLock(){
        headerLock.lock_shared();
    }
    inline void readUnlock(){
        headerLock.unlock_shared();
    }

private:


    bool mappedFile;
    string fileName;
    string folderName;
    uint64_t fileSize;
    uint64_t pagesPerFile;
    uint64_t nFiles;
    int file0Descriptor;

    std::shared_mutex pagesLock;
    uint64_t nPages;
    vector<char *> pages;
    vector<int> fileDescriptors;
    zkresult AddPages(const uint64_t nPages_);

    mutex freePagesLock;
    uint64_t firstUnusedPage;
    uint64_t numFreePages;
    vector<uint64_t> freePages;

    mutex editedPagesLock;
    std::unordered_map<uint64_t, uint64_t> editedPages;

    shared_mutex headerLock;

};

extern PageManager pageManager;

#endif