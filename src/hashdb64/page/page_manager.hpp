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
#include "config.hpp"

#define MULTIPLE_WRITES 0

class PageContext;
class PageManager
{
public:

    PageManager();
    ~PageManager();

    zkresult init(PageContext &ctx);
    zkresult addFile();
    zkresult addPages(const uint64_t nPages_);

    uint64_t getFreePage();
    void releasePage(const uint64_t pageNumber);
    uint64_t editPage(const uint64_t pageNumber);
    void flushPages(PageContext &ctx);

    inline uint64_t getNumFreePages();
    inline char *getPageAddress(const uint64_t pageNumber);

    inline void readLock(){ headerLock.lock_shared();}
    inline void readUnlock(){ headerLock.unlock_shared();}

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

    mutex freePagesLock;
    uint64_t firstUnusedPage;
    uint64_t numFreePages;
    vector<uint64_t> freePages;

    mutex editedPagesLock;
    std::unordered_map<uint64_t, uint64_t> editedPages;

    shared_mutex headerLock;

};

char* PageManager::getPageAddress(const uint64_t pageNumber)
{
    shared_lock<shared_mutex> guard(pagesLock);
    assert(pageNumber < nPages);
    uint64_t fileId = pageNumber/pagesPerFile;
    uint64_t pageInFile = pageNumber % pagesPerFile;
    return pages[fileId] + pageInFile * (uint64_t)4096;
};

uint64_t PageManager::getNumFreePages(){
#if MULTIPLE_WRITES
        lock_guard<mutex> guard(freePagesLock);
        shared_lock<shared_mutex> guard2(pagesLock);
#endif
        return numFreePages+nPages-firstUnusedPage;
    };
#endif