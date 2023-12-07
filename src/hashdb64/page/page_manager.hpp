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

class PageContext;
class PageManager
{
public:

    PageManager();
    ~PageManager();

    zkresult init(PageContext &ctx);
    zkresult reset(PageContext &ctx);
    
    uint64_t getFreePage();
    void releasePage(const uint64_t pageNumber);
    uint64_t editPage(const uint64_t pageNumber);
    void flushPages(PageContext &ctx);
    inline char *getPageAddress(const uint64_t pageNumber);
    inline uint64_t getNumFreePages();
    inline uint64_t getFirstUnusedPage();

    zkresult addFile();
    zkresult addPages(const uint64_t nPages_);

    inline void readLock(){ headerLock.lock_shared();}
    inline void readUnlock(){ headerLock.unlock_shared();}

private:

    bool mappedFile;
    string fileName;
    string folderName;
    uint64_t fileSize;
    uint64_t pagesPerFile;
    uint64_t nFiles;

    shared_mutex dbResizeLock;
    uint64_t nPages;
    vector<char *> pages;

    recursive_mutex writePagesLock;
    uint64_t firstUnusedPage;
    uint64_t numFreePages;
    vector<uint64_t> freePages;
    unordered_map<uint64_t, uint64_t> editedPages;

    shared_mutex headerLock;

};

char* PageManager::getPageAddress(const uint64_t pageNumber)
{
    shared_lock<shared_mutex> guard_pages(dbResizeLock);
    zkassertpermanent(pageNumber < nPages);
    uint64_t fileId = pageNumber/pagesPerFile;
    uint64_t pageInFile = pageNumber % pagesPerFile;
    return pages[fileId] + pageInFile * (uint64_t)4096;
};

//Note: if there is a single writter thread we assume that only the writter thread will call this function! 
uint64_t PageManager::getNumFreePages(){
#if MULTIPLE_WRITES
        lock_guard<recursive_mutex> guard_freePages(writePagesLock);
        shared_lock<shared_mutex> guard_pages(dbResizeLock);
#endif
        return numFreePages+nPages-firstUnusedPage;
    };

//Note: if there is a single writter thread we assume that only the writter thread will call this function! 
inline uint64_t PageManager::getFirstUnusedPage(){
#if MULTIPLE_WRITES
        lock_guard<recursive_mutex> guard_freePages(writePagesLock);
#endif
        return firstUnusedPage;
}

#endif