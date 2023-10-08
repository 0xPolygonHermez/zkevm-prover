#ifndef PAGE_MANAGER_HPP
#define PAGE_MANAGER_HPP

#include <cstdint>
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkresult.hpp"
#include <vector>
#include <unordered_map>
#include <mutex>
#include <vector>
#include "zkassert.hpp"
#include <cassert>


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
        return numFreePages+nPages-firstUnusedPage;
    };
    inline char *getPageAddress(const uint64_t pageNumber)
    {
        assert(pageNumber < nPages);
        uint64_t fileId = pageNumber/pagesPerFile;
        uint64_t pageInFile = pageNumber % pagesPerFile;
        return pages[fileId] + pageInFile * (uint64_t)4096;
    };


private:

    recursive_mutex mlock;

    bool mappedFile;
    string fileName;
    string folderName;
    uint64_t fileSize;
    uint64_t pagesPerFile;
    uint64_t nFiles;
    int file0Descriptor;

    uint64_t nPages;
    vector<char *> pages;


    uint64_t firstUnusedPage;
    uint64_t numFreePages;
    vector<uint64_t> freePages;
    std::unordered_map<uint64_t, uint64_t> editedPages;

    zkresult AddPages(const uint64_t nPages_);

};

extern PageManager pageManager;

#endif