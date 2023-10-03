#ifndef PAGE_MANAGER_HPP
#define PAGE_MANAGER_HPP

#include <cstdint>
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkresult.hpp"
#include <list>
#include <unordered_set>

class PageManager
{
public:
    PageManager();
    PageManager(const uint64_t nPages_);
    ~PageManager();
    inline char *getPageAddress(const uint64_t pageNumber)
    {
        return pages + pageNumber * 4096;
    };

    uint64_t getFreePage();
    void releasePage(const uint64_t pageNumber);
    uint32_t editPage(const uint32_t pageNumber);
    void flushPages();


private:
    uint32_t nPages;
    char *pages;
    
    std::list<uint32_t> freePages;
    std::unordered_set<uint32_t> editedPages;

    zkresult AddPages(const uint64_t nPages_);

};

extern PageManager pageManager;

#endif