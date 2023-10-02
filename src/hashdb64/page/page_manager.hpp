#ifndef PAGE_MANAGER_HPP
#define PAGE_MANAGER_HPP

#include <cstdint>
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkresult.hpp"
#include <list>

class PageManager
{
public:
    PageManager();
    PageManager(const uint64_t nPages_);
    ~PageManager();
    zkresult AddPages(const uint64_t nPages_);
    inline char *getPage(const uint64_t pageNumber)
    {
        return pages + pageNumber * 4096;
    };

    uint64_t getFreePage();
    void releasePage(const uint64_t pageNumber);

private:
    uint32_t nPages;
    char *pages;
    std::list<uint32_t> freePages;
};

extern PageManager pageManager;

#endif