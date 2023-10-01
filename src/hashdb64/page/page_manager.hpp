#ifndef PAGE_MANAGER_HPP
#define PAGE_MANAGER_HPP

#include <cstdint>

class PageManager
{
public:
    uint64_t getFreeMemoryPage (void);
    void     releaseMemoryPage (const uint64_t pageNumber);
};

extern PageManager pageManager;

#endif