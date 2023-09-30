#include "page_manager.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

PageManager pageManager;

uint64_t PageManager::getFreeMemoryPage (void)
{
    // TODO: allocate a large amount of memory in the constructor/init and build a list of free and busy pages
    void * address = calloc(1, 4096*2);
    if (address == NULL)
    {
        zklog.error("PageManager::getFreeMemoryPage() failed calling calloc()");
        exitProcess();
    }
    uint64_t pageNumber = ((uint64_t)address + 4095)/4096;
    return pageNumber;
}

void PageManager::releaseMemoryPage (const uint64_t pageNumber)
{
    // TODO: add pageNumber to list of free pages
}