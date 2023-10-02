#include "page_manager.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include <cstring>

PageManager pageManager;

PageManager::PageManager()
{
    nPages = 0;
    pages = NULL;
}

PageManager::PageManager(const uint64_t nPages_)
{
    nPages = 0;
    AddPages(nPages_);
}
PageManager::~PageManager(void)
{
    if (pages != NULL)
        free(pages);
}
zkresult PageManager::AddPages(const uint64_t nPages_)
{
    char *auxPages = NULL;
    auxPages = (char *)realloc(pages, nPages + nPages_ * 4096);
    if (auxPages != NULL)
    {
        pages = auxPages;
        memset(pages + nPages * 4096, 0, nPages_ * 4096);
        nPages += nPages_;
        for (uint64_t i = 0; i < nPages_; i++)
            freePages.push_back(nPages + i);
    }
    else
    {
        zklog.error("PageManager::AddPages() failed calling realloc()");
        exitProcess();
    }
    return zkresult::ZKR_SUCCESS;
}

uint64_t PageManager::getFreePage(void)
{
    uint32_t pageNumber = freePages.front();
    freePages.pop_front();
    return pageNumber;
}

void PageManager::releasePage(const uint64_t pageNumber)
{
    memset(getPage(pageNumber), 0, 4096);
    freePages.push_back(pageNumber);
}