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
    AddPages(10240);
}
PageManager::PageManager(const uint64_t nPages_)
{
    nPages = 0;
    pages = NULL;
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
    auxPages = (char *)realloc(pages, (nPages + nPages_) * 4096);
    if (auxPages != NULL)
    {
        pages = auxPages;
        memset(pages + nPages * 4096, 0, nPages_ * 4096);
        for (uint64_t i = 0; i < nPages_; i++)
            freePages.push_back(nPages + i);
        nPages += nPages_;
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
    if(freePages.empty())
    {
        AddPages(1024);
    }
    uint32_t pageNumber = freePages.front();
    freePages.pop_front();
    return pageNumber;
}
void PageManager::releasePage(const uint64_t pageNumber)
{
    memset(getPageAddress(pageNumber), 0, 4096);
    freePages.push_back(pageNumber);
}

uint32_t PageManager::editPage(const uint32_t pageNumber)
{
    std::pair<std::unordered_set<uint32_t>::iterator,bool> res = editedPages.insert(pageNumber);
    uint32_t pageNumber_ = pageNumber;
    if(res.second == true)
    {
        pageNumber_ = getFreePage();
        memcpy(getPageAddress(pageNumber_),getPageAddress(pageNumber) , 4096);
    }
    return pageNumber_;
}

void PageManager::flushPages(){
    for(unordered_set<uint32_t>::const_iterator it = editedPages.begin(); it != editedPages.end(); it++){
        releasePage(*it);
    }
    editedPages.clear();
}
