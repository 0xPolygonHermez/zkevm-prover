#include "page_manager.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include <cstring>
#include <cassert>

PageManager pageManager;
PageManager::PageManager()
{
    nPages = 0;
    pages = NULL;
    AddPages(10240);
}
PageManager::PageManager(const uint64_t nPages_)
{
    assert(nPages_ >= 2);
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
        for (uint64_t i = 2; i < nPages_; i++) //0 and 1 are reserved pages
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
    assert(pageNumber >= 2); //first two pages cannot be released
    memset(getPageAddress(pageNumber), 0, 4096);
    freePages.push_back(pageNumber);
}

uint32_t PageManager::editPage(const uint32_t pageNumber)
{
    assert(pageNumber != 1); //page 1 cannot be edited is the page used to edit the header
    if(pageNumber == 0){
        return 1;
    }
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
    memcpy(getPageAddress(0), getPageAddress(1), 4096); // copy tmp header to header
    for(unordered_set<uint32_t>::const_iterator it = editedPages.begin(); it != editedPages.end(); it++){
        releasePage(*it);
    }
    editedPages.clear();
}
