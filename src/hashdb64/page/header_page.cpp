#include "header_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "page_manager.hpp"

zkresult HeaderPage::InitEmptyPage (const uint64_t pageNumber)
{
    HeaderStruct * page = (HeaderStruct *)pageManager.getPage(pageNumber);
    //page->rootVersionPage = 0;
    return ZKR_SUCCESS;
}

void HeaderPage::Print (const uint64_t pageNumber, bool details)
{
    HeaderStruct * page = (HeaderStruct *)pageManager.getPage(pageNumber);
    zklog.info("HeaderPage::Print() pageNumber=" + to_string(pageNumber));
    zklog.info("  rootVersionPage=" + to_string(page->rootVersionPage));
}