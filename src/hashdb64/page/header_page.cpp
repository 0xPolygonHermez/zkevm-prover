#include "header_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"

zkresult HeaderPage::InitEmptyPage (const uint64_t pageNumber)
{
    HeaderStruct * page = (HeaderStruct *)(pageNumber*4096);
    page->rootVersionPage = 0;
    return ZKR_SUCCESS;
}