#ifndef HEADER_PAGE_HPP
#define HEADER_PAGE_HPP

#include <unistd.h>
#include "zkresult.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"

struct HeaderStruct
{
    uint64_t rootVersionPage;
};

class HeaderPage
{
public:
    static zkresult InitEmptyPage (const uint64_t pageNumber);

    static void Print (const uint64_t pageNumber, bool details);
};

#endif