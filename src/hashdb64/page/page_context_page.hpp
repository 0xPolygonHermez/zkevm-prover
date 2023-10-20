
#ifndef PAGE_CONTEXT_HPP
#define PAGE_CONTEXT_HPP

#include "page_manager.hpp"
#include "config.hpp"

class PageContext
{
public:

    PageManager &pageManager;
    const Config &config;

    PageContext (PageManager &pageManager_, const Config &config_ ) :
        pageManager(pageManager_), config(config_) {}; 

};
#endif