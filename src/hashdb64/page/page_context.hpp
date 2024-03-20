
#ifndef PAGE_CONTEXT_HPP
#define PAGE_CONTEXT_HPP

#include "page_manager.hpp"
#include "config.hpp"
#include "Keccak-more-compact.hpp"

class PageContext
{
public:

    PageManager &pageManager;
    const Config &config;
    uint8_t uuid[32];

    PageContext (PageManager &pageManager_, const Config &config_ ) :
        pageManager(pageManager_), config(config_) {
            string uuidString = "Polygon zkEVM HashDB64 v1.0.0";
            FIPS202_SHA3_256((uint8_t *)uuidString.c_str(), uuidString.size(), uuid);
        }; 
};
#endif