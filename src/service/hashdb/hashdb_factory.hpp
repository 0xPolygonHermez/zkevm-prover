#ifndef HASHDB_FACTORY_HPP
#define HASHDB_FACTORY_HPP

#include "goldilocks_base_field.hpp"
#include "config.hpp"
#include "hashdb_interface.hpp"

class HashDBClientFactory
{
private:
    // Disallow creating an instance of this object
    HashDBClientFactory() {}
public:
    static HashDBInterface* createHashDBClient (Goldilocks &fr, const Config &config);
    static void freeHashDBClient (HashDBInterface * pHashDB);
};

#endif