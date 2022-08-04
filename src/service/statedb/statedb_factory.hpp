#ifndef STATEDB_FACTORY_HPP
#define STATEDB_FACTORY_HPP

#include "goldilocks_base_field.hpp"
#include "config.hpp"
#include "statedb_interface.hpp"

class StateDBClientFactory
{
private:
    // Disallow creating an instance of this object
    StateDBClientFactory() {}
public:
    static StateDBInterface* createStateDBClient (Goldilocks &fr, const Config &config);
    static void freeStateDBClient (StateDBInterface * pStateDB);
};

#endif