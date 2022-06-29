#ifndef STATEDB_FACTORY_HPP
#define STATEDB_FACTORY_HPP

#include "goldilocks/goldilocks_base_field.hpp"
#include "config.hpp"
#include "statedb_client.hpp"

class StateDBClientFactory
{
private:
    // Disallow creating an instance of this object
    StateDBClientFactory() {}
public:
    static StateDBClient* createStateDBClient (Goldilocks &fr, const Config &config);       
};

#endif