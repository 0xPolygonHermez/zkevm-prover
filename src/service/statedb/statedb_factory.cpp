#include "goldilocks/goldilocks_base_field.hpp"
#include "config.hpp"
#include "statedb_factory.hpp"
#include "statedb.hpp"
#include "statedb_remote.hpp"

StateDBInterface* StateDBClientFactory::createStateDBClient (Goldilocks &fr, const Config &config)
{
    if (config.stateDBURL=="local") return new StateDB (fr, config);
    else return new StateDBRemote (fr, config);
}
