#include "goldilocks/goldilocks_base_field.hpp"
#include "config.hpp"
#include "statedb_factory.hpp"
#include "statedb_local_client.hpp"
#include "statedb_remote_client.hpp"

StateDBClient* StateDBClientFactory::createStateDBClient (Goldilocks &fr, const Config &config)
{
    if (config.stateDBURL=="local") return new StateDBLocalClient (fr, config);
    else return new StateDBRemoteClient (fr, config);
}
