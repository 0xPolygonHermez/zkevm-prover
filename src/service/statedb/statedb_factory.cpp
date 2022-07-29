#include "goldilocks/goldilocks_base_field.hpp"
#include "config.hpp"
#include "statedb_factory.hpp"
#include "statedb_local_client.hpp"
#include "statedb_remote_client.hpp"
#include "utils.hpp"

StateDBClient * pLocalClient = NULL;

StateDBClient* StateDBClientFactory::createStateDBClient (Goldilocks &fr, const Config &config)
{
    if (config.stateDBURL=="local")
    {
        if (pLocalClient == NULL)
        {
            pLocalClient = new StateDBLocalClient (fr, config);
            if (pLocalClient == NULL)
            {
                cerr << "Error: StateDBClientFactory::createStateDBClient() failed calling new StateDBLocalClient()" << endl;
                exitProcess();
            }
        }
        return pLocalClient;
    }

    StateDBClient * pRemoteClient = new StateDBRemoteClient (fr, config);
    if (pRemoteClient == NULL)
    {
        cerr << "Error: StateDBClientFactory::createStateDBClient() failed calling new StateDBRemoteClient()" << endl;
        exitProcess();
    }
    return pRemoteClient;
}
