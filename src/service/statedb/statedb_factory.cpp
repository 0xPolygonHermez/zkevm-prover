#include "goldilocks_base_field.hpp"
#include "config.hpp"
#include "statedb_factory.hpp"
#include "statedb.hpp"
#include "statedb_remote.hpp"
#include "utils.hpp"

StateDBInterface * pLocalClient = NULL;

StateDBInterface* StateDBClientFactory::createStateDBClient (Goldilocks &fr, const Config &config)
{
    if (config.stateDBURL=="local")
    {
        if (pLocalClient == NULL)
        {
            pLocalClient = new StateDB (fr, config);
            if (pLocalClient == NULL)
            {
                cerr << "Error: StateDBClientFactory::createStateDBClient() failed calling new StateDB()" << endl;
                exitProcess();
            }
        }
        return pLocalClient;
    }

    StateDBInterface * pRemoteClient = new StateDBRemote (fr, config);
    if (pRemoteClient == NULL)
    {
        cerr << "Error: StateDBClientFactory::createStateDBClient() failed calling new StateDBRemote()" << endl;
        exitProcess();
    }
    return pRemoteClient;
}
