#include "goldilocks_base_field.hpp"
#include "config.hpp"
#include "statedb_factory.hpp"
#include "statedb.hpp"
#include "statedb_remote.hpp"
#include "statedb_singleton.hpp"
#include "utils.hpp"
#include "exit_process.hpp"
#include "zklog.hpp"

StateDBInterface* StateDBClientFactory::createStateDBClient (Goldilocks &fr, const Config &config)
{
    if (config.stateDBURL=="local")
    {
        StateDBInterface * pLocalClient = stateDBSingleton.get(fr, config);
        if (pLocalClient == NULL)
        {
            zklog.error("StateDBClientFactory::createStateDBClient() failed calling new stateDBSingleton.get()");
            exitProcess();
        }
        return pLocalClient;
    }

    StateDBInterface *pRemoteClient = new StateDBRemote (fr, config);
    if (pRemoteClient == NULL)
    {
        zklog.error("StateDBClientFactory::createStateDBClient() failed calling new StateDBRemote()");
        exitProcess();
    }
    return pRemoteClient;
}

void StateDBClientFactory::freeStateDBClient (StateDBInterface * pStateDB)
{
    // Check the interface is not null
    if (pStateDB == NULL)
    {
        zklog.error("StateDBClientFactory::freeStateDBClient() called with pStateDB=NULL");
        exitProcess();
    }

    if (pStateDB != stateDBSingleton.get())
    {
        delete pStateDB;
    }
}
