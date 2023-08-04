#include "goldilocks_base_field.hpp"
#include "config.hpp"
#include "hashdb_factory.hpp"
#include "hashdb.hpp"
#include "hashdb_remote.hpp"
#include "hashdb_singleton.hpp"
#include "utils.hpp"
#include "exit_process.hpp"
#include "zklog.hpp"

HashDBInterface* HashDBClientFactory::createHashDBClient (Goldilocks &fr, const Config &config)
{
    if (config.hashDBURL=="local")
    {
        HashDBInterface * pLocalClient = hashDBSingleton.get();
        if (pLocalClient == NULL)
        {
            zklog.error("HashDBClientFactory::createHashDBClient() failed calling new hashDBSingleton.get()");
            exitProcess();
        }
        return pLocalClient;
    }

    HashDBInterface *pRemoteClient = new HashDBRemote (fr, config);
    if (pRemoteClient == NULL)
    {
        zklog.error("HashDBClientFactory::createHashDBClient() failed calling new HashDBRemote()");
        exitProcess();
    }
    return pRemoteClient;
}

void HashDBClientFactory::freeHashDBClient (HashDBInterface * pHashDB)
{
    // Check the interface is not null
    if (pHashDB == NULL)
    {
        zklog.error("HashDBClientFactory::freeHashDBClient() called with pHashDB=NULL");
        exitProcess();
    }

    if (pHashDB != hashDBSingleton.get())
    {
        delete pHashDB;
    }
}
