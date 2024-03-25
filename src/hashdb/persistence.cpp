#include "persistence.hpp"
#include "exit_process.hpp"

string persistence2string (const Persistence persistence)
{
    switch (persistence)
    {
        case PERSISTENCE_CACHE: return "CACHE";
        case PERSISTENCE_DATABASE: return "DATABASE";
        case PERSISTENCE_TEMPORARY: return "TEMPORARY";
        case PERSISTENCE_TEMPORARY_HASH: return "TEMPORARY_HASH";
        default: exitProcess();
    }
    exitProcess();
    return "";
}