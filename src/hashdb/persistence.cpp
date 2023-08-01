#include "persistence.hpp"
#include "exit_process.hpp"

string persistence2string (const Persistence persistence)
{
    switch (persistence)
    {
        case PERSISTENCE_CACHE: return "CACHE";
        case PERSISTENCE_DATABASE: return "DATABASE";
        case PERSISTENCE_TEMPORARY: return "TEMPORARY";
        default: exitProcess();
    }
    exitProcess();
    return "";
}