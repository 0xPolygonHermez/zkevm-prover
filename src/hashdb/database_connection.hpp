#ifndef DATABASE_CONNECTION_HPP
#define DATABASE_CONNECTION_HPP

#include <pqxx/pqxx>

class DatabaseConnection
{
public:
    pqxx::connection * pConnection;
    bool bInUse;
    bool bDisconnect;
    DatabaseConnection() : pConnection(NULL), bInUse(false), bDisconnect(false) {};
};

#endif