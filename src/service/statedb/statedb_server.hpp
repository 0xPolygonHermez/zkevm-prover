#ifndef STATEDB_SERVER_HPP
#define STATEDB_SERVER_HPP

#include "goldilocks_base_field.hpp"
#include "database.hpp"

class StateDBServer
{
private:    
    Goldilocks &fr;
    Config &config;
    pthread_t t;

public:
    StateDBServer (Goldilocks &fr, Config &config) : fr(fr), config(config) {}; 
    void run (void);
    void runThread (void);
    void waitForThread (void);
};

void* stateDBServerThread(void* arg);

#endif