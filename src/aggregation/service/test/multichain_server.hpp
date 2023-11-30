#ifndef MULTICHAIN_SERVER_HPP
#define MULTICHAIN_SERVER_HPP

#include "goldilocks_base_field.hpp"
#include "prover_aggregation.hpp"
#include "config.hpp"

class MultichainServer
{
    Goldilocks &fr;
    Config &config;
    pthread_t t;
public:
    MultichainServer(Goldilocks &fr, Config &config) : fr(fr), config(config) {};
    void run (void);
    void runThread (void);
    void waitForThread (void);
};

void* multichainServerThread(void* arg);

#endif