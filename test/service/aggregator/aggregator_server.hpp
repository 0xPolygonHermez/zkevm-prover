#ifndef AGGREGATOR_SERVER_HPP
#define AGGREGATOR_SERVER_HPP

#include "goldilocks_base_field.hpp"
#include "prover.hpp"
#include "config.hpp"

class AggregatorServer
{
    Goldilocks &fr;
    Config &config;
    pthread_t t;
public:
    AggregatorServer(Goldilocks &fr, Config &config) : fr(fr), config(config) {};
    void run (void);
    void runThread (void);
    void waitForThread (void);
};

void* aggregatorServerThread(void* arg);

#endif