#ifndef ZKPROVER_SERVER_MOCK_HPP
#define ZKPROVER_SERVER_MOCK_HPP

#include "goldilocks_base_field.hpp"
#include "prover.hpp"
#include "config.hpp"

class ZkServerMock
{
    Goldilocks &fr;
    Prover &prover;
    Config &config;
    pthread_t t;
public:
    ZkServerMock(Goldilocks &fr, Prover &prover, Config &config) : fr(fr), prover(prover), config(config) {};
    void run (void);
    void runThread (void);
    void waitForThread (void);
};

void* serverMockThread(void* arg);

#endif