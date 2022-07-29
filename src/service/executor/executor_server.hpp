#ifndef EXECUTOR_SERVER_HPP
#define EXECUTOR_SERVER_HPP

#include "goldilocks_base_field.hpp"
#include "prover.hpp"
#include "config.hpp"

class ExecutorServer
{
    Goldilocks &fr;
    Prover &prover;
    Config &config;
    pthread_t t;
public:
    ExecutorServer(Goldilocks &fr, Prover &prover, Config &config) : fr(fr), prover(prover), config(config) {};
    void run (void);
    void runThread (void);
    void waitForThread (void);
};

void* executorServerThread(void* arg);

#endif