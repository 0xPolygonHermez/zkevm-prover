#ifndef ZKPROVER_SERVER_HPP
#define ZKPROVER_SERVER_HPP

#include "goldilocks_base_field.hpp"
#include "prover.hpp"
#include "config.hpp"

class ZkServer
{
    Goldilocks &fr;
    Prover &prover;
    const Config &config;
    pthread_t t;
public:
    ZkServer(Goldilocks &fr, Prover &prover, const Config &config) : fr(fr), prover(prover), config(config) {};
    void run (void);
    void runThread (void);
    void waitForThread (void);
};

void* proverServerThread(void* arg);

#endif