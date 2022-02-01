#ifndef ZKPROVER_SERVER_HPP
#define ZKPROVER_SERVER_HPP

#include "ffiasm/fr.hpp"
#include "prover.hpp"
#include "config.hpp"

class ZkServer
{
    RawFr &fr;
    Prover &prover;
    Config &config;
public:
    ZkServer(RawFr &fr, Prover &prover, Config &config) : fr(fr), prover(prover), config(config) {};
    void run (void);
};

#endif