#ifndef ZKPROVER_SERVER_HPP
#define ZKPROVER_SERVER_HPP

#include "ffiasm/fr.hpp"
#include "prover.hpp"

class ZkServer
{
    RawFr &fr;
    Prover &prover;
public:
    ZkServer(RawFr &fr, Prover &prover) : fr(fr), prover(prover) {};
    void run (void);
};

#endif