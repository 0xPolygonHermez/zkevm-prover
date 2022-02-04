#ifndef ZKPROVER_CLIENT_HPP
#define ZKPROVER_CLIENT_HPP

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include "zk-prover.grpc.pb.h"
#include "proof.hpp"
#include "ffiasm/fr.hpp"
#include "prover.hpp"

void* clientThread(void* arg);

class Client
{
public:
    RawFr &fr;
    const Config &config;
    zkprover::ZKProver::Stub * stub;
    pthread_t t; // Client thread

public:
    Client (RawFr &fr, const Config &config);

    void runThread (void);
    void GetStatus (void);
    string GenProof (void);
    void GetProof (const string &uuid);
    void Cancel (const string &uuid);
    void Execute (void);
};

void* clientThread (void* arg);

#endif