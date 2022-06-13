#ifndef ZKPROVER_CLIENT_HPP
#define ZKPROVER_CLIENT_HPP

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include "zk-prover.grpc.pb.h"
#include "proof.hpp"
#include "goldilocks/goldilocks_base_field.hpp"
#include "prover.hpp"

void* clientThread(void* arg);

class Client
{
public:
    Goldilocks &fr;
    const Config &config;
    zkprover::v1::ZKProverService::Stub * stub;
    pthread_t t; // Client thread

public:
    Client (Goldilocks &fr, const Config &config);

    void runThread (void);
    void GetStatus (void);
    string GenProof (void);
    bool GetProof (const string &uuid); // Returns false if pending
    bool Cancel (const string &uuid);
    bool Execute (void);
};

void* clientThread (void* arg);

#endif