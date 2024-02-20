#ifndef HASHDB_SERVER_HPP
#define HASHDB_SERVER_HPP

#include "goldilocks_base_field.hpp"
#include "database.hpp"
#include <grpcpp/grpcpp.h>

using grpc::Server;

class HashDBServer
{
private:    
    Goldilocks &fr;
    Config &config;
    pthread_t t;
    std::unique_ptr<Server> server;

public:
    HashDBServer (Goldilocks &fr, Config &config) : fr(fr), config(config) {}; 
    void run (void);
    void runThread (void);
    void waitForThread (void);
};

void* hashDBServerThread(void* arg);

#endif