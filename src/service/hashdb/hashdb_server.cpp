#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include "hashdb_server.hpp"
#include "hashdb_service.hpp"
#include "zklog.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

void HashDBServer::run (void)
{
    ServerBuilder builder;

    // Limit the maximum number of threads to avoid memory starvation
    grpc::ResourceQuota rq;
    rq.SetMaxThreads(config.maxHashDBThreads);
    builder.SetResourceQuota(rq);

    HashDBServiceImpl service(fr, config, true, false);

    std::string server_address("0.0.0.0:" + to_string(config.hashDBServerPort));

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);

    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());

    zklog.info("HashDB server listening on " + server_address);

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

void HashDBServer::runThread (void)
{
    pthread_create(&t, NULL, hashDBServerThread, this);
}

void HashDBServer::waitForThread (void)
{
    pthread_join(t, NULL);
}

void* hashDBServerThread (void* arg)
{
    HashDBServer *pServer = (HashDBServer *)arg;
    pServer->run();
    return NULL;
}