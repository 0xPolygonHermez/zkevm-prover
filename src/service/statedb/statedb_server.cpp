#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include "statedb_server.hpp"
#include "statedb_service.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

void StateDBServer::run (void)
{
    ServerBuilder builder;

    // Limit the maximum number of threads to avoid memory starvation
    grpc::ResourceQuota rq;
    rq.SetMaxThreads(config.maxStateDBThreads);
    builder.SetResourceQuota(rq);

    StateDBServiceImpl service(fr, config, true, false);

    std::string server_address("0.0.0.0:" + to_string(config.stateDBServerPort));

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);

    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());

    std::cout << "StateDB server listening on " << server_address << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

void StateDBServer::runThread (void)
{
    pthread_create(&t, NULL, stateDBServerThread, this);
}

void StateDBServer::waitForThread (void)
{
    pthread_join(t, NULL);
}

void* stateDBServerThread (void* arg)
{
    StateDBServer *pServer = (StateDBServer *)arg;
    pServer->run();
    return NULL;
}