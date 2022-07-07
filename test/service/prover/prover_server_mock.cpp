#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "config.hpp"
#include "prover_server_mock.hpp"
#include "prover_service_mock.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

void ZkServerMock::run (void)
{
    ServerBuilder builder;
    ZKProverServiceMockImpl service(fr, prover);

    std::string server_address("0.0.0.0:" + to_string(config.proverServerMockPort));

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);

    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    
    std::cout << "Server mock listening on " << server_address << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

void ZkServerMock::runThread (void)
{
    pthread_create(&t, NULL, serverMockThread, this);
}

void ZkServerMock::waitForThread (void)
{
    pthread_join(t, NULL);
}

void* serverMockThread (void* arg)
{
    ZkServerMock *pServer = (ZkServerMock *)arg;
    pServer->run();
    return NULL;
}