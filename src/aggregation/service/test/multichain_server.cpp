#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "config.hpp"
#include "multichain_server.hpp"
#include "multichain_service.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

void MultichainServer::run (void)
{
    ServerBuilder builder;
    
    // Limit the maximum number of threads to avoid memory starvation
    grpc::ResourceQuota rq;
    rq.SetMaxThreads(4);
    builder.SetResourceQuota(rq);

    MultichainServiceImpl service(fr, config);

    std::string server_address("0.0.0.0:" + to_string(config.multichainServerPort));

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);

    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    
    std::cout << "Multichain server listening on " << server_address << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

void MultichainServer::runThread (void)
{
    pthread_create(&t, NULL, multichainServerThread, this);
}

void MultichainServer::waitForThread (void)
{
    pthread_join(t, NULL);
}

void* multichainServerThread (void* arg)
{
    MultichainServer *pServer = (MultichainServer *)arg;
    pServer->run();
    return NULL;
}