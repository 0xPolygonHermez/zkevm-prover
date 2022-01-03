#ifndef ZKPROVER_SERVER_HPP
#define ZKPROVER_SERVER_HPP

/*#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include "service.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using zkprover::ZKProver;
using zkprover::State;
using zkprover::PublicInputs;
using zkprover::PublicInputsExtended;
using zkprover::InputProver;
using zkprover::Proof;
using zkprover::ProofX;
using zkprover::NoParams;
*/
class ZkServer
{    
public:
    void run (void);
};

#endif