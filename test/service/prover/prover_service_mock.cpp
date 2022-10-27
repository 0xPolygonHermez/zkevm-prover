#include "config.hpp"
#include "prover_service_mock.hpp"
#include "input.hpp"
#include "proof.hpp"
#include "utils.hpp"

#include <grpcpp/grpcpp.h>
#include "timer.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

struct timeval lastGenProof = {0, 0};
string lastUUID;

::grpc::Status ZKProverServiceMockImpl::GetStatus(::grpc::ServerContext* context, const ::zkprover::v1::GetStatusRequest* request, ::zkprover::v1::GetStatusResponse* response)
{
    bool bComputing = (TimeDiff(lastGenProof) < prover.config.proverServerMockTimeout);

    // Set last computed request data
    response->set_last_computed_request_id(bComputing ? getUUID() : lastUUID);
    response->set_last_computed_end_time(time(NULL));

    // If computing, set the current request data
    response->set_state(bComputing ? zkprover::v1::GetStatusResponse_StatusProver_STATUS_PROVER_COMPUTING : zkprover::v1::GetStatusResponse_StatusProver_STATUS_PROVER_IDLE);
    response->set_current_computing_request_id(bComputing ? lastUUID : "");
    response->set_current_computing_start_time(time(NULL));

    // Set the versions
    response->set_version_proto("v0_0_1");
    response->set_version_server("0.0.1");

    // Set the list of pending requests uuids
    response->add_pending_request_queue_ids(getUUID());
    response->add_pending_request_queue_ids(getUUID());
    response->add_pending_request_queue_ids(getUUID());

#ifdef LOG_SERVICE
    cout << "ZKProverServiceMockImpl::GetStatus() returns: " << response->DebugString() << endl;
#endif
    return Status::OK;
}

::grpc::Status ZKProverServiceMockImpl::GenProof(::grpc::ServerContext* context, const ::zkprover::v1::GenProofRequest* request, ::zkprover::v1::GenProofResponse* response)
{
#ifdef LOG_SERVICE
    cout << "ZKProverServiceMockImpl::GenProof() called with request: " << request->DebugString() << endl;
#endif
    // Build the response as Ok, returning the UUID assigned by the prover to this request
    response->set_result(zkprover::v1::GenProofResponse_ResultGenProof_RESULT_GEN_PROOF_OK);
    lastUUID = getUUID();
    response->set_id(lastUUID);
    gettimeofday(&lastGenProof,NULL);

#ifdef LOG_SERVICE
    cout << "ZKProverServiceMockImpl::GenProof() returns: " << response->DebugString() << endl;
#endif

    return Status::OK;
}

::grpc::Status ZKProverServiceMockImpl::Cancel(::grpc::ServerContext* context, const ::zkprover::v1::CancelRequest* request, ::zkprover::v1::CancelResponse* response)
{
#ifdef LOG_SERVICE
    cout << "ZKProverServiceMockImpl::Cancel() called with request: " << request->DebugString() << endl;
#endif

    bool bComputing = (TimeDiff(lastGenProof) < prover.config.proverServerMockTimeout);

    if (bComputing && (request->id() == lastUUID ))
    {
        response->set_result(zkprover::v1::CancelResponse_ResultCancel_RESULT_CANCEL_OK);
        lastGenProof = {0,0};
    }
    else
    {
        response->set_result(zkprover::v1::CancelResponse_ResultCancel_RESULT_CANCEL_ERROR);
    }

#ifdef LOG_SERVICE
    cout << "ZKProverServiceMockImpl::Cancel() returns: " << response->DebugString() << endl;
#endif
    
    return Status::OK;
}

::grpc::Status ZKProverServiceMockImpl::GetProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::zkprover::v1::GetProofResponse, ::zkprover::v1::GetProofRequest>* stream)
{
#ifdef LOG_SERVICE
    cout << "ZKProverServiceMockImpl::GetProof() starts" << endl;
#endif
    zkprover::v1::GetProofRequest request;
    while (stream->Read(&request))
    {
#ifdef LOG_SERVICE
        cout << "ZKProverServiceMockImpl::GetProof() received: " << request.DebugString() << endl;
#endif
        bool bComputing = (TimeDiff(lastGenProof) < prover.config.proverServerMockTimeout);

        // Get the prover request UUID from the request
        string uuid = request.id();

        zkprover::v1::GetProofResponse response;

        if (!bComputing && (uuid == lastUUID))
        {
            // Request is completed
            response.set_id(uuid);
            response.set_result(zkprover::v1::GetProofResponse_ResultGetProof_RESULT_GET_PROOF_COMPLETED_OK);
            response.set_result_string("completed");

            // Convert the returned Proof to zkprover::Proof
            zkprover::v1::Proof * pProofProver = new zkprover::v1::Proof();
            pProofProver->add_proof_a("13661670604050723159190639550237390237901487387303122609079617855313706601738");
            pProofProver->add_proof_a("318870292909531730706266902424471322193388970015138106363857068613648741679");
            pProofProver->add_proof_a("1");
            zkprover::v1::ProofB *pProofB = pProofProver->add_proof_b();
            pProofB->add_proofs("697129936138216869261087581911668981951894602632341950972818743762373194907");
            pProofB->add_proofs("8382255061406857865565510718293473646307698289010939169090474571110768554297");
            pProofB = pProofProver->add_proof_b();
            pProofB->add_proofs("15430920731683674465693779067364347784717314152940718599921771157730150217435");
            pProofB->add_proofs("9973632244944366583831174453935477607483467152902406810554814671794600888188");
            pProofB = pProofProver->add_proof_b();
            pProofB->add_proofs("1");
            pProofB->add_proofs("0");
            pProofProver->add_proof_c("19319469652444706345294120534164146052521965213898291140974711293816652378032");
            pProofProver->add_proof_c("20960565072144725955004735885836324119094967998861346319897532045008317265851");
            pProofProver->add_proof_c("1");
            response.set_allocated_proof(pProofProver);

            // Set public inputs extended
            zkprover::v1::PublicInputs* pPublicInputs = new(zkprover::v1::PublicInputs);
            pPublicInputs->set_old_state_root(string2ba("0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9"));
            pPublicInputs->set_old_acc_input_hash(string2ba("0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9"));
            pPublicInputs->set_old_batch_num(1);
            pPublicInputs->set_chain_id(1000);
            pPublicInputs->set_batch_l2_data(string2ba("0x617b3a3528F9cDd6630fd3301B9c8911F7Bf063D"));
            pPublicInputs->set_global_exit_root(string2ba("0x617b3a3528F9cDd6630fd3301B9c8911F7Bf063D"));
            pPublicInputs->set_eth_timestamp(1000000);
            pPublicInputs->set_sequencer_addr("0x617b3a3528F9cDd6630fd3301B9c8911F7Bf063D");
            pPublicInputs->set_aggregator_addr("0x617b3a3528F9cDd6630fd3301B9c8911F7Bf063D");
            zkprover::v1::PublicInputsExtended* pPublicInputsExtended = new(zkprover::v1::PublicInputsExtended);
            pPublicInputsExtended->set_allocated_public_inputs(pPublicInputs);
            pPublicInputsExtended->set_new_state_root("0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9");
            pPublicInputsExtended->set_new_acc_input_hash("0x1afd6eaf13538380d99a245c2acc4a25481b54556ae080cf07d1facc0638cd8e");
            pPublicInputsExtended->set_new_local_exit_root("0x17c04c3760510b48c6012742c540a81aba4bca2f78b9d14bfd2f123e2e53ea3e");
            pPublicInputsExtended->set_new_batch_num(2);
            response.set_allocated_public_(pPublicInputsExtended);
        }
        else if (bComputing && (uuid == lastUUID))
        {
            // Request is being computed
            response.set_id(uuid);
            response.set_result(zkprover::v1::GetProofResponse_ResultGetProof_RESULT_GET_PROOF_PENDING);
            response.set_result_string("pending");
        }
        else
        {
            // Request is being computed
            response.set_id(uuid);
            response.set_result(zkprover::v1::GetProofResponse_ResultGetProof_RESULT_GET_PROOF_ERROR);
            response.set_result_string("unknown id");
        }
        
#ifdef LOG_SERVICE
        cout << "ZKProverServiceMockImpl::GetProof() sends: " << response.DebugString() << endl;
#endif
        // Return the response via the stream
        stream->Write(response);
    }

#ifdef LOG_SERVICE
    cout << "ZKProverServiceMockImpl::GetProof() done" << endl;
#endif

    return Status::OK;
}
