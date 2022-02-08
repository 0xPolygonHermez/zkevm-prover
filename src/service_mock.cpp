#include "config.hpp"
#include "service_mock.hpp"
#include "input.hpp"
#include "proof.hpp"
#include "utils.hpp"

#include <grpcpp/grpcpp.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

::grpc::Status ZKProverServiceMockImpl::GetStatus(::grpc::ServerContext* context, const ::zkprover::NoParams* request, ::zkprover::ResGetStatus* response)
{
    // Set last computed request data
    response->set_last_computed_request_id(getUUID());
    response->set_last_computed_end_time(time(NULL));

    // If computing, set the current request data
    response->set_state(zkprover::ResGetStatus_StatusProver_COMPUTING);
    response->set_current_computing_request_id(getUUID());
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

::grpc::Status ZKProverServiceMockImpl::GenProof (::grpc::ServerContext* context, const ::zkprover::InputProver* request, ::zkprover::ResGenProof* response)
{
    // Build the response as Ok, returning the UUID assigned by the prover to this request
    response->set_result(zkprover::ResGenProof_ResultGenProof_OK);
    response->set_id(getUUID());

#ifdef LOG_SERVICE
    cout << "ZKProverServiceMockImpl::GenProof() returns: " << response->DebugString() << endl;
#endif

    return Status::OK;
}

::grpc::Status ZKProverServiceMockImpl::Cancel (::grpc::ServerContext* context, const ::zkprover::RequestId* request, ::zkprover::ResCancel* response)
{
    response->set_result(zkprover::ResCancel_ResultCancel_ERROR);

#ifdef LOG_SERVICE
    cout << "ZKProverServiceMockImpl::Cancel() returns: " << response->DebugString() << endl;
#endif
    
    return Status::OK;
}

::grpc::Status ZKProverServiceMockImpl::GetProof (::grpc::ServerContext* context, const ::zkprover::RequestId* request, ::zkprover::ResGetProof* response)
{
    // Get the prover request UUID from the request
    string uuid = request->id();

    // Request is completed
    response->set_id(uuid);
    response->set_result(zkprover::ResGetProof_ResultGetProof_COMPLETED_OK);
    response->set_result_string("completed");

    // Convert the returned Proof to zkprover::Proof
    zkprover::Proof * pProofProver = new zkprover::Proof();
    pProofProver->add_proofa("13661670604050723159190639550237390237901487387303122609079617855313706601738");
    pProofProver->add_proofa("318870292909531730706266902424471322193388970015138106363857068613648741679");
    pProofProver->add_proofa("1");
    zkprover::ProofB *pProofB = pProofProver->add_proofb();
    pProofB->add_proofs("697129936138216869261087581911668981951894602632341950972818743762373194907");
    pProofB->add_proofs("8382255061406857865565510718293473646307698289010939169090474571110768554297");
    pProofB = pProofProver->add_proofb();
    pProofB->add_proofs("15430920731683674465693779067364347784717314152940718599921771157730150217435");
    pProofB->add_proofs("9973632244944366583831174453935477607483467152902406810554814671794600888188");
    pProofB = pProofProver->add_proofb();
    pProofB->add_proofs("1");
    pProofB->add_proofs("0");
    pProofProver->add_proofc("19319469652444706345294120534164146052521965213898291140974711293816652378032");
    pProofProver->add_proofc("20960565072144725955004735885836324119094967998861346319897532045008317265851");
    pProofProver->add_proofc("1");
    response->set_allocated_proof(pProofProver);

    // Set public inputs extended
    zkprover::PublicInputsExtended* pPublicInputsExtended = new(zkprover::PublicInputsExtended);
    pPublicInputsExtended->set_input_hash("0x1afd6eaf13538380d99a245c2acc4a25481b54556ae080cf07d1facc0638cd8e");
    zkprover::PublicInputs* pPublicInputs = new(zkprover::PublicInputs);
    pPublicInputs->set_old_state_root("0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9");
    pPublicInputs->set_old_local_exit_root("0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9");
    pPublicInputs->set_new_state_root("0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9");
    pPublicInputs->set_new_local_exit_root("0x17c04c3760510b48c6012742c540a81aba4bca2f78b9d14bfd2f123e2e53ea3e");
    pPublicInputs->set_sequencer_addr("0x617b3a3528F9cDd6630fd3301B9c8911F7Bf063D");
    pPublicInputs->set_batch_hash_data("0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9");
    pPublicInputs->set_chain_id(1001);
    pPublicInputs->set_batch_num(1);
    pPublicInputsExtended->set_allocated_public_inputs(pPublicInputs);
    response->set_allocated_public_(pPublicInputsExtended);
    
    prover.unlock();

#ifdef LOG_SERVICE
    cout << "ZKProverServiceMockImpl::GetProof() returns: " << response->DebugString() << endl;
#endif

    return Status::OK;
}

::grpc::Status ZKProverServiceMockImpl::Execute (::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::zkprover::ResExecute, ::zkprover::InputProver>* stream)
{
#ifdef LOG_SERVICE
    cout << "ZKProverServiceMockImpl::Execute() starts" << endl;
#endif
    zkprover::InputProver inputProver;
    while ( stream->Read(&inputProver) )
    {
        cout << "ZKProverServiceMockImpl::Execute() called with input: " << inputProver.DebugString() << endl;

        // Prepare the ResExecute response
        zkprover::ResExecute response;

        // Set the counters
        zkprover::ZkCounters * pCounters = new zkprover::ZkCounters();
        pCounters->set_ecrecover(1);
        pCounters->set_hash_poseidon(6);
        pCounters->set_hash_keccak(3);
        pCounters->set_arith(1);
        response.set_allocated_counters(pCounters);

        // Set the receipts
        response.add_receipts("receipt");

        // Set the logs
        response.add_logs("log");

        // Set the different keys values
        string key = "27c1e4f36cbcc2ac000b94b9ede123ac0fed319fcd07f829892ba1ce73df54c8";
        string value = "000000000000000000000000000000000000000000000000000000000000000100540ae2a259cb9179561cffe6a0a3852a2c1806ad894ed396a2ef16e1f10e9c0000000000000000000000000000000000000000000000006bc75e2d631000000000000000000000000000000000000000000000000000000000000000000005000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
        (*(response.mutable_diff_keys_values()))[key] = value;

        // Set the new state root
        response.set_new_state_root("0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9");
        
        // Return the response via the stream
        stream->Write(response);
    }

#ifdef LOG_SERVICE
    cout << "ZKProverServiceMockImpl::Execute() done" << endl;
#endif

    return Status::OK;
}
