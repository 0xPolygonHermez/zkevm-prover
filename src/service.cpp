#include "config.hpp"
#include "service.hpp"
#include "input.hpp"
#include "proof.hpp"
#include "utils.hpp"

#include <grpcpp/grpcpp.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

::grpc::Status ZKProverServiceImpl::GetStatus(::grpc::ServerContext* context, const ::zkprover::v1::GetStatusRequest* request, ::zkprover::v1::GetStatusResponse* response)
{
    // Lock the prover
    prover.lock();

    // Set last computed request data
    response->set_last_computed_request_id(prover.lastComputedRequestId);
    response->set_last_computed_end_time(prover.lastComputedRequestEndTime);

    // If computing, set the current request data
    if (prover.pCurrentRequest != NULL)
    {
        response->set_state(zkprover::v1::GetStatusResponse_StatusProver_STATUS_PROVER_COMPUTING);
        response->set_current_computing_request_id(prover.pCurrentRequest->uuid);
        response->set_current_computing_start_time(prover.pCurrentRequest->startTime);
    }
    else
    {
        response->set_state(zkprover::v1::GetStatusResponse_StatusProver_STATUS_PROVER_IDLE);
        response->set_current_computing_request_id("");
        response->set_current_computing_start_time(0);
    }

    // Set the versions
    response->set_version_proto("v0_0_1");
    response->set_version_server("0.0.1");

    // Set the list of pending requests uuids
    for (uint64_t i=0; i<prover.pendingRequests.size(); i++)
    {
        response->add_pending_request_queue_ids(prover.pendingRequests[i]->uuid);
    }

    // Unlock the prover
    prover.unlock();

#ifdef LOG_SERVICE
    cout << "ZKProverServiceImpl::GetStatus() returns: " << response->DebugString() << endl;
#endif
    return Status::OK;
}

::grpc::Status ZKProverServiceImpl::GenProof(::grpc::ServerContext* context, const ::zkprover::v1::GenProofRequest* request, ::zkprover::v1::GenProofResponse* response)
{
    ProverRequest * pProverRequest = new ProverRequest(fr);
    if (pProverRequest == NULL)
    {
        cerr << "ZKProverServiceImpl::GenProof() failed allocation a new ProveRequest" << endl;
        exit(-1);
    }
    cout << "ZKProverServiceImpl::GenProof() created a new prover request: " << to_string((uint64_t)pProverRequest) << endl;

    // Convert inputProver into input
    inputProver2Input(fr, request->input(), pProverRequest->input);

    // Submit the prover request
    string uuid = prover.submitRequest(pProverRequest);

    // Build the response as Ok, returning the UUID assigned by the prover to this request
    response->set_result(zkprover::v1::GenProofResponse_ResultGenProof_RESULT_GEN_PROOF_OK);
    response->set_id(uuid.c_str());

#ifdef LOG_SERVICE
    cout << "ZKProverServiceImpl::GenProof() returns: " << response->DebugString() << endl;
#endif

    return Status::OK;
}

::grpc::Status ZKProverServiceImpl::Cancel(::grpc::ServerContext* context, const ::zkprover::v1::CancelRequest* request, ::zkprover::v1::CancelResponse* response)
{
    // Get the prover request UUID
    string uuid = request->id();

    // Lock the prover
    prover.lock();

    // Map uuid to the corresponding prover request
    std::map<std::string, ProverRequest *>::iterator it = prover.requestsMap.find(uuid);
    if (it == prover.requestsMap.end())
    {
        prover.unlock();
        cerr << "ZKProverServiceImpl::Cancel() unknown uuid: " << uuid << endl;
        response->set_result(zkprover::v1::CancelResponse_ResultCancel_RESULT_CANCEL_ERROR);
        return Status::OK;
    }

    // Check if it is already completed
    if (it->second->bCompleted)
    {
        prover.unlock();
        cerr << "ZKProverServiceImpl::Cancel() already completed uuid: " << uuid << endl;
        response->set_result(zkprover::v1::CancelResponse_ResultCancel_RESULT_CANCEL_ERROR);
        return Status::OK;
    }

    // Mark the request as cancelling
    it->second->bCancelling = true;

    // Unlock the prover
    prover.unlock();

    response->set_result(zkprover::v1::CancelResponse_ResultCancel_RESULT_CANCEL_OK);

#ifdef LOG_SERVICE
    cout << "ZKProverServiceImpl::Cancel() returns: " << response->DebugString() << endl;
#endif
    
    return Status::OK;
}

::grpc::Status ZKProverServiceImpl::GetProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::zkprover::v1::GetProofResponse, ::zkprover::v1::GetProofRequest>* stream)
{
#ifdef LOG_SERVICE
    cout << "ZKProverServiceImpl::GetProof() starts" << endl;
#endif
    zkprover::v1::GetProofRequest request;
    while (stream->Read(&request))
    {
#ifdef LOG_SERVICE
        cout << "ZKProverServiceImpl::GetProof() received: " << request.DebugString() << endl;
#endif
        // Get the prover request UUID from the request
        string uuid = request.id();

        zkprover::v1::GetProofResponse response;

        // Lock the prover
        prover.lock();

        // Map uuid to the corresponding prover request
        std::map<std::string, ProverRequest *>::iterator it = prover.requestsMap.find(uuid);
        if (it == prover.requestsMap.end())
        {
            prover.unlock();
            cerr << "ZKProverServiceImpl::GetProof() unknown uuid:" << uuid << endl;
            response.set_result(zkprover::v1::GetProofResponse_ResultGetProof_RESULT_GET_PROOF_ERROR);
            return Status::OK;
        }
        ProverRequest * pProverRequest = it->second;

        // Check if it is already completed
        if (!pProverRequest->bCompleted)
        {
            prover.unlock();
            cerr << "ZKProverServiceImpl::GetProof() not completed uuid=" << uuid << endl;
            response.set_result(zkprover::v1::GetProofResponse_ResultGetProof_RESULT_GET_PROOF_PENDING);
            return Status::OK;
        }

        // Request is completed
        response.set_id(uuid);
        response.set_result(zkprover::v1::GetProofResponse_ResultGetProof_RESULT_GET_PROOF_COMPLETED_OK);
        response.set_result_string("completed");

        // Convert the returned Proof to zkprover::Proof
        zkprover::v1::Proof * pProofProver = new zkprover::v1::Proof();
        proof2ProofProver(fr, pProverRequest->proof, *pProofProver);
        response.set_allocated_proof(pProofProver);

        // Set public inputs extended
        zkprover::v1::PublicInputsExtended* pPublicInputsExtended = new(zkprover::v1::PublicInputsExtended);
        pPublicInputsExtended->set_input_hash(pProverRequest->proof.publicInputsExtended.inputHash);
        zkprover::v1::PublicInputs* pPublicInputs = new(zkprover::v1::PublicInputs);
        pPublicInputs->set_old_state_root(pProverRequest->proof.publicInputsExtended.publicInputs.oldStateRoot);
        pPublicInputs->set_old_local_exit_root(pProverRequest->proof.publicInputsExtended.publicInputs.oldLocalExitRoot);
        pPublicInputs->set_new_state_root(pProverRequest->proof.publicInputsExtended.publicInputs.newStateRoot);
        pPublicInputs->set_new_local_exit_root(pProverRequest->proof.publicInputsExtended.publicInputs.newLocalExitRoot);
        pPublicInputs->set_sequencer_addr(pProverRequest->proof.publicInputsExtended.publicInputs.sequencerAddr);
        pPublicInputs->set_batch_hash_data(pProverRequest->proof.publicInputsExtended.publicInputs.batchHashData);
        pPublicInputs->set_chain_id(pProverRequest->proof.publicInputsExtended.publicInputs.chainId);
        pPublicInputs->set_batch_num(pProverRequest->proof.publicInputsExtended.publicInputs.batchNum);
        pPublicInputs->set_block_num(pProverRequest->proof.publicInputsExtended.publicInputs.blockNum);
        pPublicInputs->set_eth_timestamp(pProverRequest->proof.publicInputsExtended.publicInputs.timestamp);
        pPublicInputsExtended->set_allocated_public_inputs(pPublicInputs);
        response.set_allocated_public_(pPublicInputsExtended);
        
        prover.unlock();

#ifdef LOG_SERVICE
        cout << "ZKProverServiceImpl::GetProof() sends: " << response.DebugString() << endl;
#endif
        // Return the response via the stream
        stream->Write(response);
    }

#ifdef LOG_SERVICE
    cout << "ZKProverServiceImpl::GetProof() done" << endl;
#endif

    return Status::OK;
}

::grpc::Status ZKProverServiceImpl::Execute(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::zkprover::v1::ExecuteResponse, ::zkprover::v1::ExecuteRequest>* stream)
{
#ifdef LOG_SERVICE
    cout << "ZKProverServiceImpl::Execute() starts" << endl;
#endif
    zkprover::v1::ExecuteRequest request;
    while ( stream->Read(&request) )
    {
        ProverRequest proverRequest(fr);
#ifdef LOG_SERVICE
        cout << "ZKProverServiceImpl::Execute() called with input: " << request.DebugString() << endl;
#endif
        // Convert inputProver into input
        inputProver2Input(fr, request.input(), proverRequest.input);

        // Call the prover execute method
        prover.execute(&proverRequest);

        // Prepare the ResExecute response
        zkprover::v1::ExecuteResponse response;

        // Set the counters
        zkprover::v1::ZkCounters * pCounters = new zkprover::v1::ZkCounters();
        pCounters->set_ecrecover(proverRequest.counters.ecRecover);
        pCounters->set_hash_poseidon(proverRequest.counters.hashPoseidon);
        pCounters->set_hash_keccak(proverRequest.counters.hashKeccak);
        pCounters->set_arith(proverRequest.counters.arith);
        response.set_allocated_counters(pCounters);

        // Set the receipts
        for (uint64_t i=0; i<proverRequest.receipts.size(); i++)
        {
            response.add_receipts(proverRequest.receipts[i]);
        }

        // Set the logs
        for (uint64_t i=0; i<proverRequest.logs.size(); i++)
        {
            response.add_logs(proverRequest.logs[i]);
        }

        // Set the different keys values
        map< string, vector<Goldilocks::Element>>::const_iterator it;
        for (it=proverRequest.db.dbNew.begin(); it!=proverRequest.db.dbNew.end(); it++)
        {
            string key;
            key = NormalizeToNFormat(it->first, 64);
            string value;
            for (uint64_t i=0; i<it->second.size(); i++)
            {
                value += NormalizeToNFormat(fr.toString(it->second[i], 16), 64);
            }
            (*(response.mutable_diff_keys_values()))[key] = value;
        }

        // Set the new state root
        response.set_new_state_root(proverRequest.input.publicInputs.newStateRoot);

#ifdef LOG_SERVICE
        cout << "ZKProverServiceImpl::Execute() sends: " << response.DebugString() << endl;
#endif
        // Return the response via the stream
        stream->Write(response);
    }

#ifdef LOG_SERVICE
    cout << "ZKProverServiceImpl::Execute() done" << endl;
#endif

    return Status::OK;
}

::grpc::Status ZKProverServiceImpl::SynchronizeBatchProposal(::grpc::ServerContext* context, const ::zkprover::v1::SynchronizeBatchProposalRequest* request, ::zkprover::v1::SynchronizeBatchProposalResponse* response)
{
#ifdef LOG_SERVICE
    cout << "ZKProverServiceImpl::SynchronizeBatchProposal() called with request: " << request->DebugString() << endl;
#endif

    //zkprover::v1::Receipt * pReceipt = response->add_receipts();
    //zkprover::v1::Log * pLog = pReceipt->add_logs();

#ifdef LOG_SERVICE
    cout << "ZKProverServiceImpl::SynchronizeBatchProposal() returns: " << response->DebugString() << endl;
#endif
    
    return Status::OK;
}