#include "config.hpp"

#ifdef RUN_GRPC_SERVER

#include "service.hpp"
#include "input.hpp"
#include "proof.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

::grpc::Status ZKProverServiceImpl::GetStatus (::grpc::ServerContext* context, const ::zkprover::NoParams* request, ::zkprover::State* response)
{
    response->set_status(status);
    return Status::OK;
}

::grpc::Status ZKProverServiceImpl::GenProof (::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::zkprover::Proof, ::zkprover::InputProver>* stream)
{
    zkprover::InputProver inputProver;
    while ( !bCancelling && stream->Read(&inputProver) ) // TODO: Should this be a loop?  Laia conversation -> they are not ending the connection, but sending several "calculate" messages
    {
        status = zkprover::State::PENDING;

        Input input;
        
        // Parse message
        input.message = inputProver.message();

        // Parse public inputs
        zkprover::PublicInputs publicInputs = inputProver.publicinputs();
        input.publicInputs.oldStateRoot = publicInputs.oldstateroot();
        input.publicInputs.oldLocalExitRoot = publicInputs.oldlocalexitroot();
        input.publicInputs.newStateRoot = publicInputs.newstateroot();
        input.publicInputs.newLocalExitRoot = publicInputs.newlocalexitroot();
        input.publicInputs.sequencerAddr = publicInputs.sequenceraddr();
        input.publicInputs.batchHashData = publicInputs.batchhashdata();
        input.publicInputs.chainId = publicInputs.chainid();
        input.publicInputs.batchNum = publicInputs.batchnum();
        
        // Parse global exit root
        input.globalExitRoot = inputProver.globalexitroot();

        // Parse transactions list
        for (int i=0; i<inputProver.txs_size(); i++)
        {
            input.txs.push_back(inputProver.txs(i));
        }

        // Parse keys map
        google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> > keys;
        keys = inputProver.keys();
        google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> >::iterator it;
        for (it=keys.begin(); it!=keys.end(); it++)
        {
            input.keys[it->first] = it->second;
        }

        // Execute and obtain the proof
        Proof proof;
        
        // Convert from Proof to zkprover::Proof
        zkprover::Proof proofProver;

        // Set proofA
        for (uint64_t i=0; i<proof.proofA.size(); i++)
        {
            proofProver.add_proofa(proof.proofA[i]);
        }

        // Set proofB
        for (uint64_t i=0; i<proof.proofB.size(); i++)
        {
            zkprover::ProofX *pProofX = proofProver.add_proofb();
            for (uint64_t j=0; j<proof.proofB[i].proof.size(); j++)
            {
                pProofX->add_proof(proof.proofB[i].proof[j]);
            }
        }

        // Set proofC
        for (uint64_t i=0; i<proof.proofC.size(); i++)
        {
            proofProver.add_proofc(proof.proofC[i]);
        }

        // Set public inputs extended
        zkprover::PublicInputsExtended* pPublicInputsExtended = new(zkprover::PublicInputsExtended);
        pPublicInputsExtended->set_inputhash(proof.publicInputsExtended.inputHash);
        zkprover::PublicInputs* pPublicInputs = new(zkprover::PublicInputs);
        pPublicInputs->set_oldstateroot(proof.publicInputsExtended.publicInputs.oldStateRoot);
        pPublicInputs->set_oldlocalexitroot(proof.publicInputsExtended.publicInputs.oldLocalExitRoot);
        pPublicInputs->set_newstateroot(proof.publicInputsExtended.publicInputs.newStateRoot);
        pPublicInputs->set_newlocalexitroot(proof.publicInputsExtended.publicInputs.newLocalExitRoot);
        pPublicInputs->set_sequenceraddr(proof.publicInputsExtended.publicInputs.sequencerAddr);
        pPublicInputs->set_batchhashdata(proof.publicInputsExtended.publicInputs.batchHashData);
        pPublicInputs->set_chainid(proof.publicInputsExtended.publicInputs.chainId);
        pPublicInputs->set_batchnum(proof.publicInputsExtended.publicInputs.batchNum);
        pPublicInputsExtended->set_allocated_publicinputs(pPublicInputs);
        proofProver.set_allocated_publicinputsextended(pPublicInputsExtended);

        // Return the prover data
        stream->Write(proofProver);

        // Store a copy of the proof to return in GetProof() service call
        lastProof = proofProver;

        status = zkprover::State::FINISHED;
    }

    status = zkprover::State::IDLE;

    return Status::OK;
}

::grpc::Status ZKProverServiceImpl::Cancel (::grpc::ServerContext* context, const ::zkprover::NoParams* request, ::zkprover::State* response)
{
    switch (status)
    {
        case zkprover::State::PENDING:
        {
            bCancelling = true;
            return Status::OK;
        }
        case zkprover::State::IDLE:
        case zkprover::State::ERROR:
        case zkprover::State::FINISHED:
        default:
        {
            break;
        }
    }
    return Status::OK;
}

::grpc::Status ZKProverServiceImpl::GetProof (::grpc::ServerContext* context, const ::zkprover::NoParams* request, ::zkprover::Proof* response)
{
    *response = lastProof;
    return Status::OK;
}

#endif