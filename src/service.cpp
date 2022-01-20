#include "config.hpp"
#include "service.hpp"
#include "input.hpp"
#include "proof.hpp"

#include <grpcpp/grpcpp.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

::grpc::Status ZKProverServiceImpl::GetStatus (::grpc::ServerContext* context, const ::zkprover::NoParams* request, ::zkprover::State* response)
{
    response->set_status(status);
#ifdef LOG_SERVICE
    cout << "ZKProverServiceImpl::GetStatus() returning " << status << endl;
#endif
    return Status::OK;
}

::grpc::Status ZKProverServiceImpl::GenProof (::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::zkprover::Proof, ::zkprover::InputProver>* stream)
{
#ifdef LOG_SERVICE
    cout << "ZKProverServiceImpl::GenProof() starts" << endl;
#endif
    zkprover::InputProver inputProver;
    while ( !bCancelling && stream->Read(&inputProver) )
    {
        status = zkprover::State::PENDING;

        // Convert inputProver into input
        Input input(fr);
        inputProver2Input(inputProver, input);

        // Call the prover and obtain the proof
        Proof proof;
        prover.prove(input, proof);
        
        // Convert from Proof to zkprover::Proof
        zkprover::Proof proofProver;
        proof2ProofProver(proof, proofProver);

        // Return the prover data
        stream->Write(proofProver);

        // Store a copy of the proof to return in GetProof() service call
        lastProof = proofProver;

        status = zkprover::State::FINISHED;
    }

    status = zkprover::State::IDLE;
#ifdef LOG_SERVICE
    cout << "ZKProverServiceImpl::GenProof() ends" << endl;
#endif

    return Status::OK;
}

::grpc::Status ZKProverServiceImpl::Cancel (::grpc::ServerContext* context, const ::zkprover::NoParams* request, ::zkprover::State* response)
{
#ifdef LOG_SERVICE
    cout << "ZKProverServiceImpl::Cancel()" << endl;
#endif
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
#ifdef LOG_SERVICE
    cout << "ZKProverServiceImpl::GetProof()" << endl;
#endif
    *response = lastProof;
    return Status::OK;
}

void ZKProverServiceImpl::inputProver2Input ( zkprover::InputProver &inputProver, Input &input)
{
    // Parse message
    input.message = inputProver.message();
#ifdef LOG_RPC_INPUT
    cout << "inputProver2Input() got:" << endl;
    cout << "input.message: " << input.message << endl;
#endif

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
#ifdef LOG_RPC_INPUT
    cout << "input.publicInputs.oldStateRoot: " << input.publicInputs.oldStateRoot << endl;
    cout << "input.publicInputs.oldLocalExitRoot: " << input.publicInputs.oldLocalExitRoot << endl;
    cout << "input.publicInputs.newStateRoot: " << input.publicInputs.newStateRoot << endl;
    cout << "input.publicInputs.newLocalExitRoot: " << input.publicInputs.newLocalExitRoot << endl;
    cout << "input.publicInputs.sequencerAddr: " << input.publicInputs.sequencerAddr << endl;
    cout << "input.publicInputs.batchHashData: " << input.publicInputs.batchHashData << endl;
    cout << "input.publicInputs.chainId: " << to_string(input.publicInputs.chainId) << endl;
    cout << "input.publicInputs.batchNum: " << to_string(input.publicInputs.batchNum) << endl;
#endif

    // Parse global exit root
    input.globalExitRoot = inputProver.globalexitroot();
#ifdef LOG_RPC_INPUT
    cout << "input.globalExitRoot: " << input.globalExitRoot << endl;
#endif

    // Parse transactions list
    for (int i=0; i<inputProver.txs_size(); i++)
    {
        input.txStrings.push_back(inputProver.txs(i));
#ifdef LOG_RPC_INPUT
        cout << "input.txStrings[" << to_string(i) << "]: " << input.txStrings[i] << endl;
#endif
    }

    // Parse keys map
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> > keys;
    keys = inputProver.keys();
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> >::iterator it;
    for (it=keys.begin(); it!=keys.end(); it++)
    {
        input.keys[it->first] = it->second;
#ifdef LOG_RPC_INPUT
        cout << "input.keys[" << it->first << "]: " << input.keys[it->first] << endl;
#endif
    }
}

void ZKProverServiceImpl::proof2ProofProver (Proof &proof, zkprover::Proof &proofProver)
{
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
}