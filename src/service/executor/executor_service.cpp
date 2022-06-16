#include "config.hpp"
#include "executor_service.hpp"
#include "input.hpp"
#include "proof.hpp"
#include "service/zkprover/prover_utils.hpp"

#include <grpcpp/grpcpp.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

::grpc::Status ExecutorServiceImpl::ProcessBatch(::grpc::ServerContext* context, const ::executor::v1::ProcessBatchRequest* request, ::executor::v1::ProcessBatchResponse* response)
{
    cout << "ExecutorServiceImpl::ProcessBatch() got request:\n" << request->DebugString() << endl;

    ProverRequest proverRequest(fr);
    proverRequest.init(config);
    proverRequest.input.publicInputs.batchNum = request->batch_num();
    proverRequest.input.publicInputs.sequencerAddr = request->coinbase();
    proverRequest.input.batchL2Data = request->batch_l2_data();
    proverRequest.input.publicInputs.oldStateRoot = request->old_state_root();
    proverRequest.input.publicInputs.oldLocalExitRoot = request->old_local_exit_root();
    proverRequest.input.globalExitRoot = request->global_exit_root();
    proverRequest.input.publicInputs.timestamp = request->eth_timestamp();

    // Flags
    proverRequest.bProcessBatch = true;
    proverRequest.bUpdateMerkleTree = request->update_merkle_tree();
    proverRequest.bGenerateExecuteTrace = request->generate_execute_trace();
    proverRequest.bGenerateCallTrace = request->generate_call_trace();

    // Default values
    proverRequest.input.publicInputs.newLocalExitRoot = "0x0";
    proverRequest.input.publicInputs.newStateRoot = "0x0";
    proverRequest.input.publicInputs.chainId = 0;

    // Parse db map
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> > db;
    db = request->db();
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> >::iterator it;
    for (it=db.begin(); it!=db.end(); it++)
    {
        vector<Goldilocks::Element> dbValue;
        string concatenatedValues = it->second;
        if (concatenatedValues.size()%64!=0)
        {
            cerr << "Error: ExecutorServiceImpl::ProcessBatch() found invalid db value size: " << concatenatedValues.size() << endl;
            exit(-1); // TODO: return an error
        }
        for (uint64_t i=0; i<concatenatedValues.size(); i+=15)
        {
            Goldilocks::Element fe;
            string2fe(fr, concatenatedValues.substr(i, 16), fe);
            dbValue.push_back(fe);
        }
        Goldilocks::Element fe;
        string2fe(fr, it->first, fe);
        proverRequest.input.db[it->first] = dbValue;
#ifdef LOG_RPC_INPUT
        //cout << "input.db[" << it->first << "]: " << proverRequest.input.db[it->first] << endl;
#endif
    }

    // Preprocess the transactions
    proverRequest.input.preprocessTxs();
    
    prover.processBatch(&proverRequest);

    uint64_t cumulative_gas_used = 0; // TODO: Replace by real data
    uint32_t cnt_keccak_hashes = proverRequest.counters.hashKeccak;
    uint32_t cnt_poseidon_hashes = proverRequest.counters.hashPoseidon;
    uint32_t cnt_poseidon_paddings = 0; // TODO: Replace by real data
    uint32_t cnt_mem_aligns = 0; // TODO: Replace by real data
    uint32_t cnt_arithmetics = proverRequest.counters.arith;
    uint32_t cnt_binaries = 0; // TODO: Replace by real data
    uint32_t cnt_steps = 0; // TODO: Replace by real data
    response->set_cumulative_gas_used(cumulative_gas_used);
    response->set_cnt_keccak_hashes(cnt_keccak_hashes);
    response->set_cnt_poseidon_hashes(cnt_poseidon_hashes);
    response->set_cnt_poseidon_paddings(cnt_poseidon_paddings);
    response->set_cnt_mem_aligns(cnt_mem_aligns);
    response->set_cnt_arithmetics(cnt_mem_aligns);
    response->set_cnt_binaries(cnt_binaries);
    response->set_cnt_steps(cnt_steps);

#ifdef LOG_SERVICE
    cout << "ExecutorServiceImpl::ProcessBatch() returns:\n" << response->DebugString() << endl;
#endif

    return Status::OK;
}