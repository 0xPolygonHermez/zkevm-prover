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
    proverRequest.input.batchL2Data = request->batch_l2_data();
    proverRequest.input.publicInputs.oldStateRoot = request->old_state_root();
    proverRequest.input.globalExitRoot = request->global_exit_root();
    proverRequest.input.publicInputs.timestamp = request->eth_timestamp();
    bool update_merkle_tree = request->update_merkle_tree();
    bool generate_execute_trace = request->generate_execute_trace();
    bool generate_call_trace = request->generate_call_trace();

    proverRequest.input.publicInputs.oldLocalExitRoot = "0x0";
    proverRequest.input.publicInputs.newLocalExitRoot = "0x0";
    proverRequest.input.publicInputs.sequencerAddr = "0x0";
    
    prover.execute(&proverRequest);

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