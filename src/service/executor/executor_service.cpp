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

    /**/

    response->set_cumulative_gas_used(0); // TODO: Replace by real data
    response->set_cnt_keccak_hashes(proverRequest.counters.hashKeccak);
    response->set_cnt_poseidon_hashes(proverRequest.counters.hashPoseidon);
    response->set_cnt_poseidon_paddings(0); // TODO: Replace by real data
    response->set_cnt_mem_aligns(0); // TODO: Replace by real data
    response->set_cnt_arithmetics(proverRequest.counters.arith);
    response->set_cnt_binaries(0); // TODO: Replace by real data
    response->set_cnt_steps(0); // TODO: Replace by real data
    response->set_new_state_root(""); // TODO: Replace by real data
    response->set_new_local_exit_root(""); // TODO: Replace by real data
    for (uint64_t tx=0; tx<3; tx++)
    {
        executor::v1::ProcessTransactionResponse * pProcessTransactionResponse = response->add_responses();
        pProcessTransactionResponse->set_tx_hash(""); // TODO: Replace by real data
        pProcessTransactionResponse->set_type(0); // Type indicates legacy transaction; it will be always 0 (legacy) in the executor
        pProcessTransactionResponse->set_return_value(""); // Returned data from the runtime (function result or data supplied with revert opcode) // TODO: Replace by real data
        pProcessTransactionResponse->set_gas_left(0); // Total gas left as result of execution // TODO: Replace by real data
        pProcessTransactionResponse->set_gas_used(0); // Total gas used as result of execution or gas estimation // TODO: Replace by real data
        pProcessTransactionResponse->set_gas_refunded(0); // Total gas refunded as result of execution // TODO: Replace by real data
        pProcessTransactionResponse->set_error(""); // Any error encountered during the execution // TODO: Replace by real data
        pProcessTransactionResponse->set_create_address(""); // New SC Address in case of SC creation // TODO: Replace by real data
        pProcessTransactionResponse->set_state_root(""); // TODO: Replace by real data
        pProcessTransactionResponse->set_unprocessed_transaction(false); // Indicates if this tx didn't fit into the batch // TODO: Replace by real data
        for (uint64_t log=0; log<3; log++)
        {
            executor::v1::Log * pLog = pProcessTransactionResponse->add_logs();
            pLog->set_address(""); // Address of the contract that generated the event // TODO: Replace by real data
            for (uint64_t topic=0; topic<3; topic++)
            {
                std::string * pTopic = pLog->add_topics();
                *pTopic = ""; // List of topics provided by the contract // TODO: Replace by real data
            }
            pLog->set_data(""); // Supplied by the contract, usually ABI-encoded // TODO: Replace by real data
            pLog->set_batch_number(0); // Batch in which the transaction was included // TODO: Replace by real data
            pLog->set_tx_hash(""); // Hash of the transaction // TODO: Replace by real data
            pLog->set_tx_index(0); // Index of the transaction in the block // TODO: Replace by real data
            pLog->set_batch_hash(""); // Hash of the batch in which the transaction was included // TODO: Replace by real data
            pLog->set_index(0); // Index of the log in the block // TODO: Replace by real data
        }
        for (uint64_t trace=0; trace<3; trace++)
        {
            executor::v1::ExecutionTraceStep * pExecutionTraceStep = pProcessTransactionResponse->add_execution_trace();
            pExecutionTraceStep->set_pc(0); // Program Counter
            pExecutionTraceStep->set_op(""); // OpCode
            pExecutionTraceStep->set_remaining_gas(0);
            pExecutionTraceStep->set_gas_cost(0); // Gas cost of the operation
            pExecutionTraceStep->set_memory(""); // Content of memory
            pExecutionTraceStep->set_memory_size(0);
            for (uint64_t stack=0; stack<3; stack++)
                pExecutionTraceStep->add_stack(0); // Content of the stack
            pExecutionTraceStep->set_return_data("");
            google::protobuf::Map<std::string, std::string> * pStorage = pExecutionTraceStep->mutable_storage();
            for (uint64_t storage=0; storage<3; storage++)
                (*pStorage)[to_string(storage)] = to_string(storage); // Content of the storage
            pExecutionTraceStep->set_depth(0); // Call depth
            pExecutionTraceStep->set_gas_refund(0);
            pExecutionTraceStep->set_error("");
        }
        executor::v1::CallTrace * pCallTrace = new executor::v1::CallTrace();
        executor::v1::TransactionContext * pTransactionContext = pCallTrace->mutable_context();
        pTransactionContext->set_type(""); // CALL or CREATE
        pTransactionContext->set_from(""); // Sender of the transaction
        pTransactionContext->set_to(""); // Target of the transaction
        pTransactionContext->set_data(""); // Input data of the transaction
        pTransactionContext->set_gas(0);
        pTransactionContext->set_value(0);
        pTransactionContext->set_batch(""); // Hash of the batch in which the transaction was included
        pTransactionContext->set_output(""); // Returned data from the runtime (function result or data supplied with revert opcode)
        pTransactionContext->set_gas_used(0); // Total gas used as result of execution
        pTransactionContext->set_execution_time(0);
        pTransactionContext->set_old_state_root(""); // Starting state root
        for (uint64_t step=0; step<3; step++)
        {
            executor::v1::TransactionStep * pTransactionStep = pCallTrace->add_steps();
            pTransactionStep->set_state_root("");
            pTransactionStep->set_depth(0); // Call depth
            pTransactionStep->set_pc(0); // Program counter
            pTransactionStep->set_gas(0); // Remaining gas
            pTransactionStep->set_gas_cost(0); // Gas cost of the operation
            pTransactionStep->set_gas_refund(0); // Gas refunded during the operation
            pTransactionStep->set_op(0); // Opcode
            for (uint64_t stack=0; stack<3; stack++)
                pTransactionStep->add_stack(0); // Content of the stack
            pTransactionStep->set_memory(""); // Content of the memory
            pTransactionStep->set_return_data("");
            executor::v1::Contract * pContract = pTransactionStep->mutable_contract(); // Contract information
            pContract->set_address("");
            pContract->set_caller("");
            pContract->set_value(0);
            pContract->set_data("");
            pTransactionStep->set_error("");
        }
        pProcessTransactionResponse->set_allocated_call_trace(pCallTrace);
    }


#ifdef LOG_SERVICE
    cout << "ExecutorServiceImpl::ProcessBatch() returns:\n" << response->DebugString() << endl;
#endif

    return Status::OK;
}