#include "config.hpp"
#include "executor_service.hpp"
#include "input.hpp"
#include "proof.hpp"
#include "service/zkprover/prover_utils.hpp"
#include "full_tracer.hpp"

#include <grpcpp/grpcpp.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

::grpc::Status ExecutorServiceImpl::ProcessBatch(::grpc::ServerContext* context, const ::executor::v1::ProcessBatchRequest* request, ::executor::v1::ProcessBatchResponse* response)
{
#ifdef LOG_SERVICE
    cout << "ExecutorServiceImpl::ProcessBatch() got request:\n" << request->DebugString() << endl;
#endif

    ProverRequest proverRequest(fr);
    proverRequest.init(config);
    proverRequest.input.publicInputs.batchNum = request->batch_num();
    proverRequest.input.publicInputs.sequencerAddr = request->coinbase();
    proverRequest.input.batchL2Data = "0x" + ba2string(request->batch_l2_data());
    cout << "ExecutorServiceImpl::ProcessBatch() got batchL2Data=" << proverRequest.input.batchL2Data << endl;
    proverRequest.input.publicInputs.oldStateRoot = "0x" + ba2string(request->old_state_root());
    cout << "ExecutorServiceImpl::ProcessBatch() got oldStateRoot=" << proverRequest.input.publicInputs.oldStateRoot << endl;
    proverRequest.input.publicInputs.oldLocalExitRoot = "0x" + ba2string(request->old_local_exit_root());
    cout << "ExecutorServiceImpl::ProcessBatch() got oldLocalExitRoot=" << proverRequest.input.publicInputs.oldLocalExitRoot << endl;
    proverRequest.input.globalExitRoot = "0x" + ba2string(request->global_exit_root());
    cout << "ExecutorServiceImpl::ProcessBatch() got globalExitRoot=" << proverRequest.input.globalExitRoot << endl;
    //string aux = request->global_exit_root();
    proverRequest.input.publicInputs.timestamp = request->eth_timestamp();

    // Flags
    proverRequest.bProcessBatch = true;
    proverRequest.bUpdateMerkleTree = request->update_merkle_tree();
    proverRequest.bGenerateExecuteTrace = request->generate_execute_trace();
    proverRequest.bGenerateCallTrace = request->generate_call_trace();

    // Default values
    proverRequest.input.publicInputs.newLocalExitRoot = "0x0";
    proverRequest.input.publicInputs.newStateRoot = "0x0";

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
        for (uint64_t i=0; i<concatenatedValues.size(); i+=16)
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
    
    response->set_cumulative_gas_used(proverRequest.fullTracer.finalTrace.cumulative_gas_used);
    response->set_cnt_keccak_hashes(proverRequest.counters.keccakF);
    response->set_cnt_poseidon_hashes(proverRequest.counters.poseidonG);
    response->set_cnt_poseidon_paddings(proverRequest.counters.paddingPG);
    response->set_cnt_mem_aligns(proverRequest.counters.memAlign);
    response->set_cnt_arithmetics(proverRequest.counters.arith);
    response->set_cnt_binaries(proverRequest.counters.binary);
    response->set_cnt_steps(proverRequest.counters.steps);
    response->set_new_state_root(proverRequest.fullTracer.finalTrace.new_state_root);
    response->set_new_local_exit_root(proverRequest.fullTracer.finalTrace.new_local_exit_root);
    vector<Response> &responses(proverRequest.fullTracer.finalTrace.responses);
    for (uint64_t tx=0; tx<responses.size(); tx++)
    {
        executor::v1::ProcessTransactionResponse * pProcessTransactionResponse = response->add_responses();
        pProcessTransactionResponse->set_tx_hash(responses[tx].tx_hash);
        pProcessTransactionResponse->set_type(responses[tx].type); // Type indicates legacy transaction; it will be always 0 (legacy) in the executor
        pProcessTransactionResponse->set_return_value(responses[tx].return_value); // Returned data from the runtime (function result or data supplied with revert opcode)
        pProcessTransactionResponse->set_gas_left(responses[tx].gas_left); // Total gas left as result of execution
        pProcessTransactionResponse->set_gas_used(responses[tx].gas_used); // Total gas used as result of execution or gas estimation
        pProcessTransactionResponse->set_gas_refunded(responses[tx].gas_refunded); // Total gas refunded as result of execution
        pProcessTransactionResponse->set_error(responses[tx].error); // Any error encountered during the execution
        pProcessTransactionResponse->set_create_address(responses[tx].create_address); // New SC Address in case of SC creation
        pProcessTransactionResponse->set_state_root(responses[tx].state_root);
        pProcessTransactionResponse->set_unprocessed_transaction(responses[tx].unprocessed_transaction); // Indicates if this tx didn't fit into the batch
        for (uint64_t log=0; log<responses[tx].call_trace.context.logs.size(); log++)
        {
            executor::v1::Log * pLog = pProcessTransactionResponse->add_logs();
            pLog->set_address(responses[tx].logs[log].address); // Address of the contract that generated the event
            for (uint64_t topic=0; topic<responses[tx].call_trace.context.logs[log].topics.size(); topic++)
            {
                std::string * pTopic = pLog->add_topics();
                *pTopic = responses[tx].call_trace.context.logs[log].topics[topic]; // List of topics provided by the contract
            }
            // data is a vector of strings :(
            pLog->set_data(responses[tx].logs[log].data[0]); // Supplied by the contract, usually ABI-encoded // TODO: Replace by real data
            pLog->set_batch_number(responses[tx].logs[log].batch_number); // Batch in which the transaction was included
            pLog->set_tx_hash(responses[tx].logs[log].tx_hash); // Hash of the transaction
            pLog->set_tx_index(responses[tx].logs[log].tx_index); // Index of the transaction in the block
            pLog->set_batch_hash(responses[tx].logs[log].batch_hash); // Hash of the batch in which the transaction was included
            pLog->set_index(responses[tx].logs[log].index); // Index of the log in the block
        }
        if (proverRequest.bGenerateExecuteTrace)
        {
            for (uint64_t trace=0; trace<responses[tx].call_trace.steps.size(); trace++)
            {
                executor::v1::ExecutionTraceStep * pExecutionTraceStep = pProcessTransactionResponse->add_execution_trace();
                pExecutionTraceStep->set_pc(responses[tx].call_trace.steps[trace].pc); // Program Counter
                pExecutionTraceStep->set_op(responses[tx].call_trace.steps[trace].opcode); // OpCode
                pExecutionTraceStep->set_remaining_gas(0);
                pExecutionTraceStep->set_gas_cost(responses[tx].call_trace.steps[trace].gasCost); // Gas cost of the operation
                pExecutionTraceStep->set_memory(""); // Content of memory
                pExecutionTraceStep->set_memory_size(0);
                for (uint64_t stack=0; stack<responses[tx].call_trace.steps[trace].stack.size() ; stack++)
                    pExecutionTraceStep->add_stack(responses[tx].call_trace.steps[trace].stack[stack]); // Content of the stack
                pExecutionTraceStep->set_return_data("");
                google::protobuf::Map<std::string, std::string>  * pStorage = pExecutionTraceStep->mutable_storage();
                map<string,string>::iterator it;
                for (it=responses[tx].call_trace.steps[trace].storage.begin(); it!=responses[tx].call_trace.steps[trace].storage.end(); it++)
                    (*pStorage)[it->first] = it->second; // Content of the storage
                pExecutionTraceStep->set_depth(responses[tx].call_trace.steps[trace].depth); // Call depth
                pExecutionTraceStep->set_gas_refund(responses[tx].call_trace.steps[trace].refund);
                pExecutionTraceStep->set_error(responses[tx].call_trace.steps[trace].error);
            }
        }
        if (proverRequest.bGenerateCallTrace)
        {
            executor::v1::CallTrace * pCallTrace = new executor::v1::CallTrace();
            executor::v1::TransactionContext * pTransactionContext = pCallTrace->mutable_context();
            pTransactionContext->set_type(responses[tx].call_trace.context.type); // "CALL" or "CREATE"
            pTransactionContext->set_from(responses[tx].call_trace.context.from); // Sender of the transaction
            pTransactionContext->set_to(responses[tx].call_trace.context.to); // Target of the transaction
            pTransactionContext->set_data(responses[tx].call_trace.context.data); // Input data of the transaction
            pTransactionContext->set_gas(responses[tx].call_trace.context.gas);
            pTransactionContext->set_gas_price(responses[tx].call_trace.context.gasPrice);
            pTransactionContext->set_value(responses[tx].call_trace.context.value);
            pTransactionContext->set_batch(responses[tx].call_trace.context.batch); // Hash of the batch in which the transaction was included
            pTransactionContext->set_output(responses[tx].call_trace.context.output); // Returned data from the runtime (function result or data supplied with revert opcode)
            pTransactionContext->set_gas_used(responses[tx].call_trace.context.gas_used); // Total gas used as result of execution
            pTransactionContext->set_execution_time(responses[tx].call_trace.context.execution_time);
            pTransactionContext->set_old_state_root(responses[tx].call_trace.context.old_state_root); // Starting state root
            for (uint64_t step=0; step<responses[tx].call_trace.steps.size(); step++)
            {
                executor::v1::TransactionStep * pTransactionStep = pCallTrace->add_steps();
                pTransactionStep->set_state_root(responses[tx].call_trace.steps[step].state_root);
                pTransactionStep->set_depth(responses[tx].call_trace.steps[step].depth); // Call depth
                pTransactionStep->set_pc(responses[tx].call_trace.steps[step].pc); // Program counter
                pTransactionStep->set_gas(responses[tx].call_trace.steps[step].remaining_gas); // Remaining gas
                pTransactionStep->set_gas_cost(responses[tx].call_trace.steps[step].gasCost); // Gas cost of the operation
                pTransactionStep->set_gas_refund(responses[tx].call_trace.steps[step].refund); // Gas refunded during the operation
                pTransactionStep->set_op(responses[tx].call_trace.steps[step].op); // Opcode
                for (uint64_t stack=0; stack<3; stack++)
                    pTransactionStep->add_stack(0); // Content of the stack
                pTransactionStep->set_memory(responses[tx].call_trace.steps[step].memory); // Content of the memory
                pTransactionStep->set_return_data(responses[tx].call_trace.steps[step].return_data[0]); // TODO
                executor::v1::Contract * pContract = pTransactionStep->mutable_contract(); // Contract information
                pContract->set_address(responses[tx].call_trace.steps[step].contract.address);
                pContract->set_caller(responses[tx].call_trace.steps[step].contract.caller);
                pContract->set_value(responses[tx].call_trace.steps[step].contract.value);
                pContract->set_data(responses[tx].call_trace.steps[step].contract.data);
                pContract->set_gas(responses[tx].call_trace.steps[step].contract.gas);
                pTransactionStep->set_error(responses[tx].call_trace.steps[step].error);
            }
            pProcessTransactionResponse->set_allocated_call_trace(pCallTrace);
        }
    }


#ifdef LOG_SERVICE
    cout << "ExecutorServiceImpl::ProcessBatch() returns:\n" << response->DebugString() << endl;
#endif

    return Status::OK;
}