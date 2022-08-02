#include "config.hpp"
#include "executor_service.hpp"
#include "input.hpp"
#include "proof.hpp"
#include "service/prover/prover_utils.hpp"
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

    // Create and init an instance of ProverRequest
    ProverRequest proverRequest(fr);
    proverRequest.init(config);

    // Get batchNum
    proverRequest.input.publicInputs.batchNum = request->batch_num();

    // Get sequencerAddr
    proverRequest.input.publicInputs.sequencerAddr = Add0xIfMissing(request->coinbase());
    if (proverRequest.input.publicInputs.sequencerAddr.size() > (2 + 40))
    {
        cerr << "Error: ExecutorServiceImpl::ProcessBatch() got sequencer address too long, size=" << proverRequest.input.publicInputs.sequencerAddr.size() << endl;
        return Status::CANCELLED;
    }
    cout << "ExecutorServiceImpl::ProcessBatch() got sequencerAddr=" << proverRequest.input.publicInputs.sequencerAddr << endl;

    // Get batchL2Data
    proverRequest.input.batchL2Data = "0x" + ba2string(request->batch_l2_data());
    cout << "ExecutorServiceImpl::ProcessBatch() got batchL2Data=" << proverRequest.input.batchL2Data << endl;

    // Get oldStateRoot
    proverRequest.input.publicInputs.oldStateRoot = "0x" + ba2string(request->old_state_root());
    if (proverRequest.input.publicInputs.oldStateRoot.size() > (2 + 64))
    {
        cerr << "Error: ExecutorServiceImpl::ProcessBatch() got oldStateRoot too long, size=" << proverRequest.input.publicInputs.oldStateRoot.size() << endl;
        return Status::CANCELLED;
    }
    cout << "ExecutorServiceImpl::ProcessBatch() got oldStateRoot=" << proverRequest.input.publicInputs.oldStateRoot << endl;

    // Get oldLocalExitRoot
    proverRequest.input.publicInputs.oldLocalExitRoot = "0x" + ba2string(request->old_local_exit_root());
    if (proverRequest.input.publicInputs.oldLocalExitRoot.size() > (2 + 64))
    {
        cerr << "Error: ExecutorServiceImpl::ProcessBatch() got oldLocalExitRoot too long, size=" << proverRequest.input.publicInputs.oldLocalExitRoot.size() << endl;
        return Status::CANCELLED;
    }
    cout << "ExecutorServiceImpl::ProcessBatch() got oldLocalExitRoot=" << proverRequest.input.publicInputs.oldLocalExitRoot << endl;

    // Get globalExitRoot
    proverRequest.input.globalExitRoot = "0x" + ba2string(request->global_exit_root());
    if (proverRequest.input.globalExitRoot.size() > (2 + 64))
    {
        cerr << "Error: ExecutorServiceImpl::ProcessBatch() got globalExitRoot too long, size=" << proverRequest.input.globalExitRoot.size() << endl;
        return Status::CANCELLED;
    }
    cout << "ExecutorServiceImpl::ProcessBatch() got globalExitRoot=" << proverRequest.input.globalExitRoot << endl;

    // Get timestamp
    proverRequest.input.publicInputs.timestamp = request->eth_timestamp();
    cout << "ExecutorServiceImpl::ProcessBatch() got timestamp=" << proverRequest.input.publicInputs.timestamp << endl;

    // Get from
    proverRequest.input.from = Add0xIfMissing(request->from());
    if (proverRequest.input.from.size() > (2 + 40))
    {
        cerr << "Error: ExecutorServiceImpl::ProcessBatch() got from too long, size=" << proverRequest.input.from.size() << endl;
        return Status::CANCELLED;
    }
    cout << "ExecutorServiceImpl::ProcessBatch() got from=" << proverRequest.input.from << endl;

    // Flags
    proverRequest.bProcessBatch = true;
    proverRequest.bUpdateMerkleTree = request->update_merkle_tree();
    proverRequest.txHashToGenerateExecuteTrace = "0x" + ba2string(request->tx_hash_to_generate_execute_trace());
    proverRequest.txHashToGenerateCallTrace = "0x" + ba2string(request->tx_hash_to_generate_call_trace());

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
            return Status::CANCELLED;
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

    // Parse contracts data
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> > contractsBytecode;
    contractsBytecode = request->contracts_bytecode();
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> >::iterator itp;
    for (itp=contractsBytecode.begin(); itp!=contractsBytecode.end(); itp++)
    {
        vector<uint8_t> dbValue;
        string contractValue = string2ba(itp->second);
        for (uint64_t i=0; i<contractValue.size(); i++)
        {
            dbValue.push_back(contractValue.at(i));
        }
        proverRequest.input.contractsBytecode[itp->first] = dbValue;
#ifdef LOG_RPC_INPUT
        //cout << "proverRequest.input.contractsBytecode[" << itp->first << "]: " << itp->second << endl;
#endif
    }     

    // Preprocess the transactions
    proverRequest.input.preprocessTxs();

    prover.processBatch(&proverRequest);

    if (proverRequest.result != ZKR_SUCCESS)
    {
        cerr << "Error: ExecutorServiceImpl::ProcessBatch() detected proverRequest.result=" << proverRequest.result << "=" << zkresult2string(proverRequest.result) << endl;
        return Status::CANCELLED;
    }
    
    response->set_cumulative_gas_used(proverRequest.fullTracer.finalTrace.cumulative_gas_used);
    response->set_cnt_keccak_hashes(proverRequest.counters.keccakF);
    response->set_cnt_poseidon_hashes(proverRequest.counters.poseidonG);
    response->set_cnt_poseidon_paddings(proverRequest.counters.paddingPG);
    response->set_cnt_mem_aligns(proverRequest.counters.memAlign);
    response->set_cnt_arithmetics(proverRequest.counters.arith);
    response->set_cnt_binaries(proverRequest.counters.binary);
    response->set_cnt_steps(proverRequest.counters.steps);
    response->set_new_state_root(string2ba(proverRequest.fullTracer.finalTrace.new_state_root));
    response->set_new_local_exit_root(string2ba(proverRequest.fullTracer.finalTrace.new_local_exit_root));
    vector<Response> &responses(proverRequest.fullTracer.finalTrace.responses);
    for (uint64_t tx=0; tx<responses.size(); tx++)
    {
        executor::v1::ProcessTransactionResponse * pProcessTransactionResponse = response->add_responses();
        pProcessTransactionResponse->set_tx_hash(string2ba(responses[tx].tx_hash));
        pProcessTransactionResponse->set_type(responses[tx].type); // Type indicates legacy transaction; it will be always 0 (legacy) in the executor
        pProcessTransactionResponse->set_return_value(string2ba(responses[tx].return_value)); // Returned data from the runtime (function result or data supplied with revert opcode)
        pProcessTransactionResponse->set_gas_left(responses[tx].gas_left); // Total gas left as result of execution
        pProcessTransactionResponse->set_gas_used(responses[tx].gas_used); // Total gas used as result of execution or gas estimation
        pProcessTransactionResponse->set_gas_refunded(responses[tx].gas_refunded); // Total gas refunded as result of execution
        pProcessTransactionResponse->set_error(string2error(responses[tx].error)); // Any error encountered during the execution
        pProcessTransactionResponse->set_create_address(responses[tx].create_address); // New SC Address in case of SC creation
        pProcessTransactionResponse->set_state_root(string2ba(responses[tx].state_root));
        pProcessTransactionResponse->set_unprocessed_transaction(responses[tx].unprocessed_transaction); // Indicates if this tx didn't fit into the batch
        for (uint64_t log=0; log<responses[tx].logs.size(); log++)
        {
            executor::v1::Log * pLog = pProcessTransactionResponse->add_logs();
            pLog->set_address(responses[tx].logs[log].address); // Address of the contract that generated the event
            for (uint64_t topic=0; topic<responses[tx].logs[log].topics.size(); topic++)
            {
                std::string * pTopic = pLog->add_topics();
                *pTopic = string2ba(responses[tx].logs[log].topics[topic]); // List of topics provided by the contract
            }
            // data is a vector of strings :(
            //pLog->set_data(string2ba(responses[tx].logs[log].data[0])); // Supplied by the contract, usually ABI-encoded // TODO: Replace by real data
            pLog->set_batch_number(responses[tx].logs[log].batch_number); // Batch in which the transaction was included
            pLog->set_tx_hash(string2ba(responses[tx].logs[log].tx_hash)); // Hash of the transaction
            pLog->set_tx_index(responses[tx].logs[log].tx_index); // Index of the transaction in the block
            pLog->set_batch_hash(string2ba(responses[tx].logs[log].batch_hash)); // Hash of the batch in which the transaction was included
            pLog->set_index(responses[tx].logs[log].index); // Index of the log in the block
        }
        if (proverRequest.txHashToGenerateExecuteTrace == responses[tx].tx_hash)
        {
            for (uint64_t trace=0; trace<responses[tx].call_trace.steps.size(); trace++)
            {
                executor::v1::ExecutionTraceStep * pExecutionTraceStep = pProcessTransactionResponse->add_execution_trace();
                pExecutionTraceStep->set_pc(responses[tx].call_trace.steps[trace].pc); // Program Counter
                pExecutionTraceStep->set_op(responses[tx].call_trace.steps[trace].opcode); // OpCode
                pExecutionTraceStep->set_remaining_gas(0);
                pExecutionTraceStep->set_gas_cost(responses[tx].call_trace.steps[trace].gasCost); // Gas cost of the operation
                pExecutionTraceStep->set_memory(string2ba(responses[tx].call_trace.steps[trace].memory)); // Content of memory
                pExecutionTraceStep->set_memory_size(responses[tx].call_trace.steps[trace].memory_size);
                for (uint64_t stack=0; stack<responses[tx].call_trace.steps[trace].stack.size() ; stack++)
                    pExecutionTraceStep->add_stack(responses[tx].call_trace.steps[trace].stack[stack]); // Content of the stack
                //pExecutionTraceStep->set_return_data(string2ba(responses[tx].call_trace.steps[trace].return_data[0])); TODO
                google::protobuf::Map<std::string, std::string>  * pStorage = pExecutionTraceStep->mutable_storage();
                map<string,string>::iterator it;
                for (it=responses[tx].call_trace.steps[trace].storage.begin(); it!=responses[tx].call_trace.steps[trace].storage.end(); it++)
                    (*pStorage)[it->first] = it->second; // Content of the storage
                pExecutionTraceStep->set_depth(responses[tx].call_trace.steps[trace].depth); // Call depth
                pExecutionTraceStep->set_gas_refund(responses[tx].call_trace.steps[trace].refund);
                pExecutionTraceStep->set_error(string2error(responses[tx].call_trace.steps[trace].error));
            }
        }
        if (proverRequest.txHashToGenerateCallTrace == responses[tx].tx_hash)
        {
            executor::v1::CallTrace * pCallTrace = new executor::v1::CallTrace();
            executor::v1::TransactionContext * pTransactionContext = pCallTrace->mutable_context();
            pTransactionContext->set_type(responses[tx].call_trace.context.type); // "CALL" or "CREATE"
            pTransactionContext->set_from(responses[tx].call_trace.context.from); // Sender of the transaction
            pTransactionContext->set_to(responses[tx].call_trace.context.to); // Target of the transaction
            pTransactionContext->set_data(string2ba(responses[tx].call_trace.context.data)); // Input data of the transaction
            pTransactionContext->set_gas(responses[tx].call_trace.context.gas);
            pTransactionContext->set_gas_price(responses[tx].call_trace.context.gasPrice);
            pTransactionContext->set_value(responses[tx].call_trace.context.value);
            pTransactionContext->set_batch(string2ba(responses[tx].call_trace.context.batch)); // Hash of the batch in which the transaction was included
            pTransactionContext->set_output(string2ba(responses[tx].call_trace.context.output)); // Returned data from the runtime (function result or data supplied with revert opcode)
            pTransactionContext->set_gas_used(responses[tx].call_trace.context.gas_used); // Total gas used as result of execution
            pTransactionContext->set_execution_time(responses[tx].call_trace.context.execution_time);
            pTransactionContext->set_old_state_root(string2ba(responses[tx].call_trace.context.old_state_root)); // Starting state root
            for (uint64_t step=0; step<responses[tx].call_trace.steps.size(); step++)
            {
                executor::v1::TransactionStep * pTransactionStep = pCallTrace->add_steps();
                pTransactionStep->set_state_root(string2ba(responses[tx].call_trace.steps[step].state_root));
                pTransactionStep->set_depth(responses[tx].call_trace.steps[step].depth); // Call depth
                pTransactionStep->set_pc(responses[tx].call_trace.steps[step].pc); // Program counter
                pTransactionStep->set_gas(responses[tx].call_trace.steps[step].remaining_gas); // Remaining gas
                pTransactionStep->set_gas_cost(responses[tx].call_trace.steps[step].gasCost); // Gas cost of the operation
                pTransactionStep->set_gas_refund(responses[tx].call_trace.steps[step].refund); // Gas refunded during the operation
                pTransactionStep->set_op(responses[tx].call_trace.steps[step].op); // Opcode
                //for (uint64_t stack=0; stack<3; stack++)
                //    pTransactionStep->add_stack(0); // Content of the stack   TODO
                pTransactionStep->set_memory(string2ba(responses[tx].call_trace.steps[step].memory)); // Content of the memory
                //pTransactionStep->set_return_data(string2ba(responses[tx].call_trace.steps[step].return_data[0])); // TODO
                executor::v1::Contract * pContract = pTransactionStep->mutable_contract(); // Contract information
                pContract->set_address(responses[tx].call_trace.steps[step].contract.address);
                pContract->set_caller(responses[tx].call_trace.steps[step].contract.caller);
                pContract->set_value(responses[tx].call_trace.steps[step].contract.value);
                pContract->set_data(string2ba(responses[tx].call_trace.steps[step].contract.data));
                pContract->set_gas(responses[tx].call_trace.steps[step].contract.gas);
                pTransactionStep->set_error(string2error(responses[tx].call_trace.steps[step].error));
            }
            pProcessTransactionResponse->set_allocated_call_trace(pCallTrace);
        }
    }


#ifdef LOG_SERVICE
    cout << "ExecutorServiceImpl::ProcessBatch() returns:\n" << response->DebugString() << endl;
#endif

    return Status::OK;
}

::executor::v1::Error ExecutorServiceImpl::string2error (string &errorString)
{
    if (errorString == "OOG") return ::executor::v1::ERROR_OUT_OF_GAS;
    if (errorString == "revert") return ::executor::v1::ERROR_EXECUTION_REVERTED;
    if (errorString == "invalid") return ::executor::v1::ERROR_INVALID_TX;
    if (errorString == "overflow") return ::executor::v1::ERROR_STACK_OVERFLOW;
    if (errorString == "underflow") return ::executor::v1::ERROR_STACK_UNDERFLOW;
    if (errorString == "OOC") return ::executor::v1::ERROR_OUT_OF_COUNTERS;
    if (errorString == "") return ::executor::v1::ERROR_UNSPECIFIED;
    cerr << "Error: ExecutorServiceImpl::string2error() found invalid error string=" << errorString << endl;
    exitProcess();
    return ::executor::v1::ERROR_UNSPECIFIED;
}