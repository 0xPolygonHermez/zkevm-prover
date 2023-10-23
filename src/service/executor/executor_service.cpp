#include "config.hpp"
#include "executor_service.hpp"
#include "input.hpp"
#include "proof.hpp"
#include "zklog.hpp"
#include <grpcpp/grpcpp.h>
#include "exit_process.hpp"
#include "utils.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

::grpc::Status ExecutorServiceImpl::ProcessBatch(::grpc::ServerContext* context, const ::executor::v1::ProcessBatchRequest* request, ::executor::v1::ProcessBatchResponse* response)
{
    // If the process is exising, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

    //TimerStart(EXECUTOR_PROCESS_BATCH);
    struct timeval EXECUTOR_PROCESS_BATCH_start;
    gettimeofday(&EXECUTOR_PROCESS_BATCH_start,NULL);

#ifdef LOG_SERVICE
    zklog.info("ExecutorServiceImpl::ProcessBatch() got request:\n" + request->DebugString());
#endif

#ifdef LOG_TIME
    lock();
    if ( (firstTotalTime.tv_sec == 0) && (firstTotalTime.tv_usec == 0) )
    {
        gettimeofday(&firstTotalTime, NULL);
        lastTotalTime = firstTotalTime;
    }
    unlock();
#endif

    // Create and init an instance of ProverRequest
    ProverRequest proverRequest(fr, config, prt_processBatch);

    // Save request to file
    if (config.saveRequestToFile)
    {
        string2file(request->DebugString(), proverRequest.filePrefix + "executor_request.txt");
    }

    // Get external request ID
    proverRequest.contextId = request->context_id();

    // Build log tags
    LogTag logTag("context_id", proverRequest.contextId);
    proverRequest.tags.emplace_back(logTag);

    // PUBLIC INPUTS

    // Get oldStateRoot
    if (request->old_state_root().size() > 32)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatch() got oldStateRoot too long, size=" + to_string(request->old_state_root().size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_OLD_STATE_ROOT);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    ba2scalar(proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot, request->old_state_root());

    // Get oldAccInputHash
    if (request->old_acc_input_hash().size() > 32)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatch() got oldAccInputHash too long, size=" + to_string(request->old_acc_input_hash().size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_OLD_ACC_INPUT_HASH);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    ba2scalar(proverRequest.input.publicInputsExtended.publicInputs.oldAccInputHash, request->old_acc_input_hash());

    // Get batchNum
    proverRequest.input.publicInputsExtended.publicInputs.oldBatchNum = request->old_batch_num();

    // Get chain ID
    proverRequest.input.publicInputsExtended.publicInputs.chainID = request->chain_id();
    if (proverRequest.input.publicInputsExtended.publicInputs.chainID == 0)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatch() got chainID = 0", &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_CHAIN_ID);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }

    // Get fork ID
    proverRequest.input.publicInputsExtended.publicInputs.forkID = request->fork_id();

    // Create full tracer based on fork ID
    proverRequest.CreateFullTracer();
    if (proverRequest.result != ZKR_SUCCESS)
    {
        response->set_error(zkresult2error(proverRequest.result));
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }

    // Get batchL2Data
    if (request->batch_l2_data().size() > MAX_BATCH_L2_DATA_SIZE)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatch() found batchL2Data.size()=" + to_string(request->batch_l2_data().size()) + " > MAX_BATCH_L2_DATA_SIZE=" + to_string(MAX_BATCH_L2_DATA_SIZE), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_BATCH_L2_DATA);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    proverRequest.input.publicInputsExtended.publicInputs.batchL2Data = request->batch_l2_data();

    // Get globalExitRoot
    if (request->global_exit_root().size() > 32)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatch() got globalExitRoot too long, size=" + to_string(request->global_exit_root().size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_GLOBAL_EXIT_ROOT);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    ba2scalar(proverRequest.input.publicInputsExtended.publicInputs.globalExitRoot, request->global_exit_root());

    // Get timestamp
    proverRequest.input.publicInputsExtended.publicInputs.timestamp = request->eth_timestamp();

    // Get sequencer address
    string auxString = Remove0xIfPresent(request->coinbase());
    if (auxString.size() > 40)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatch() got sequencer address too long, size=" + to_string(auxString.size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_COINBASE);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    if (!stringIsHex(auxString))
    {
        zklog.error("ExecutorServiceImpl::ProcessBatch() got sequencer address not hex, coinbase=" + auxString, &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_COINBASE);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    proverRequest.input.publicInputsExtended.publicInputs.sequencerAddr.set_str(auxString, 16);

    // ROOT

    // Get from
    proverRequest.input.from = Add0xIfMissing(request->from());
    if (proverRequest.input.from.size() > (2 + 40))
    {
        zklog.error("ExecutorServiceImpl::ProcessBatch() got from too long, size=" + to_string(proverRequest.input.from.size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_FROM);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    if (!stringIs0xHex(proverRequest.input.from))
    {
        zklog.error("ExecutorServiceImpl::ProcessBatch() got from not hex, size=" + proverRequest.input.from, &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_FROM);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }

    // Flags
    proverRequest.input.bUpdateMerkleTree = request->update_merkle_tree();
    proverRequest.input.bGetKeys = request->get_keys();
    if (proverRequest.input.bGetKeys && (proverRequest.input.publicInputsExtended.publicInputs.forkID < 5))
    {
        zklog.error("ExecutorServiceImpl::ProcessBatch() got get_keys=true but fork_id=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.forkID) + "<5", &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_GET_KEY);
        TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;

    }


    // Trace config
    if (request->has_trace_config())
    {
        proverRequest.input.traceConfig.bEnabled = true;
        const executor::v1::TraceConfig & traceConfig = request->trace_config();
        if (traceConfig.disable_storage())
        {
            proverRequest.input.traceConfig.bDisableStorage = true;
        }
        if (traceConfig.disable_stack())
        {
            proverRequest.input.traceConfig.bDisableStack = true;
        }
        if (traceConfig.enable_memory())
        {
            proverRequest.input.traceConfig.bEnableMemory = true;
        }
        if (traceConfig.enable_return_data())
        {
            proverRequest.input.traceConfig.bEnableReturnData = true;
        }
        if (traceConfig.tx_hash_to_generate_execute_trace().size() > 0)
        {
            proverRequest.input.traceConfig.txHashToGenerateExecuteTrace = Add0xIfMissing(ba2string(traceConfig.tx_hash_to_generate_execute_trace()));
        }
        if (traceConfig.tx_hash_to_generate_call_trace().size() > 0)
        {
            proverRequest.input.traceConfig.txHashToGenerateCallTrace = Add0xIfMissing(ba2string(traceConfig.tx_hash_to_generate_call_trace()));
        }
        proverRequest.input.traceConfig.calculateFlags();
    }

    // Default values
    proverRequest.input.publicInputsExtended.newStateRoot = "0x0";
    proverRequest.input.publicInputsExtended.newAccInputHash = "0x0";
    proverRequest.input.publicInputsExtended.newLocalExitRoot = "0x0";
    proverRequest.input.publicInputsExtended.newBatchNum = 0;

    // Parse db map
    const google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> > &db = request->db();
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> >::const_iterator it;
    string key;
    for (it=db.begin(); it!=db.end(); it++)
    {
        // Get key
        key = it->first;
        Remove0xIfPresentNoCopy(key);
        if (key.size() > 64)
        {
            zklog.error("ExecutorServiceImpl::ProcessBatch() got db key too long, size=" + to_string(key.size()), &proverRequest.tags);
            response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DB_KEY);
            //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
            return Status::OK;
        }
        if (!stringIsHex(key))
        {
            zklog.error("ExecutorServiceImpl::ProcessBatch() got db key not hex, key=" + key, &proverRequest.tags);
            response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DB_KEY);
            //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
            return Status::OK;
        }
        PrependZerosNoCopy(key, 64);

        // Get value
        vector<Goldilocks::Element> dbValue;
        string concatenatedValues = it->second;
        if (!stringIsHex(concatenatedValues))
        {
            zklog.error("ExecutorServiceImpl::ProcessBatch() found db value not hex: " + concatenatedValues, &proverRequest.tags);
            response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DB_VALUE);
            //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
            return Status::OK;
        }
        if (concatenatedValues.size()%16!=0)
        {
            zklog.error("ExecutorServiceImpl::ProcessBatch() found invalid db value size: " + to_string(concatenatedValues.size()), &proverRequest.tags);
            response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DB_VALUE);
            //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
            return Status::OK;
        }
        for (uint64_t i=0; i<concatenatedValues.size(); i+=16)
        {
            Goldilocks::Element fe;
            string2fe(fr, concatenatedValues.substr(i, 16), fe);
            dbValue.push_back(fe);
        }
        
        // Save key-value
        proverRequest.input.db[key] = dbValue;

#ifdef LOG_SERVICE_EXECUTOR_INPUT
        //zklog.info("input.db[" + key + "]: " + proverRequest.input.db[key], &proverRequest.tags);
#endif
    }

    // Parse contracts data
    const google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> > &contractsBytecode = request->contracts_bytecode();
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> >::const_iterator itp;
    for (itp=contractsBytecode.begin(); itp!=contractsBytecode.end(); itp++)
    {
        // Get key
        key = itp->first;
        Remove0xIfPresentNoCopy(key);
        if (key.size() > (64))
        {
            zklog.error("ExecutorServiceImpl::ProcessBatch() got contracts key too long, size=" + to_string(key.size()), &proverRequest.tags);
            response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_CONTRACTS_BYTECODE_KEY);
            //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
            return Status::OK;
        }
        if (!stringIsHex(key))
        {
            zklog.error("ExecutorServiceImpl::ProcessBatch() got contracts key not hex, key=" + key, &proverRequest.tags);
            response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_CONTRACTS_BYTECODE_KEY);
            //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
            return Status::OK;
        }
        PrependZerosNoCopy(key, 64);

        // Get value
        if (!stringIsHex(Remove0xIfPresent(itp->second)))
        {
            zklog.error("ExecutorServiceImpl::ProcessBatch() got contracts value not hex, value=" + itp->second, &proverRequest.tags);
            response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_CONTRACTS_BYTECODE_VALUE);
            //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
            return Status::OK;
        }
        vector<uint8_t> dbValue;
        string contractValue = string2ba(itp->second);
        for (uint64_t i=0; i<contractValue.size(); i++)
        {
            dbValue.push_back(contractValue.at(i));
        }

        // Save key-value
        proverRequest.input.contractsBytecode[key] = dbValue;

#ifdef LOG_SERVICE_EXECUTOR_INPUT
        //zklog.info("proverRequest.input.contractsBytecode[" + itp->first + "]: " + itp->second, &proverRequest.tags);
#endif
    }

    // Get no counters flag
    proverRequest.input.bNoCounters = request->no_counters();

#ifdef LOG_SERVICE_EXECUTOR_INPUT
    zklog.info(string("ExecutorServiceImpl::ProcessBatch() got") +
        " sequencerAddr=" + proverRequest.input.publicInputsExtended.publicInputs.sequencerAddr.get_str(16) +
        " batchL2DataLength=" + to_string(request->batch_l2_data().size()) +
        " batchL2Data=0x" + ba2string(proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.substr(0, 10)) + "..." + ba2string(proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.substr(zkmax(int64_t(0),int64_t(proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.size())-10), proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.size())) +
        " oldStateRoot=" + proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot.get_str(16) +
        " oldAccInputHash=" + proverRequest.input.publicInputsExtended.publicInputs.oldAccInputHash.get_str(16) +
        " oldBatchNum=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.oldBatchNum) +
        " chainId=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.chainID) +
        " forkId=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.forkID) +
            (((proverRequest.input.publicInputsExtended.publicInputs.forkID >= 5) && config.useMainExecC) ? " C" :
             ((proverRequest.input.publicInputsExtended.publicInputs.forkID >= 4) && config.useMainExecGenerated) ? " generated" : " native") +
        " globalExitRoot=" + proverRequest.input.publicInputsExtended.publicInputs.globalExitRoot.get_str(16) +
        " timestamp=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.timestamp) +
        " from=" + proverRequest.input.from +
        " bUpdateMerkleTree=" + to_string(proverRequest.input.bUpdateMerkleTree) +
        " bNoCounters=" + to_string(proverRequest.input.bNoCounters) +
        " traceConfig=" + proverRequest.input.traceConfig.toString() +
        " UUID=" + proverRequest.uuid, &proverRequest.tags);
#endif

    if (config.logExecutorServerInputJson)
    {
        // Log the input file content
        json inputJson;
        proverRequest.input.save(inputJson);
        string inputJsonString = inputJson.dump();
        replace(inputJsonString.begin(), inputJsonString.end(), '"', '\'');
        zklog.info("ExecutorServiceImpl::ProcessBatch() Input=" + inputJsonString, &proverRequest.tags);
    }

    prover.processBatch(&proverRequest);

    //TimerStart(EXECUTOR_PROCESS_BATCH_BUILD_RESPONSE);

    if (proverRequest.result != ZKR_SUCCESS)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatch() detected proverRequest.result=" + to_string(proverRequest.result) + "=" + zkresult2string(proverRequest.result), &proverRequest.tags);
    }
    
    response->set_error(zkresult2error(proverRequest.result));
    response->set_cumulative_gas_used(proverRequest.pFullTracer->get_cumulative_gas_used());
    response->set_cnt_keccak_hashes(proverRequest.counters.keccakF);
    response->set_cnt_poseidon_hashes(proverRequest.counters.poseidonG);
    response->set_cnt_poseidon_paddings(proverRequest.counters.paddingPG);
    response->set_cnt_mem_aligns(proverRequest.counters.memAlign);
    response->set_cnt_arithmetics(proverRequest.counters.arith);
    response->set_cnt_binaries(proverRequest.counters.binary);
    response->set_cnt_steps(proverRequest.counters.steps);
    response->set_new_state_root(string2ba(proverRequest.pFullTracer->get_new_state_root()));
    response->set_new_acc_input_hash(string2ba(proverRequest.pFullTracer->get_new_acc_input_hash()));
    response->set_new_local_exit_root(string2ba(proverRequest.pFullTracer->get_new_local_exit_root()));
    response->set_flush_id(proverRequest.flushId);
    response->set_stored_flush_id(proverRequest.lastSentFlushId);
    response->set_prover_id(config.proverID);
    
    unordered_map<string, InfoReadWrite> * p_read_write_addresses = proverRequest.pFullTracer->get_read_write_addresses();
    if (p_read_write_addresses != NULL)
    {
        unordered_map<string, InfoReadWrite>::const_iterator itRWA;
        for (itRWA=p_read_write_addresses->begin(); itRWA != p_read_write_addresses->end(); itRWA++)
        {
            executor::v1::InfoReadWrite infoReadWrite;
            google::protobuf::Map<std::string, executor::v1::InfoReadWrite> * pReadWriteAddresses = response->mutable_read_write_addresses();
            infoReadWrite.set_balance(itRWA->second.balance);
            infoReadWrite.set_nonce(itRWA->second.nonce);
            (*pReadWriteAddresses)[itRWA->first] = infoReadWrite;
        }
    }

    vector<Response> &responses = proverRequest.pFullTracer->get_responses();
    for (uint64_t tx=0; tx<responses.size(); tx++)
    {
        // Remember the previous memory sent for each TX, and send only increments
        string previousMemory;

        executor::v1::ProcessTransactionResponse * pProcessTransactionResponse = response->add_responses();
        pProcessTransactionResponse->set_tx_hash(string2ba(responses[tx].tx_hash));
        pProcessTransactionResponse->set_rlp_tx(responses[tx].rlp_tx);
        pProcessTransactionResponse->set_type(responses[tx].type); // Type indicates legacy transaction; it will be always 0 (legacy) in the executor
        pProcessTransactionResponse->set_return_value(string2ba(responses[tx].return_value)); // Returned data from the runtime (function result or data supplied with revert opcode)
        pProcessTransactionResponse->set_gas_left(responses[tx].gas_left); // Total gas left as result of execution
        pProcessTransactionResponse->set_gas_used(responses[tx].gas_used); // Total gas used as result of execution or gas estimation
        pProcessTransactionResponse->set_gas_refunded(responses[tx].gas_refunded); // Total gas refunded as result of execution
        pProcessTransactionResponse->set_error(string2error(responses[tx].error)); // Any error encountered during the execution
        pProcessTransactionResponse->set_create_address(responses[tx].create_address); // New SC Address in case of SC creation
        pProcessTransactionResponse->set_state_root(string2ba(responses[tx].state_root));
        pProcessTransactionResponse->set_effective_percentage(responses[tx].effective_percentage);
        pProcessTransactionResponse->set_effective_gas_price(responses[tx].effective_gas_price);
        pProcessTransactionResponse->set_has_balance_opcode(responses[tx].has_balance_opcode);
        pProcessTransactionResponse->set_has_gasprice_opcode(responses[tx].has_gasprice_opcode);
        for (uint64_t log=0; log<responses[tx].logs.size(); log++)
        {
            executor::v1::Log * pLog = pProcessTransactionResponse->add_logs();
            pLog->set_address(responses[tx].logs[log].address); // Address of the contract that generated the event
            for (uint64_t topic=0; topic<responses[tx].logs[log].topics.size(); topic++)
            {
                std::string * pTopic = pLog->add_topics();
                *pTopic = string2ba(responses[tx].logs[log].topics[topic]); // List of topics provided by the contract
            }
            string dataConcatenated;
            for (uint64_t data=0; data<responses[tx].logs[log].data.size(); data++)
                dataConcatenated += responses[tx].logs[log].data[data];
            pLog->set_data(string2ba(dataConcatenated)); // Supplied by the contract, usually ABI-encoded
            pLog->set_batch_number(responses[tx].logs[log].batch_number); // Batch in which the transaction was included
            pLog->set_tx_hash(string2ba(responses[tx].logs[log].tx_hash)); // Hash of the transaction
            pLog->set_tx_index(responses[tx].logs[log].tx_index); // Index of the transaction in the block
            pLog->set_batch_hash(string2ba(responses[tx].logs[log].batch_hash)); // Hash of the batch in which the transaction was included
            pLog->set_index(responses[tx].logs[log].index); // Index of the log in the block
        }
        if (proverRequest.input.traceConfig.bEnabled && (proverRequest.input.traceConfig.txHashToGenerateExecuteTrace == responses[tx].tx_hash))
        {
            for (uint64_t step=0; step<responses[tx].execution_trace.size(); step++)
            {
                executor::v1::ExecutionTraceStep * pExecutionTraceStep = pProcessTransactionResponse->add_execution_trace();
                pExecutionTraceStep->set_pc(responses[tx].execution_trace[step].pc); // Program Counter
                // opcode can be null if bNoCounters=true
                if (responses[tx].execution_trace[step].opcode != NULL)
                {
                    pExecutionTraceStep->set_op(responses[tx].execution_trace[step].opcode); // OpCode
                }
                pExecutionTraceStep->set_remaining_gas(responses[tx].execution_trace[step].gas);
                pExecutionTraceStep->set_gas_cost(responses[tx].execution_trace[step].gas_cost); // Gas cost of the operation
                pExecutionTraceStep->set_memory_size(responses[tx].execution_trace[step].memory_size);
                pExecutionTraceStep->set_memory_offset(responses[tx].execution_trace[step].memory_offset);
                pExecutionTraceStep->set_memory(responses[tx].execution_trace[step].memory);
                for (uint64_t stack=0; stack<responses[tx].execution_trace[step].stack.size() ; stack++)
                    pExecutionTraceStep->add_stack(responses[tx].execution_trace[step].stack[stack].get_str(16)); // Content of the stack
                string dataConcatenated;
                for (uint64_t data=0; data<responses[tx].execution_trace[step].return_data.size(); data++)
                    dataConcatenated += responses[tx].execution_trace[step].return_data[data];
                pExecutionTraceStep->set_return_data(string2ba(dataConcatenated));
                google::protobuf::Map<std::string, std::string> * pStorage = pExecutionTraceStep->mutable_storage();
                unordered_map<string,string>::iterator it;
                for (it=responses[tx].execution_trace[step].storage.begin(); it!=responses[tx].execution_trace[step].storage.end(); it++)
                    (*pStorage)[it->first] = it->second; // Content of the storage
                pExecutionTraceStep->set_depth(responses[tx].execution_trace[step].depth); // Call depth
                pExecutionTraceStep->set_gas_refund(responses[tx].execution_trace[step].gas_refund);
                pExecutionTraceStep->set_error(string2error(responses[tx].execution_trace[step].error));
            }
        }
        if (proverRequest.input.traceConfig.bEnabled && (proverRequest.input.traceConfig.txHashToGenerateCallTrace == responses[tx].tx_hash))
        {
            executor::v1::CallTrace * pCallTrace = new executor::v1::CallTrace();
            executor::v1::TransactionContext * pTransactionContext = pCallTrace->mutable_context();
            pTransactionContext->set_type(responses[tx].call_trace.context.type); // "CALL" or "CREATE"
            pTransactionContext->set_from(responses[tx].call_trace.context.from); // Sender of the transaction
            pTransactionContext->set_to(responses[tx].call_trace.context.to); // Target of the transaction
            pTransactionContext->set_data(string2ba(responses[tx].call_trace.context.data)); // Input data of the transaction
            pTransactionContext->set_gas(responses[tx].call_trace.context.gas);
            pTransactionContext->set_gas_price(Add0xIfMissing(responses[tx].call_trace.context.gas_price.get_str(16)));
            pTransactionContext->set_value(Add0xIfMissing(responses[tx].call_trace.context.value.get_str(16)));
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
                pTransactionStep->set_gas(responses[tx].call_trace.steps[step].gas); // Remaining gas
                pTransactionStep->set_gas_cost(responses[tx].call_trace.steps[step].gas_cost); // Gas cost of the operation
                pTransactionStep->set_gas_refund(responses[tx].call_trace.steps[step].gas_refund); // Gas refunded during the operation
                pTransactionStep->set_op(responses[tx].call_trace.steps[step].op); // Opcode
                for (uint64_t stack=0; stack<responses[tx].call_trace.steps[step].stack.size() ; stack++)
                    pTransactionStep->add_stack(responses[tx].call_trace.steps[step].stack[stack].get_str(16)); // Content of the stack
                pTransactionStep->set_memory_size(responses[tx].call_trace.steps[step].memory_size);
                pTransactionStep->set_memory_offset(responses[tx].call_trace.steps[step].memory_offset);
                pTransactionStep->set_memory(responses[tx].call_trace.steps[step].memory);
                string dataConcatenated;
                for (uint64_t data=0; data<responses[tx].call_trace.steps[step].return_data.size(); data++)
                    dataConcatenated += responses[tx].call_trace.steps[step].return_data[data];
                pTransactionStep->set_return_data(string2ba(dataConcatenated));
                executor::v1::Contract * pContract = pTransactionStep->mutable_contract(); // Contract information
                pContract->set_address(responses[tx].call_trace.steps[step].contract.address);
                pContract->set_caller(responses[tx].call_trace.steps[step].contract.caller);
                pContract->set_value(Add0xIfMissing(responses[tx].call_trace.steps[step].contract.value.get_str(16)));
                pContract->set_data(string2ba(responses[tx].call_trace.steps[step].contract.data));
                pContract->set_gas(responses[tx].call_trace.steps[step].contract.gas);
                pContract->set_type(responses[tx].call_trace.steps[step].contract.type);
                pTransactionStep->set_error(string2error(responses[tx].call_trace.steps[step].error));
            }
            pProcessTransactionResponse->set_allocated_call_trace(pCallTrace);
        }
    }

    // Return accessed keys, if requested
    if (proverRequest.input.bGetKeys)
    {
        unordered_set<string>::const_iterator it;
        for (it = proverRequest.nodesKeys.begin(); it != proverRequest.nodesKeys.end(); it++)
        {
            response->add_nodes_keys(string2ba(it->c_str()));
        }
        for (it = proverRequest.programKeys.begin(); it != proverRequest.programKeys.end(); it++)
        {
            response->add_program_keys(string2ba(it->c_str()));
        }
    }

#ifdef LOG_SERVICE_EXECUTOR_OUTPUT
    {
        string s = "ExecutorServiceImpl::ProcessBatch() returns result=" + to_string(response->error()) +
            " old_state_root=" + proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot.get_str(16) +
            " new_state_root=" + proverRequest.pFullTracer->get_new_state_root() +
            " new_acc_input_hash=" + proverRequest.pFullTracer->get_new_acc_input_hash() +
            " new_local_exit_root=" + proverRequest.pFullTracer->get_new_local_exit_root() +
            " old_batch_num=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.oldBatchNum) +
            " steps=" + to_string(proverRequest.counters.steps) +
            " gasUsed=" + to_string(proverRequest.pFullTracer->get_cumulative_gas_used()) +
            " counters.keccakF=" + to_string(proverRequest.counters.keccakF) +
            " counters.poseidonG=" + to_string(proverRequest.counters.poseidonG) +
            " counters.paddingPG=" + to_string(proverRequest.counters.paddingPG) +
            " counters.memAlign=" + to_string(proverRequest.counters.memAlign) +
            " counters.arith=" + to_string(proverRequest.counters.arith) +
            " counters.binary=" + to_string(proverRequest.counters.binary) +
            " flush_id=" + to_string(proverRequest.flushId) +
            " last_sent_flush_id=" + to_string(proverRequest.lastSentFlushId) +
            " nTxs=" + to_string(responses.size());
         if (config.logExecutorServerTxs)
         {
            for (uint64_t tx=0; tx<responses.size(); tx++)
            {
                s += " tx[" + to_string(tx) + "].hash=" + responses[tx].tx_hash +
                    " stateRoot=" + responses[tx].state_root +
                    " gasUsed=" + to_string(responses[tx].gas_used) +
                    " gasLeft=" + to_string(responses[tx].gas_left) +
                    " gasUsed+gasLeft=" + to_string(responses[tx].gas_used + responses[tx].gas_left) +
                    " gasRefunded=" + to_string(responses[tx].gas_refunded) +
                    " result=" + responses[tx].error;
            }
         }
        zklog.info(s, &proverRequest.tags);
    }
#endif

    if (config.logExecutorServerResponses)
    {
        zklog.info("ExecutorServiceImpl::ProcessBatch() returns:\n" + response->DebugString(), &proverRequest.tags);
    }

    //TimerStopAndLog(EXECUTOR_PROCESS_BATCH_BUILD_RESPONSE);
    
    //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
    struct timeval EXECUTOR_PROCESS_BATCH_stop;
    gettimeofday(&EXECUTOR_PROCESS_BATCH_stop,NULL);

    if (config.saveResponseToFile)
    {
        //TimerStart(EXECUTOR_PROCESS_BATCH_SAVING_RESPONSE_TO_FILE);
        //zklog.info("ExecutorServiceImpl::ProcessBatch() returns response of size=" + to_string(response->ByteSizeLong()), &proverRequest.tags);
        string2file(response->DebugString(), proverRequest.filePrefix + "executor_response.txt");
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH_SAVING_RESPONSE_TO_FILE);
    }

    if (config.opcodeTracer)
    {
        map<uint8_t, vector<Opcode>> opcodeMap;
        vector<Opcode> &info(proverRequest.pFullTracer->get_info());
        zklog.info("Received " + to_string(info.size()) + " opcodes:", &proverRequest.tags);
        for (uint64_t i=0; i<info.size(); i++)
        {
            if (opcodeMap.find(info[i].op) == opcodeMap.end())
            {
                vector<Opcode> aux;
                opcodeMap[info[i].op] = aux;
            }
            opcodeMap[info[i].op].push_back(info[i]);
        }
        string s;
        map<uint8_t, vector<Opcode>>::iterator opcodeMapIt;
        for (opcodeMapIt = opcodeMap.begin(); opcodeMapIt != opcodeMap.end(); opcodeMapIt++)
        {
            s += "    0x" + byte2string(opcodeMapIt->first) + "=" + opcodeMapIt->second[0].opcode + " called " + to_string(opcodeMapIt->second.size()) + " times";

            uint64_t opcodeTotalGas = 0;
            s += " gas=";
            for (uint64_t i=0; i<opcodeMapIt->second.size(); i++)
            {
                s += to_string(opcodeMapIt->second[i].gas_cost) + ",";
                opcodeTotalGas += opcodeMapIt->second[i].gas_cost;
            }

            uint64_t opcodeTotalDuration = 0;
            s += " duration=";
            for (uint64_t i=0; i<opcodeMapIt->second.size(); i++)
            {
                s += to_string(opcodeMapIt->second[i].duration) + ",";
                opcodeTotalDuration += opcodeMapIt->second[i].duration;
            }

            s += " TP=" + to_string((double(opcodeTotalGas)*1000000)/double(opcodeTotalDuration)) + "gas/s";
        }
        zklog.info(s, &proverRequest.tags);
    }

    // Calculate the throughput, for this ProcessBatch call, and for all calls
#ifdef LOG_TIME
    lock();
    counter++;
    uint64_t execGas = response->cumulative_gas_used();
    totalGas += execGas;
    uint64_t execBytes = request->batch_l2_data().size();
    totalBytes += execBytes;
    uint64_t execTX = responses.size();
    totalTX += execTX;
    double execTime = double(TimeDiff(EXECUTOR_PROCESS_BATCH_start, EXECUTOR_PROCESS_BATCH_stop))/1000000;
    totalTime += execTime;
    struct timeval now;
    gettimeofday(&now, NULL);
    double timeSinceLastTotal = zkmax(1, double(TimeDiff(lastTotalTime, now))/1000000);
    if (timeSinceLastTotal >= 10.0)
    {
        totalTPG = double(totalGas - lastTotalGas)/timeSinceLastTotal;
        totalTPB = double(totalBytes - lastTotalBytes)/timeSinceLastTotal;
        totalTPTX = double(totalTX - lastTotalTX)/timeSinceLastTotal;
        lastTotalGas = totalGas;
        lastTotalBytes = totalBytes;
        lastTotalTX = totalTX;
        lastTotalTime = now;
    }
    double timeSinceFirstTotal = zkmax(1, double(TimeDiff(firstTotalTime, now))/1000000);
    double TPG = double(totalGas)/timeSinceFirstTotal;
    double TPB = double(totalBytes)/timeSinceFirstTotal;
    double TPTX = double(totalTX)/timeSinceFirstTotal;
    
    uint64_t nfd = getNumberOfFileDescriptors();

    zklog.info("ExecutorServiceImpl::ProcessBatch() done counter=" + to_string(counter) + " B=" + to_string(execBytes) + " TX=" + to_string(execTX) + " gas=" + to_string(execGas) + " time=" + to_string(execTime) +
        " TP=" + to_string(double(execBytes)/execTime) + "B/s=" + to_string(double(execTX)/execTime) + "TX/s=" + to_string(double(execGas)/execTime) + "gas/s=" + to_string(double(execGas)/double(execBytes)) + "gas/B" +
        " totalTP(10s)=" + to_string(totalTPB) + "B/s=" + to_string(totalTPTX) + "TX/s=" + to_string(totalTPG) + "gas/s=" + to_string(totalTPG/zkmax(1,totalTPB)) + "gas/B" +
        " totalTP(ever)=" + to_string(TPB) + "B/s=" + to_string(TPTX) + "TX/s=" + to_string(TPG) + "gas/s=" + to_string(TPG/zkmax(1,TPB)) + "gas/B" +
        " totalTime=" + to_string(totalTime) +
        " filedesc=" + to_string(nfd),
        &proverRequest.tags);
    
    // If the TP in gas/s is < threshold, log the input, unless it has been done before
    if (!config.logExecutorServerInput && (config.logExecutorServerInputGasThreshold > 0) && ((double(execGas)/execTime) < config.logExecutorServerInputGasThreshold))
    {
        json inputJson;
        proverRequest.input.save(inputJson);
        string inputJsonString = inputJson.dump();
        replace(inputJsonString.begin(), inputJsonString.end(), '"', '\'');
        zklog.info("TP=" + to_string(double(execGas)/execTime) + "gas/s Input=" + inputJsonString, &proverRequest.tags);
    }
    unlock();
#endif

    return Status::OK;
}

::grpc::Status ExecutorServiceImpl::GetFlushStatus (::grpc::ServerContext* context, const ::google::protobuf::Empty* request, ::executor::v1::GetFlushStatusResponse* response)
{
    // If the process is exising, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

    uint64_t storedFlushId;
    uint64_t storingFlushId;
    uint64_t lastFlushId;
    uint64_t pendingToFlushNodes;
    uint64_t pendingToFlushProgram;
    uint64_t storingNodes;
    uint64_t storingProgram;
    string proverId;

    pHashDB->getFlushStatus(storedFlushId, storingFlushId, lastFlushId, pendingToFlushNodes, pendingToFlushProgram, storingNodes, storingProgram, proverId);
    
    response->set_stored_flush_id(storedFlushId);
    response->set_storing_flush_id(storingFlushId);
    response->set_last_flush_id(lastFlushId);
    response->set_pending_to_flush_nodes(pendingToFlushNodes);
    response->set_pending_to_flush_program(pendingToFlushProgram);
    response->set_storing_nodes(storingNodes);
    response->set_storing_program(storingProgram);
    response->set_prover_id(proverId);

    return Status::OK;
}

#ifdef PROCESS_BATCH_STREAM

::grpc::Status ExecutorServiceImpl::ProcessBatchStream (::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::executor::v1::ProcessBatchResponse, ::executor::v1::ProcessBatchRequest>* stream)
{
    // If the process is exising, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

    TimerStart(PROCESS_BATCH_STREAM);

#ifdef LOG_SERVICE
    zklog.info("ExecutorServiceImpl::ProcessBatchStream() stream starts");
#endif
    executor::v1::ProcessBatchRequest processBatchRequest;
    executor::v1::ProcessBatchResponse processBatchResponse;
    bool bResult;
    ::grpc::Status grpcStatus;
    uint64_t numberOfRequests = 0;

    while (true)
    {
        // Clear variables
        processBatchRequest.Clear();
        processBatchResponse.Clear();

        // Receive the next ProcessBatchRequest
        bResult = stream->Read(&processBatchRequest);
        if (!bResult)
        {
            zklog.error("ExecutorServiceImpl::ProcessBatchStream() failed calling stream->Read(processBatchRequest) numberOfRequests=" + to_string(numberOfRequests));
            TimerStopAndLog(PROCESS_BATCH_STREAM);
            return Status::CANCELLED;
        }

        // Call ProcessBatch
        grpcStatus = ProcessBatch(context, &processBatchRequest, &processBatchResponse);
        if (!grpcStatus.ok())
        {
            zklog.error("ExecutorServiceImpl::ProcessBatchStream() failed calling ProcessBatch() numberOfRequests=" + to_string(numberOfRequests));
            TimerStopAndLog(PROCESS_BATCH_STREAM);
            return grpcStatus;
        }

        // Send the response
        bResult = stream->Write(processBatchResponse);
        if (!bResult)
        {
            zklog.error("ExecutorServiceImpl::ProcessBatchStream() failed calling stream->Write(processBatchResponse) numberOfRequests=" + to_string(numberOfRequests));
            TimerStopAndLog(PROCESS_BATCH_STREAM);
            return Status::CANCELLED;
        }

        // Increment number of requests
        numberOfRequests++;
    }
}

#endif

::executor::v1::RomError ExecutorServiceImpl::string2error (string &errorString)
{
    if (errorString == "OOG"                              ) return ::executor::v1::ROM_ERROR_OUT_OF_GAS;
    if (errorString == "revert"                           ) return ::executor::v1::ROM_ERROR_EXECUTION_REVERTED;
    if (errorString == "overflow"                         ) return ::executor::v1::ROM_ERROR_STACK_OVERFLOW;
    if (errorString == "underflow"                        ) return ::executor::v1::ROM_ERROR_STACK_UNDERFLOW;
    if (errorString == "OOCS"                             ) return ::executor::v1::ROM_ERROR_OUT_OF_COUNTERS_STEP;
    if (errorString == "OOCK"                             ) return ::executor::v1::ROM_ERROR_OUT_OF_COUNTERS_KECCAK;
    if (errorString == "OOCB"                             ) return ::executor::v1::ROM_ERROR_OUT_OF_COUNTERS_BINARY;
    if (errorString == "OOCM"                             ) return ::executor::v1::ROM_ERROR_OUT_OF_COUNTERS_MEM;
    if (errorString == "OOCA"                             ) return ::executor::v1::ROM_ERROR_OUT_OF_COUNTERS_ARITH;
    if (errorString == "OOCPA"                            ) return ::executor::v1::ROM_ERROR_OUT_OF_COUNTERS_PADDING;
    if (errorString == "OOCPO"                            ) return ::executor::v1::ROM_ERROR_OUT_OF_COUNTERS_POSEIDON;
    if (errorString == "intrinsic_invalid_signature"      ) return ::executor::v1::ROM_ERROR_INTRINSIC_INVALID_SIGNATURE;
    if (errorString == "intrinsic_invalid_chain_id"       ) return ::executor::v1::ROM_ERROR_INTRINSIC_INVALID_CHAIN_ID;
    if (errorString == "intrinsic_invalid_nonce"          ) return ::executor::v1::ROM_ERROR_INTRINSIC_INVALID_NONCE;
    if (errorString == "intrinsic_invalid_gas_limit"      ) return ::executor::v1::ROM_ERROR_INTRINSIC_INVALID_GAS_LIMIT;
    if (errorString == "intrinsic_invalid_gas_overflow"   ) return ::executor::v1::ROM_ERROR_INTRINSIC_TX_GAS_OVERFLOW;
    if (errorString == "intrinsic_invalid_balance"        ) return ::executor::v1::ROM_ERROR_INTRINSIC_INVALID_BALANCE;
    if (errorString == "intrinsic_invalid_batch_gas_limit") return ::executor::v1::ROM_ERROR_INTRINSIC_INVALID_BATCH_GAS_LIMIT;
    if (errorString == "intrinsic_invalid_sender_code"    ) return ::executor::v1::ROM_ERROR_INTRINSIC_INVALID_SENDER_CODE;
    if (errorString == "invalidRLP"                       ) return ::executor::v1::ROM_ERROR_INVALID_RLP;
    if (errorString == "invalidJump"                      ) return ::executor::v1::ROM_ERROR_INVALID_JUMP;
    if (errorString == "invalidOpcode"                    ) return ::executor::v1::ROM_ERROR_INVALID_OPCODE;
    if (errorString == "invalidAddressCollision"          ) return ::executor::v1::ROM_ERROR_CONTRACT_ADDRESS_COLLISION;
    if (errorString == "invalidStaticTx"                  ) return ::executor::v1::ROM_ERROR_INVALID_STATIC;
    if (errorString == "invalidCodeSize"                  ) return ::executor::v1::ROM_ERROR_MAX_CODE_SIZE_EXCEEDED;
    if (errorString == "invalidCodeStartsEF"              ) return ::executor::v1::ROM_ERROR_INVALID_BYTECODE_STARTS_EF;
    if (errorString == "invalid_fork_id"                  ) return ::executor::v1::ROM_ERROR_UNSUPPORTED_FORK_ID;
    if (errorString == ""                                 ) return ::executor::v1::ROM_ERROR_NO_ERROR;
    zklog.error("ExecutorServiceImpl::string2error() found invalid error string=" + errorString);
    exitProcess();
    return ::executor::v1::ROM_ERROR_UNSPECIFIED;
}

::executor::v1::ExecutorError ExecutorServiceImpl::zkresult2error (zkresult &result)
{
    switch (result)
    {
    case ZKR_SUCCESS:                                       return ::executor::v1::EXECUTOR_ERROR_NO_ERROR;
    case ZKR_SM_MAIN_OOC_ARITH:                             return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_COUNTERS_OVERFLOW_ARITH;
    case ZKR_SM_MAIN_OOC_BINARY:                            return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_COUNTERS_OVERFLOW_BINARY;
    case ZKR_SM_MAIN_OOC_KECCAK_F:                          return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_COUNTERS_OVERFLOW_KECCAK;
    case ZKR_SM_MAIN_OOC_MEM_ALIGN:                         return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_COUNTERS_OVERFLOW_MEM;
    case ZKR_SM_MAIN_OOC_PADDING_PG:                        return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_COUNTERS_OVERFLOW_PADDING;
    case ZKR_SM_MAIN_OOC_POSEIDON_G:                        return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_COUNTERS_OVERFLOW_POSEIDON;
    case ZKR_SM_MAIN_INVALID_FORK_ID:                       return ::executor::v1::EXECUTOR_ERROR_UNSUPPORTED_FORK_ID;
    case ZKR_SM_MAIN_BALANCE_MISMATCH:                      return ::executor::v1::EXECUTOR_ERROR_BALANCE_MISMATCH;
    case ZKR_SM_MAIN_FEA2SCALAR:                            return ::executor::v1::EXECUTOR_ERROR_FEA2SCALAR;
    case ZKR_SM_MAIN_TOS32:                                 return ::executor::v1::EXECUTOR_ERROR_TOS32;
    case ZKR_DB_ERROR:                                      return ::executor::v1::EXECUTOR_ERROR_DB_ERROR;
    case ZKR_SM_MAIN_INVALID_UNSIGNED_TX:                   return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_INVALID_UNSIGNED_TX;
    case ZKR_SM_MAIN_INVALID_NO_COUNTERS:                   return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_INVALID_NO_COUNTERS;
    case ZKR_SM_MAIN_ARITH_ECRECOVER_DIVIDE_BY_ZERO:        return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_ARITH_ECRECOVER_DIVIDE_BY_ZERO;
    case ZKR_SM_MAIN_ADDRESS_OUT_OF_RANGE:                  return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_ADDRESS_OUT_OF_RANGE;
    case ZKR_SM_MAIN_ADDRESS_NEGATIVE:                      return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_ADDRESS_NEGATIVE;
    case ZKR_SM_MAIN_STORAGE_INVALID_KEY:                   return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_STORAGE_INVALID_KEY;
    case ZKR_SM_MAIN_HASHK:                                 return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHK;
    case ZKR_SM_MAIN_HASHK_SIZE_OUT_OF_RANGE:               return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHK_SIZE_OUT_OF_RANGE;
    case ZKR_SM_MAIN_HASHK_POSITION_NEGATIVE:               return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHK_POSITION_NEGATIVE;
    case ZKR_SM_MAIN_HASHK_POSITION_PLUS_SIZE_OUT_OF_RANGE: return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHK_POSITION_PLUS_SIZE_OUT_OF_RANGE;
    case ZKR_SM_MAIN_HASHKDIGEST_ADDRESS_NOT_FOUND:         return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHKDIGEST_ADDRESS_NOT_FOUND;
    case ZKR_SM_MAIN_HASHKDIGEST_NOT_COMPLETED:             return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHKDIGEST_NOT_COMPLETED;
    case ZKR_SM_MAIN_HASHP:                                 return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHP;
    case ZKR_SM_MAIN_HASHP_SIZE_OUT_OF_RANGE:               return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHP_SIZE_OUT_OF_RANGE;
    case ZKR_SM_MAIN_HASHP_POSITION_NEGATIVE:               return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHP_POSITION_NEGATIVE;
    case ZKR_SM_MAIN_HASHP_POSITION_PLUS_SIZE_OUT_OF_RANGE: return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHP_POSITION_PLUS_SIZE_OUT_OF_RANGE;
    case ZKR_SM_MAIN_HASHPDIGEST_ADDRESS_NOT_FOUND:         return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHPDIGEST_ADDRESS_NOT_FOUND;
    case ZKR_SM_MAIN_HASHPDIGEST_NOT_COMPLETED:             return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHPDIGEST_NOT_COMPLETED;
    case ZKR_SM_MAIN_MEMALIGN_OFFSET_OUT_OF_RANGE:          return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_MEMALIGN_OFFSET_OUT_OF_RANGE;
    case ZKR_SM_MAIN_MULTIPLE_FREEIN:                       return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_MULTIPLE_FREEIN;
    case ZKR_SM_MAIN_ASSERT:                                return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_ASSERT;
    case ZKR_SM_MAIN_MEMORY:                                return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_MEMORY;
    case ZKR_SM_MAIN_STORAGE_READ_MISMATCH:                 return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_STORAGE_READ_MISMATCH;
    case ZKR_SM_MAIN_STORAGE_WRITE_MISMATCH:                return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_STORAGE_WRITE_MISMATCH;
    case ZKR_SM_MAIN_HASHK_VALUE_MISMATCH:                  return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHK_VALUE_MISMATCH;
    case ZKR_SM_MAIN_HASHK_PADDING_MISMATCH:                return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHK_PADDING_MISMATCH;
    case ZKR_SM_MAIN_HASHK_SIZE_MISMATCH:                   return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHK_SIZE_MISMATCH;
    case ZKR_SM_MAIN_HASHKLEN_LENGTH_MISMATCH:              return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHKLEN_LENGTH_MISMATCH;
    case ZKR_SM_MAIN_HASHKLEN_CALLED_TWICE:                 return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHKLEN_CALLED_TWICE;
    case ZKR_SM_MAIN_HASHKDIGEST_NOT_FOUND:                 return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHKDIGEST_NOT_FOUND;
    case ZKR_SM_MAIN_HASHKDIGEST_DIGEST_MISMATCH:           return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHKDIGEST_DIGEST_MISMATCH;
    case ZKR_SM_MAIN_HASHKDIGEST_CALLED_TWICE:              return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHKDIGEST_CALLED_TWICE;
    case ZKR_SM_MAIN_HASHP_VALUE_MISMATCH:                  return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHP_VALUE_MISMATCH;
    case ZKR_SM_MAIN_HASHP_PADDING_MISMATCH:                return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHP_PADDING_MISMATCH;
    case ZKR_SM_MAIN_HASHP_SIZE_MISMATCH:                   return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHP_SIZE_MISMATCH;
    case ZKR_SM_MAIN_HASHPLEN_LENGTH_MISMATCH:              return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHPLEN_LENGTH_MISMATCH;
    case ZKR_SM_MAIN_HASHPLEN_CALLED_TWICE:                 return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHPLEN_CALLED_TWICE;
    case ZKR_SM_MAIN_HASHPDIGEST_DIGEST_MISMATCH:           return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHPDIGEST_DIGEST_MISMATCH;
    case ZKR_SM_MAIN_HASHPDIGEST_CALLED_TWICE:              return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHPDIGEST_CALLED_TWICE;
    case ZKR_SM_MAIN_ARITH_MISMATCH:                        return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_ARITH_MISMATCH;
    case ZKR_SM_MAIN_ARITH_ECRECOVER_MISMATCH:              return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_ARITH_ECRECOVER_MISMATCH;
    case ZKR_SM_MAIN_BINARY_ADD_MISMATCH:                   return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_BINARY_ADD_MISMATCH;
    case ZKR_SM_MAIN_BINARY_SUB_MISMATCH:                   return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_BINARY_SUB_MISMATCH;
    case ZKR_SM_MAIN_BINARY_LT_MISMATCH:                    return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_BINARY_LT_MISMATCH;
    case ZKR_SM_MAIN_BINARY_SLT_MISMATCH:                   return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_BINARY_SLT_MISMATCH;
    case ZKR_SM_MAIN_BINARY_EQ_MISMATCH:                    return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_BINARY_EQ_MISMATCH;
    case ZKR_SM_MAIN_BINARY_AND_MISMATCH:                   return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_BINARY_AND_MISMATCH;
    case ZKR_SM_MAIN_BINARY_OR_MISMATCH:                    return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_BINARY_OR_MISMATCH;
    case ZKR_SM_MAIN_BINARY_XOR_MISMATCH:                   return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_BINARY_XOR_MISMATCH;
    case ZKR_SM_MAIN_MEMALIGN_WRITE_MISMATCH:               return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_MEMALIGN_WRITE_MISMATCH;
    case ZKR_SM_MAIN_MEMALIGN_WRITE8_MISMATCH:              return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_MEMALIGN_WRITE8_MISMATCH;
    case ZKR_SM_MAIN_MEMALIGN_READ_MISMATCH:                return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_MEMALIGN_READ_MISMATCH;
    case ZKR_SM_MAIN_S33:                                   return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_JMPN_OUT_OF_RANGE;
    case ZKR_SM_MAIN_OUT_OF_STEPS:                          return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_COUNTERS_OVERFLOW_STEPS;
    case ZKR_SM_MAIN_HASHK_READ_OUT_OF_RANGE:               return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHK_READ_OUT_OF_RANGE;
    case ZKR_SM_MAIN_HASHP_READ_OUT_OF_RANGE:               return ::executor::v1::EXECUTOR_ERROR_SM_MAIN_HASHP_READ_OUT_OF_RANGE;

    default:                                                return ::executor::v1::EXECUTOR_ERROR_UNSPECIFIED;
    }
}