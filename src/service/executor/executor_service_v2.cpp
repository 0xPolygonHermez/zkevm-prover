#include "config.hpp"
#include "executor_service.hpp"
#include "input.hpp"
#include "proof.hpp"
#include "zklog.hpp"
#include <grpcpp/grpcpp.h>
#include "exit_process.hpp"
#include "utils.hpp"
#include "witness.hpp"
#include "data_stream.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

::grpc::Status ExecutorServiceImpl::ProcessBatchV2 (::grpc::ServerContext* context, const ::executor::v1::ProcessBatchRequestV2* request, ::executor::v1::ProcessBatchResponseV2* response)
{
    // If the process is exiting, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

    //TimerStart(EXECUTOR_PROCESS_BATCH);
    struct timeval EXECUTOR_PROCESS_BATCH_start;
    gettimeofday(&EXECUTOR_PROCESS_BATCH_start,NULL);

#ifdef LOG_SERVICE
    zklog.info("ExecutorServiceImpl::ProcessBatchV2() got request:\n" + request->DebugString());
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
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() got oldStateRoot too long, size=" + to_string(request->old_state_root().size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_OLD_STATE_ROOT);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    ba2scalar(proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot, request->old_state_root());

    // Get oldAccInputHash
    if (request->old_acc_input_hash().size() > 32)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() got oldAccInputHash too long, size=" + to_string(request->old_acc_input_hash().size()), &proverRequest.tags);
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
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() got chainID = 0", &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_CHAIN_ID);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }

    // Get fork ID
    proverRequest.input.publicInputsExtended.publicInputs.forkID = request->fork_id();
    if (proverRequest.input.publicInputsExtended.publicInputs.forkID < 7)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() got invalid fork ID =" + to_string(proverRequest.input.publicInputsExtended.publicInputs.forkID), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_UNSUPPORTED_FORK_ID);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }

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
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() found batchL2Data.size()=" + to_string(request->batch_l2_data().size()) + " > MAX_BATCH_L2_DATA_SIZE=" + to_string(MAX_BATCH_L2_DATA_SIZE), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_BATCH_L2_DATA);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    proverRequest.input.publicInputsExtended.publicInputs.batchL2Data = request->batch_l2_data();

    // Get globalExitRoot
    if (request->l1_info_root().size() > 32)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() got l1_info_root too long, size=" + to_string(request->l1_info_root().size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_L1_INFO_ROOT);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    ba2scalar(proverRequest.input.publicInputsExtended.publicInputs.l1InfoRoot, request->l1_info_root());

    // Get forcedBlockHashL1
    if (request->forced_blockhash_l1().size() > 32)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() got forced_blockhash_l1 too long, size=" + to_string(request->forced_blockhash_l1().size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_FORCED_BLOCKHASH_L1);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    ba2scalar(proverRequest.input.publicInputsExtended.publicInputs.forcedBlockHashL1, request->forced_blockhash_l1());

    // Get timestamp limit
    proverRequest.input.publicInputsExtended.publicInputs.timestampLimit = request->timestamp_limit();

    // Get sequencer address
    string auxString = Remove0xIfPresent(request->coinbase());
    if (auxString.size() > 40)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() got sequencer address too long, size=" + to_string(auxString.size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_COINBASE);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    if (!stringIsHex(auxString))
    {
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() got sequencer address not hex, coinbase=" + auxString, &proverRequest.tags);
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
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() got from too long, size=" + to_string(proverRequest.input.from.size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_FROM);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    if (!stringIs0xHex(proverRequest.input.from))
    {
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() got from not hex, size=" + proverRequest.input.from, &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_FROM);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }

    // Flags
    proverRequest.input.bUpdateMerkleTree = request->update_merkle_tree();
    if (proverRequest.input.bUpdateMerkleTree && config.dbReadOnly)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() got bUpdateMerkleTree=true while dbReadOnly=true", &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_UPDATE_MERKLE_TREE);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    proverRequest.input.bNoCounters = request->no_counters();
    proverRequest.input.bGetKeys = request->get_keys();
    proverRequest.input.bSkipVerifyL1InfoRoot = request->skip_verify_l1_info_root();
    proverRequest.input.bSkipFirstChangeL2Block = request->skip_first_change_l2_block();
    proverRequest.input.bSkipWriteBlockInfoRoot = request->skip_write_block_info_root();

    // Trace config
    if (request->has_trace_config())
    {
        proverRequest.input.traceConfig.bEnabled = true;
        const executor::v1::TraceConfigV2 & traceConfig = request->trace_config();
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
        if (traceConfig.tx_hash_to_generate_full_trace().size() > 0)
        {
            proverRequest.input.traceConfig.txHashToGenerateFullTrace = Add0xIfMissing(ba2string(traceConfig.tx_hash_to_generate_full_trace()));
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
            zklog.error("ExecutorServiceImpl::ProcessBatchV2() got db key too long, size=" + to_string(key.size()), &proverRequest.tags);
            response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DB_KEY);
            //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
            return Status::OK;
        }
        if (!stringIsHex(key))
        {
            zklog.error("ExecutorServiceImpl::ProcessBatchV2() got db key not hex, key=" + key, &proverRequest.tags);
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
            zklog.error("ExecutorServiceImpl::ProcessBatchV2() found db value not hex: " + concatenatedValues, &proverRequest.tags);
            response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DB_VALUE);
            //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
            return Status::OK;
        }
        if (concatenatedValues.size()%16!=0)
        {
            zklog.error("ExecutorServiceImpl::ProcessBatchV2() found invalid db value size: " + to_string(concatenatedValues.size()), &proverRequest.tags);
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
            zklog.error("ExecutorServiceImpl::ProcessBatchV2() got contracts key too long, size=" + to_string(key.size()), &proverRequest.tags);
            response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_CONTRACTS_BYTECODE_KEY);
            //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
            return Status::OK;
        }
        if (!stringIsHex(key))
        {
            zklog.error("ExecutorServiceImpl::ProcessBatchV2() got contracts key not hex, key=" + key, &proverRequest.tags);
            response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_CONTRACTS_BYTECODE_KEY);
            //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
            return Status::OK;
        }
        PrependZerosNoCopy(key, 64);

        // Get value
        if (!stringIsHex(Remove0xIfPresent(itp->second)))
        {
            zklog.error("ExecutorServiceImpl::ProcessBatchV2() got contracts value not hex, value=" + itp->second, &proverRequest.tags);
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

    // Parse L1 info tree data
    const google::protobuf::Map<google::protobuf::uint32, executor::v1::L1DataV2> &l1InfoTreeData = request->l1_info_tree_data();
    google::protobuf::Map<google::protobuf::uint32, executor::v1::L1DataV2>::const_iterator itl;
    for (itl=l1InfoTreeData.begin(); itl!=l1InfoTreeData.end(); itl++)
    {
        // Get index
        uint64_t index = itl->first;

        // Get L1 data
        L1Data l1Data;
        const executor::v1::L1DataV2 &l1DataV2 = itl->second;
        if (l1DataV2.global_exit_root().size() > 32)
        {
            zklog.error("ExecutorServiceImpl::ProcessBatchV2() got l1DataV2.global_exit_root() too long, size=" + to_string(l1DataV2.global_exit_root().size()), &proverRequest.tags);
            response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_L1_DATA_V2_GLOBAL_EXIT_ROOT);
            return Status::OK;
        }
        ba2scalar(l1Data.globalExitRoot, l1DataV2.global_exit_root());
        if (l1DataV2.block_hash_l1().size() > 32)
        {
            zklog.error("ExecutorServiceImpl::ProcessBatchV2() got l1DataV2.block_hash_l1() too long, size=" + to_string(l1DataV2.block_hash_l1().size()), &proverRequest.tags);
            response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_L1_DATA_V2_BLOCK_HASH_L1);
            return Status::OK;
        }
        ba2scalar(l1Data.blockHashL1, l1DataV2.block_hash_l1());
        l1Data.minTimestamp = l1DataV2.min_timestamp();
        for (int64_t i=0; i<l1DataV2.smt_proof_size(); i++)
        {
            mpz_class auxScalar;
            if (l1DataV2.smt_proof(i).size() > 32)
            {
                zklog.error("ExecutorServiceImpl::ProcessBatchV2() got l1DataV2.smt_proof(i) too long, size=" + to_string(l1DataV2.smt_proof(i).size()), &proverRequest.tags);
                response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_L1_SMT_PROOF);
                return Status::OK;
            }
            ba2scalar(auxScalar, l1DataV2.smt_proof(i));
            l1Data.smtProof.emplace_back(auxScalar);
        }

        // Store it
        proverRequest.input.l1InfoTreeData[index] = l1Data;
    }

    if (request->state_override_size() > 0)
    {
        google::protobuf::Map<std::string, executor::v1::OverrideAccountV2>::const_iterator it;
        for (it = request->state_override().begin(); it != request->state_override().end(); it++)
        {
            OverrideEntry overrideEntry;

            // Get balance
            if (it->second.balance().size() > 32)
            {
                zklog.error("ExecutorServiceImpl::ProcessBatchV2() got state override balance too long, size=" + to_string(it->second.balance().size()), &proverRequest.tags);
                response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_BALANCE);
                return Status::OK;
            }

            if (it->second.balance().size() > 0)
            {
                overrideEntry.balance.set_str(ba2string(it->second.balance()), 16);
                overrideEntry.bBalance = true;
            }

            // Get nonce
            overrideEntry.nonce = it->second.nonce();

            // Get code
            ba2ba(it->second.code(), overrideEntry.code);

            // Get state
            if (it->second.state_size() > 0)
            {
                google::protobuf::Map<std::string, std::string>::const_iterator itState;
                for (itState = it->second.state().begin(); itState != it->second.state().end(); itState++)
                {
                    // Get a valid key
                    string keyString = Remove0xIfPresent(itState->first);
                    if (keyString.size() > 64)
                    {
                        zklog.error("ExecutorServiceImpl::ProcessBatch() got state override key too long, size=" + to_string(keyString.size()), &proverRequest.tags);
                        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DB_KEY);
                        return Status::OK;
                    }
                    if (!stringIsHex(keyString))
                    {
                        zklog.error("ExecutorServiceImpl::ProcessBatch() got state override key not hex, key=" + keyString, &proverRequest.tags);
                        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DB_KEY);
                        return Status::OK;
                    }
                    keyString = NormalizeToNFormat(keyString, 64);

                    // Get a valid value
                    string valueString = Remove0xIfPresent(itState->second);
                    if (valueString.size() > 64)
                    {
                        zklog.error("ExecutorServiceImpl::ProcessBatch() got state override value too long, size=" + to_string(valueString.size()), &proverRequest.tags);
                        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DB_VALUE);
                        return Status::OK;
                    }
                    if (!stringIsHex(valueString))
                    {
                        zklog.error("ExecutorServiceImpl::ProcessBatch() got state override value not hex, value=" + valueString, &proverRequest.tags);
                        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DB_VALUE);
                        return Status::OK;
                    }
                    mpz_class valueScalar;
                    valueScalar.set_str(valueString, 16);

                    // Store the value
                    overrideEntry.state[keyString] = valueScalar;
                }
            }

            // Get state diff
            if (it->second.state_diff_size() > 0)
            {
                google::protobuf::Map<std::string, std::string>::const_iterator itState;
                for (itState = it->second.state_diff().begin(); itState != it->second.state_diff().end(); itState++)
                {
                    // Get a valid key
                    string keyString = Remove0xIfPresent(itState->first);
                    if (keyString.size() > 64)
                    {
                        zklog.error("ExecutorServiceImpl::ProcessBatch() got state override key too long, size=" + to_string(keyString.size()), &proverRequest.tags);
                        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DB_KEY);
                        return Status::OK;
                    }
                    if (!stringIsHex(keyString))
                    {
                        zklog.error("ExecutorServiceImpl::ProcessBatch() got state override key not hex, key=" + keyString, &proverRequest.tags);
                        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DB_KEY);
                        return Status::OK;
                    }
                    keyString = NormalizeToNFormat(keyString, 64);

                    // Get a valid value
                    string valueString = Remove0xIfPresent(itState->second);
                    if (valueString.size() > 64)
                    {
                        zklog.error("ExecutorServiceImpl::ProcessBatch() got state override value too long, size=" + to_string(valueString.size()), &proverRequest.tags);
                        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DB_VALUE);
                        return Status::OK;
                    }
                    if (!stringIsHex(valueString))
                    {
                        zklog.error("ExecutorServiceImpl::ProcessBatch() got state override value not hex, value=" + valueString, &proverRequest.tags);
                        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DB_VALUE);
                        return Status::OK;
                    }
                    mpz_class valueScalar;
                    valueScalar.set_str(valueString, 16);

                    // Store the value
                    overrideEntry.stateDiff[keyString] = valueScalar;
                }
            }

            // Store the override entry
            proverRequest.input.stateOverride[it->first] = overrideEntry;
        }
    }

    proverRequest.input.debug.gasLimit = request->debug().gas_limit();
    
    if (request->debug().new_state_root().size() > 32)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() got new_state_root too long, size=" + to_string(request->debug().new_state_root().size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_NEW_STATE_ROOT);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    ba2scalar(proverRequest.input.publicInputsExtended.newStateRoot, request->debug().new_state_root());
    if (request->debug().new_acc_input_hash().size() > 32)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() got new_acc_input_hash too long, size=" + to_string(request->debug().new_acc_input_hash().size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_NEW_ACC_INPUT_HASH);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    ba2scalar(proverRequest.input.publicInputsExtended.newAccInputHash, request->debug().new_acc_input_hash());
    if (request->debug().new_local_exit_root().size() > 32)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() got new_local_exit_root too long, size=" + to_string(request->debug().new_local_exit_root().size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_NEW_LOCAL_EXIT_ROOT);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    ba2scalar(proverRequest.input.publicInputsExtended.newLocalExitRoot, request->debug().new_local_exit_root());
    proverRequest.input.publicInputsExtended.newBatchNum = request->debug().new_batch_num();

#ifdef LOG_SERVICE_EXECUTOR_INPUT
    string l1InfoTreeDataString = " l1InfoTreeData.size=" + to_string(proverRequest.input.l1InfoTreeData.size()) + "=";
    unordered_map<uint64_t, L1Data>::const_iterator itl1;
    for (itl1 = proverRequest.input.l1InfoTreeData.begin(); itl1 != proverRequest.input.l1InfoTreeData.end(); itl1++)
    {
        l1InfoTreeDataString += to_string(itl1->first) + ",";
    }
    zklog.info(string("ExecutorServiceImpl::ProcessBatchV2() got") +
        " sequencerAddr=" + proverRequest.input.publicInputsExtended.publicInputs.sequencerAddr.get_str(16) +
        " batchL2DataLength=" + to_string(request->batch_l2_data().size()) +
        " batchL2Data=0x" + ba2string(proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.substr(0, 10)) + "..." + ba2string(proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.substr(zkmax(int64_t(0),int64_t(proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.size())-10), proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.size())) +
        " oldStateRoot=" + proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot.get_str(16) +
        " oldAccInputHash=" + proverRequest.input.publicInputsExtended.publicInputs.oldAccInputHash.get_str(16) +
        " oldBatchNum=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.oldBatchNum) +
        " chainId=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.chainID) +
        " forkId=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.forkID) +
            (((proverRequest.input.publicInputsExtended.publicInputs.forkID >= 7) && config.useMainExecC) ? " C" :
#ifdef MAIN_SM_EXECUTOR_GENERATED_CODE
             ((proverRequest.input.publicInputsExtended.publicInputs.forkID >= 4) && config.useMainExecGenerated) ? " generated" :
#endif
             " native") +
        " globalExitRoot=" + proverRequest.input.publicInputsExtended.publicInputs.globalExitRoot.get_str(16) +
        " timestampLimit=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.timestampLimit) +
        " from=" + proverRequest.input.from +
        " bUpdateMerkleTree=" + to_string(proverRequest.input.bUpdateMerkleTree) +
        " bNoCounters=" + to_string(proverRequest.input.bNoCounters) +
        " bGetKeys=" + to_string(proverRequest.input.bGetKeys) +
        " bSkipVerifyL1InfoRoot=" + to_string(proverRequest.input.bSkipVerifyL1InfoRoot) +
        " bSkipFirstChangeL2Block=" + to_string(proverRequest.input.bSkipFirstChangeL2Block) +
        " bSkipWriteBlockInfoRoot=" + to_string(proverRequest.input.bSkipWriteBlockInfoRoot) +
        " traceConfig=" + proverRequest.input.traceConfig.toString() +
        " UUID=" + proverRequest.uuid +
        " gasLimit=" + to_string(proverRequest.input.debug.gasLimit) +
        l1InfoTreeDataString +
        " stateOverride.size=" + to_string(proverRequest.input.stateOverride.size())
        , &proverRequest.tags);
#endif

    if (config.logExecutorServerInputJson)
    {
        // Log the input file content
        json inputJson;
        proverRequest.input.save(inputJson);
        string inputJsonString = inputJson.dump();
        replace(inputJsonString.begin(), inputJsonString.end(), '"', '\'');
        zklog.info("ExecutorServiceImpl::ProcessBatchV2() Input=" + inputJsonString, &proverRequest.tags);
    }

    prover.processBatch(&proverRequest);

    //TimerStart(EXECUTOR_PROCESS_BATCH_BUILD_RESPONSE);

    if (proverRequest.result != ZKR_SUCCESS)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() detected proverRequest.result=" + to_string(proverRequest.result) + "=" + zkresult2string(proverRequest.result), &proverRequest.tags);
    }
    
    response->set_error(zkresult2error(proverRequest.result));
    response->set_gas_used(proverRequest.pFullTracer->get_gas_used());

    response->set_cnt_keccak_hashes(proverRequest.counters.keccakF);
    response->set_cnt_poseidon_hashes(proverRequest.counters.poseidonG);
    response->set_cnt_poseidon_paddings(proverRequest.counters.paddingPG);
    response->set_cnt_mem_aligns(proverRequest.counters.memAlign);
    response->set_cnt_arithmetics(proverRequest.counters.arith);
    response->set_cnt_binaries(proverRequest.counters.binary);
    response->set_cnt_sha256_hashes(proverRequest.counters.sha256F);
    response->set_cnt_steps(proverRequest.counters.steps);
    
    response->set_cnt_reserve_keccak_hashes(proverRequest.counters_reserve.keccakF);
    response->set_cnt_reserve_poseidon_hashes(proverRequest.counters_reserve.poseidonG);
    response->set_cnt_reserve_poseidon_paddings(proverRequest.counters_reserve.paddingPG);
    response->set_cnt_reserve_mem_aligns(proverRequest.counters_reserve.memAlign);
    response->set_cnt_reserve_arithmetics(proverRequest.counters_reserve.arith);
    response->set_cnt_reserve_binaries(proverRequest.counters_reserve.binary);
    response->set_cnt_reserve_sha256_hashes(proverRequest.counters_reserve.sha256F);
    response->set_cnt_reserve_steps(proverRequest.counters_reserve.steps);

    response->set_new_state_root(string2ba(proverRequest.pFullTracer->get_new_state_root()));
    response->set_new_acc_input_hash(string2ba(proverRequest.pFullTracer->get_new_acc_input_hash()));
    response->set_new_local_exit_root(string2ba(proverRequest.pFullTracer->get_new_local_exit_root()));
    response->set_flush_id(proverRequest.flushId);
    response->set_stored_flush_id(proverRequest.lastSentFlushId);
    response->set_prover_id(config.proverID);
    response->set_fork_id(proverRequest.input.publicInputsExtended.publicInputs.forkID);
    response->set_error_rom(string2error(proverRequest.pFullTracer->get_error()));
    response->set_invalid_batch(proverRequest.pFullTracer->get_invalid_batch());
    
    unordered_map<string, InfoReadWrite> * p_read_write_addresses = proverRequest.pFullTracer->get_read_write_addresses();
    if (p_read_write_addresses != NULL)
    {
        unordered_map<string, InfoReadWrite>::const_iterator itRWA;
        for (itRWA=p_read_write_addresses->begin(); itRWA != p_read_write_addresses->end(); itRWA++)
        {
            executor::v1::InfoReadWriteV2 infoReadWrite;
            google::protobuf::Map<std::string, executor::v1::InfoReadWriteV2> * pReadWriteAddresses = response->mutable_read_write_addresses();
            infoReadWrite.set_balance(itRWA->second.balance);
            infoReadWrite.set_nonce(itRWA->second.nonce);
            (*pReadWriteAddresses)[itRWA->first] = infoReadWrite;
        }
    }

    vector<Block> &block_responses = proverRequest.pFullTracer->get_block_responses();
    uint64_t nTxs = 0;

    for (uint64_t block=0; block<block_responses.size(); block++)
    {
        executor::v1::ProcessBlockResponseV2 * pProcessBlockResponse = response->add_block_responses();

        pProcessBlockResponse->set_block_hash(string2ba(block_responses[block].block_hash));
        pProcessBlockResponse->set_block_hash_l1(string2ba(block_responses[block].block_hash_l1));
        pProcessBlockResponse->set_block_info_root(string2ba(block_responses[block].block_info_root));
        pProcessBlockResponse->set_block_number(block_responses[block].block_number);
        pProcessBlockResponse->set_coinbase(block_responses[block].coinbase);
        pProcessBlockResponse->set_gas_limit(block_responses[block].gas_limit);
        pProcessBlockResponse->set_gas_used(block_responses[block].gas_used);
        pProcessBlockResponse->set_ger(string2ba(block_responses[block].ger));
        pProcessBlockResponse->set_parent_hash(string2ba(block_responses[block].parent_hash));
        pProcessBlockResponse->set_timestamp(block_responses[block].timestamp);
        pProcessBlockResponse->set_error(string2error(block_responses[block].error));

        for (uint64_t log=0; log<block_responses[block].logs.size(); log++)
        {
            executor::v1::LogV2 * pLog = pProcessBlockResponse->add_logs();
            pLog->set_address(block_responses[block].logs[log].address); // Address of the contract that generated the event
            for (uint64_t topic=0; topic<block_responses[block].logs[log].topics.size(); topic++)
            {
                std::string * pTopic = pLog->add_topics();
                *pTopic = string2ba(block_responses[block].logs[log].topics[topic]); // List of topics provided by the contract
            }
            string dataConcatenated;
            for (uint64_t data=0; data<block_responses[block].logs[log].data.size(); data++)
                dataConcatenated += block_responses[block].logs[log].data[data];
            pLog->set_data(string2ba(dataConcatenated)); // Supplied by the contract, usually ABI-encoded
            //pLog->set_batch_number(block_responses[tblockx].logs[log].batch_number); // Batch in which the transaction was included
            pLog->set_tx_hash(string2ba(block_responses[block].logs[log].tx_hash)); // Hash of the transaction
            pLog->set_tx_hash_l2(string2ba(block_responses[block].logs[log].tx_hash_l2)); // Hash of the transaction in layer 2
            pLog->set_tx_index(block_responses[block].logs[log].tx_index); // Index of the transaction in the block
            //pLog->set_batch_hash(string2ba(block_responses[block].logs[log].batch_hash)); // Hash of the batch in which the transaction was included
            pLog->set_index(block_responses[block].logs[log].index); // Index of the log in the block
        }

        vector<ResponseV2> &responses = block_responses[block].responses;
        nTxs += responses.size();

        for (uint64_t tx=0; tx<block_responses[block].responses.size(); tx++)
        {
            // Remember the previous memory sent for each TX, and send only increments
            string previousMemory;

            executor::v1::ProcessTransactionResponseV2 * pProcessTransactionResponse = pProcessBlockResponse->add_responses();

            //executor::v1::ProcessTransactionResponse * pProcessTransactionResponse = response->add_responses();
            pProcessTransactionResponse->set_tx_hash(string2ba(responses[tx].tx_hash));
            pProcessTransactionResponse->set_tx_hash_l2(string2ba(responses[tx].tx_hash_l2));
            pProcessTransactionResponse->set_rlp_tx(responses[tx].rlp_tx);
            pProcessTransactionResponse->set_type(responses[tx].type); // Type indicates legacy transaction; it will be always 0 (legacy) in the executor
            pProcessTransactionResponse->set_return_value(string2ba(responses[tx].return_value)); // Returned data from the runtime (function result or data supplied with revert opcode)
            pProcessTransactionResponse->set_gas_left(responses[tx].gas_left); // Total gas left as result of execution
            pProcessTransactionResponse->set_gas_used(responses[tx].gas_used); // Total gas used as result of execution or gas estimation
            pProcessTransactionResponse->set_gas_refunded(responses[tx].gas_refunded); // Total gas refunded as result of execution
            pProcessTransactionResponse->set_error(string2error(responses[tx].error)); // Any error encountered during the execution
            pProcessTransactionResponse->set_create_address(responses[tx].create_address); // New SC Address in case of SC creation
            pProcessTransactionResponse->set_state_root(string2ba(responses[tx].state_root));
            pProcessTransactionResponse->set_status(responses[tx].status);
            pProcessTransactionResponse->set_effective_percentage(responses[tx].effective_percentage);
            pProcessTransactionResponse->set_effective_gas_price(responses[tx].effective_gas_price);
            pProcessTransactionResponse->set_has_balance_opcode(responses[tx].has_balance_opcode);
            pProcessTransactionResponse->set_has_gasprice_opcode(responses[tx].has_gasprice_opcode);
            pProcessTransactionResponse->set_cumulative_gas_used(responses[tx].cumulative_gas_used);
            
            for (uint64_t log=0; log<responses[tx].logs.size(); log++)
            {
                executor::v1::LogV2 * pLog = pProcessTransactionResponse->add_logs();
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
                //pLog->set_batch_number(responses[tx].logs[log].batch_number); // Batch in which the transaction was included
                pLog->set_tx_hash(string2ba(responses[tx].logs[log].tx_hash)); // Hash of the transaction
                pLog->set_tx_index(responses[tx].logs[log].tx_index); // Index of the transaction in the block
                //pLog->set_batch_hash(string2ba(responses[tx].logs[log].batch_hash)); // Hash of the batch in which the transaction was included
                pLog->set_index(responses[tx].logs[log].index); // Index of the log in the block
            }
            if (proverRequest.input.traceConfig.bEnabled && (proverRequest.input.traceConfig.txHashToGenerateFullTrace == responses[tx].tx_hash))
            {
                executor::v1::FullTraceV2 * pFullTrace = new executor::v1::FullTraceV2();
                executor::v1::TransactionContextV2 * pTransactionContext = pFullTrace->mutable_context();
                pTransactionContext->set_type(responses[tx].full_trace.context.type); // "CALL" or "CREATE"
                pTransactionContext->set_from(responses[tx].full_trace.context.from); // Sender of the transaction
                pTransactionContext->set_to(responses[tx].full_trace.context.to); // Target of the transaction
                pTransactionContext->set_data(string2ba(responses[tx].full_trace.context.data)); // Input data of the transaction
                pTransactionContext->set_gas(responses[tx].full_trace.context.gas);
                pTransactionContext->set_gas_price(Add0xIfMissing(responses[tx].full_trace.context.gas_price.get_str(16)));
                pTransactionContext->set_value(Add0xIfMissing(responses[tx].full_trace.context.value.get_str(16)));
                //pTransactionContext->set_batch(string2ba(responses[tx].full_trace.context.batch)); // Hash of the batch in which the transaction was included
                pTransactionContext->set_output(string2ba(responses[tx].full_trace.context.output)); // Returned data from the runtime (function result or data supplied with revert opcode)
                pTransactionContext->set_gas_used(responses[tx].full_trace.context.gas_used); // Total gas used as result of execution
                pTransactionContext->set_execution_time(responses[tx].full_trace.context.execution_time);
                pTransactionContext->set_old_state_root(string2ba(responses[tx].full_trace.context.old_state_root)); // Starting state root
                pTransactionContext->set_chain_id(responses[tx].full_trace.context.chainId);
                pTransactionContext->set_tx_index(responses[tx].full_trace.context.txIndex);
                for (uint64_t step=0; step<responses[tx].full_trace.steps.size(); step++)
                {
                    executor::v1::TransactionStepV2 * pTransactionStep = pFullTrace->add_steps();
                    pTransactionStep->set_state_root(string2ba(responses[tx].full_trace.steps[step].state_root));
                    pTransactionStep->set_depth(responses[tx].full_trace.steps[step].depth); // Call depth
                    pTransactionStep->set_pc(responses[tx].full_trace.steps[step].pc); // Program counter
                    pTransactionStep->set_gas(responses[tx].full_trace.steps[step].gas); // Remaining gas
                    pTransactionStep->set_gas_cost(responses[tx].full_trace.steps[step].gas_cost); // Gas cost of the operation
                    pTransactionStep->set_gas_refund(responses[tx].full_trace.steps[step].gas_refund); // Gas refunded during the operation
                    pTransactionStep->set_op(responses[tx].full_trace.steps[step].op); // Opcode
                    for (uint64_t stack=0; stack<responses[tx].full_trace.steps[step].stack.size() ; stack++)
                        pTransactionStep->add_stack(responses[tx].full_trace.steps[step].stack[stack].get_str(16)); // Content of the stack
                    pTransactionStep->set_memory_size(responses[tx].full_trace.steps[step].memory_size);
                    pTransactionStep->set_memory_offset(responses[tx].full_trace.steps[step].memory_offset);
                    pTransactionStep->set_memory(responses[tx].full_trace.steps[step].memory);
                    string dataConcatenated;
                    for (uint64_t data=0; data<responses[tx].full_trace.steps[step].return_data.size(); data++)
                        dataConcatenated += responses[tx].full_trace.steps[step].return_data[data];
                    pTransactionStep->set_return_data(string2ba(dataConcatenated));
                    executor::v1::ContractV2 * pContract = pTransactionStep->mutable_contract(); // Contract information
                    pContract->set_address(responses[tx].full_trace.steps[step].contract.address);
                    pContract->set_caller(responses[tx].full_trace.steps[step].contract.caller);
                    pContract->set_value(Add0xIfMissing(responses[tx].full_trace.steps[step].contract.value.get_str(16)));
                    pContract->set_data(string2ba(responses[tx].full_trace.steps[step].contract.data));
                    pContract->set_gas(responses[tx].full_trace.steps[step].contract.gas);
                    pContract->set_type(responses[tx].full_trace.steps[step].contract.type);
                    pTransactionStep->set_error(string2error(responses[tx].full_trace.steps[step].error));

                    google::protobuf::Map<std::string, std::string> * pStorage = pTransactionStep->mutable_storage();
                    unordered_map<string,string>::iterator it;
                    for (it=responses[tx].full_trace.steps[step].storage.begin(); it!=responses[tx].full_trace.steps[step].storage.end(); it++)
                        (*pStorage)[it->first] = it->second; // Content of the storage
                }
                pProcessTransactionResponse->set_allocated_full_trace(pFullTrace);
            }
        }
    }

    // Return accessed keys, if requested
    if (proverRequest.input.bGetKeys)
    {
        unordered_set<string>::const_iterator it;
        for (it = proverRequest.nodesKeys.begin(); it != proverRequest.nodesKeys.end(); it++)
        {
            response->add_smt_keys(string2ba(it->c_str()));
        }
        for (it = proverRequest.programKeys.begin(); it != proverRequest.programKeys.end(); it++)
        {
            response->add_program_keys(string2ba(it->c_str()));
        }
    }

#ifdef LOG_SERVICE_EXECUTOR_OUTPUT
    {
        string s = "ExecutorServiceImpl::ProcessBatchV2() returns result=" + to_string(response->error()) +
            " old_state_root=" + proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot.get_str(16) +
            " new_state_root=" + proverRequest.pFullTracer->get_new_state_root() +
            " new_acc_input_hash=" + proverRequest.pFullTracer->get_new_acc_input_hash() +
            " new_local_exit_root=" + proverRequest.pFullTracer->get_new_local_exit_root() +
            " old_batch_num=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.oldBatchNum) +
            " steps=" + to_string(proverRequest.counters.steps) +
            " gasUsed=" + to_string(proverRequest.pFullTracer->get_gas_used()) +
            " counters.keccakF=" + to_string(proverRequest.counters.keccakF) +
            " counters.poseidonG=" + to_string(proverRequest.counters.poseidonG) +
            " counters.paddingPG=" + to_string(proverRequest.counters.paddingPG) +
            " counters.memAlign=" + to_string(proverRequest.counters.memAlign) +
            " counters.arith=" + to_string(proverRequest.counters.arith) +
            " counters.binary=" + to_string(proverRequest.counters.binary) +
            " counters.sha256F=" + to_string(proverRequest.counters.sha256F) +
            " flush_id=" + to_string(proverRequest.flushId) +
            " last_sent_flush_id=" + to_string(proverRequest.lastSentFlushId) +
            " nBlocks=" + to_string(block_responses.size()) +
            " nTxs=" + to_string(nTxs);
         if (config.logExecutorServerTxs)
         {
            for (uint64_t block=0; block<block_responses.size(); block++)
            {
                s += " block[" + to_string(block) + "].hash=" + block_responses[block].block_hash + " blockNumber=" + to_string(block_responses[block].block_number);
                vector<ResponseV2> &responses = block_responses[block].responses;
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
         }
        zklog.info(s, &proverRequest.tags);
    }
#endif

    if (config.logExecutorServerResponses)
    {
        zklog.info("ExecutorServiceImpl::ProcessBatchV2() returns:\n" + response->DebugString(), &proverRequest.tags);
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
    uint64_t execGas = response->gas_used();
    totalGas += execGas;
    uint64_t execBytes = request->batch_l2_data().size();
    totalBytes += execBytes;
    uint64_t execTX = nTxs;
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

    zklog.info("ExecutorServiceImpl::ProcessBatchV2() done counter=" + to_string(counter) + " B=" + to_string(execBytes) + " TX=" + to_string(execTX) + " gas=" + to_string(execGas) + " time=" + to_string(execTime) +
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


::grpc::Status ExecutorServiceImpl::ProcessStatelessBatchV2 (::grpc::ServerContext* context, const ::executor::v1::ProcessStatelessBatchRequestV2* request, ::executor::v1::ProcessBatchResponseV2* response)
{
    zkresult zkr;

    // If the process is exiting, do not start new activities
    if (bExitingProcess)
    {
        return Status::CANCELLED;
    }

    //TimerStart(EXECUTOR_PROCESS_BATCH);
    struct timeval EXECUTOR_PROCESS_BATCH_start;
    gettimeofday(&EXECUTOR_PROCESS_BATCH_start,NULL);

#ifdef LOG_SERVICE
    zklog.info("ExecutorServiceImpl::ProcessStatelessBatchV2() got request:\n" + request->DebugString());
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

    // Get witness
    proverRequest.input.publicInputsExtended.publicInputs.witness = request->witness();
    if (proverRequest.input.publicInputsExtended.publicInputs.witness.empty())
    {
        zklog.error("ExecutorServiceImpl::ProcessStatelessBatchV2() got an empty witness", &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_WITNESS);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    //zklog.info("witness.size=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.witness.size()) + " witness=0x" + ba2string(proverRequest.input.publicInputsExtended.publicInputs.witness.substr(0, 10)) + "..." + ba2string(proverRequest.input.publicInputsExtended.publicInputs.witness.substr(zkmax(int64_t(0),int64_t(proverRequest.input.publicInputsExtended.publicInputs.witness.size())-10), proverRequest.input.publicInputsExtended.publicInputs.witness.size())));

    // Parse witness and get db, programs and old state root
    zkr = witness2db(proverRequest.input.publicInputsExtended.publicInputs.witness, proverRequest.input.db, proverRequest.input.contractsBytecode, proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("ExecutorServiceImpl::ProcessStatelessBatchV2() failed calling witness2db() result=" + zkresult2string(zkr), &proverRequest.tags);
        response->set_error(zkresult2error(zkr));
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }

    // Get data stream
    proverRequest.input.publicInputsExtended.publicInputs.dataStream = request->data_stream();
    if (proverRequest.input.publicInputsExtended.publicInputs.dataStream.empty())
    {
        zklog.error("ExecutorServiceImpl::ProcessStatelessBatchV2() got an empty data stream", &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DATA_STREAM);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }

    // Parse data stream and get a binary structure
    DataStreamBatch batch;
    zkr = dataStream2batch(proverRequest.input.publicInputsExtended.publicInputs.dataStream, batch);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("ExecutorServiceImpl::ProcessStatelessBatchV2() failed calling dataStream2batch() result=" + zkresult2string(zkr), &proverRequest.tags);
        response->set_error(zkresult2error(zkr));
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    if (batch.blocks.empty())
    {
        zklog.error("ExecutorServiceImpl::ProcessStatelessBatchV2() called dataStream2batch() but got zero blocks", &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DATA_STREAM);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }

    // Get batchL2Data
    zkr = dataStreamBatch2batchL2Data(batch, proverRequest.input.publicInputsExtended.publicInputs.batchL2Data);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("ExecutorServiceImpl::ProcessStatelessBatchV2() failed calling dataStreamBatch2batchL2Data() result=" + zkresult2string(zkr), &proverRequest.tags);
        response->set_error(zkresult2error(zkr));
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    if (proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.size() > MAX_BATCH_L2_DATA_SIZE)
    {
        zklog.error("ExecutorServiceImpl::ProcessBatchV2() found batchL2Data.size()=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.size()) + " > MAX_BATCH_L2_DATA_SIZE=" + to_string(MAX_BATCH_L2_DATA_SIZE), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_BATCH_L2_DATA);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }

    // Get oldAccInputHash
    if (request->old_acc_input_hash().size() > 32)
    {
        zklog.error("ExecutorServiceImpl::ProcessStatelessBatchV2() got oldAccInputHash too long, size=" + to_string(request->old_acc_input_hash().size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_OLD_ACC_INPUT_HASH);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    ba2scalar(proverRequest.input.publicInputsExtended.publicInputs.oldAccInputHash, request->old_acc_input_hash());

    // Get old batch number
    if (batch.batchNumber == 0)
    {
        zklog.error("ExecutorServiceImpl::ProcessStatelessBatchV2() called dataStream2batch() but got batch.batchNumber=0", &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_DATA_STREAM);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    proverRequest.input.publicInputsExtended.publicInputs.oldBatchNum = batch.batchNumber - 1;

    // Get chain ID
    proverRequest.input.publicInputsExtended.publicInputs.chainID = batch.chainId;
    if (proverRequest.input.publicInputsExtended.publicInputs.chainID == 0)
    {
        zklog.error("ExecutorServiceImpl::ProcessStatelessBatchV2() got chainID = 0", &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_CHAIN_ID);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }

    // Get fork ID
    proverRequest.input.publicInputsExtended.publicInputs.forkID = batch.forkId;
    if (proverRequest.input.publicInputsExtended.publicInputs.forkID < 7)
    {
        zklog.error("ExecutorServiceImpl::ProcessStatelessBatchV2() got invalid fork ID =" + to_string(proverRequest.input.publicInputsExtended.publicInputs.forkID), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_UNSUPPORTED_FORK_ID);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }

    // Create full tracer based on fork ID
    proverRequest.CreateFullTracer();
    if (proverRequest.result != ZKR_SUCCESS)
    {
        response->set_error(zkresult2error(proverRequest.result));
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }

    // Get L1 info root
    if (request->l1_info_root().size() > 32)
    {
        zklog.error("ExecutorServiceImpl::ProcessStatelessBatchV2() got l1_info_root too long, size=" + to_string(request->l1_info_root().size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_L1_INFO_ROOT);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    ba2scalar(proverRequest.input.publicInputsExtended.publicInputs.l1InfoRoot, request->l1_info_root());

    // Get forced block hash L1
    if (request->forced_blockhash_l1().size() > 32)
    {
        zklog.error("ExecutorServiceImpl::ProcessStatelessBatchV2() got forced_blockhash_l1 too long, size=" + to_string(request->forced_blockhash_l1().size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_FORCED_BLOCKHASH_L1);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    ba2scalar(proverRequest.input.publicInputsExtended.publicInputs.forcedBlockHashL1, request->forced_blockhash_l1());

    // Get timestamp limit
    proverRequest.input.publicInputsExtended.publicInputs.timestampLimit = request->timestamp_limit();

    // Get sequencer address
    string auxString = Remove0xIfPresent(request->coinbase());
    if (auxString.size() > 40)
    {
        zklog.error("ExecutorServiceImpl::ProcessStatelessBatchV2() got sequencer address too long, size=" + to_string(auxString.size()), &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_COINBASE);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    if (!stringIsHex(auxString))
    {
        zklog.error("ExecutorServiceImpl::ProcessStatelessBatchV2() got sequencer address not hex, coinbase=" + auxString, &proverRequest.tags);
        response->set_error(executor::v1::EXECUTOR_ERROR_INVALID_COINBASE);
        //TimerStopAndLog(EXECUTOR_PROCESS_BATCH);
        return Status::OK;
    }
    proverRequest.input.publicInputsExtended.publicInputs.sequencerAddr.set_str(auxString, 16);

    // ROOT

    // Leave from empty
    //proverRequest.input.from = "0x";

    // Flags
    proverRequest.input.bUpdateMerkleTree = true;
    proverRequest.input.bNoCounters = false;
    proverRequest.input.bGetKeys = false;
    proverRequest.input.bSkipVerifyL1InfoRoot = true;
    proverRequest.input.bSkipFirstChangeL2Block = false;
    proverRequest.input.bSkipWriteBlockInfoRoot = false;

    // Default values
    proverRequest.input.publicInputsExtended.newStateRoot = "0x0";
    proverRequest.input.publicInputsExtended.newAccInputHash = "0x0";
    proverRequest.input.publicInputsExtended.newLocalExitRoot = "0x0";
    proverRequest.input.publicInputsExtended.newBatchNum = 0;

#ifdef LOG_SERVICE_EXECUTOR_INPUT
    string l1InfoTreeDataString = " l1InfoTreeData.size=" + to_string(proverRequest.input.l1InfoTreeData.size()) + "=";
    unordered_map<uint64_t, L1Data>::const_iterator itl1;
    for (itl1 = proverRequest.input.l1InfoTreeData.begin(); itl1 != proverRequest.input.l1InfoTreeData.end(); itl1++)
    {
        l1InfoTreeDataString += to_string(itl1->first) + ",";
    }
    zklog.info(string("ExecutorServiceImpl::ProcessStatelessBatchV2() got") +
        " sequencerAddr=" + proverRequest.input.publicInputsExtended.publicInputs.sequencerAddr.get_str(16) +
        " batchL2DataLength=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.size()) +
        " batchL2Data=0x" + ba2string(proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.substr(0, 10)) + "..." + ba2string(proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.substr(zkmax(int64_t(0),int64_t(proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.size())-10), proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.size())) +
        " oldStateRoot=" + proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot.get_str(16) +
        " oldAccInputHash=" + proverRequest.input.publicInputsExtended.publicInputs.oldAccInputHash.get_str(16) +
        " oldBatchNum=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.oldBatchNum) +
        " chainId=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.chainID) +
        " forkId=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.forkID) +
            (((proverRequest.input.publicInputsExtended.publicInputs.forkID >= 7) && config.useMainExecC) ? " C" :
#ifdef MAIN_SM_EXECUTOR_GENERATED_CODE
             ((proverRequest.input.publicInputsExtended.publicInputs.forkID >= 4) && config.useMainExecGenerated) ? " generated" :
#endif
             " native") +
        " globalExitRoot=" + proverRequest.input.publicInputsExtended.publicInputs.globalExitRoot.get_str(16) +
        " timestampLimit=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.timestampLimit) +
        " from=" + proverRequest.input.from +
        " bUpdateMerkleTree=" + to_string(proverRequest.input.bUpdateMerkleTree) +
        " bNoCounters=" + to_string(proverRequest.input.bNoCounters) +
        " bGetKeys=" + to_string(proverRequest.input.bGetKeys) +
        " bSkipVerifyL1InfoRoot=" + to_string(proverRequest.input.bSkipVerifyL1InfoRoot) +
        " bSkipFirstChangeL2Block=" + to_string(proverRequest.input.bSkipFirstChangeL2Block) +
        " bSkipWriteBlockInfoRoot=" + to_string(proverRequest.input.bSkipWriteBlockInfoRoot) +
        " traceConfig=" + proverRequest.input.traceConfig.toString() +
        " UUID=" + proverRequest.uuid +
        " gasLimit=" + to_string(proverRequest.input.debug.gasLimit) +
        l1InfoTreeDataString +
        " stateOverride.size=" + to_string(proverRequest.input.stateOverride.size())
        , &proverRequest.tags);
#endif

    if (config.logExecutorServerInputJson)
    {
        // Log the input file content
        json inputJson;
        proverRequest.input.save(inputJson);
        string inputJsonString = inputJson.dump();
        replace(inputJsonString.begin(), inputJsonString.end(), '"', '\'');
        zklog.info("ExecutorServiceImpl::ProcessStatelessBatchV2() Input=" + inputJsonString, &proverRequest.tags);
    }

    prover.processBatch(&proverRequest);

    //TimerStart(EXECUTOR_PROCESS_BATCH_BUILD_RESPONSE);

    if (proverRequest.result != ZKR_SUCCESS)
    {
        zklog.error("ExecutorServiceImpl::ProcessStatelessBatchV2() detected proverRequest.result=" + to_string(proverRequest.result) + "=" + zkresult2string(proverRequest.result), &proverRequest.tags);
    }
    
    response->set_error(zkresult2error(proverRequest.result));
    response->set_gas_used(proverRequest.pFullTracer->get_gas_used());

    response->set_cnt_keccak_hashes(proverRequest.counters.keccakF);
    response->set_cnt_poseidon_hashes(proverRequest.counters.poseidonG);
    response->set_cnt_poseidon_paddings(proverRequest.counters.paddingPG);
    response->set_cnt_mem_aligns(proverRequest.counters.memAlign);
    response->set_cnt_arithmetics(proverRequest.counters.arith);
    response->set_cnt_binaries(proverRequest.counters.binary);
    response->set_cnt_sha256_hashes(proverRequest.counters.sha256F);
    response->set_cnt_steps(proverRequest.counters.steps);
    
    response->set_cnt_reserve_keccak_hashes(proverRequest.counters_reserve.keccakF);
    response->set_cnt_reserve_poseidon_hashes(proverRequest.counters_reserve.poseidonG);
    response->set_cnt_reserve_poseidon_paddings(proverRequest.counters_reserve.paddingPG);
    response->set_cnt_reserve_mem_aligns(proverRequest.counters_reserve.memAlign);
    response->set_cnt_reserve_arithmetics(proverRequest.counters_reserve.arith);
    response->set_cnt_reserve_binaries(proverRequest.counters_reserve.binary);
    response->set_cnt_reserve_sha256_hashes(proverRequest.counters_reserve.sha256F);
    response->set_cnt_reserve_steps(proverRequest.counters_reserve.steps);

    response->set_new_state_root(string2ba(proverRequest.pFullTracer->get_new_state_root()));
    response->set_new_acc_input_hash(string2ba(proverRequest.pFullTracer->get_new_acc_input_hash()));
    response->set_new_local_exit_root(string2ba(proverRequest.pFullTracer->get_new_local_exit_root()));
    response->set_flush_id(proverRequest.flushId);
    response->set_stored_flush_id(proverRequest.lastSentFlushId);
    response->set_prover_id(config.proverID);
    response->set_fork_id(proverRequest.input.publicInputsExtended.publicInputs.forkID);
    response->set_error_rom(string2error(proverRequest.pFullTracer->get_error()));
    response->set_invalid_batch(proverRequest.pFullTracer->get_invalid_batch());
    
    unordered_map<string, InfoReadWrite> * p_read_write_addresses = proverRequest.pFullTracer->get_read_write_addresses();
    if (p_read_write_addresses != NULL)
    {
        unordered_map<string, InfoReadWrite>::const_iterator itRWA;
        for (itRWA=p_read_write_addresses->begin(); itRWA != p_read_write_addresses->end(); itRWA++)
        {
            executor::v1::InfoReadWriteV2 infoReadWrite;
            google::protobuf::Map<std::string, executor::v1::InfoReadWriteV2> * pReadWriteAddresses = response->mutable_read_write_addresses();
            infoReadWrite.set_balance(itRWA->second.balance);
            infoReadWrite.set_nonce(itRWA->second.nonce);
            (*pReadWriteAddresses)[itRWA->first] = infoReadWrite;
        }
    }

    vector<Block> &block_responses = proverRequest.pFullTracer->get_block_responses();
    uint64_t nTxs = 0;

    for (uint64_t block=0; block<block_responses.size(); block++)
    {
        executor::v1::ProcessBlockResponseV2 * pProcessBlockResponse = response->add_block_responses();

        pProcessBlockResponse->set_block_hash(string2ba(block_responses[block].block_hash));
        pProcessBlockResponse->set_block_hash_l1(string2ba(block_responses[block].block_hash_l1));
        pProcessBlockResponse->set_block_info_root(string2ba(block_responses[block].block_info_root));
        pProcessBlockResponse->set_block_number(block_responses[block].block_number);
        pProcessBlockResponse->set_coinbase(block_responses[block].coinbase);
        pProcessBlockResponse->set_gas_limit(block_responses[block].gas_limit);
        pProcessBlockResponse->set_gas_used(block_responses[block].gas_used);
        pProcessBlockResponse->set_ger(string2ba(block_responses[block].ger));
        pProcessBlockResponse->set_parent_hash(string2ba(block_responses[block].parent_hash));
        pProcessBlockResponse->set_timestamp(block_responses[block].timestamp);
        pProcessBlockResponse->set_error(string2error(block_responses[block].error));

        for (uint64_t log=0; log<block_responses[block].logs.size(); log++)
        {
            executor::v1::LogV2 * pLog = pProcessBlockResponse->add_logs();
            pLog->set_address(block_responses[block].logs[log].address); // Address of the contract that generated the event
            for (uint64_t topic=0; topic<block_responses[block].logs[log].topics.size(); topic++)
            {
                std::string * pTopic = pLog->add_topics();
                *pTopic = string2ba(block_responses[block].logs[log].topics[topic]); // List of topics provided by the contract
            }
            string dataConcatenated;
            for (uint64_t data=0; data<block_responses[block].logs[log].data.size(); data++)
                dataConcatenated += block_responses[block].logs[log].data[data];
            pLog->set_data(string2ba(dataConcatenated)); // Supplied by the contract, usually ABI-encoded
            //pLog->set_batch_number(block_responses[tblockx].logs[log].batch_number); // Batch in which the transaction was included
            pLog->set_tx_hash(string2ba(block_responses[block].logs[log].tx_hash)); // Hash of the transaction
            pLog->set_tx_hash_l2(string2ba(block_responses[block].logs[log].tx_hash_l2)); // Hash of the transaction in layer 2
            pLog->set_tx_index(block_responses[block].logs[log].tx_index); // Index of the transaction in the block
            //pLog->set_batch_hash(string2ba(block_responses[block].logs[log].batch_hash)); // Hash of the batch in which the transaction was included
            pLog->set_index(block_responses[block].logs[log].index); // Index of the log in the block
        }

        vector<ResponseV2> &responses = block_responses[block].responses;
        nTxs += responses.size();

        for (uint64_t tx=0; tx<block_responses[block].responses.size(); tx++)
        {
            // Remember the previous memory sent for each TX, and send only increments
            string previousMemory;

            executor::v1::ProcessTransactionResponseV2 * pProcessTransactionResponse = pProcessBlockResponse->add_responses();

            //executor::v1::ProcessTransactionResponse * pProcessTransactionResponse = response->add_responses();
            pProcessTransactionResponse->set_tx_hash(string2ba(responses[tx].tx_hash));
            pProcessTransactionResponse->set_tx_hash_l2(string2ba(responses[tx].tx_hash_l2));
            pProcessTransactionResponse->set_rlp_tx(responses[tx].rlp_tx);
            pProcessTransactionResponse->set_type(responses[tx].type); // Type indicates legacy transaction; it will be always 0 (legacy) in the executor
            pProcessTransactionResponse->set_return_value(string2ba(responses[tx].return_value)); // Returned data from the runtime (function result or data supplied with revert opcode)
            pProcessTransactionResponse->set_gas_left(responses[tx].gas_left); // Total gas left as result of execution
            pProcessTransactionResponse->set_gas_used(responses[tx].gas_used); // Total gas used as result of execution or gas estimation
            pProcessTransactionResponse->set_gas_refunded(responses[tx].gas_refunded); // Total gas refunded as result of execution
            pProcessTransactionResponse->set_error(string2error(responses[tx].error)); // Any error encountered during the execution
            pProcessTransactionResponse->set_create_address(responses[tx].create_address); // New SC Address in case of SC creation
            pProcessTransactionResponse->set_state_root(string2ba(responses[tx].state_root));
            pProcessTransactionResponse->set_status(responses[tx].status);
            pProcessTransactionResponse->set_effective_percentage(responses[tx].effective_percentage);
            pProcessTransactionResponse->set_effective_gas_price(responses[tx].effective_gas_price);
            pProcessTransactionResponse->set_has_balance_opcode(responses[tx].has_balance_opcode);
            pProcessTransactionResponse->set_has_gasprice_opcode(responses[tx].has_gasprice_opcode);
            pProcessTransactionResponse->set_cumulative_gas_used(responses[tx].cumulative_gas_used);
            
            for (uint64_t log=0; log<responses[tx].logs.size(); log++)
            {
                executor::v1::LogV2 * pLog = pProcessTransactionResponse->add_logs();
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
                //pLog->set_batch_number(responses[tx].logs[log].batch_number); // Batch in which the transaction was included
                pLog->set_tx_hash(string2ba(responses[tx].logs[log].tx_hash)); // Hash of the transaction
                pLog->set_tx_index(responses[tx].logs[log].tx_index); // Index of the transaction in the block
                //pLog->set_batch_hash(string2ba(responses[tx].logs[log].batch_hash)); // Hash of the batch in which the transaction was included
                pLog->set_index(responses[tx].logs[log].index); // Index of the log in the block
            }
            if (proverRequest.input.traceConfig.bEnabled && (proverRequest.input.traceConfig.txHashToGenerateFullTrace == responses[tx].tx_hash))
            {
                executor::v1::FullTraceV2 * pFullTrace = new executor::v1::FullTraceV2();
                executor::v1::TransactionContextV2 * pTransactionContext = pFullTrace->mutable_context();
                pTransactionContext->set_type(responses[tx].full_trace.context.type); // "CALL" or "CREATE"
                pTransactionContext->set_from(responses[tx].full_trace.context.from); // Sender of the transaction
                pTransactionContext->set_to(responses[tx].full_trace.context.to); // Target of the transaction
                pTransactionContext->set_data(string2ba(responses[tx].full_trace.context.data)); // Input data of the transaction
                pTransactionContext->set_gas(responses[tx].full_trace.context.gas);
                pTransactionContext->set_gas_price(Add0xIfMissing(responses[tx].full_trace.context.gas_price.get_str(16)));
                pTransactionContext->set_value(Add0xIfMissing(responses[tx].full_trace.context.value.get_str(16)));
                //pTransactionContext->set_batch(string2ba(responses[tx].full_trace.context.batch)); // Hash of the batch in which the transaction was included
                pTransactionContext->set_output(string2ba(responses[tx].full_trace.context.output)); // Returned data from the runtime (function result or data supplied with revert opcode)
                pTransactionContext->set_gas_used(responses[tx].full_trace.context.gas_used); // Total gas used as result of execution
                pTransactionContext->set_execution_time(responses[tx].full_trace.context.execution_time);
                pTransactionContext->set_old_state_root(string2ba(responses[tx].full_trace.context.old_state_root)); // Starting state root
                pTransactionContext->set_chain_id(responses[tx].full_trace.context.chainId);
                pTransactionContext->set_tx_index(responses[tx].full_trace.context.txIndex);
                for (uint64_t step=0; step<responses[tx].full_trace.steps.size(); step++)
                {
                    executor::v1::TransactionStepV2 * pTransactionStep = pFullTrace->add_steps();
                    pTransactionStep->set_state_root(string2ba(responses[tx].full_trace.steps[step].state_root));
                    pTransactionStep->set_depth(responses[tx].full_trace.steps[step].depth); // Call depth
                    pTransactionStep->set_pc(responses[tx].full_trace.steps[step].pc); // Program counter
                    pTransactionStep->set_gas(responses[tx].full_trace.steps[step].gas); // Remaining gas
                    pTransactionStep->set_gas_cost(responses[tx].full_trace.steps[step].gas_cost); // Gas cost of the operation
                    pTransactionStep->set_gas_refund(responses[tx].full_trace.steps[step].gas_refund); // Gas refunded during the operation
                    pTransactionStep->set_op(responses[tx].full_trace.steps[step].op); // Opcode
                    for (uint64_t stack=0; stack<responses[tx].full_trace.steps[step].stack.size() ; stack++)
                        pTransactionStep->add_stack(responses[tx].full_trace.steps[step].stack[stack].get_str(16)); // Content of the stack
                    pTransactionStep->set_memory_size(responses[tx].full_trace.steps[step].memory_size);
                    pTransactionStep->set_memory_offset(responses[tx].full_trace.steps[step].memory_offset);
                    pTransactionStep->set_memory(responses[tx].full_trace.steps[step].memory);
                    string dataConcatenated;
                    for (uint64_t data=0; data<responses[tx].full_trace.steps[step].return_data.size(); data++)
                        dataConcatenated += responses[tx].full_trace.steps[step].return_data[data];
                    pTransactionStep->set_return_data(string2ba(dataConcatenated));
                    executor::v1::ContractV2 * pContract = pTransactionStep->mutable_contract(); // Contract information
                    pContract->set_address(responses[tx].full_trace.steps[step].contract.address);
                    pContract->set_caller(responses[tx].full_trace.steps[step].contract.caller);
                    pContract->set_value(Add0xIfMissing(responses[tx].full_trace.steps[step].contract.value.get_str(16)));
                    pContract->set_data(string2ba(responses[tx].full_trace.steps[step].contract.data));
                    pContract->set_gas(responses[tx].full_trace.steps[step].contract.gas);
                    pContract->set_type(responses[tx].full_trace.steps[step].contract.type);
                    pTransactionStep->set_error(string2error(responses[tx].full_trace.steps[step].error));

                    google::protobuf::Map<std::string, std::string> * pStorage = pTransactionStep->mutable_storage();
                    unordered_map<string,string>::iterator it;
                    for (it=responses[tx].full_trace.steps[step].storage.begin(); it!=responses[tx].full_trace.steps[step].storage.end(); it++)
                        (*pStorage)[it->first] = it->second; // Content of the storage
                }
                pProcessTransactionResponse->set_allocated_full_trace(pFullTrace);
            }
        }
    }

    // Return accessed keys, if requested
    if (proverRequest.input.bGetKeys)
    {
        unordered_set<string>::const_iterator it;
        for (it = proverRequest.nodesKeys.begin(); it != proverRequest.nodesKeys.end(); it++)
        {
            response->add_smt_keys(string2ba(it->c_str()));
        }
        for (it = proverRequest.programKeys.begin(); it != proverRequest.programKeys.end(); it++)
        {
            response->add_program_keys(string2ba(it->c_str()));
        }
    }

#ifdef LOG_SERVICE_EXECUTOR_OUTPUT
    {
        string s = "ExecutorServiceImpl::ProcessStatelessBatchV2() returns result=" + to_string(response->error()) +
            " old_state_root=" + proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot.get_str(16) +
            " new_state_root=" + proverRequest.pFullTracer->get_new_state_root() +
            " new_acc_input_hash=" + proverRequest.pFullTracer->get_new_acc_input_hash() +
            " new_local_exit_root=" + proverRequest.pFullTracer->get_new_local_exit_root() +
            " old_batch_num=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.oldBatchNum) +
            " steps=" + to_string(proverRequest.counters.steps) +
            " gasUsed=" + to_string(proverRequest.pFullTracer->get_gas_used()) +
            " counters.keccakF=" + to_string(proverRequest.counters.keccakF) +
            " counters.poseidonG=" + to_string(proverRequest.counters.poseidonG) +
            " counters.paddingPG=" + to_string(proverRequest.counters.paddingPG) +
            " counters.memAlign=" + to_string(proverRequest.counters.memAlign) +
            " counters.arith=" + to_string(proverRequest.counters.arith) +
            " counters.binary=" + to_string(proverRequest.counters.binary) +
            " counters.sha256F=" + to_string(proverRequest.counters.sha256F) +
            " flush_id=" + to_string(proverRequest.flushId) +
            " last_sent_flush_id=" + to_string(proverRequest.lastSentFlushId) +
            " nBlocks=" + to_string(block_responses.size()) +
            " nTxs=" + to_string(nTxs);
         if (config.logExecutorServerTxs)
         {
            for (uint64_t block=0; block<block_responses.size(); block++)
            {
                s += " block[" + to_string(block) + "].hash=" + block_responses[block].block_hash + " blockNumber=" + to_string(block_responses[block].block_number);
                vector<ResponseV2> &responses = block_responses[block].responses;
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
         }
        zklog.info(s, &proverRequest.tags);
    }
#endif

    if (config.logExecutorServerResponses)
    {
        zklog.info("ExecutorServiceImpl::ProcessStatelessBatchV2() returns:\n" + response->DebugString(), &proverRequest.tags);
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
    uint64_t execGas = response->gas_used();
    totalGas += execGas;
    uint64_t execBytes = proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.size();
    totalBytes += execBytes;
    uint64_t execTX = nTxs;
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

    zklog.info("ExecutorServiceImpl::ProcessStatelessBatchV2() done counter=" + to_string(counter) + " B=" + to_string(execBytes) + " TX=" + to_string(execTX) + " gas=" + to_string(execGas) + " time=" + to_string(execTime) +
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