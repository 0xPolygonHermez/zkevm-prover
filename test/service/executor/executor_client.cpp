
#include <nlohmann/json.hpp>
#include "executor_client.hpp"
#include "hashdb_singleton.hpp"
#include "zkmax.hpp"
#include "check_tree.hpp"
#include "state_manager_64.hpp"

using namespace std;
using json = nlohmann::json;

ExecutorClient::ExecutorClient (Goldilocks &fr, const Config &config) :
    fr(fr),
    config(config)
{
    // Set channel option to receive large messages
    grpc::ChannelArguments channelArguments;
    channelArguments.SetMaxReceiveMessageSize(1024*1024*1024);

    // Create channel
    std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(config.executorClientHost + ":" + to_string(config.executorClientPort), grpc::InsecureChannelCredentials(), channelArguments);

    // Create stub (i.e. client)
    stub = new executor::v1::ExecutorService::Stub(channel);
}

ExecutorClient::~ExecutorClient()
{
    delete stub;
}

void ExecutorClient::runThread (void)
{
    // Allow service to initialize
    sleep(1);

    pthread_create(&t, NULL, executorClientThread, this);
}

int64_t ExecutorClient::waitForThread (void)
{
    void * pResult;
    pthread_join(t, &pResult);
    return (int64_t)pResult;
}

void ExecutorClient::runThreads (void)
{
    // Allow service to initialize
    sleep(1);

    for (uint64_t i=0; i<EXECUTOR_CLIENT_MULTITHREAD_N_THREADS; i++)
    {
        pthread_create(&threads[i], NULL, executorClientThreads, this);
    }
}

int64_t ExecutorClient::waitForThreads (void)
{
    int64_t iTotalResult = 0;
    for (uint64_t i=0; i<EXECUTOR_CLIENT_MULTITHREAD_N_THREADS; i++)
    {
        void * pResult;
        pthread_join(threads[i], &pResult);
        int64_t iResult = (int64_t)pResult;
        if (iResult != 0)
        {
            iTotalResult = iResult;
        }
    }
    return iTotalResult;
}

bool ExecutorClient::ProcessBatch (const string &inputFile)
{
    // Get a  HashDB interface
    HashDBInterface* pHashDB = HashDBClientFactory::createHashDBClient(fr, config);
    zkassertpermanent(pHashDB != NULL);

    TimerStart(EXECUTOR_CLIENT_PROCESS_BATCH);

    if (inputFile.size() == 0)
    {
        cerr << "Error: ExecutorClient::ProcessBatch() found inputFile empty" << endl;
        exit(-1);
    }

    Input input(fr);
    json inputJson;
    file2json(inputFile, inputJson);
    zkresult zkResult = input.load(inputJson);
    if (zkResult != ZKR_SUCCESS)
    {
        cerr << "Error: ExecutorClient::ProcessBatch() failed calling input.load() zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
        exit(-1);
    }

    // Flags
    bool update_merkle_tree = true;
    bool get_keys = false;
    bool no_counters = input.bNoCounters;
    if (input.stepsN > 0)
    {
        no_counters = true;
    }
    
    // Resulting new state root
    string newStateRoot;

    vector<string> blockStateRoots;

    if ((input.publicInputsExtended.publicInputs.forkID >= 1) &&
        (input.publicInputsExtended.publicInputs.forkID <= 6))
    {
        ::executor::v1::ProcessBatchRequest request;
        request.set_coinbase(Add0xIfMissing(input.publicInputsExtended.publicInputs.sequencerAddr.get_str(16)));
        request.set_batch_l2_data(input.publicInputsExtended.publicInputs.batchL2Data);
        request.set_old_state_root(scalar2ba(input.publicInputsExtended.publicInputs.oldStateRoot));
        request.set_old_acc_input_hash(scalar2ba(input.publicInputsExtended.publicInputs.oldAccInputHash));
        request.set_global_exit_root(scalar2ba(input.publicInputsExtended.publicInputs.globalExitRoot));
        request.set_eth_timestamp(input.publicInputsExtended.publicInputs.timestamp);
        request.set_update_merkle_tree(update_merkle_tree);
        request.set_chain_id(input.publicInputsExtended.publicInputs.chainID);
        request.set_fork_id(input.publicInputsExtended.publicInputs.forkID);
        request.set_from(input.from);
        request.set_no_counters(no_counters);
        if (input.traceConfig.bEnabled)
        {
            executor::v1::TraceConfig * pTraceConfig = request.mutable_trace_config();
            pTraceConfig->set_disable_storage(input.traceConfig.bDisableStorage);
            pTraceConfig->set_disable_stack(input.traceConfig.bDisableStack);
            pTraceConfig->set_enable_memory(input.traceConfig.bEnableMemory);
            pTraceConfig->set_enable_return_data(input.traceConfig.bEnableReturnData);
            pTraceConfig->set_tx_hash_to_generate_full_trace(string2ba(input.traceConfig.txHashToGenerateFullTrace));
        }
        request.set_old_batch_num(input.publicInputsExtended.publicInputs.oldBatchNum);

        // Parse keys map
        DatabaseMap::MTMap::const_iterator it;
        for (it=input.db.begin(); it!=input.db.end(); it++)
        {
            string key = NormalizeToNFormat(it->first, 64);
            string value;
            vector<Goldilocks::Element> dbValue = it->second;
            for (uint64_t i=0; i<dbValue.size(); i++)
            {
                value += NormalizeToNFormat(fr.toString(dbValue[i], 16), 16);
            }
            (*request.mutable_db())[key] = value;
        }

        // Parse contracts data
        DatabaseMap::ProgramMap::const_iterator itp;
        for (itp=input.contractsBytecode.begin(); itp!=input.contractsBytecode.end(); itp++)
        {
            string key = NormalizeToNFormat(itp->first, 64);
            string value;
            vector<uint8_t> contractValue = itp->second;
            for (uint64_t i=0; i<contractValue.size(); i++)
            {
                value += byte2string(contractValue[i]);
            }
            (*request.mutable_contracts_bytecode())[key] = value;
        }

        ::executor::v1::ProcessBatchResponse processBatchResponse;
        for (uint64_t i=0; i<config.executorClientLoops; i++)
        {
            if (config.executorClientResetDB)
            {
                pHashDB->resetDB();
            }
            else if (i == 1)
            {
                request.clear_db();
                request.clear_contracts_bytecode();
            }
            ::grpc::ClientContext context;
            ::grpc::Status grpcStatus = stub->ProcessBatch(&context, request, &processBatchResponse);
            if (grpcStatus.error_code() != grpc::StatusCode::OK)
            {
                cerr << "Error: ExecutorClient::ProcessBatch() failed calling server i=" << i << " error=" << grpcStatus.error_code() << "=" << grpcStatus.error_message() << endl;
                return false;
            }
            if (processBatchResponse.error() != executor::v1::EXECUTOR_ERROR_NO_ERROR)
            {
                cerr << "Error: ExecutorClient::ProcessBatch() failed i=" << i << " error=" << processBatchResponse.error() << endl;
                return false;
            }
            newStateRoot = ba2string(processBatchResponse.new_state_root());

    #ifdef LOG_SERVICE
            cout << "ExecutorClient::ProcessBatch() got:\n" << response.DebugString() << endl;
    #endif
        }

        // Wait until the returned flush ID has been stored to database
        if ((config.databaseURL != "local") && (processBatchResponse.stored_flush_id() != processBatchResponse.flush_id()))
        {
            executor::v1::GetFlushStatusResponse getFlushStatusResponse;
            do
            {
                usleep(10000);
                google::protobuf::Empty request;
                ::grpc::ClientContext context;
                ::grpc::Status grpcStatus = stub->GetFlushStatus(&context, request, &getFlushStatusResponse);
                if (grpcStatus.error_code() != grpc::StatusCode::OK)
                {
                    cerr << "Error: ExecutorClient::ProcessBatch() failed calling GetFlushStatus()" << endl;
                    break;
                }
            } while (getFlushStatusResponse.stored_flush_id() < processBatchResponse.flush_id());
            zklog.info("ExecutorClient::ProcessBatch() successfully stored returned flush id=" + to_string(processBatchResponse.flush_id()));
        }
    }
    else if (input.publicInputsExtended.publicInputs.witness.empty())
    {
        ::executor::v1::ProcessBatchRequestV2 request;
        request.set_coinbase(Add0xIfMissing(input.publicInputsExtended.publicInputs.sequencerAddr.get_str(16)));
        request.set_batch_l2_data(input.publicInputsExtended.publicInputs.batchL2Data);
        request.set_old_state_root(scalar2ba(input.publicInputsExtended.publicInputs.oldStateRoot));
        request.set_old_acc_input_hash(scalar2ba(input.publicInputsExtended.publicInputs.oldAccInputHash));
        request.set_l1_info_root(scalar2ba(input.publicInputsExtended.publicInputs.l1InfoRoot));
        request.set_timestamp_limit(input.publicInputsExtended.publicInputs.timestampLimit);
        request.set_forced_blockhash_l1(scalar2ba(input.publicInputsExtended.publicInputs.forcedBlockHashL1));
        request.set_update_merkle_tree(update_merkle_tree);
        request.set_no_counters(no_counters);
        request.set_get_keys(get_keys);
        request.set_skip_verify_l1_info_root(input.bSkipVerifyL1InfoRoot);
        request.set_skip_first_change_l2_block(input.bSkipFirstChangeL2Block);
        request.set_skip_write_block_info_root(input.bSkipWriteBlockInfoRoot);
        request.set_chain_id(input.publicInputsExtended.publicInputs.chainID);
        request.set_fork_id(input.publicInputsExtended.publicInputs.forkID);
        request.set_from(input.from);
        executor::v1::DebugV2 *pDebug = NULL;
        if (input.publicInputsExtended.newStateRoot != 0)
        {
            if(pDebug == NULL)
            {
                pDebug = new executor::v1::DebugV2();
            }
            pDebug->set_new_state_root(scalar2ba(input.publicInputsExtended.newStateRoot));
        }
        if (input.publicInputsExtended.newAccInputHash != 0){
            if(pDebug == NULL)
            {
                pDebug = new executor::v1::DebugV2();
            }
            pDebug->set_new_acc_input_hash(scalar2ba(input.publicInputsExtended.newAccInputHash));
        }
        if (input.publicInputsExtended.newLocalExitRoot != 0){
            if(pDebug == NULL)
            {
                pDebug = new executor::v1::DebugV2();
            }
            pDebug->set_new_local_exit_root(scalar2ba(input.publicInputsExtended.newLocalExitRoot));
        }
        if (input.publicInputsExtended.newBatchNum != 0){
            if(pDebug == NULL)
            {
                pDebug = new executor::v1::DebugV2();
            }
            pDebug->set_new_batch_num(input.publicInputsExtended.newBatchNum);
        }
        if (input.debug.gasLimit != 0)
        {
            if(pDebug == NULL)
            {
                pDebug = new executor::v1::DebugV2();
            }
            pDebug->set_gas_limit(input.debug.gasLimit);
        }
        if( pDebug != NULL ){
            request.set_allocated_debug(pDebug);
        }


        unordered_map<uint64_t, L1Data>::const_iterator itL1Data;
        for (itL1Data = input.l1InfoTreeData.begin(); itL1Data != input.l1InfoTreeData.end(); itL1Data++)
        {
            executor::v1::L1DataV2 l1Data;
            l1Data.set_global_exit_root(string2ba(itL1Data->second.globalExitRoot.get_str(16)));
            l1Data.set_block_hash_l1(string2ba(itL1Data->second.blockHashL1.get_str(16)));
            l1Data.set_min_timestamp(itL1Data->second.minTimestamp);
            for (uint64_t i=0; i<itL1Data->second.smtProof.size(); i++)
            {
                l1Data.add_smt_proof(string2ba(itL1Data->second.smtProof[i].get_str(16)));
            }
            (*request.mutable_l1_info_tree_data())[itL1Data->first] = l1Data;
        }
        if (input.traceConfig.bEnabled)
        {
            executor::v1::TraceConfigV2 * pTraceConfig = request.mutable_trace_config();
            pTraceConfig->set_disable_storage(input.traceConfig.bDisableStorage);
            pTraceConfig->set_disable_stack(input.traceConfig.bDisableStack);
            pTraceConfig->set_enable_memory(input.traceConfig.bEnableMemory);
            pTraceConfig->set_enable_return_data(input.traceConfig.bEnableReturnData);
            pTraceConfig->set_tx_hash_to_generate_full_trace(string2ba(input.traceConfig.txHashToGenerateFullTrace));
        }
        request.set_old_batch_num(input.publicInputsExtended.publicInputs.oldBatchNum);

        // Parse keys map
        DatabaseMap::MTMap::const_iterator it;
        for (it=input.db.begin(); it!=input.db.end(); it++)
        {
            string key = NormalizeToNFormat(it->first, 64);
            string value;
            vector<Goldilocks::Element> dbValue = it->second;
            for (uint64_t i=0; i<dbValue.size(); i++)
            {
                value += NormalizeToNFormat(fr.toString(dbValue[i], 16), 16);
            }
            (*request.mutable_db())[key] = value;
        }

        // Parse contracts data
        DatabaseMap::ProgramMap::const_iterator itp;
        for (itp=input.contractsBytecode.begin(); itp!=input.contractsBytecode.end(); itp++)
        {
            string key = NormalizeToNFormat(itp->first, 64);
            string value;
            vector<uint8_t> contractValue = itp->second;
            for (uint64_t i=0; i<contractValue.size(); i++)
            {
                value += byte2string(contractValue[i]);
            }
            (*request.mutable_contracts_bytecode())[key] = value;
        }

        ::executor::v1::ProcessBatchResponseV2 processBatchResponse;
        for (uint64_t i=0; i<config.executorClientLoops; i++)
        {
            if (config.executorClientResetDB)
            {
                pHashDB->resetDB();
            }
            else if (i == 1)
            {
                request.clear_db();
                request.clear_contracts_bytecode();
            }
            ::grpc::ClientContext context;
            ::grpc::Status grpcStatus = stub->ProcessBatchV2(&context, request, &processBatchResponse);
            if (grpcStatus.error_code() != grpc::StatusCode::OK)
            {
                cerr << "Error: ExecutorClient::ProcessBatch() failed calling server i=" << i << " error=" << grpcStatus.error_code() << "=" << grpcStatus.error_message() << endl;
                break;
            }
            if (processBatchResponse.error() != executor::v1::EXECUTOR_ERROR_NO_ERROR)
            {
                cerr << "Error: ExecutorClient::ProcessBatch() failed i=" << i << " error=" << processBatchResponse.error() << endl;
                return false;
            }
            newStateRoot = ba2string(processBatchResponse.new_state_root());

    #ifdef LOG_SERVICE
            cout << "ExecutorClient::ProcessBatch() got:\n" << response.DebugString() << endl;
    #endif
        }

        // Store the hash at the end of each block
        for (int64_t b=0; b < processBatchResponse.block_responses().size(); b++)
        {
            blockStateRoots.emplace_back(ba2string(processBatchResponse.block_responses()[b].block_hash()));
        }

        // Wait until the returned flush ID has been stored to database
        if ((config.databaseURL != "local") && (processBatchResponse.stored_flush_id() != processBatchResponse.flush_id()))
        {
            executor::v1::GetFlushStatusResponse getFlushStatusResponse;
            do
            {
                usleep(10000);
                google::protobuf::Empty request;
                ::grpc::ClientContext context;
                ::grpc::Status grpcStatus = stub->GetFlushStatus(&context, request, &getFlushStatusResponse);
                if (grpcStatus.error_code() != grpc::StatusCode::OK)
                {
                    cerr << "Error: ExecutorClient::ProcessBatch() failed calling GetFlushStatus()" << endl;
                    break;
                }
            } while (getFlushStatusResponse.stored_flush_id() < processBatchResponse.flush_id());
            zklog.info("ExecutorClient::ProcessBatch() successfully stored returned flush id=" + to_string(processBatchResponse.flush_id()));
        }
    }
    else if (!input.publicInputsExtended.publicInputs.witness.empty()) // Stateless
    {
        ::executor::v1::ProcessStatelessBatchRequestV2 request;
        request.set_witness(input.publicInputsExtended.publicInputs.witness);
        request.set_data_stream(input.publicInputsExtended.publicInputs.dataStream);
        request.set_coinbase(Add0xIfMissing(input.publicInputsExtended.publicInputs.sequencerAddr.get_str(16)));
        request.set_old_acc_input_hash(scalar2ba(input.publicInputsExtended.publicInputs.oldAccInputHash));
        request.set_l1_info_root(scalar2ba(input.publicInputsExtended.publicInputs.l1InfoRoot));
        request.set_timestamp_limit(input.publicInputsExtended.publicInputs.timestampLimit);
        request.set_forced_blockhash_l1(scalar2ba(input.publicInputsExtended.publicInputs.forcedBlockHashL1));

        ::executor::v1::ProcessBatchResponseV2 processBatchResponse;
        for (uint64_t i=0; i<config.executorClientLoops; i++)
        {
            if (config.executorClientResetDB)
            {
                pHashDB->resetDB();
            }
            else if (i == 1)
            {
                //request.clear_db();
                //request.clear_contracts_bytecode();
            }
            ::grpc::ClientContext context;
            ::grpc::Status grpcStatus = stub->ProcessStatelessBatchV2(&context, request, &processBatchResponse);
            if (grpcStatus.error_code() != grpc::StatusCode::OK)
            {
                cerr << "Error: ExecutorClient::ProcessBatch() failed calling server i=" << i << " error=" << grpcStatus.error_code() << "=" << grpcStatus.error_message() << endl;
                break;
            }
            if (processBatchResponse.error() != executor::v1::EXECUTOR_ERROR_NO_ERROR)
            {
                cerr << "Error: ExecutorClient::ProcessBatch() failed i=" << i << " error=" << processBatchResponse.error() << endl;
                return false;
            }
            newStateRoot = ba2string(processBatchResponse.new_state_root());
            zklog.info("ExecutorClient::ProcessBatch() newStateRoot=" + newStateRoot);

    #ifdef LOG_SERVICE
            cout << "ExecutorClient::ProcessBatch() got:\n" << response.DebugString() << endl;
    #endif
        }
        
        // Store the hash at the end of each block
        for (int64_t b=0; b < processBatchResponse.block_responses().size(); b++)
        {
            blockStateRoots.emplace_back(ba2string(processBatchResponse.block_responses()[b].block_hash()));
        }

        // Wait until the returned flush ID has been stored to database
        if ((config.databaseURL != "local") && (processBatchResponse.stored_flush_id() != processBatchResponse.flush_id()))
        {
            executor::v1::GetFlushStatusResponse getFlushStatusResponse;
            do
            {
                usleep(10000);
                google::protobuf::Empty request;
                ::grpc::ClientContext context;
                ::grpc::Status grpcStatus = stub->GetFlushStatus(&context, request, &getFlushStatusResponse);
                if (grpcStatus.error_code() != grpc::StatusCode::OK)
                {
                    cerr << "Error: ExecutorClient::ProcessBatch() failed calling GetFlushStatus()" << endl;
                    break;
                }
            } while (getFlushStatusResponse.stored_flush_id() < processBatchResponse.flush_id());
            zklog.info("ExecutorClient::ProcessBatch() successfully stored returned flush id=" + to_string(processBatchResponse.flush_id()));
        }
    }

    if (input.publicInputsExtended.newStateRoot != 0)
    {
        mpz_class newStateRootScalar;
        newStateRootScalar.set_str(Remove0xIfPresent(newStateRoot), 16);
        if (input.publicInputsExtended.newStateRoot != newStateRootScalar)
        {
            zklog.error("ExecutorClient::ProcessBatch() returned newStateRoot=" + newStateRoot + " != input.publicInputsExtended.newStateRoot=" + input.publicInputsExtended.newStateRoot.get_str(16) + " inputFile=" + inputFile);
            return false;
        }
    }

    if (config.executorClientCheckNewStateRoot)
    {
        if (config.hashDB64)
        {            
            //if (StateManager64::isVirtualStateRoot(newStateRoot))
            {
                TimerStart(CONSOLIDATE_STATE);

                Goldilocks::Element virtualStateRoot[4];
                string2fea(fr, newStateRoot, virtualStateRoot);
                Goldilocks::Element consolidatedStateRoot[4];
                uint64_t flushId, storedFlushId;
                zkresult zkr = pHashDB->consolidateState(virtualStateRoot, update_merkle_tree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, consolidatedStateRoot, flushId, storedFlushId);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("ExecutorClient::ProcessBatch() failed calling pHashDB->consolidateState() result=" + zkresult2string(zkr));
                    return false;
                }
                newStateRoot = fea2string(fr, consolidatedStateRoot);

                TimerStopAndLog(CONSOLIDATE_STATE);
            }
        }

        TimerStart(CHECK_NEW_STATE_ROOT);

        if (newStateRoot.size() == 0)
        {
            zklog.error("ExecutorClient::ProcessBatch() found newStateRoot emty");
            return false;
        }

        HashDB &hashDB = *hashDBSingleton.get();

        if (config.hashDB64)
        {
            Database64 &db = hashDB.db64;
            zkresult zkr = db.PrintTree(newStateRoot);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("ExecutorClient::ProcessBatch() failed calling db.PrintTree() result=" + zkresult2string(zkr));
                return false;
            }
        }
        else if (NormalizeToNFormat(input.publicInputsExtended.publicInputs.oldStateRoot.get_str(16), 64) !=
                 NormalizeToNFormat(newStateRoot, 64))
        {
            blockStateRoots.emplace_back(newStateRoot);
            vector<string>::iterator it = unique(blockStateRoots.begin(), blockStateRoots.end());
            blockStateRoots.resize(distance(blockStateRoots.begin(), it));

            Database &db = hashDB.db;
            for (uint64_t b = 0; b < blockStateRoots.size(); b++)
            {
                db.clearCache();

                CheckTreeCounters checkTreeCounters;

                zklog.info("ExecutorClient::ProcessBatch() checking state of block=" + to_string(b) + " root=" + blockStateRoots[b]);

                zkresult result = CheckTree(db, blockStateRoots[b], 0, checkTreeCounters, "");
                if (result != ZKR_SUCCESS)
                {
                    zklog.error("ExecutorClient::ProcessBatch() failed calling CheckTree() result=" + zkresult2string(result));
                    return false;
                }

                zklog.info("intermediateNodes=" + to_string(checkTreeCounters.intermediateNodes));
                zklog.info("leafNodes=" + to_string(checkTreeCounters.leafNodes));
                zklog.info("values=" + to_string(checkTreeCounters.values));
                zklog.info("maxLevel=" + to_string(checkTreeCounters.maxLevel));
            }
        }

        TimerStopAndLog(CHECK_NEW_STATE_ROOT);

    }

    TimerStopAndLog(EXECUTOR_CLIENT_PROCESS_BATCH);

    return true;
}

bool ProcessDirectory (ExecutorClient *pClient, const string &directoryName, uint64_t &fileCounter, uint64_t &directoryCounter, uint64_t &skippedFileCounter, uint64_t &skippedDirectoryCounter, bool skipping)
{
    // Get files sorted alphabetically from the folder
    vector<string> files = getFolderFiles(directoryName, true);

    // Process each input file in order
    for (size_t i = 0; i < files.size(); i++)
    {
        // Get full file name
        string inputFile = directoryName + files[i];

        // Check file existence
        if (!fileExists(inputFile))
        {
            zklog.error("ProcessDirectory() found invalid file or directory with name=" + inputFile);
            exitProcess();
        }

        // Mark if this is a directory
        bool isDirectory = fileIsDirectory(inputFile);

        // Skip some files that we know are failing
        if ( skipping 
             /* Files and directories expected to be skipped */
             || (inputFile.find("ignore") != string::npos) // Ignore tests masked as "ignore"
             || (inputFile.find("-list.json") != string::npos) // Ignore tests masked as "-list"
#ifndef MULTI_ROM_TEST
             || (inputFile.find("tests-30M") != string::npos) // Ignore tests that require a rom with a different gas limit
#endif
             || (inputFile.find("rlp-error/test-length-data_1.json") != string::npos) // batchL2Data.size()=120119 > MAX_BATCH_L2_DATA_SIZE=120000
             || (inputFile.find("rlp-error/test-length-data_2.json") != string::npos) // batchL2Data.size()=120118 > MAX_BATCH_L2_DATA_SIZE=120000
             || (inputFile.find("ethereum-tests/GeneralStateTests/stMemoryStressTest/mload32bitBound_return2_0.json") != string::npos) // executor.v1.ProcessBatchResponseV2 exceeded maximum protobuf size of 2GB: 4294968028
             || (inputFile.find("ethereum-tests/GeneralStateTests/stMemoryStressTest/mload32bitBound_return_0.json") != string::npos) // executor.v1.ProcessBatchResponseV2 exceeded maximum protobuf size of 2GB: 4294968028
             || (inputFile.find("inputs-executor/ethereum-tests/GeneralStateTests/stCreate2/create2collisionCode_0.json") != string::npos)
             || (inputFile.find("inputs-executor/ethereum-tests/GeneralStateTests/stCreate2/create2collisionNonce_0.json") != string::npos)
             || (inputFile.find("inputs-executor/ethereum-tests/GeneralStateTests/stCreate2/create2noCash_2.json") != string::npos)
           )
        {
            zklog.warning("ProcessDirectory() skipping file=" + inputFile + " fileCounter=" + to_string(fileCounter) + " skippedFileCounter=" + to_string(skippedFileCounter));
            if (isDirectory)
            {
                skippedDirectoryCounter++;
                bool bResult = ProcessDirectory(pClient, inputFile + "/", fileCounter, directoryCounter, skippedFileCounter, skippedDirectoryCounter, true);
                if (bResult == false)
                {
                    return false;
                }
            }
            else
            {
                skippedFileCounter++;
            }
            continue;
        }

        // If file is a directory, call recursively
        if (isDirectory)
        {
            directoryCounter++;
            bool bResult = ProcessDirectory(pClient, inputFile + "/", fileCounter, directoryCounter, skippedFileCounter, skippedDirectoryCounter, skipping);
            if (bResult == false)
            {
                return false;
            }
            continue;
        }

        // File exists and it is not a directory
        fileCounter++;
        zklog.info("ProcessDirectory() fileCounter=" + to_string(fileCounter) + " inputFile=" + inputFile);
        bool bResult = pClient->ProcessBatch(inputFile);
        if (!bResult)
        {
            zklog.error("ProcessDirectory() failed fileCounter=" + to_string(fileCounter) + " inputFile=" + inputFile);
            return false;
        }
    }
    return true;
}

void* executorClientThread (void* arg)
{
    zklog.info("executorClientThread() started");
    int64_t result = 0;
    string uuid;
    ExecutorClient *pClient = (ExecutorClient *)arg;

    TimerStart(EXECUTOR_CLIENT_THREAD);
    
    // Execute should block and succeed
    cout << "executorClientThread() calling pClient->ProcessBatch()" << endl;

    if (config.inputFile.back() == '/')
    {
        uint64_t fileCounter = 0;
        uint64_t directoryCounter = 0;
        uint64_t skippedFileCounter = 0;
        uint64_t skippedDirectoryCounter = 0;
        bool bResult = ProcessDirectory(pClient, config.inputFile, fileCounter, directoryCounter, skippedFileCounter, skippedDirectoryCounter, false);
        if (!bResult)
        {
            zklog.error("executorClientThread() failed calling ProcessDirectory()");
            result = -1;
        }
        zklog.info("executorClientThread() called ProcessDirectory() and got directories=" + to_string(directoryCounter) + " files=" + to_string(fileCounter) + " skippedFiles=" + to_string(skippedFileCounter) + " percentage=" + to_string((fileCounter*100)/zkmax(1, fileCounter + skippedFileCounter)) + "% skippedDirectories=" + to_string(skippedDirectoryCounter));
    }
    else
    {
        bool bResult = pClient->ProcessBatch(config.inputFile);
        if (!bResult)
        {
            zklog.error("executorClientThread() failed calling ProcessBatch()");
            result = -1;
        }
    }

    TimerStopAndLog(EXECUTOR_CLIENT_THREAD);

    return (void *)result;
}

void* executorClientThreads (void* arg)
{
    zklog.info("executorClientThreads() started");
    int64_t result = 0;
    string uuid;
    ExecutorClient *pClient = (ExecutorClient *)arg;

    // Execute should block and succeed
    //cout << "executorClientThreads() calling pClient->ProcessBatch()" << endl;
    for(uint64_t i=0; i<EXECUTOR_CLIENT_MULTITHREAD_N_FILES; i++)
    {        
        if (config.inputFile.back() == '/')
        {
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile, true);
            // Process each input file in order
            for (size_t i = 0; i < files.size(); i++)
            {
                string inputFile = config.inputFile + files[i];
                zklog.info("executorClientThreads() inputFile=" + inputFile);
                bool bResult = pClient->ProcessBatch(inputFile);
                if (!bResult)
                {
                    zklog.error("executorClientThreads() failed i=" + to_string(i) + " inputFile=" + inputFile);
                    result = -1;
                    break;
                }
            }
        }
        else
        {
            bool bResult = pClient->ProcessBatch(config.inputFile);
            if (!bResult)
            {
                zklog.error("executorClientThreads() failed calling ProcessBatch()");
                result = -1;
            }
        }
    }

    return (void *)result;
}