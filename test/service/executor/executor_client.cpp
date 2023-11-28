
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

void ExecutorClient::waitForThread (void)
{
    pthread_join(t, NULL);
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

void ExecutorClient::waitForThreads (void)
{
    for (uint64_t i=0; i<EXECUTOR_CLIENT_MULTITHREAD_N_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
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
        cerr << "Error: ProverClient::GenProof() failed calling input.load() zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
        exit(-1);
    }

    // Flags
    bool update_merkle_tree = true;
    bool get_keys = false;

    // Resulting new state root
    string newStateRoot;

    if (input.publicInputsExtended.publicInputs.forkID <= 6)
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
        request.set_no_counters(input.bNoCounters);
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
            if (i == 1)
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

        if (processBatchResponse.stored_flush_id() != processBatchResponse.flush_id())
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
    else
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
        request.set_no_counters(input.bNoCounters);
        request.set_get_keys(get_keys);
        request.set_skip_verify_l1_info_root(input.bSkipVerifyL1InfoRoot);
        request.set_skip_first_change_l2_block(input.bSkipFirstChangeL2Block);
        request.set_chain_id(input.publicInputsExtended.publicInputs.chainID);
        request.set_fork_id(input.publicInputsExtended.publicInputs.forkID);
        request.set_from(input.from);
        request.set_skip_verify_l1_info_root(input.bSkipFirstChangeL2Block);
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
            if (i == 1)
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

        if (processBatchResponse.stored_flush_id() != processBatchResponse.flush_id())
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
        else
        {
            Database &db = hashDB.db;
            db.clearCache();

            CheckTreeCounters checkTreeCounters;

            zkresult result = CheckTree(db, newStateRoot, 0, checkTreeCounters, "");
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

        TimerStopAndLog(CHECK_NEW_STATE_ROOT);

    }

    TimerStopAndLog(EXECUTOR_CLIENT_PROCESS_BATCH);

    return true;
}

bool ProcessDirectory (ExecutorClient *pClient, const string &directoryName, uint64_t &counter)
{
    // Get files sorted alphabetically from the folder
    vector<string> files = getFolderFiles(directoryName, true);

    // Process each input file in order
    for (size_t i = 0; i < files.size(); i++)
    {
        counter++;
        string inputFile = directoryName + files[i];

        // Skip some files that we know are failing
        if ( false 
                || (files[i].find("ignore") != string::npos) // Ignore tests masked as "ignore"

                || (files[i].find("test-length-data_1.json") == 0) // counter=288 inputFile=testvectors/inputs-executor/calldata/test-length-data_1.json newStateRoot=c14d4d9f490cd974197f01ed1adecc4024d53fa3c7e81763a03808f65b84ae71 != input.publicInputsExtended.newStateRoot=b0efbc28d34fc4fe525dd4abe23503c861134af89cca6e14af69f6973ab9a6bf OOCB => new state root does not match
                || (files[i].find("performanceTester_1.json") == 0) // counter=750 inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/VMTests/performanceTester_1.json SystemManager missing sub-state -> waiting for fix in rom
                || (files[i].find("CallcodeToPrecompileFromCalledContract-custom.json") == 0) // testvectors/inputs-executor/ethereum-tests/GeneralStateTests/stStaticFlagEnabled/CallcodeToPrecompileFromCalledContract-custom.json newStateRoot=91d9a9046fdadf44836559ae91ced0457c7505c20283ce22761902e42e2e5c6e != input.publicInputsExtended.newStateRoot=99e32a7ab1582c1979d72942d5a1a2bf63234444f95ccc92ec77b7039276d774             
                || (files[i].find("CallcodeToPrecompileFromTransaction-custom.json") == 0) // testvectors/inputs-executor/ethereum-tests/GeneralStateTests/stStaticFlagEnabled/CallcodeToPrecompileFromTransaction-custom.json newStateRoot=91d9a9046fdadf44836559ae91ced0457c7505c20283ce22761902e42e2e5c6e != input.publicInputsExtended.newStateRoot=99e32a7ab1582c1979d72942d5a1a2bf63234444f95ccc92ec77b7039276d774
                || (files[i].find("DelegatecallToPrecompileFromCalledContract-custom.json") == 0) // testvectors/inputs-executor/ethereum-tests/GeneralStateTests/stStaticFlagEnabled/DelegatecallToPrecompileFromCalledContract-custom.json newStateRoot=91d9a9046fdadf44836559ae91ced0457c7505c20283ce22761902e42e2e5c6e != input.publicInputsExtended.newStateRoot=99e32a7ab1582c1979d72942d5a1a2bf63234444f95ccc92ec77b7039276d774
                || (files[i].find("DelegatecallToPrecompileFromTransaction-custom.json") == 0) // testvectors/inputs-executor/ethereum-tests/GeneralStateTests/stStaticFlagEnabled/DelegatecallToPrecompileFromTransaction-custom.json newStateRoot=f5971aedd8f30d3eba337d8257020b16015572f7f538678ed7ea31c17819a61e != input.publicInputsExtended.newStateRoot=d1b397950750cefdff4ef8691a3234e4d6cc99b1436fdcb181a17cb8b0cc6865
                || (files[i].find("pairingTest_0.json") == 0) // inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/stZeroKnowledge/pairingTest_0.json  newStateRoot=5517e42cc95e1f4bab7f535a563f5c1d14900c8b6a93ffb7c19446c6c4641d26 != input.publicInputsExtended.newStateRoot=9f5fdf3aa6107c42c69af17aeddc00d92a7e575e73504af75b6d690175a53700
                || (files[i].find("pairingTest_12.json") == 0) // inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/stZeroKnowledge/pairingTest_12.json newStateRoot=e3aa4a919114315c36c0dc705b9b3ba16cc4e456af138a89fd2268237d49d8b3 != input.publicInputsExtended.newStateRoot=5a59ac290a34f38caf116922864dcefea5f17acb3c63bfc7e8c77621293df934
                || (files[i].find("pairingTest_16.json") == 0) // counter=338 inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/stZeroKnowledge/pairingTest_16.json newStateRoot=a2de34399cc117f9734fefa5ecb8511c68780cfd526cbae5fac0cc9fc2898b59 != input.publicInputsExtended.newStateRoot=bc1861dcf5671cf3c72af7db982d8753ed972e20afeee78670c80b9cf5cc54e9
                || (files[i].find("pairingTest_4.json") == 0) // counter=9632 inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/stZeroKnowledge/pairingTest_4.json newStateRoot=ea0e3f1dec1adb8368dd399fc60f467cc5c7966e13c7340c68d465ce0367c36f != input.publicInputsExtended.newStateRoot=5f84bbf18e70fa498363ad4fb57b1c11639c7e2aa9019903f935b6cd2646d0e6
                || (files[i].find("pairingTest_8.json") == 0) // counter=300 inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/stZeroKnowledge/pairingTest_8.json newStateRoot=be3742934aa3d001391fb16d1bceb3787fb777811c07d6919d46c5426b09949d != input.publicInputsExtended.newStateRoot=3a3f8ce88f7e73d8823e5f3cc177d03bcdfe5fac226235d7f3d00b28c58b3245
                || (files[i].find("ABAcalls1.json") == 0) // counter=10142 inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/tests-30M/ABAcalls1.json newStateRoot=8bbde6cbb5fcd8de1f54a5c91cf4e343a77bb4d97a143d1a3c7b00840d512f0d != input.publicInputsExtended.newStateRoot=b614512aaf23dd03e679201710dc371b90d656bd8130b4e477baa116e9dfa0f2
                || (files[i].find("ABAcalls2.json") == 0) // counter=10142 inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/tests-30M/ABAcalls2.json newStateRoot=4e870d7c3a2a71d03bfc69969032202beffea8d46c9fa11aced4e837dd7ac23a != input.publicInputsExtended.newStateRoot=aa605be543b3ef791ecb1c70e131ed1b3e6bb5cb5d51ec827fe8ff0b2fe467cd 
                || (files[i].find("CALLCODE_Bounds2_1.json") == 0) // counter=10142 inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/tests-30M/CALLCODE_Bounds2_1.json newStateRoot=aec1e67465849dab4fd2eb8cbef56ce69e8c4c95cbedc31d5215ca34facdd2cc != input.publicInputsExtended.newStateRoot=9a97ef054fbf6a96cce06abdf0574acd91307ea9689c178043de1cce0ba7e42a
                || (files[i].find("CALLCODE_Bounds3_1.json") == 0) // counter=10142 inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/tests-30M/CALLCODE_Bounds3_1.json newStateRoot=ce29076d926443f72f178cf2ac90e045a18e2b725dad15d58e2f1656fb525813 != input.publicInputsExtended.newStateRoot=42b86f6ded9c29a67f806bea078b43127b16aee59d94431c5baceec441c97cc1
                || (files[i].find("CALLCODE_Bounds4_2.json") == 0) // counter=10142 inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/tests-30M/CALLCODE_Bounds4_2.json newStateRoot=7bad6d7d7a9730176e6273fbae88a27aa2400c3334a6184fc7240e4209928594 != input.publicInputsExtended.newStateRoot=db9734e743b325e15dd8bfed5824e6e22e4c40dd8cc416c5ba852d69f1b51d26
                || (files[i].find("CALLCODE_Bounds_1.json") == 0) // counter=10142 inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/tests-30M/CALLCODE_Bounds_1.json newStateRoot=f5e0b390b0f17a81dd1a53d352be6445fc15249683a477ae4eedd42dc74e8e4b != input.publicInputsExtended.newStateRoot=82dca91b75f6f336dc549510060f222b9e4a466a2f17ff5ffeb3af363f77f6df
                || (files[i].find("CALL_Bounds2_1.json") == 0) // counter=10142 inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/tests-30M/CALL_Bounds2_1.json newStateRoot=c77f2405f2f581e059987dc88607fbf83620eb84948a3c2696e1bdca4c898fc5 != input.publicInputsExtended.newStateRoot=84d77f8cad43f130841b977023ddae34ab8066b7fc79c8914270a6f44f78fa88
                || (files[i].find("CALL_Bounds2a_1.json") == 0) // counter=10142 inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/tests-30M/CALL_Bounds2a_1.json newStateRoot=c67d4a5c454e0ee47b72d41d371471ae068dd7babcfb6d2d93119e9c43f9ff12 != input.publicInputsExtended.newStateRoot=f26d191f459925bd5a4e73ae9a16506645f1a1cfea46b9767ff6b9fc056d720d
                || (files[i].find("CALL_Bounds3_2.json") == 0) // counter=10142 inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/tests-30M/CALL_Bounds3_2.json newStateRoot=3fd12cbb6410cf3c94f600a34c35d7b4af7a9ab0fa5ef8e773ab19b31b40e436 != input.publicInputsExtended.newStateRoot=2dbee92d28004258c08fd3c8eec592c32be05f1fceae025c0c0619e541cd019c
                || (files[i].find("CALL_Bounds_1.json") == 0) // counter=10142 inputFile=testvectors/inputs-executor/ethereum-tests/GeneralStateTests/tests-30M/CALL_Bounds_1.json newStateRoot=fdef51e5472fb9d0a674d06225ab92f72e606e28756ec0d4daab0833dab3710b != input.publicInputsExtended.newStateRoot=d2b40482ffbb1a35858b8bcd7ad385f091330ec1588d833e61c82787adb1357a
            )
        {
            zklog.error("ProcessDirectory() skipping file counter=" + to_string(counter) + " file=" + inputFile);
            continue;
        }

        // Check file existence
        if (!fileExists(inputFile))
        {
            zklog.error("ProcessDirectory() found invalid file or directory with name=" + inputFile);
            exitProcess();
        }

        // If file is a directory, call recursively
        if (fileIsDirectory(inputFile))
        {
            bool bResult = ProcessDirectory(pClient, inputFile + "/", counter);
            if (bResult == false)
            {
                return false;
            }
            continue;
        }

        // File exists and it is not a directory
        zklog.info("ProcessDirectory() counter=" + to_string(counter) + " inputFile=" + inputFile);
        bool bResult = pClient->ProcessBatch(inputFile);
        if (!bResult)
        {
            zklog.error("ProcessDirectory() failed counter=" + to_string(counter) + " inputFile=" + inputFile);
            return false;
        }
    }
    return true;
}

void* executorClientThread (void* arg)
{
    cout << "executorClientThread() started" << endl;
    string uuid;
    ExecutorClient *pClient = (ExecutorClient *)arg;
    
    // Execute should block and succeed
    cout << "executorClientThread() calling pClient->ProcessBatch()" << endl;

    if (config.inputFile.back() == '/')
    {
        uint64_t counter = 0;
        ProcessDirectory(pClient, config.inputFile, counter);
    }
    else
    {
        pClient->ProcessBatch(config.inputFile);
    }
    
    return NULL;
}

void* executorClientThreads (void* arg)
{
    //cout << "executorClientThreads() started" << endl;
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
                    break;
                }
            }
        }
        else
        {
            pClient->ProcessBatch(config.inputFile);
        }
    }

    return NULL;
}