
#include <nlohmann/json.hpp>
#include "executor_client.hpp"
#include "hashdb_singleton.hpp"
#include "zkmax.hpp"
#include "check_tree.hpp"
#include "check_tree_64.hpp"

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

bool ExecutorClient::ProcessBatch (void)
{
    TimerStart(EXECUTOR_CLIENT_PROCESS_BATCH);

    if (config.inputFile.size() == 0)
    {
        cerr << "Error: ExecutorClient::ProcessBatch() found config.inputFile empty" << endl;
        exit(-1);
    }
    ::executor::v1::ProcessBatchRequest request;
    Input input(fr);
    json inputJson;
    file2json(config.inputFile, inputJson);
    zkresult zkResult = input.load(inputJson);
    if (zkResult != ZKR_SUCCESS)
    {
        cerr << "Error: ProverClient::GenProof() failed calling input.load() zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
        exit(-1);
    }

    bool update_merkle_tree = true;
    bool get_keys = false;

    //request.set_batch_num(input.publicInputs.batchNum);
    request.set_coinbase(Add0xIfMissing(input.publicInputsExtended.publicInputs.sequencerAddr.get_str(16)));
    request.set_batch_l2_data(input.publicInputsExtended.publicInputs.batchL2Data);
    request.set_old_state_root(scalar2ba(input.publicInputsExtended.publicInputs.oldStateRoot));
    request.set_old_acc_input_hash(scalar2ba(input.publicInputsExtended.publicInputs.oldAccInputHash));
    request.set_global_exit_root(scalar2ba(input.publicInputsExtended.publicInputs.globalExitRoot));
    request.set_eth_timestamp(input.publicInputsExtended.publicInputs.timestamp);
    request.set_update_merkle_tree(update_merkle_tree);
    request.set_get_keys(get_keys);
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
        pTraceConfig->set_tx_hash_to_generate_execute_trace(string2ba(input.traceConfig.txHashToGenerateExecuteTrace));
        pTraceConfig->set_tx_hash_to_generate_call_trace(string2ba(input.traceConfig.txHashToGenerateCallTrace));
        //request.set_tx_hash_to_generate_execute_trace(string2ba(input.traceConfig.txHashToGenerateExecuteTrace));
        //request.set_tx_hash_to_generate_call_trace(string2ba(input.traceConfig.txHashToGenerateCallTrace));
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
    string newStateRoot;
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
            break;
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
            sleep(1);
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

    if (config.executorClientCheckNewStateRoot)
    {
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

            CheckTreeCounters64 checkTreeCounters;

            zkresult result = CheckTree64(db, newStateRoot, 0, checkTreeCounters);
            if (result != ZKR_SUCCESS)
            {
                zklog.error("ExecutorClient::ProcessBatch() failed calling ClimbTree64() result=" + zkresult2string(result));
                return false;
            }

            zklog.info("intermediateNodes=" + to_string(checkTreeCounters.intermediateNodes));
            zklog.info("leafNodes=" + to_string(checkTreeCounters.leafNodes));
            zklog.info("values=" + to_string(checkTreeCounters.values));
            zklog.info("maxLevel=" + to_string(checkTreeCounters.maxLevel));
        }
        else
        {
            Database &db = hashDB.db;
            db.clearCache();

            CheckTreeCounters checkTreeCounters;

            zkresult result = CheckTree(db, newStateRoot, 0, checkTreeCounters);
            if (result != ZKR_SUCCESS)
            {
                zklog.error("ExecutorClient::ProcessBatch() failed calling ClimbTree() result=" + zkresult2string(result));
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

void* executorClientThread (void* arg)
{
    cout << "executorClientThread() started" << endl;
    string uuid;
    ExecutorClient *pClient = (ExecutorClient *)arg;
    
    // Execute should block and succeed
    cout << "executorClientThread() calling pClient->ProcessBatch()" << endl;
    pClient->ProcessBatch();
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
        pClient->ProcessBatch();
    }

    return NULL;
}