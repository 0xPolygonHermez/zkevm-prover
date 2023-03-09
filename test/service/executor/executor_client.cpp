
#include <nlohmann/json.hpp>
#include "executor_client.hpp"

using namespace std;
using json = nlohmann::json;

ExecutorClient::ExecutorClient (Goldilocks &fr, const Config &config) :
    fr(fr),
    config(config)
{
    // Create channel
    std::shared_ptr<grpc_impl::Channel> channel = ::grpc::CreateChannel(config.executorClientHost + ":" + to_string(config.executorClientPort), grpc::InsecureChannelCredentials());

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
    ::grpc::ClientContext context;
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

    //request.set_batch_num(input.publicInputs.batchNum);
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

    ::executor::v1::ProcessBatchResponse response;
    std::unique_ptr<grpc::ClientReaderWriter<executor::v1::ProcessBatchRequest, executor::v1::ProcessBatchResponse>> readerWriter;
    ::grpc::Status grpcStatus = stub->ProcessBatch(&context, request, &response);
    if (grpcStatus.error_code() != grpc::StatusCode::OK)
    {
        cerr << "Error: ExecutorClient::ProcessBatch() failed calling server" << endl;
    }

#ifdef LOG_SERVICE
    cout << "ExecutorClient::ProcessBatch() got:\n" << response.DebugString() << endl;
#endif

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