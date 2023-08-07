
#include <nlohmann/json.hpp>
#include "executor_client.hpp"
#include "hashdb_singleton.hpp"
#include "zkmax.hpp"
#include "scalar.hpp"

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
    std::shared_ptr<grpc_impl::Channel> channel = grpc::CreateCustomChannel(config.executorClientHost + ":" + to_string(config.executorClientPort), grpc::InsecureChannelCredentials(), channelArguments);

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

        TimerStopAndLog(CHECK_NEW_STATE_ROOT);

    }

    TimerStopAndLog(EXECUTOR_CLIENT_PROCESS_BATCH);

    return true;
}

zkresult CheckTree (Database &db, const string &key, uint64_t level, CheckTreeCounters &checkTreeCounters)
{
    checkTreeCounters.maxLevel = zkmax(checkTreeCounters.maxLevel, level);
    Goldilocks::Element vkey[4], vkeyf[4];
    Goldilocks fr;
    string2fea(fr, key,vkey);
    vkeyf[3] = vkey[0];
    vkeyf[2] = vkey[1];
    vkeyf[1] = vkey[2];
    vkeyf[0] = vkey[3]; 

    vector<Goldilocks::Element> value;
    zkresult result = db.read(key,vkeyf, value, NULL, false);
    if (result != ZKR_SUCCESS)
    {
        zklog.error("CheckTree() failed key=" + key + " level=" + to_string(level));
        return result;
    }
    if (value.size() != 12)
    {
        zklog.error("CheckTree() invalid value.size=" + to_string(value.size()) + " key=" + key + " level=" + to_string(level));
        return ZKR_UNSPECIFIED;
    }

    if (!db.fr.isZero(value[9]))
    {
        zklog.error("CheckTree() fe9 not zero value.size=" + to_string(value.size()) + " key=" + key + " level=" + to_string(level) + " fe9=" + db.fr.toString(value[9],16));
        return ZKR_UNSPECIFIED;
    }

    if (!db.fr.isZero(value[10]))
    {
        zklog.error("CheckTree() fe10 not zero value.size=" + to_string(value.size()) + " key=" + key + " level=" + to_string(level) + " fe10=" + db.fr.toString(value[10],16));
        return ZKR_UNSPECIFIED;
    }

    if (!db.fr.isZero(value[11]))
    {
        zklog.error("CheckTree() fe11 not zero value.size=" + to_string(value.size()) + " key=" + key + " level=" + to_string(level) + " fe11=" + db.fr.toString(value[11],16));
        return ZKR_UNSPECIFIED;
    }

    uint64_t fe8 = db.fr.toU64(value[8]);

    if (fe8 == 0) // Intermediate node
    {
        checkTreeCounters.intermediateNodes++;

        string hashLeft = fea2string(db.fr, value[0], value[1], value[2], value[3]);
        if (hashLeft == "0")
        {
            return ZKR_SUCCESS;
        }
        result = CheckTree(db, hashLeft, level+1, checkTreeCounters);
        if (result != ZKR_SUCCESS)
        {
            return result;
        }
        string hashRight = fea2string(db.fr, value[4], value[5], value[6], value[7]);
        if (hashRight == "0")
        {
            return ZKR_SUCCESS;
        }
        result = CheckTree(db, hashRight, level+1, checkTreeCounters);
        return result;
    }
    else if (fe8 == 1) // Leaf node
    {
        checkTreeCounters.leafNodes++;

        level++;
        string valueHash = fea2string(db.fr, value[4], value[5], value[6], value[7]);
        value.clear();
        Goldilocks::Element vHash[4];
        vHash[0] = value[4];
        vHash[1] = value[5];
        vHash[2] = value[6];
        vHash[3] = value[7];
        zkresult result = db.read(valueHash, vHash, value, NULL, false);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("CheckTree() failed key=" + valueHash + " level=" + to_string(level));
            return result;
        }
        if (value.size() != 12)
        {
            zklog.error("CheckTree() found value for key=" + valueHash + " at level=" + to_string(level) + " with incorrect size=" + to_string(value.size()));
            /*zklog.error("valueL=" + fea2string(db.fr, value[0], value[1], value[2], value[3]));
            zklog.error("valueH=" + fea2string(db.fr, value[4], value[5], value[6], value[7]));
            PoseidonGoldilocks poseidon;
            Goldilocks::Element valueFea[12];
            valueFea[0] = value[0];
            valueFea[1] = value[1];
            valueFea[2] = value[2];
            valueFea[3] = value[3];
            valueFea[4] = value[4];
            valueFea[5] = value[5];
            valueFea[6] = value[6];
            valueFea[7] = value[7];
            valueFea[8] = db.fr.zero();
            valueFea[9] = db.fr.zero();
            valueFea[10] = db.fr.zero();
            valueFea[11] = db.fr.zero();
            Goldilocks::Element hashFea[4];
            poseidon.hash(hashFea, valueFea);
            zklog.info("poseidon=" + fea2string(db.fr, hashFea));*/
            //return ZKR_UNSPECIFIED;
        }
        checkTreeCounters.maxLevel = zkmax(checkTreeCounters.maxLevel, level);
        checkTreeCounters.values++;
        return ZKR_SUCCESS;
    }
    else
    {
        zklog.error("CheckTree() failed key=" + key + " level=" + to_string(level) + " invalid fe8=" + to_string(fe8));
        return ZKR_UNSPECIFIED;
    }
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