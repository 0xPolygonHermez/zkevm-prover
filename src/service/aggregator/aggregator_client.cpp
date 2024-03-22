
#include <nlohmann/json.hpp>
#include "aggregator_client.hpp"
#include "zklog.hpp"
#include "watchdog.hpp"
#include "zklog.hpp"
#include "witness.hpp"
#include "data_stream.hpp"

using namespace std;
using json = nlohmann::json;


/*
The proof generation process follows this schema:

genBatchProof
             \
              > genAggregatedBatchProof
             /    \
genBatchProof      \
                    \
genBlobInnerProof -----> genBlobOuterProof
                                          \
                                           > genAggregatedBlobOuterProof --> genFinalProof
                                          /
                         genBlobOuterProof

In other words:
- batch + batch => aggregated batch
- aggregated batch + batch => aggregated batch
- blob inner + aggregated batch => blob outer
- blob outer + blob outer => aggregated blob outer
- aggregated blob outer => final
*/


AggregatorClient::AggregatorClient (Goldilocks &fr, const Config &config, Prover &prover) :
    fr(fr),
    config(config),
    prover(prover)
{
    grpc::ChannelArguments channelArguments;
    channelArguments.SetMaxReceiveMessageSize(100*1024*1024);
    channelArguments.SetMaxReceiveMessageSize((config.aggregatorClientMaxRecvMsgSize == 0) ? -1 : config.aggregatorClientMaxRecvMsgSize);

    // Create channel
    std::shared_ptr<grpc::Channel> channel = ::grpc::CreateCustomChannel(config.aggregatorClientHost + ":" + to_string(config.aggregatorClientPort), grpc::InsecureChannelCredentials(), channelArguments);

    // Create stub (i.e. client)
    stub = new aggregator::v1::AggregatorService::Stub(channel);
}

void AggregatorClient::runThread (void)
{
    zklog.info("AggregatorClient::runThread() creating aggregatorClientThread");
    pthread_create(&t, NULL, aggregatorClientThread, this);
}

void AggregatorClient::waitForThread (void)
{
    pthread_join(t, NULL);
}

/**************/
/* GET STATUS */
/**************/

bool AggregatorClient::GetStatus (::aggregator::v1::GetStatusResponse &getStatusResponse)
{
    // Lock the prover
    prover.lock();

    // Set last computed request data
    getStatusResponse.set_last_computed_request_id(prover.lastComputedRequestId);
    getStatusResponse.set_last_computed_end_time(prover.lastComputedRequestEndTime);

    // If computing, set the current request data
    if ((prover.pCurrentRequest != NULL) || (prover.pendingRequests.size() > 0))
    {
        getStatusResponse.set_status(aggregator::v1::GetStatusResponse_Status_STATUS_COMPUTING);
        if (prover.pCurrentRequest != NULL)
        {
            getStatusResponse.set_current_computing_request_id(prover.pCurrentRequest->uuid);
            getStatusResponse.set_current_computing_start_time(prover.pCurrentRequest->startTime);
        }
        else
        {
            getStatusResponse.set_current_computing_request_id("");
            getStatusResponse.set_current_computing_start_time(0);
        }
    }
    else
    {
        getStatusResponse.set_status(aggregator::v1::GetStatusResponse_Status_STATUS_IDLE);
        getStatusResponse.set_current_computing_request_id("");
        getStatusResponse.set_current_computing_start_time(0);
    }

    // Set the versions
    getStatusResponse.set_version_proto("v0_0_1");
    getStatusResponse.set_version_server("0.0.1");

    // Set the list of pending requests uuids
    for (uint64_t i=0; i<prover.pendingRequests.size(); i++)
    {
        getStatusResponse.add_pending_request_queue_ids(prover.pendingRequests[i]->uuid);
    }

    // Unlock the prover
    prover.unlock();

    // Set the prover id
    getStatusResponse.set_prover_id(config.proverID);

    // Set the prover name
    getStatusResponse.set_prover_name(config.proverName);

    // Set the number of cores
    getStatusResponse.set_number_of_cores(getNumberOfCores());

    // Set the system memory details
    MemoryInfo memoryInfo;
    getMemoryInfo(memoryInfo);
    getStatusResponse.set_total_memory(memoryInfo.total);
    getStatusResponse.set_free_memory(memoryInfo.free);
    getStatusResponse.set_fork_id(PROVER_FORK_ID);

#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GetStatus() returns: " + getStatusResponse.DebugString());
#endif
    return true;
}

/*******************/
/* GEN BATCH PROOF */
/*******************/

bool AggregatorClient::GenBatchProof (const aggregator::v1::GenBatchProofRequest &genBatchProofRequest, aggregator::v1::GenBatchProofResponse &genBatchProofResponse)
{
#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenBatchProof() called with request: " + genBatchProofRequest.DebugString());
#endif
    ProverRequest * pProverRequest = new ProverRequest(fr, config, prt_genBatchProof);
    if (pProverRequest == NULL)
    {
        zklog.error("AggregatorClient::GenBatchProof() failed allocation a new ProveRequest");
        exitProcess();
    }
#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenBatchProof() created a new prover request: " + to_string((uint64_t)pProverRequest));
#endif

    // Parse public inputs

    // Get oldStateRoot
    if (genBatchProofRequest.input().public_inputs().old_state_root().size() > 32)
    {
        zklog.error("AggregatorClient::GenBatchProof() got oldStateRoot too long, size=" + to_string(genBatchProofRequest.input().public_inputs().old_state_root().size()));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.oldStateRoot, genBatchProofRequest.input().public_inputs().old_state_root());

    // Get oldAccInputHash
    if (genBatchProofRequest.input().public_inputs().old_batch_acc_input_hash().size() > 32)
    {
        zklog.error("AggregatorClient::GenBatchProof() got oldAccInputHash too long, size=" + to_string(genBatchProofRequest.input().public_inputs().old_batch_acc_input_hash().size()));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.oldAccInputHash, genBatchProofRequest.input().public_inputs().old_batch_acc_input_hash());

    // Get previousL1InfoTreeRoot
    if (genBatchProofRequest.input().public_inputs().previous_l1_info_tree_root().size() > 32)
    {
        zklog.error("AggregatorClient::GenBatchProof() got previousL1InfoTreeRoot too long, size=" + to_string(genBatchProofRequest.input().public_inputs().previous_l1_info_tree_root().size()));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.previousL1InfoTreeRoot, genBatchProofRequest.input().public_inputs().previous_l1_info_tree_root());

    // Get previousL1InfoTreeIndex
    pProverRequest->input.publicInputsExtended.publicInputs.previousL1InfoTreeIndex = genBatchProofRequest.input().public_inputs().previous_l1_info_tree_index();

    // Get chain ID
    pProverRequest->input.publicInputsExtended.publicInputs.chainID = genBatchProofRequest.input().public_inputs().chain_id();
    if (pProverRequest->input.publicInputsExtended.publicInputs.chainID == 0)
    {
        zklog.error("AggregatorClient::GenBatchProof() got chainID = 0");
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Get fork ID
    pProverRequest->input.publicInputsExtended.publicInputs.forkID = genBatchProofRequest.input().public_inputs().fork_id();
    if (pProverRequest->input.publicInputsExtended.publicInputs.forkID != PROVER_FORK_ID)
    {
        zklog.error("AggregatorClient::GenBatchProof() got an invalid prover ID=" + to_string(pProverRequest->input.publicInputsExtended.publicInputs.forkID) + " different from expected=" + to_string(PROVER_FORK_ID));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Create full tracer based on fork ID
    pProverRequest->CreateFullTracer();
    if (pProverRequest->result != ZKR_SUCCESS)
    {
        zklog.error("AggregatorClient::GenBatchProof() failed calling pProverRequest->CreateFullTracer() result=" + to_string(pProverRequest->result) + "=" + zkresult2string(pProverRequest->result));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Get batch L2 data
    if (genBatchProofRequest.input().public_inputs().batch_l2_data().size() > MAX_BATCH_L2_DATA_SIZE)
    {
        zklog.error("AggregatorClient::GenBatchProof() found batchL2Data.size()=" + to_string(genBatchProofRequest.input().public_inputs().batch_l2_data().size()) + " > MAX_BATCH_L2_DATA_SIZE=" + to_string(MAX_BATCH_L2_DATA_SIZE));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    pProverRequest->input.publicInputsExtended.publicInputs.batchL2Data = genBatchProofRequest.input().public_inputs().batch_l2_data();

    // Get sequencer address
    string auxString = Remove0xIfPresent(genBatchProofRequest.input().public_inputs().sequencer_addr());
    if (auxString.size() > 40)
    {
        zklog.error("AggregatorClient::GenBatchProof() got sequencerAddr too long, size=" + to_string(auxString.size()));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    if (!stringIsHex(auxString))
    {
        zklog.error("AggregatorClient::GenBatchProof() got sequencer address not hex, sequencer_addr=" + auxString);
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    pProverRequest->input.publicInputsExtended.publicInputs.sequencerAddr.set_str(auxString, 16);

    // Get type
    pProverRequest->input.publicInputsExtended.publicInputs.type = genBatchProofRequest.input().public_inputs().type();

    // Get forcedHashData
    if (genBatchProofRequest.input().public_inputs().forced_hash_data().size() > 32)
    {
        zklog.error("AggregatorClient::GenBatchProof() got forcedHashData too long, size=" + to_string(genBatchProofRequest.input().public_inputs().forced_hash_data().size()));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.forcedHashData, genBatchProofRequest.input().public_inputs().forced_hash_data());

    // Get forced data global exit root
    if (genBatchProofRequest.input().public_inputs().forced_data().global_exit_root().size() > 32)
    {
        zklog.error("AggregatorClient::GenBatchProof() got forced data global exit root too long, size=" + to_string(genBatchProofRequest.input().public_inputs().forced_data().global_exit_root().size()));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.forcedData.globalExitRoot, genBatchProofRequest.input().public_inputs().forced_data().global_exit_root());

    // Get forced data block hash L1
    if (genBatchProofRequest.input().public_inputs().forced_data().block_hash_l1().size() > 32)
    {
        zklog.error("AggregatorClient::GenBatchProof() got forced data block hah L1 too long, size=" + to_string(genBatchProofRequest.input().public_inputs().forced_data().block_hash_l1().size()));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.forcedData.blockHashL1, genBatchProofRequest.input().public_inputs().forced_data().block_hash_l1());

    // Get forced data minimum timestamp
    pProverRequest->input.publicInputsExtended.publicInputs.forcedData.minTimestamp = genBatchProofRequest.input().public_inputs().forced_data().min_timestamp();

    // Get aggregator address
    auxString = Remove0xIfPresent(genBatchProofRequest.input().public_inputs().aggregator_addr());
    if (auxString.size() > 40)
    {
        zklog.error("AggregatorClient::GenBatchProof() got aggregator address too long, size=" + to_string(auxString.size()));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    if (!stringIsHex(auxString))
    {
        zklog.error("AggregatorClient::GenBatchProof() got aggregator address not hex, sequencer_addr=" + auxString);
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    pProverRequest->input.publicInputsExtended.publicInputs.aggregatorAddress.set_str(auxString, 16);

    // Parse L1 info tree data
    const google::protobuf::Map<google::protobuf::uint32, aggregator::v1::L1Data> &l1InfoTreeData = genBatchProofRequest.input().public_inputs().l1_info_tree_data();
    google::protobuf::Map<google::protobuf::uint32, aggregator::v1::L1Data>::const_iterator itl;
    for (itl=l1InfoTreeData.begin(); itl!=l1InfoTreeData.end(); itl++)
    {
        // Get index
        uint64_t index = itl->first;

        // Get L1 data
        L1Data l1Data;
        const aggregator::v1::L1Data &l1DataV3 = itl->second;
        if (l1DataV3.global_exit_root().size() > 32)
        {
            zklog.error("AggregatorClient::GenBatchProof() got L1 Data global exit root too long, index=" + to_string(itl->first) + " size=" + to_string(l1DataV3.global_exit_root().size()));
            genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
            return false;
        }
        ba2scalar(l1Data.globalExitRoot, l1DataV3.global_exit_root());
        if (l1DataV3.block_hash_l1().size() > 32)
        {
            zklog.error("AggregatorClient::GenBatchProof() got L1 Data block hash L1 too long, index=" + to_string(itl->first) + " size=" + to_string(l1DataV3.block_hash_l1().size()));
            genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
            return false;
        }
        ba2scalar(l1Data.blockHashL1, l1DataV3.block_hash_l1());
        l1Data.minTimestamp = l1DataV3.min_timestamp();
        for (int64_t i=0; i<l1DataV3.smt_proof_size(); i++)
        {
            mpz_class auxScalar;
            if (l1DataV3.smt_proof(i).size() > 32)
            {
                zklog.error("AggregatorClient::GenBatchProof() got L1 Data SMT proof too long, index=" + to_string(itl->first) + " i=" + to_string(i) + " size=" + to_string(l1DataV3.smt_proof(i).size()));
                genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
                return false;
            }
            ba2scalar(auxScalar, l1DataV3.smt_proof(i));
            l1Data.smtProof.emplace_back(auxScalar);
        }
        if (l1DataV3.initial_historic_root().size() > 32)
        {
            zklog.error("AggregatorClient::GenBatchProof() got L1 Data initial historic root too long, index=" + to_string(itl->first) + " size=" + to_string(l1DataV3.initial_historic_root().size()));
            genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
            return false;
        }
        ba2scalar(l1Data.initialHistoricRoot, l1DataV3.initial_historic_root());

        // Store it
        pProverRequest->input.l1InfoTreeData[index] = l1Data;
    }

    // Parse keys map
    const google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> > &db = genBatchProofRequest.input().db();
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> >::const_iterator it;
    string key;
    for (it=db.begin(); it!=db.end(); it++)
    {
        // Get key
        key = it->first;
        Remove0xIfPresentNoCopy(key);
        if (key.size() > 64)
        {
            zklog.error("AggregatorClient::GenBatchProof() got db key too long, size=" + to_string(key.size()));
            genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
            return false;
        }
        if (!stringIsHex(key))
        {
            zklog.error("AggregatorClient::GenBatchProof() got db key not hex, key=" + key);
            genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
            return false;
        }
        PrependZerosNoCopy(key, 64);

        // Get value
        vector<Goldilocks::Element> dbValue;
        string concatenatedValues = it->second;
        if (!stringIsHex(concatenatedValues))
        {
            zklog.error("AggregatorClient::GenBatchProof() found db value not hex: " + concatenatedValues);
            genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
            return false;
        }
        if (concatenatedValues.size()%16!=0)
        {
            zklog.error("AggregatorClient::GenBatchProof() found invalid db value size: " + to_string(concatenatedValues.size()));
            genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
            return false;
        }
        for (uint64_t i=0; i<concatenatedValues.size(); i+=16)
        {
            Goldilocks::Element fe;
            string2fe(fr, concatenatedValues.substr(i, 16), fe);
            dbValue.push_back(fe);
        }
        
        // Save key-value
        pProverRequest->input.db[key] = dbValue;
    }

    // Parse contracts data
    const google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> > &contractsBytecode = genBatchProofRequest.input().contracts_bytecode();
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> >::const_iterator itp;
    for (itp=contractsBytecode.begin(); itp!=contractsBytecode.end(); itp++)
    {
        // Get key
        key = itp->first;
        Remove0xIfPresentNoCopy(key);
        if (key.size() > (64))
        {
            zklog.error("AggregatorClient::GenBatchProof() got contracts key too long, size=" + to_string(key.size()));
            genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
            return false;
        }
        if (!stringIsHex(key))
        {
            zklog.error("AggregatorClient::GenBatchProof() got contracts key not hex, key=" + key);
            genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
            return false;
        }
        PrependZerosNoCopy(key, 64);
        
        // Get value
        if (!stringIsHex(itp->second))
        {
            zklog.error("AggregatorClient::GenBatchProof() got contracts value not hex, value=" + itp->second);
            genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
            return false;
        }
        vector<uint8_t> dbValue;
        string contractValue = string2ba(itp->second);
        for (uint64_t i=0; i<contractValue.size(); i++)
        {
            dbValue.push_back(contractValue.at(i));
        }

        // Save key-value
        pProverRequest->input.contractsBytecode[key] = dbValue;
    }

    // Submit the prover request
    string uuid = prover.submitRequest(pProverRequest);

    // Build the response as Ok, returning the UUID assigned by the prover to this request
    genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_OK);
    genBatchProofResponse.set_id(uuid.c_str());

#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenBatchProof() returns: " + genBatchProofResponse.DebugString());
#endif
    return true;
}

/*****************************/
/* GEN STATELESS BATCH PROOF */
/*****************************/

bool AggregatorClient::GenStatelessBatchProof (const aggregator::v1::GenStatelessBatchProofRequest &genStatelessBatchProofRequest, aggregator::v1::GenBatchProofResponse &genBatchProofResponse)
{
#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenStatelessBatchProof() called with request: " + genStatelessBatchProofRequest.DebugString());
#endif
    ProverRequest * pProverRequest = new ProverRequest(fr, config, prt_genBatchProof);
    if (pProverRequest == NULL)
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() failed allocation a new ProveRequest");
        exitProcess();
    }
#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenStatelessBatchProof() created a new prover request: " + to_string((uint64_t)pProverRequest));
#endif

    // Parse public inputs
    
    // Get witness
    const string &witness = genStatelessBatchProofRequest.input().public_inputs().witness();
    if (witness.empty())
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() got an empty witness", &pProverRequest->tags);
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Parse witness and get db, programs and old state root
    zkresult zkr;
    zkr = witness2db(witness, pProverRequest->input.db, pProverRequest->input.contractsBytecode, pProverRequest->input.publicInputsExtended.publicInputs.oldStateRoot);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() failed calling witness2db() result=" + zkresult2string(zkr), &pProverRequest->tags);
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Get data stream
    const string &dataStream = genStatelessBatchProofRequest.input().public_inputs().data_stream();
    if (dataStream.empty())
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() got an empty data stream", &pProverRequest->tags);
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Parse data stream and get a binary structure
    DataStreamBatch batch;
    zkr = dataStream2batch(dataStream, batch);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() failed calling dataStream2batch() result=" + zkresult2string(zkr), &pProverRequest->tags);
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    if (batch.blocks.empty())
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() called dataStream2batch() but got zero blocks=", &pProverRequest->tags);
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Get batchL2Data
    zkr = dataStreamBatch2batchL2Data(batch, pProverRequest->input.publicInputsExtended.publicInputs.batchL2Data);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() failed calling dataStreamBatch2batchL2Data() result=" + zkresult2string(zkr), &pProverRequest->tags);
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    if (pProverRequest->input.publicInputsExtended.publicInputs.batchL2Data.size() > MAX_BATCH_L2_DATA_SIZE)
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() found batchL2Data.size()=" + to_string(pProverRequest->input.publicInputsExtended.publicInputs.batchL2Data.size()) + " > MAX_BATCH_L2_DATA_SIZE=" + to_string(MAX_BATCH_L2_DATA_SIZE), &pProverRequest->tags);
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Get oldAccInputHash
    if (genStatelessBatchProofRequest.input().public_inputs().old_acc_input_hash().size() > 32)
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() got oldAccInputHash too long, size=" + to_string(genStatelessBatchProofRequest.input().public_inputs().old_acc_input_hash().size()));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.oldAccInputHash, genStatelessBatchProofRequest.input().public_inputs().old_acc_input_hash());

    // Get oldBatchNum
    pProverRequest->input.publicInputsExtended.publicInputs.oldBatchNum = batch.batchNumber;

    // Get chain ID
    pProverRequest->input.publicInputsExtended.publicInputs.chainID = batch.chainId;
    if (pProverRequest->input.publicInputsExtended.publicInputs.chainID == 0)
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() got chainID = 0");
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Get fork ID
    pProverRequest->input.publicInputsExtended.publicInputs.forkID = batch.forkId;
    if (pProverRequest->input.publicInputsExtended.publicInputs.forkID != PROVER_FORK_ID)
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() got an invalid prover ID=" + to_string(pProverRequest->input.publicInputsExtended.publicInputs.forkID) + " different from expected=" + to_string(PROVER_FORK_ID));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Create full tracer based on fork ID
    pProverRequest->CreateFullTracer();
    if (pProverRequest->result != ZKR_SUCCESS)
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() failed calling pProverRequest->CreateFullTracer() result=" + to_string(pProverRequest->result) + "=" + zkresult2string(pProverRequest->result));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Get L1 info root
    if (genStatelessBatchProofRequest.input().public_inputs().l1_info_root().size() > 32)
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() got l1_info_root too long, size=" + to_string(genStatelessBatchProofRequest.input().public_inputs().l1_info_root().size()));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.l1InfoRoot, genStatelessBatchProofRequest.input().public_inputs().l1_info_root());

    // Get forced block hash L1
    if (genStatelessBatchProofRequest.input().public_inputs().forced_blockhash_l1().size() > 32)
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() got forced_blockhash_l1 too long, size=" + to_string(genStatelessBatchProofRequest.input().public_inputs().forced_blockhash_l1().size()));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.forcedBlockHashL1, genStatelessBatchProofRequest.input().public_inputs().forced_blockhash_l1());

    // Get timestamp limit
    pProverRequest->input.publicInputsExtended.publicInputs.timestampLimit = genStatelessBatchProofRequest.input().public_inputs().timestamp_limit();

    // Get sequencer address
    string auxString = Remove0xIfPresent(genStatelessBatchProofRequest.input().public_inputs().sequencer_addr());
    if (auxString.size() > 40)
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() got sequencerAddr too long, size=" + to_string(auxString.size()));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    if (!stringIsHex(auxString))
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() got sequencer address not hex, sequencer_addr=" + auxString);
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    pProverRequest->input.publicInputsExtended.publicInputs.sequencerAddr.set_str(auxString, 16);

    // Get aggregator address
    auxString = Remove0xIfPresent(genStatelessBatchProofRequest.input().public_inputs().aggregator_addr());
    if (auxString.size() > 40)
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() got aggregator address too long, size=" + to_string(auxString.size()));
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    if (!stringIsHex(auxString))
    {
        zklog.error("AggregatorClient::GenStatelessBatchProof() got aggregator address not hex, sequencer_addr=" + auxString);
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    pProverRequest->input.publicInputsExtended.publicInputs.aggregatorAddress.set_str(auxString, 16);

    // Parse L1 info tree data
    const google::protobuf::Map<google::protobuf::uint32, aggregator::v1::L1Data> &l1InfoTreeData = genStatelessBatchProofRequest.input().public_inputs().l1_info_tree_data();
    google::protobuf::Map<google::protobuf::uint32, aggregator::v1::L1Data>::const_iterator itl;
    for (itl=l1InfoTreeData.begin(); itl!=l1InfoTreeData.end(); itl++)
    {
        // Get index
        uint64_t index = itl->first;

        // Get L1 data
        L1Data l1Data;
        const aggregator::v1::L1Data &l1DataV2 = itl->second;
        if (l1DataV2.global_exit_root().size() > 32)
        {
            zklog.error("AggregatorClient::GenStatelessBatchProof()() got l1DataV2.global_exit_root() too long, size=" + to_string(l1DataV2.global_exit_root().size()), &(pProverRequest->tags));
            genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
            return false;
        }
        ba2scalar(l1Data.globalExitRoot, l1DataV2.global_exit_root());
        if (l1DataV2.block_hash_l1().size() > 32)
        {
            zklog.error("AggregatorClient::GenStatelessBatchProof()() got l1DataV2.block_hash_l1() too long, size=" + to_string(l1DataV2.block_hash_l1().size()), &(pProverRequest->tags));
            genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
            return false;
        }
        ba2scalar(l1Data.blockHashL1, l1DataV2.block_hash_l1());
        l1Data.minTimestamp = l1DataV2.min_timestamp();
        for (int64_t i=0; i<l1DataV2.smt_proof_size(); i++)
        {
            mpz_class auxScalar;
            if (l1DataV2.smt_proof(i).size() > 32)
            {
                zklog.error("AggregatorClient::GenStatelessBatchProof()() got l1DataV2.smt_proof(i) too long, size=" + to_string(l1DataV2.smt_proof(i).size()), &(pProverRequest->tags));
                genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
                return false;
            }
            ba2scalar(auxScalar, l1DataV2.smt_proof(i));
            l1Data.smtProof.emplace_back(auxScalar);
        }

        // Store it
        pProverRequest->input.l1InfoTreeData[index] = l1Data;
    }

    // ROOT

    // Get from
    pProverRequest->input.from = "0x0";

    // Flags
    pProverRequest->input.bUpdateMerkleTree = false;
    pProverRequest->input.bNoCounters = false;
    pProverRequest->input.bGetKeys = false;
    pProverRequest->input.bSkipVerifyL1InfoRoot = false;
    pProverRequest->input.bSkipFirstChangeL2Block = false;
    pProverRequest->input.bSkipWriteBlockInfoRoot = false;

    // Default values
    pProverRequest->input.publicInputsExtended.newStateRoot = "0x0";
    pProverRequest->input.publicInputsExtended.newAccInputHash = "0x0";
    pProverRequest->input.publicInputsExtended.newLocalExitRoot = "0x0";
    pProverRequest->input.publicInputsExtended.newBatchNum = 0;

    // Submit the prover request
    string uuid = prover.submitRequest(pProverRequest);

    // Build the response as Ok, returning the UUID assigned by the prover to this request
    genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_OK);
    genBatchProofResponse.set_id(uuid.c_str());

#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenStatelessBatchProof() returns: " + genBatchProofResponse.DebugString());
#endif
    return true;
}

/******************************/
/* GEN AGGREGATED BATCH PROOF */
/******************************/

bool AggregatorClient::GenAggregatedBatchProof (const aggregator::v1::GenAggregatedBatchProofRequest &genAggregatedProofRequest, aggregator::v1::GenAggregatedBatchProofResponse &genAggregatedProofResponse)
{
#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenAggregatedProof() called with request: " + genAggregatedProofRequest.DebugString());
#endif
    ProverRequest * pProverRequest = new ProverRequest(fr, config, prt_genAggregatedBatchProof);
    if (pProverRequest == NULL)
    {
        zklog.error("AggregatorClient::GenAggregatedProof() failed allocation a new ProveRequest");
        exitProcess();
    }
#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenAggregatedProof() created a new prover request: " + to_string((uint64_t)pProverRequest));
#endif

    // Set the 2 inputs
    pProverRequest->aggregatedBatchProofInput1 = json::parse(genAggregatedProofRequest.recursive_proof_1());
    pProverRequest->aggregatedBatchProofInput2 = json::parse(genAggregatedProofRequest.recursive_proof_2());

    // Submit the prover request
    string uuid = prover.submitRequest(pProverRequest);

    // Build the response as Ok, returning the UUID assigned by the prover to this request
    genAggregatedProofResponse.set_result(aggregator::v1::Result::RESULT_OK);
    genAggregatedProofResponse.set_id(uuid.c_str());

#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenAggregatedProof() returns: " + genAggregatedProofResponse.DebugString());
#endif
    return true;
}

/************************/
/* GEN BLOB INNER PROOF */
/************************/

bool AggregatorClient::GenBlobInnerProof (const aggregator::v1::GenBlobInnerProofRequest &genBlobInnerProofRequest, aggregator::v1::GenBlobInnerProofResponse &genBlobInnerProofResponse)
{
#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenBlobInnerProof() called with request: " + genBlobOuterProofRequest.DebugString());
#endif
    ProverRequest * pProverRequest = new ProverRequest(fr, config, prt_genBlobInnerProof);
    if (pProverRequest == NULL)
    {
        zklog.error("AggregatorClient::GenBlobInnerProof() failed allocation a new ProveRequest");
        exitProcess();
    }
#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenBlobInnerProof() created a new prover request: " + to_string((uint64_t)pProverRequest));
#endif

    // Get oldBlobStateRoot
    if (genBlobInnerProofRequest.input().public_inputs().old_blob_state_root().size() > 32)
    {
        zklog.error("AggregatorClient::GenBlobInnerProof() got oldBlobStateRoot too long, size=" + to_string(genBlobInnerProofRequest.input().public_inputs().old_blob_state_root().size()));
        genBlobInnerProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.oldBlobStateRoot, genBlobInnerProofRequest.input().public_inputs().old_blob_state_root());

    // Get oldBlobAccInputHash
    if (genBlobInnerProofRequest.input().public_inputs().old_blob_acc_input_hash().size() > 32)
    {
        zklog.error("AggregatorClient::GenBlobInnerProof() got oldBlobAccInputHash too long, size=" + to_string(genBlobInnerProofRequest.input().public_inputs().old_blob_acc_input_hash().size()));
        genBlobInnerProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.oldBlobAccInputHash, genBlobInnerProofRequest.input().public_inputs().old_blob_acc_input_hash());

    // Get oldBlobNum
    pProverRequest->input.publicInputsExtended.publicInputs.oldBlobNum = genBlobInnerProofRequest.input().public_inputs().old_num_blob();

    // Get oldStateRoot
    if (genBlobInnerProofRequest.input().public_inputs().old_state_root().size() > 32)
    {
        zklog.error("AggregatorClient::GenBlobInnerProof() got oldStateRoot too long, size=" + to_string(genBlobInnerProofRequest.input().public_inputs().old_state_root().size()));
        genBlobInnerProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.oldStateRoot, genBlobInnerProofRequest.input().public_inputs().old_state_root());

    // Get fork ID
    pProverRequest->input.publicInputsExtended.publicInputs.forkID = genBlobInnerProofRequest.input().public_inputs().fork_id();
    if (pProverRequest->input.publicInputsExtended.publicInputs.forkID < 9)
    {
        zklog.error("AggregatorClient::GenBlobInnerProof() got oldStateRoot too long, size=" + to_string(pProverRequest->input.publicInputsExtended.publicInputs.forkID));
        genBlobInnerProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Create full tracer based on fork ID
    pProverRequest->CreateFullTracer();
    if (pProverRequest->result != ZKR_SUCCESS)
    {
        zklog.error("AggregatorClient::GenBlobInnerProof() failed calling pProverRequest->CreateFullTracer() result=" + to_string(pProverRequest->result) + "=" + zkresult2string(pProverRequest->result));
        genBlobInnerProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Get lastL1InfoTreeIndex
    pProverRequest->input.publicInputsExtended.publicInputs.lastL1InfoTreeIndex = genBlobInnerProofRequest.input().public_inputs().last_l1_info_tree_index();

    // Get lastL1InfoTreeRoot
    if (genBlobInnerProofRequest.input().public_inputs().last_l1_info_tree_root().size() > 32)
    {
        zklog.error("AggregatorClient::GenBlobInnerProof() got lastL1InfoTreeRoot too long, size=" + to_string(genBlobInnerProofRequest.input().public_inputs().last_l1_info_tree_root().size()));
        genBlobInnerProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.lastL1InfoTreeRoot, genBlobInnerProofRequest.input().public_inputs().last_l1_info_tree_root());

    // Get sequencer address
    string auxString = Remove0xIfPresent(genBlobInnerProofRequest.input().public_inputs().sequencer_addr());
    if (auxString.size() > 40)
    {
        zklog.error("AggregatorClient::GenBlobInnerProof() got sequencerAddr too long, size=" + to_string(auxString.size()));
        genBlobInnerProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    if (!stringIsHex(auxString))
    {
        zklog.error("AggregatorClient::GenBlobInnerProof() got sequencerAddr not hex, value=" + auxString);
        genBlobInnerProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    pProverRequest->input.publicInputsExtended.publicInputs.sequencerAddr.set_str(auxString, 16);

    // Get timestamp limit
    pProverRequest->input.publicInputsExtended.publicInputs.timestampLimit = genBlobInnerProofRequest.input().public_inputs().timestamp_limit();

    // Get zkGasLimit
    if (genBlobInnerProofRequest.input().public_inputs().zk_gas_limit().size() > 32)
    {
        zklog.error("AggregatorClient::GenBlobInnerProof() got zkGasLimit too long, size=" + to_string(genBlobInnerProofRequest.input().public_inputs().zk_gas_limit().size()));
        genBlobInnerProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.zkGasLimit, genBlobInnerProofRequest.input().public_inputs().zk_gas_limit());

    // Get type
    pProverRequest->input.publicInputsExtended.publicInputs.type = genBlobInnerProofRequest.input().public_inputs().type();

    // Get pointZ
    if (genBlobInnerProofRequest.input().public_inputs().point_z().size() > 32)
    {
        zklog.error("AggregatorClient::GenBlobInnerProof() got pointZ too long, size=" + to_string(genBlobInnerProofRequest.input().public_inputs().point_z().size()));
        genBlobInnerProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.pointZ, genBlobInnerProofRequest.input().public_inputs().point_z());

    // Get pointY
    if (genBlobInnerProofRequest.input().public_inputs().point_y().size() > 32)
    {
        zklog.error("AggregatorClient::GenBlobInnerProof() got pointY too long, size=" + to_string(genBlobInnerProofRequest.input().public_inputs().point_y().size()));
        genBlobInnerProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.pointY, genBlobInnerProofRequest.input().public_inputs().point_y());

    // Get blobData
    if (genBlobInnerProofRequest.input().public_inputs().blob_data().size() > MAX_BLOB_DATA_SIZE)
    {
        zklog.error("AggregatorClient::GenBlobInnerProof() got blobData too long, size=" + to_string(genBlobInnerProofRequest.input().public_inputs().blob_data().size()));
        genBlobInnerProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    pProverRequest->input.publicInputsExtended.publicInputs.blobData = genBlobInnerProofRequest.input().public_inputs().blob_data();

    // Get forcedHashData
    if (genBlobInnerProofRequest.input().public_inputs().forced_hash_data().size() > 32)
    {
        zklog.error("AggregatorClient::GenBlobInnerProof() got forcedHashData too long, size=" + to_string(genBlobInnerProofRequest.input().public_inputs().forced_hash_data().size()));
        genBlobInnerProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.forcedHashData, genBlobInnerProofRequest.input().public_inputs().forced_hash_data());

    // Submit the prover request
    string uuid = prover.submitRequest(pProverRequest);

    // Build the response as Ok, returning the UUID assigned by the prover to this request
    genBlobInnerProofResponse.set_result(aggregator::v1::Result::RESULT_OK);
    genBlobInnerProofResponse.set_id(uuid.c_str());

#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenBlobInnerProof() returns: " + genBlobInnerProofResponse.DebugString());
#endif
    return true;
}

/************************/
/* GEN BLOB OUTER PROOF */
/************************/

bool AggregatorClient::GenBlobOuterProof (const aggregator::v1::GenBlobOuterProofRequest &genBlobOuterProofRequest, aggregator::v1::GenBlobOuterProofResponse &genBlobOuterProofResponse)
{
#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenBlobOuterProof() called with request: " + genBlobOuterProofRequest.DebugString());
#endif
    ProverRequest * pProverRequest = new ProverRequest(fr, config, prt_genBlobOuterProof);
    if (pProverRequest == NULL)
    {
        zklog.error("AggregatorClient::GenBlobOuterProof() failed allocation a new ProveRequest");
        exitProcess();
    }
#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenBlobOuterProof() created a new prover request: " + to_string((uint64_t)pProverRequest));
#endif

    // Set the 2 inputs
    pProverRequest->blobOuterProofInputBatch = json::parse(genBlobOuterProofRequest.batch_proof());
    pProverRequest->blobOuterProofInputBlobInner = json::parse(genBlobOuterProofRequest.blob_inner_proof());

    // Submit the prover request
    string uuid = prover.submitRequest(pProverRequest);

    // Build the response as Ok, returning the UUID assigned by the prover to this request
    genBlobOuterProofResponse.set_result(aggregator::v1::Result::RESULT_OK);
    genBlobOuterProofResponse.set_id(uuid.c_str());

#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenBlobOuterProof() returns: " + genAggregatedProofResponse.DebugString());
#endif
    return true;
}

/***********************************/
/* GEN AGGREGATED BLOB OUTER PROOF */
/***********************************/

bool AggregatorClient::GenAggregatedBlobOuterProof (const aggregator::v1::GenAggregatedBlobOuterProofRequest &genAggregatedBlobOuterProofRequest, aggregator::v1::GenAggregatedBlobOuterProofResponse &genAggregatedBlobOuterProofResponse)
{
#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenAggregatedBlobOuterProof() called with request: " + genAggregatedBlobOuterProofRequest.DebugString());
#endif
    ProverRequest * pProverRequest = new ProverRequest(fr, config, prt_genAggregatedBlobOuterProof);
    if (pProverRequest == NULL)
    {
        zklog.error("AggregatorClient::GenAggregatedBlobOuterProof() failed allocation a new ProveRequest");
        exitProcess();
    }
#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenAggregatedBlobOuterProof() created a new prover request: " + to_string((uint64_t)pProverRequest));
#endif

    // Set the 2 inputs
    pProverRequest->aggregatedBlobOuterProofInput1 = json::parse(genAggregatedBlobOuterProofRequest.recursive_proof_1());
    pProverRequest->aggregatedBlobOuterProofInput2 = json::parse(genAggregatedBlobOuterProofRequest.recursive_proof_2());

    // Submit the prover request
    string uuid = prover.submitRequest(pProverRequest);

    // Build the response as Ok, returning the UUID assigned by the prover to this request
    genAggregatedBlobOuterProofResponse.set_result(aggregator::v1::Result::RESULT_OK);
    genAggregatedBlobOuterProofResponse.set_id(uuid.c_str());

#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenAggregatedBlobOuterProof() returns: " + genAggregatedBlobOuterProofResponse.DebugString());
#endif
    return true;
}

/*******************/
/* GEN FINAL PROOF */
/*******************/

bool AggregatorClient::GenFinalProof (const aggregator::v1::GenFinalProofRequest &genFinalProofRequest, aggregator::v1::GenFinalProofResponse &genFinalProofResponse)
{
#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenFinalProof() called with request: " + genFinalProofRequest.DebugString());
#endif
    ProverRequest * pProverRequest = new ProverRequest(fr, config, prt_genFinalProof);
    if (pProverRequest == NULL)
    {
        zklog.error("AggregatorClient::GenFinalProof() failed allocation a new ProveRequest");
        exitProcess();
    }
#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenFinalProof() created a new prover request: " + to_string((uint64_t)pProverRequest));
#endif

    // Set the input
    pProverRequest->finalProofInput = json::parse(genFinalProofRequest.recursive_proof());

    // Set the aggregator address
    string auxString = Remove0xIfPresent(genFinalProofRequest.aggregator_addr());
    if (auxString.size() > 40)
    {
        zklog.error("AggregatorClient::GenFinalProof() got aggregator address too long, size=" + to_string(auxString.size()));
        genFinalProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    pProverRequest->input.publicInputsExtended.publicInputs.aggregatorAddress.set_str(auxString, 16);

    // Submit the prover request
    string uuid = prover.submitRequest(pProverRequest);

    // Build the response as Ok, returning the UUID assigned by the prover to this request
    genFinalProofResponse.set_result(aggregator::v1::Result::RESULT_OK);
    genFinalProofResponse.set_id(uuid.c_str());

#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GenFinalProof() returns: " + genFinalProofResponse.DebugString());
#endif
    return true;
}

/**********/
/* CANCEL */
/**********/

bool AggregatorClient::Cancel (const aggregator::v1::CancelRequest &cancelRequest, aggregator::v1::CancelResponse &cancelResponse)
{
    // Get the cancel request UUID
    string uuid = cancelRequest.id();

    // Lock the prover
    prover.lock();

    // Map uuid to the corresponding prover request
    std::unordered_map<std::string, ProverRequest *>::iterator it = prover.requestsMap.find(uuid);
    if (it == prover.requestsMap.end())
    {
        prover.unlock();
        zklog.error("AggregatorClient::Cancel() unknown uuid: " + uuid);
        cancelResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Check if it is already completed
    if (it->second->bCompleted)
    {
        prover.unlock();
        zklog.error("AggregatorClient::Cancel() already completed uuid: " + uuid);
        cancelResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Mark the request as cancelling
    it->second->bCancelling = true;

    // Unlock the prover
    prover.unlock();

    cancelResponse.set_result(aggregator::v1::Result::RESULT_OK);

#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::Cancel() returns: " + cancelResponse.DebugString());
#endif
    return true;
}

/*************/
/* GET PROOF */
/*************/

bool AggregatorClient::GetProof (const aggregator::v1::GetProofRequest &getProofRequest, aggregator::v1::GetProofResponse &getProofResponse)
{
#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GetProof() received request: " + getProofRequest.DebugString());
#endif
    // Get the prover request UUID from the request
    string uuid = getProofRequest.id();

    // Lock the prover
    prover.lock();

    // Map uuid to the corresponding prover request
    std::unordered_map<std::string, ProverRequest *>::iterator it = prover.requestsMap.find(uuid);

    // If UUID is not found, return the proper error
    if (it == prover.requestsMap.end())
    {
        zklog.error("AggregatorClient::GetProof() invalid uuid:" + uuid);
        getProofResponse.set_result(aggregator::v1::GetProofResponse_Result_RESULT_ERROR);
        getProofResponse.set_result_string("invalid UUID");
    }
    else
    {
        ProverRequest * pProverRequest = it->second;

        // If request is not completed, return the proper result
        if (!pProverRequest->bCompleted)
        {
            //zklog.error("ZKProverServiceImpl::GetProof() not completed uuid=" + uuid);
            getProofResponse.set_result(aggregator::v1::GetProofResponse_Result_RESULT_PENDING);
            getProofResponse.set_result_string("pending");
        }
        // If request is completed, return the proof
        else
        {
            // Request is completed
            getProofResponse.set_id(uuid);
            if (pProverRequest->result != ZKR_SUCCESS)
            {
                getProofResponse.set_result(aggregator::v1::GetProofResponse_Result_RESULT_COMPLETED_ERROR);
                getProofResponse.set_result_string("completed_error");
            }
            else
            {
                getProofResponse.set_result(aggregator::v1::GetProofResponse_Result_RESULT_COMPLETED_OK);
                getProofResponse.set_result_string("completed");
            }

            switch (pProverRequest->type)
            {
                case prt_genFinalProof:
                {
                    aggregator::v1::FinalProof * pFinalProof = new aggregator::v1::FinalProof();
                    zkassert(pFinalProof != NULL);

                    pFinalProof->set_proof(pProverRequest->proof.getStringProof());

                    // Set public inputs extended
                    aggregator::v1::PublicInputs* pPublicInputs = new(aggregator::v1::PublicInputs);
                    pPublicInputs->set_old_state_root(scalar2ba(pProverRequest->proof.publicInputsExtended.publicInputs.oldStateRoot));
                    pPublicInputs->set_old_batch_acc_input_hash(scalar2ba(pProverRequest->proof.publicInputsExtended.publicInputs.oldAccInputHash));
                    pPublicInputs->set_previous_l1_info_tree_root(scalar2ba(pProverRequest->proof.publicInputsExtended.publicInputs.previousL1InfoTreeRoot));
                    pPublicInputs->set_previous_l1_info_tree_index(pProverRequest->proof.publicInputsExtended.publicInputs.previousL1InfoTreeIndex);
                    pPublicInputs->set_chain_id(pProverRequest->proof.publicInputsExtended.publicInputs.chainID);
                    pPublicInputs->set_fork_id(pProverRequest->proof.publicInputsExtended.publicInputs.forkID);
                    pPublicInputs->set_batch_l2_data(pProverRequest->proof.publicInputsExtended.publicInputs.batchL2Data);
                    pPublicInputs->set_type(pProverRequest->proof.publicInputsExtended.publicInputs.type);
                    pPublicInputs->set_sequencer_addr(Add0xIfMissing(pProverRequest->proof.publicInputsExtended.publicInputs.sequencerAddr.get_str(16)));
                    pPublicInputs->set_forced_hash_data(scalar2ba(pProverRequest->proof.publicInputsExtended.publicInputs.forcedHashData));
                    //ForcedData
                    pPublicInputs->set_aggregator_addr(Add0xIfMissing(pProverRequest->proof.publicInputsExtended.publicInputs.aggregatorAddress.get_str(16)));
                    // L1InfoTreeData
                    aggregator::v1::PublicInputsExtended* pPublicInputsExtended = new(aggregator::v1::PublicInputsExtended);
                    pPublicInputsExtended->set_allocated_public_inputs(pPublicInputs);
                    pPublicInputsExtended->set_new_state_root(scalar2ba(pProverRequest->proof.publicInputsExtended.newStateRoot));
                    pPublicInputsExtended->set_new_batch_acc_input_hash(scalar2ba(pProverRequest->proof.publicInputsExtended.newAccInputHash));
                    pPublicInputsExtended->set_new_local_exit_root(scalar2ba(pProverRequest->proof.publicInputsExtended.newLocalExitRoot));
                    pPublicInputsExtended->set_current_l1_info_tree_root(scalar2ba(pProverRequest->proof.publicInputsExtended.currentL1InfoTreeRoot));
                    pPublicInputsExtended->set_current_l1_info_tree_index(pProverRequest->proof.publicInputsExtended.currentL1InfoTreeIndex);
                    pFinalProof->set_allocated_public_(pPublicInputsExtended);

                    getProofResponse.set_allocated_final_proof(pFinalProof);

                    break;
                }
                case prt_genBatchProof:
                {
                    string recursiveProof = pProverRequest->batchProofOutput.dump();
                    getProofResponse.set_recursive_proof(recursiveProof);
                    break;
                }
                case prt_genAggregatedBatchProof:
                {
                    string recursiveProof = pProverRequest->aggregatedBatchProofOutput.dump();
                    getProofResponse.set_recursive_proof(recursiveProof);
                    break;
                }
                default:
                {
                    zklog.error("AggregatorClient::GetProof() invalid pProverRequest->type=" + to_string(pProverRequest->type));
                    exitProcess();
                }
            }
        }
    }

    prover.unlock();

#ifdef LOG_SERVICE
    zklog.info("AggregatorClient::GetProof() sends response: " + getProofResponse.DebugString());
#endif
    return true;
}

/**********/
/* THREAD */
/**********/

void* aggregatorClientThread(void* arg)
{
    zklog.info("aggregatorClientThread() started");
    string uuid;
    AggregatorClient *pAggregatorClient = (AggregatorClient *)arg;
    Watchdog watchdog;
    uint64_t numberOfStreams = 0;

    while (true)
    {
        // Control the number of streams does not exceed the maximum
        if ((pAggregatorClient->config.aggregatorClientMaxStreams > 0) && (numberOfStreams >= pAggregatorClient->config.aggregatorClientMaxStreams))
        {
            zklog.info("aggregatorClientThread() killing process since we reached the maximum number of streams=" + to_string(pAggregatorClient->config.aggregatorClientMaxStreams));
            exit(0);
        }
        numberOfStreams++;

        ::grpc::ClientContext context;

        std::unique_ptr<grpc::ClientReaderWriter<aggregator::v1::ProverMessage, aggregator::v1::AggregatorMessage>> readerWriter;
        readerWriter = pAggregatorClient->stub->Channel(&context);
        watchdog.start(pAggregatorClient->config.aggregatorClientWatchdogTimeout);
        bool bResult;
        while (true)
        {
            ::aggregator::v1::AggregatorMessage aggregatorMessage;
            ::aggregator::v1::ProverMessage proverMessage;

            // Read a new aggregator message
            watchdog.restart();
            bResult = readerWriter->Read(&aggregatorMessage);
            if (!bResult)
            {
                zklog.error("aggregatorClientThread() failed calling readerWriter->Read(&aggregatorMessage)");
                break;
            }
            watchdog.restart();
            
            switch (aggregatorMessage.request_case())
            {
                case aggregator::v1::AggregatorMessage::RequestCase::kGetProofRequest:
                    break;
                case aggregator::v1::AggregatorMessage::RequestCase::kGetStatusRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kGenBatchProofRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kGenBlobInnerProofRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kCancelRequest:
                    zklog.info("aggregatorClientThread() got: " + aggregatorMessage.ShortDebugString());
                    break;
                case aggregator::v1::AggregatorMessage::RequestCase::kGenStatelessBatchProofRequest:
                    zklog.info("aggregatorClientThread() got genStatelessBatchProof() request");
                    break;
                case aggregator::v1::AggregatorMessage::RequestCase::kGenAggregatedBatchProofRequest:
                    zklog.info("aggregatorClientThread() got genAggregatedBatchProof() request");
                    break;
                case aggregator::v1::AggregatorMessage::RequestCase::kGenBlobOuterProofRequest:
                    zklog.info("aggregatorClientThread() got genBlobOuter() request");
                    break;
                case aggregator::v1::AggregatorMessage::RequestCase::kGenAggregatedBlobOuterProofRequest:
                    zklog.info("aggregatorClientThread() got genAggregatedBlobOuterProof() request");
                    break;
                case aggregator::v1::AggregatorMessage::RequestCase::kGenFinalProofRequest:
                    zklog.info("aggregatorClientThread() got genFinalProof() request");
                    break;
                default:
                    break;
            }

            // We return the same ID we got in the aggregator message
            proverMessage.set_id(aggregatorMessage.id());

            string filePrefix = pAggregatorClient->config.outputPath + "/" + getTimestamp() + "_" + aggregatorMessage.id() + ".";

            if (pAggregatorClient->config.saveRequestToFile)
            {
                string2file(aggregatorMessage.DebugString(), filePrefix + "aggregator_request.txt");
            }

            switch (aggregatorMessage.request_case())
            {
                case aggregator::v1::AggregatorMessage::RequestCase::kGetStatusRequest:
                {
                    // Allocate a new get status response
                    aggregator::v1::GetStatusResponse * pGetStatusResponse = new aggregator::v1::GetStatusResponse();
                    zkassertpermanent(pGetStatusResponse != NULL);

                    // Call GetStatus
                    pAggregatorClient->GetStatus(*pGetStatusResponse);

                    // Set the get status response
                    proverMessage.set_allocated_get_status_response(pGetStatusResponse);
                    break;
                }

                case aggregator::v1::AggregatorMessage::RequestCase::kGenBatchProofRequest:
                {
                    // Allocate a new gen batch proof response
                    aggregator::v1::GenBatchProofResponse * pGenBatchProofResponse = new aggregator::v1::GenBatchProofResponse();
                    zkassertpermanent(pGenBatchProofResponse != NULL);

                    // Call GenBatchProof
                    pAggregatorClient->GenBatchProof(aggregatorMessage.gen_batch_proof_request(), *pGenBatchProofResponse);

                    // Set the gen batch proof response
                    proverMessage.set_allocated_gen_batch_proof_response(pGenBatchProofResponse);
                    break;
                }

                case aggregator::v1::AggregatorMessage::RequestCase::kGenStatelessBatchProofRequest:
                {
                    // Allocate a new gen batch proof response
                    aggregator::v1::GenBatchProofResponse * pGenBatchProofResponse = new aggregator::v1::GenBatchProofResponse();
                    zkassertpermanent(pGenBatchProofResponse != NULL);

                    // Call GenBatchProof
                    pAggregatorClient->GenStatelessBatchProof(aggregatorMessage.gen_stateless_batch_proof_request(), *pGenBatchProofResponse);

                    // Set the gen batch proof response
                    proverMessage.set_allocated_gen_batch_proof_response(pGenBatchProofResponse);
                    break;
                }

                case aggregator::v1::AggregatorMessage::RequestCase::kGenAggregatedBatchProofRequest:
                {
                    // Allocate a new gen aggregated batch proof response
                    aggregator::v1::GenAggregatedBatchProofResponse * pGenAggregatedBatchProofResponse = new aggregator::v1::GenAggregatedBatchProofResponse();
                    zkassertpermanent(pGenAggregatedBatchProofResponse != NULL);

                    // Call GenAggregatedBatchProof
                    pAggregatorClient->GenAggregatedBatchProof(aggregatorMessage.gen_aggregated_batch_proof_request(), *pGenAggregatedBatchProofResponse);

                    // Set the gen aggregated batch proof response
                    proverMessage.set_allocated_gen_aggregated_batch_proof_response(pGenAggregatedBatchProofResponse);
                    break;
                }

                case aggregator::v1::AggregatorMessage::RequestCase::kGenBlobInnerProofRequest:
                {
                    // Allocate a new gen blob inner proof response
                    aggregator::v1::GenBlobInnerProofResponse * pGenBlobInnerProofResponse = new aggregator::v1::GenBlobInnerProofResponse();
                    zkassertpermanent(pGenBlobInnerProofResponse != NULL);

                    // Call GenBlobInner
                    pAggregatorClient->GenBlobInnerProof(aggregatorMessage.gen_blob_inner_proof_request(), *pGenBlobInnerProofResponse);

                    // Set the gen batch proof response
                    proverMessage.set_allocated_gen_blob_inner_proof_response(pGenBlobInnerProofResponse);
                    break;
                }

                case aggregator::v1::AggregatorMessage::RequestCase::kGenBlobOuterProofRequest:
                {
                    // Allocate a new gen blob outer proof response
                    aggregator::v1::GenBlobOuterProofResponse * pGenBlobOuterProofResponse = new aggregator::v1::GenBlobOuterProofResponse();
                    zkassertpermanent(pGenBlobOuterProofResponse != NULL);

                    // Call GenBlobOuter
                    pAggregatorClient->GenBlobOuterProof(aggregatorMessage.gen_blob_outer_proof_request(), *pGenBlobOuterProofResponse);

                    // Set the gen blob outer proof response
                    proverMessage.set_allocated_gen_blob_outer_proof_response(pGenBlobOuterProofResponse);
                    break;
                }

                case aggregator::v1::AggregatorMessage::RequestCase::kGenAggregatedBlobOuterProofRequest:
                {
                    // Allocate a new gen aggregated blob outer proof response
                    aggregator::v1::GenAggregatedBlobOuterProofResponse * pGenAggregatedBlobOuterProofResponse = new aggregator::v1::GenAggregatedBlobOuterProofResponse();
                    zkassertpermanent(pGenAggregatedBlobOuterProofResponse != NULL);

                    // Call GenAggregatedBlobOuterProof
                    pAggregatorClient->GenAggregatedBlobOuterProof(aggregatorMessage.gen_aggregated_blob_outer_proof_request(), *pGenAggregatedBlobOuterProofResponse);

                    // Set the gen aggregated blob outer proof response
                    proverMessage.set_allocated_gen_aggregated_blob_outer_proof_response(pGenAggregatedBlobOuterProofResponse);
                    break;
                }

                case aggregator::v1::AggregatorMessage::RequestCase::kGenFinalProofRequest:
                {
                    // Allocate a new gen final proof response
                    aggregator::v1::GenFinalProofResponse * pGenFinalProofResponse = new aggregator::v1::GenFinalProofResponse();
                    zkassertpermanent(pGenFinalProofResponse != NULL);

                    // Call GenFinalProof
                    pAggregatorClient->GenFinalProof(aggregatorMessage.gen_final_proof_request(), *pGenFinalProofResponse);

                    // Set the gen final proof response
                    proverMessage.set_allocated_gen_final_proof_response(pGenFinalProofResponse);
                    break;
                }

                case aggregator::v1::AggregatorMessage::RequestCase::kCancelRequest:
                {
                    // Allocate a new cancel response
                    aggregator::v1::CancelResponse * pCancelResponse = new aggregator::v1::CancelResponse();
                    zkassertpermanent(pCancelResponse != NULL);

                    // Call Cancel
                    pAggregatorClient->Cancel(aggregatorMessage.cancel_request(), *pCancelResponse);

                    // Set the cancel response
                    proverMessage.set_allocated_cancel_response(pCancelResponse);
                    break;
                }

                case aggregator::v1::AggregatorMessage::RequestCase::kGetProofRequest:
                {
                    // Allocate a new cancel response
                    aggregator::v1::GetProofResponse * pGetProofResponse = new aggregator::v1::GetProofResponse();
                    zkassertpermanent(pGetProofResponse != NULL);

                    // Call GetProof
                    pAggregatorClient->GetProof(aggregatorMessage.get_proof_request(), *pGetProofResponse);

                    // Set the get proof response
                    proverMessage.set_allocated_get_proof_response(pGetProofResponse);
                    break;
                }

                default:
                {
                    zklog.error("aggregatorClientThread() received an invalid type=" + to_string(aggregatorMessage.request_case()));
                    break;
                }
            }

            // Write the prover message
            watchdog.restart();
            bResult = readerWriter->Write(proverMessage);
            if (!bResult)
            {
                zklog.error("aggregatorClientThread() failed calling readerWriter->Write(proverMessage)");
                break;
            }
            watchdog.restart();
            
            switch (aggregatorMessage.request_case())
            {
                case aggregator::v1::AggregatorMessage::RequestCase::kGetStatusRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kGenBatchProofRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kGenStatelessBatchProofRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kGenAggregatedBatchProofRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kGenFinalProofRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kCancelRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kGenBlobInnerProofRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kGenBlobOuterProofRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kGenAggregatedBlobOuterProofRequest:
                    zklog.info("aggregatorClientThread() sent: " + proverMessage.ShortDebugString());
                    break;
                case aggregator::v1::AggregatorMessage::RequestCase::kGetProofRequest:
                    if (proverMessage.get_proof_response().result() != aggregator::v1::GetProofResponse_Result_RESULT_PENDING)
                        zklog.info("aggregatorClientThread() getProof() response sent; result=" + proverMessage.get_proof_response().result_string());
                    break;
                default:
                    break;
            }

            if (pAggregatorClient->config.saveResponseToFile)
            {
                string2file(proverMessage.DebugString(), filePrefix + "aggregator_response.txt");
            }
        }
        watchdog.stop();
        zklog.info("aggregatorClientThread() channel broken; will retry in 5 seconds");
        sleep(5);
    }
    return NULL;
}
