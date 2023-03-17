
#include <nlohmann/json.hpp>
#include "aggregator_client.hpp"

using namespace std;
using json = nlohmann::json;

AggregatorClient::AggregatorClient (Goldilocks &fr, const Config &config, Prover &prover) :
    fr(fr),
    config(config),
    prover(prover)
{
    // Create channel
    std::shared_ptr<grpc_impl::Channel> channel = ::grpc::CreateChannel(config.aggregatorClientHost + ":" + to_string(config.aggregatorClientPort), grpc::InsecureChannelCredentials());

    // Create stub (i.e. client)
    stub = new aggregator::v1::AggregatorService::Stub(channel);
}

void AggregatorClient::runThread (void)
{
    cout << "AggregatorClient::runThread() creating aggregatorClientThread" << endl;
    pthread_create(&t, NULL, aggregatorClientThread, this);
}

void AggregatorClient::waitForThread (void)
{
    pthread_join(t, NULL);
}

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
    cout << "AggregatorClient::GetStatus() returns: " << getStatusResponse.DebugString() << endl;
#endif
    return true;
}

bool AggregatorClient::GenBatchProof (const aggregator::v1::GenBatchProofRequest &genBatchProofRequest, aggregator::v1::GenBatchProofResponse &genBatchProofResponse)
{
#ifdef LOG_SERVICE
    cout << "AggregatorClient::GenBatchProof() called with request: " << genBatchProofRequest.DebugString() << endl;
#endif
    ProverRequest * pProverRequest = new ProverRequest(fr, config, prt_genBatchProof);
    if (pProverRequest == NULL)
    {
        cerr << "Error: AggregatorClient::GenBatchProof() failed allocation a new ProveRequest" << endl;
        exitProcess();
    }
#ifdef LOG_SERVICE
    cout << "AggregatorClient::GenBatchProof() created a new prover request: " << to_string((uint64_t)pProverRequest) << endl;
#endif

    // Parse public inputs

    string auxString;

    auxString = ba2string(genBatchProofRequest.input().public_inputs().old_state_root());
    if (auxString.size() > 64)
    {
        cerr << "Error: AggregatorClient::GenProof() got oldStateRoot too long, size=" << auxString.size() << endl;
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    pProverRequest->input.publicInputsExtended.publicInputs.oldStateRoot.set_str(auxString, 16);

    auxString = ba2string(genBatchProofRequest.input().public_inputs().old_acc_input_hash());
    if (auxString.size() > 64)
    {
        cerr << "Error: AggregatorClient::GenProof() got oldAccInputHash too long, size=" << auxString.size() << endl;
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    pProverRequest->input.publicInputsExtended.publicInputs.oldAccInputHash.set_str(auxString, 16);

    pProverRequest->input.publicInputsExtended.publicInputs.oldBatchNum = genBatchProofRequest.input().public_inputs().old_batch_num();

    pProverRequest->input.publicInputsExtended.publicInputs.chainID = genBatchProofRequest.input().public_inputs().chain_id();
    if (pProverRequest->input.publicInputsExtended.publicInputs.chainID == 0)
    {
        cerr << "Error: AggregatorClient::GenProof() got chainID = 0" << endl;
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    pProverRequest->input.publicInputsExtended.publicInputs.forkID = genBatchProofRequest.input().public_inputs().fork_id();

    if (pProverRequest->input.publicInputsExtended.publicInputs.forkID != PROVER_FORK_ID)
    {
        cerr << "Error: AggregatorClient::GenProof() got an invalid prover ID=" << pProverRequest->input.publicInputsExtended.publicInputs.forkID << " different from expected=" << PROVER_FORK_ID << endl;
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Create full tracer based on fork ID
    pProverRequest->CreateFullTracer();
    if (pProverRequest->result != ZKR_SUCCESS)
    {
        cerr << "Error: AggregatorClient::GenProof() failed calling pProverRequest->CreateFullTracer() result=" << pProverRequest->result << "=" << zkresult2string(pProverRequest->result) << endl;
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    pProverRequest->input.publicInputsExtended.publicInputs.batchL2Data = genBatchProofRequest.input().public_inputs().batch_l2_data();
    if (pProverRequest->input.publicInputsExtended.publicInputs.batchL2Data.size() > MAX_BATCH_L2_DATA_SIZE)
    {
        cerr << "Error: AggregatorClient::GenProof() found batchL2Data.size()=" << pProverRequest->input.publicInputsExtended.publicInputs.batchL2Data.size() << " > MAX_BATCH_L2_DATA_SIZE=" << MAX_BATCH_L2_DATA_SIZE << endl;
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    if (genBatchProofRequest.input().public_inputs().global_exit_root().size() > 32)
    {
        cerr << "Error: AggregatorClient::GenProof() got globalExitRoot too long, size=" << genBatchProofRequest.input().public_inputs().global_exit_root().size() << endl;
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    ba2scalar(pProverRequest->input.publicInputsExtended.publicInputs.globalExitRoot, genBatchProofRequest.input().public_inputs().global_exit_root());

    pProverRequest->input.publicInputsExtended.publicInputs.timestamp = genBatchProofRequest.input().public_inputs().eth_timestamp();

    auxString = Remove0xIfPresent(genBatchProofRequest.input().public_inputs().sequencer_addr());
    if (auxString.size() > 40)
    {
        cerr << "Error: AggregatorClient::GenProof() got sequencerAddr too long, size=" << auxString.size() << endl;
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    pProverRequest->input.publicInputsExtended.publicInputs.sequencerAddr.set_str(auxString, 16);

    auxString = Remove0xIfPresent(genBatchProofRequest.input().public_inputs().aggregator_addr());
    if (auxString.size() > 40)
    {
        cerr << "Error: AggregatorClient::GenProof() got aggregator address too long, size=" << auxString.size() << endl;
        genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }
    pProverRequest->input.publicInputsExtended.publicInputs.aggregatorAddress.set_str(auxString, 16);

    // Parse keys map
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> > db;
    db = genBatchProofRequest.input().db();
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> >::iterator it;
    for (it=db.begin(); it!=db.end(); it++)
    {
        if (it->first.size() > (64))
        {
            cerr << "Error: AggregatorClient::GenBatchProof() got db key too long, size=" << it->first.size() << endl;
            genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
            return false;
        }
        vector<Goldilocks::Element> dbValue;
        string concatenatedValues = it->second;
        if (concatenatedValues.size()%16!=0)
        {
            cerr << "Error: AggregatorClient::GenBatchProof() found invalid db value size: " << concatenatedValues.size() << endl;
            genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
            return false;
        }
        for (uint64_t i=0; i<concatenatedValues.size(); i+=16)
        {
            Goldilocks::Element fe;
            string2fe(fr, concatenatedValues.substr(i, 16), fe);
            dbValue.push_back(fe);
        }
        pProverRequest->input.db[it->first] = dbValue;
    }

    // Parse contracts data
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> > contractsBytecode;
    contractsBytecode = genBatchProofRequest.input().contracts_bytecode();
    google::protobuf::Map<std::__cxx11::basic_string<char>, std::__cxx11::basic_string<char> >::iterator itp;
    for (itp=contractsBytecode.begin(); itp!=contractsBytecode.end(); itp++)
    {
        /*if (it->first.size() != (2+64))
        {
            cerr << "Error: ZKProverServiceImpl::GenProof() got contracts bytecode key too long, size=" << it->first.size() << endl;
            return Status::CANCELLED;
        }*/
        vector<uint8_t> dbValue;
        string contractValue = string2ba(itp->second);
        for (uint64_t i=0; i<contractValue.size(); i++)
        {
            dbValue.push_back(contractValue.at(i));
        }
        pProverRequest->input.contractsBytecode[itp->first] = dbValue;
    }

    // Submit the prover request
    string uuid = prover.submitRequest(pProverRequest);

    // Build the response as Ok, returning the UUID assigned by the prover to this request
    genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_OK);
    genBatchProofResponse.set_id(uuid.c_str());

#ifdef LOG_SERVICE
    cout << "AggregatorClient::GenBatchProof() returns: " << genBatchProofResponse.DebugString() << endl;
#endif
    return true;
}

bool AggregatorClient::GenAggregatedProof (const aggregator::v1::GenAggregatedProofRequest &genAggregatedProofRequest, aggregator::v1::GenAggregatedProofResponse &genAggregatedProofResponse)
{
#ifdef LOG_SERVICE
    cout << "AggregatorClient::GenAggregatedProof() called with request: " << genAggregatedProofRequest.DebugString() << endl;
#endif
    ProverRequest * pProverRequest = new ProverRequest(fr, config, prt_genAggregatedProof);
    if (pProverRequest == NULL)
    {
        cerr << "Error: AggregatorClient::GenAggregatedProof() failed allocation a new ProveRequest" << endl;
        exitProcess();
    }
#ifdef LOG_SERVICE
    cout << "AggregatorClient::GenAggregatedProof() created a new prover request: " << to_string((uint64_t)pProverRequest) << endl;
#endif

    // Set the 2 inputs
    pProverRequest->aggregatedProofInput1 = json::parse(genAggregatedProofRequest.recursive_proof_1());
    pProverRequest->aggregatedProofInput2 = json::parse(genAggregatedProofRequest.recursive_proof_2());

    // Submit the prover request
    string uuid = prover.submitRequest(pProverRequest);

    // Build the response as Ok, returning the UUID assigned by the prover to this request
    genAggregatedProofResponse.set_result(aggregator::v1::Result::RESULT_OK);
    genAggregatedProofResponse.set_id(uuid.c_str());

#ifdef LOG_SERVICE
    cout << "AggregatorClient::GenAggregatedProof() returns: " << genAggregatedProofResponse.DebugString() << endl;
#endif
    return true;
}

bool AggregatorClient::GenFinalProof (const aggregator::v1::GenFinalProofRequest &genFinalProofRequest, aggregator::v1::GenFinalProofResponse &genFinalProofResponse)
{
#ifdef LOG_SERVICE
    cout << "AggregatorClient::GenFinalProof() called with request: " << genFinalProofRequest.DebugString() << endl;
#endif
    ProverRequest * pProverRequest = new ProverRequest(fr, config, prt_genFinalProof);
    if (pProverRequest == NULL)
    {
        cerr << "Error: AggregatorClient::GenFinalProof() failed allocation a new ProveRequest" << endl;
        exitProcess();
    }
#ifdef LOG_SERVICE
    cout << "AggregatorClient::GenFinalProof() created a new prover request: " << to_string((uint64_t)pProverRequest) << endl;
#endif

    // Set the input
    pProverRequest->finalProofInput = json::parse(genFinalProofRequest.recursive_proof());

    // Set the aggregator address
    string auxString = Remove0xIfPresent(genFinalProofRequest.aggregator_addr());
    if (auxString.size() > 40)
    {
        cerr << "Error: AggregatorClient::GenFinalProof() got aggregator address too long, size=" << auxString.size() << endl;
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
    cout << "AggregatorClient::GenFinalProof() returns: " << genFinalProofResponse.DebugString() << endl;
#endif
    return true;
}

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
        cerr << "Error: AggregatorClient::Cancel() unknown uuid: " << uuid << endl;
        cancelResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Check if it is already completed
    if (it->second->bCompleted)
    {
        prover.unlock();
        cerr << "Error: AggregatorClient::Cancel() already completed uuid: " << uuid << endl;
        cancelResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
        return false;
    }

    // Mark the request as cancelling
    it->second->bCancelling = true;

    // Unlock the prover
    prover.unlock();

    cancelResponse.set_result(aggregator::v1::Result::RESULT_OK);

#ifdef LOG_SERVICE
    cout << "AggregatorClient::Cancel() returns: " << cancelResponse.DebugString() << endl;
#endif
    return true;
}

bool AggregatorClient::GetProof (const aggregator::v1::GetProofRequest &getProofRequest, aggregator::v1::GetProofResponse &getProofResponse)
{
#ifdef LOG_SERVICE
    cout << "AggregatorClient::GetProof() received request: " << getProofRequest.DebugString();
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
        cerr << "Error: AggregatorClient::GetProof() invalid uuid:" << uuid << endl;
        getProofResponse.set_result(aggregator::v1::GetProofResponse_Result_RESULT_ERROR);
        getProofResponse.set_result_string("invalid UUID");
    }
    else
    {
        ProverRequest * pProverRequest = it->second;

        // If request is not completed, return the proper result
        if (!pProverRequest->bCompleted)
        {
            //cerr << "Error: ZKProverServiceImpl::GetProof() not completed uuid=" << uuid << endl;
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
                    pPublicInputs->set_old_acc_input_hash(scalar2ba(pProverRequest->proof.publicInputsExtended.publicInputs.oldAccInputHash));
                    pPublicInputs->set_old_batch_num(pProverRequest->proof.publicInputsExtended.publicInputs.oldBatchNum);
                    pPublicInputs->set_chain_id(pProverRequest->proof.publicInputsExtended.publicInputs.chainID);
                    pPublicInputs->set_fork_id(pProverRequest->proof.publicInputsExtended.publicInputs.forkID);
                    pPublicInputs->set_batch_l2_data(pProverRequest->proof.publicInputsExtended.publicInputs.batchL2Data);
                    pPublicInputs->set_global_exit_root(scalar2ba(pProverRequest->proof.publicInputsExtended.publicInputs.globalExitRoot));
                    pPublicInputs->set_eth_timestamp(pProverRequest->proof.publicInputsExtended.publicInputs.timestamp);
                    pPublicInputs->set_sequencer_addr(Add0xIfMissing(pProverRequest->proof.publicInputsExtended.publicInputs.sequencerAddr.get_str(16)));
                    pPublicInputs->set_aggregator_addr(Add0xIfMissing(pProverRequest->proof.publicInputsExtended.publicInputs.aggregatorAddress.get_str(16)));
                    aggregator::v1::PublicInputsExtended* pPublicInputsExtended = new(aggregator::v1::PublicInputsExtended);
                    pPublicInputsExtended->set_allocated_public_inputs(pPublicInputs);
                    pPublicInputsExtended->set_new_state_root(scalar2ba(pProverRequest->proof.publicInputsExtended.newStateRoot));
                    pPublicInputsExtended->set_new_acc_input_hash(scalar2ba(pProverRequest->proof.publicInputsExtended.newAccInputHash));
                    pPublicInputsExtended->set_new_local_exit_root(scalar2ba(pProverRequest->proof.publicInputsExtended.newLocalExitRoot));
                    pPublicInputsExtended->set_new_batch_num(pProverRequest->proof.publicInputsExtended.newBatchNum);
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
                case prt_genAggregatedProof:
                {
                    string recursiveProof = pProverRequest->aggregatedProofOutput.dump();
                    getProofResponse.set_recursive_proof(recursiveProof);
                    break;
                }
                default:
                {
                    cerr << "Error: AggregatorClient::GetProof() invalid pProverRequest->type=" << pProverRequest->type << endl;
                    exitProcess();
                }
            }
        }
    }
    
    prover.unlock();

#ifdef LOG_SERVICE
    cout << "AggregatorClient::GetProof() sends response: " << getProofResponse.DebugString();
#endif
    return true;
}

void* aggregatorClientThread(void* arg)
{
    cout << "aggregatorClientThread() started" << endl;
    string uuid;
    AggregatorClient *pAggregatorClient = (AggregatorClient *)arg;

    while (true)
    {
        ::grpc::ClientContext context;
        std::unique_ptr<grpc::ClientReaderWriter<aggregator::v1::ProverMessage, aggregator::v1::AggregatorMessage>> readerWriter;
        readerWriter = pAggregatorClient->stub->Channel(&context);
        bool bResult;
        while (true)
        {
            ::aggregator::v1::AggregatorMessage aggregatorMessage;
            ::aggregator::v1::ProverMessage proverMessage;

            // Read a new aggregator message
            bResult = readerWriter->Read(&aggregatorMessage);
            if (!bResult)
            {
                cerr << "Error: aggregatorClientThread() failed calling readerWriter->Read(&aggregatorMessage)" << endl;
                break;
            }
            
            switch (aggregatorMessage.request_case())
            {
                case aggregator::v1::AggregatorMessage::RequestCase::kGetProofRequest:
                    break;
                case aggregator::v1::AggregatorMessage::RequestCase::kGetStatusRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kGenBatchProofRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kCancelRequest:
                    cout << "aggregatorClientThread() got: " << aggregatorMessage.ShortDebugString() << endl;
                    break;
                case aggregator::v1::AggregatorMessage::RequestCase::kGenAggregatedProofRequest:
                    cout << "aggregatorClientThread() got genAggregatedProof() request" << endl;
                    break;
                case aggregator::v1::AggregatorMessage::RequestCase::kGenFinalProofRequest:
                    cout << "aggregatorClientThread() got genFinalProof() request" << endl;
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
                    zkassert(pGetStatusResponse != NULL);

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
                    zkassert(pGenBatchProofResponse != NULL);

                    // Call GenBatchProof
                    pAggregatorClient->GenBatchProof(aggregatorMessage.gen_batch_proof_request(), *pGenBatchProofResponse);

                    // Set the gen batch proof response
                    proverMessage.set_allocated_gen_batch_proof_response(pGenBatchProofResponse);
                    break;
                }

                case aggregator::v1::AggregatorMessage::RequestCase::kGenAggregatedProofRequest:
                {
                    // Allocate a new gen aggregated proof response
                    aggregator::v1::GenAggregatedProofResponse * pGenAggregatedProofResponse = new aggregator::v1::GenAggregatedProofResponse();
                    zkassert(pGenAggregatedProofResponse != NULL);

                    // Call GenAggregatedProof
                    pAggregatorClient->GenAggregatedProof(aggregatorMessage.gen_aggregated_proof_request(), *pGenAggregatedProofResponse);

                    // Set the gen aggregated proof response
                    proverMessage.set_allocated_gen_aggregated_proof_response(pGenAggregatedProofResponse);
                    break;
                }

                case aggregator::v1::AggregatorMessage::RequestCase::kGenFinalProofRequest:
                {
                    // Allocate a new gen final proof response
                    aggregator::v1::GenFinalProofResponse * pGenFinalProofResponse = new aggregator::v1::GenFinalProofResponse();
                    zkassert(pGenFinalProofResponse != NULL);

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
                    zkassert(pCancelResponse != NULL);

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
                    zkassert(pGetProofResponse != NULL);

                    // Call GetProof
                    pAggregatorClient->GetProof(aggregatorMessage.get_proof_request(), *pGetProofResponse);

                    // Set the get proof response
                    proverMessage.set_allocated_get_proof_response(pGetProofResponse);
                    break;
                }

                default:
                {
                    cerr << "Error: aggregatorClientThread() received an invalid type=" << aggregatorMessage.request_case() << endl;
                    break;
                }
            }

            // Write the prover message
            bResult = readerWriter->Write(proverMessage);
            if (!bResult)
            {
                cerr << "Error: aggregatorClientThread() failed calling readerWriter->Write(proverMessage)" << endl;
                break;
            }
            
            switch (aggregatorMessage.request_case())
            {
                case aggregator::v1::AggregatorMessage::RequestCase::kGetStatusRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kGenBatchProofRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kGenAggregatedProofRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kGenFinalProofRequest:
                case aggregator::v1::AggregatorMessage::RequestCase::kCancelRequest:
                    cout << "aggregatorClientThread() sent: " << proverMessage.ShortDebugString() << endl;
                    break;
                case aggregator::v1::AggregatorMessage::RequestCase::kGetProofRequest:
                    if (proverMessage.get_proof_response().result() != aggregator::v1::GetProofResponse_Result_RESULT_PENDING)
                        cout << "aggregatorClientThread() getProof() response sent; result=" << proverMessage.get_proof_response().result_string() << endl;
                    break;
                default:
                    break;
            }
            
            if (pAggregatorClient->config.saveResponseToFile)
            {
                string2file(proverMessage.DebugString(), filePrefix + "aggregator_response.txt");
            }
        }
        cout << "aggregatorClientThread() channel broken; will retry in 5 seconds" << endl;
        sleep(5);
    }
    return NULL;
}