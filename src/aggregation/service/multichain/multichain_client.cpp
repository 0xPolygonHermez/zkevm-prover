
#include <nlohmann/json.hpp>
#include "multichain_client.hpp"
#include "zklog.hpp"
#include "scalar.hpp"
#include "watchdog.hpp"

using namespace std;
using json = nlohmann::json;

MultichainClient::MultichainClient (Goldilocks &fr, const Config &config, ProverAggregation &prover) :
    fr(fr),
    config(config),
    prover(prover)
{
    // Create channel
    std::shared_ptr<grpc::Channel> channel = ::grpc::CreateChannel(config.multichainClientHost + ":" + to_string(config.multichainClientPort), grpc::InsecureChannelCredentials());

    // Create stub (i.e. client)
    stub = new multichain::v1::MultichainService::Stub(channel);
}

void MultichainClient::runThread (void)
{
    zklog.info("MultichainClient::runThread() creating multichainClientThread");
    pthread_create(&t, NULL, multichainClientThread, this);
}

void MultichainClient::waitForThread (void)
{
    pthread_join(t, NULL);
}

bool MultichainClient::GetStatus (::multichain::v1::GetStatusResponse &getStatusResponse)
{
    // Lock the prover
    prover.lock();

    // Set last computed request data
    getStatusResponse.set_last_computed_request_id(prover.lastComputedRequestId);
    getStatusResponse.set_last_computed_end_time(prover.lastComputedRequestEndTime);

    // If computing, set the current request data
    if ((prover.pCurrentRequest != NULL) || (prover.pendingRequests.size() > 0))
    {
        getStatusResponse.set_status(multichain::v1::GetStatusResponse_Status_STATUS_COMPUTING);
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
        getStatusResponse.set_status(multichain::v1::GetStatusResponse_Status_STATUS_IDLE);
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
    zklog.info("MultichainClient::GetStatus() returns: " + getStatusResponse.DebugString());
#endif
    return true;
}

bool MultichainClient::CalculateSha256(const multichain::v1::CalculateSha256Request &calculateSha256Request, multichain::v1::CalculateSha256Response &calculateSha256Response) {
    #ifdef LOG_SERVICE
    zklog.info("MultichainClient::CalculateSha256() called with request: " + calculateSha256Request.DebugString());
#endif
    ProverAggregationRequest * pProverAggregationRequest = new ProverAggregationRequest(fr, config, prt_calculateHash);
    if (pProverAggregationRequest == NULL)
    {
        zklog.error("MultichainClient::CalculateSha256() failed allocation a new ProveRequest");
        exitProcess();
    }
#ifdef LOG_SERVICE
    zklog.info("MultichainClient::CalculateSha256() created a new prover request: " + to_string((uint64_t)pProverAggregationRequest));
#endif

    // Set the 2 inputs
    pProverAggregationRequest->chainPublicsInput = json::parse(calculateSha256Request.publics());
    
    if(calculateSha256Request.previous_hash().size() > 0) {
        pProverAggregationRequest->prevHashInput = json::parse(calculateSha256Request.previous_hash());
    }

    // Call calculate Hash prover
    prover.calculateHash(pProverAggregationRequest);

    // Set the output hash as part of the response
    calculateSha256Response.set_out_hash(pProverAggregationRequest->hashOutput.dump());

#ifdef LOG_SERVICE
    zklog.info("MultichainClient::CalculateSha256() returns: " + calculateSha256Response.DebugString());
#endif
    return true;
}

bool MultichainClient::GenPrepareMultichainProof (const multichain::v1::GenPrepareMultichainProofRequest &genPrepareMultichainProofRequest, multichain::v1::GenPrepareMultichainProofResponse &genPrepareMultichainProofResponse)
{
#ifdef LOG_SERVICE
    zklog.info("MultichainClient::GenPrepareMultichainProof() called with request: " + genPrepareMultichainProofRequest.DebugString());
#endif
    ProverAggregationRequest * pProverAggregationRequest = new ProverAggregationRequest(fr, config, prt_genPrepareMultichainProof);
    if (pProverAggregationRequest == NULL)
    {
        zklog.error("MultichainClient::GenPrepareMultichainProof() failed allocation a new ProveRequest");
        exitProcess();
    }
#ifdef LOG_SERVICE
    zklog.info("MultichainClient::GenPrepareMultichainProof() created a new prover request: " + to_string((uint64_t)pProverAggregationRequest));
#endif

    // Set the 2 inputs
    pProverAggregationRequest->multichainPrepProofInput = json::parse(genPrepareMultichainProofRequest.recursive_proof());
    
    if(genPrepareMultichainProofRequest.previous_hash().size() > 0) {
        pProverAggregationRequest->multichainPrepPrevHashInput = json::parse(genPrepareMultichainProofRequest.previous_hash());
    }

    // Submit the prover request
    string uuid = prover.submitRequest(pProverAggregationRequest);

    // Build the response as Ok, returning the UUID assigned by the prover to this request
    genPrepareMultichainProofResponse.set_result(multichain::v1::Result::RESULT_OK);
    genPrepareMultichainProofResponse.set_id(uuid.c_str());

#ifdef LOG_SERVICE
    zklog.info("MultichainClient::GenPrepareMultichainProof() returns: " + genPrepareMultichainProofResponse.DebugString());
#endif
    return true;
}

bool MultichainClient::GenAggregatedMultichainProof (const multichain::v1::GenAggregatedMultichainProofRequest &genAggregatedMultichainProofRequest, multichain::v1::GenAggregatedMultichainProofResponse &genAggregatedMultichainProofResponse)
{
#ifdef LOG_SERVICE
    zklog.info("MultichainClient::GenAggregatedMultichainProof() called with request: " + genAggregatedMultichainProofRequest.DebugString());
#endif
    ProverAggregationRequest * pProverAggregationRequest = new ProverAggregationRequest(fr, config, prt_genAggregatedMultichainProof);
    if (pProverAggregationRequest == NULL)
    {
        zklog.error("MultichainClient::GenAggregatedMultichainProof() failed allocation a new ProveRequest");
        exitProcess();
    }
#ifdef LOG_SERVICE
    zklog.info("MultichainClient::GenAggregatedMultichainProof() created a new prover request: " + to_string((uint64_t)pProverAggregationRequest));
#endif

    // Set the 2 inputs
    pProverAggregationRequest->aggregatedMultichainProofInput1 = json::parse(genAggregatedMultichainProofRequest.multichain_proof_1());
    pProverAggregationRequest->aggregatedMultichainProofInput2 = json::parse(genAggregatedMultichainProofRequest.multichain_proof_2());

    // Submit the prover request
    string uuid = prover.submitRequest(pProverAggregationRequest);

    // Build the response as Ok, returning the UUID assigned by the prover to this request
    genAggregatedMultichainProofResponse.set_result(multichain::v1::Result::RESULT_OK);
    genAggregatedMultichainProofResponse.set_id(uuid.c_str());

#ifdef LOG_SERVICE
    zklog.info("MultichainClient::GenAggregatedMultichainProof() returns: " + genAggregatedMultichainProofResponse.DebugString());
#endif
    return true;
}

bool MultichainClient::GenFinalMultichainProof (const multichain::v1::GenFinalMultichainProofRequest &genFinalMultichainProofRequest, multichain::v1::GenFinalMultichainProofResponse &genFinalMultichainProofResponse)
{
#ifdef LOG_SERVICE
    zklog.info("MultichainClient::GenFinalMultichainProof() called with request: " + genFinalMultichainProofRequest.DebugString());
#endif
    ProverAggregationRequest * pProverAggregationRequest = new ProverAggregationRequest(fr, config, prt_genFinalMultichainProof);
    if (pProverAggregationRequest == NULL)
    {
        zklog.error("MultichainClient::GenFinalMultichainProof() failed allocation a new ProveRequest");
        exitProcess();
    }
#ifdef LOG_SERVICE
    zklog.info("MultichainClient::GenFinalMultichainProof() created a new prover request: " + to_string((uint64_t)pProverAggregationRequest));
#endif

    // Set the input
    pProverAggregationRequest->finalMultichainProofInput = json::parse(genFinalMultichainProofRequest.multichain_proof());

    // Set the aggregator address
    string auxString = Remove0xIfPresent(genFinalMultichainProofRequest.aggregator_addr());
    if (auxString.size() > 40)
    {
        zklog.error("MultichainClient::GenFinalMultichainProof() got aggregator address too long, size=" + to_string(auxString.size()));
        genFinalMultichainProofResponse.set_result(multichain::v1::Result::RESULT_ERROR);
        return false;
    }
    pProverAggregationRequest->aggregatorAddress = auxString;

    // Submit the prover request
    string uuid = prover.submitRequest(pProverAggregationRequest);

    // Build the response as Ok, returning the UUID assigned by the prover to this request
    genFinalMultichainProofResponse.set_result(multichain::v1::Result::RESULT_OK);
    genFinalMultichainProofResponse.set_id(uuid.c_str());

#ifdef LOG_SERVICE
    zklog.info("MultichainClient::GenFinalMultichainProof() returns: " + genFinalMultichainProofResponse.DebugString());
#endif
    return true;
}

bool MultichainClient::Cancel (const multichain::v1::CancelRequest &cancelRequest, multichain::v1::CancelResponse &cancelResponse)
{
    // Get the cancel request UUID
    string uuid = cancelRequest.id();

    // Lock the prover
    prover.lock();

    // Map uuid to the corresponding prover request
    std::unordered_map<std::string, ProverAggregationRequest *>::iterator it = prover.requestsMap.find(uuid);
    if (it == prover.requestsMap.end())
    {
        prover.unlock();
        zklog.error("MultichainClient::Cancel() unknown uuid: " + uuid);
        cancelResponse.set_result(multichain::v1::Result::RESULT_ERROR);
        return false;
    }

    // Check if it is already completed
    if (it->second->bCompleted)
    {
        prover.unlock();
        zklog.error("MultichainClient::Cancel() already completed uuid: " + uuid);
        cancelResponse.set_result(multichain::v1::Result::RESULT_ERROR);
        return false;
    }

    // Mark the request as cancelling
    it->second->bCancelling = true;

    // Unlock the prover
    prover.unlock();

    cancelResponse.set_result(multichain::v1::Result::RESULT_OK);

#ifdef LOG_SERVICE
    zklog.info("MultichainClient::Cancel() returns: " + cancelResponse.DebugString());
#endif
    return true;
}

bool MultichainClient::GetProof (const multichain::v1::GetProofRequest &getProofRequest, multichain::v1::GetProofResponse &getProofResponse)
{
#ifdef LOG_SERVICE
    zklog.info("MultichainClient::GetProof() received request: " + getProofRequest.DebugString());
#endif
    // Get the prover request UUID from the request
    string uuid = getProofRequest.id();

    // Lock the prover
    prover.lock();

    // Map uuid to the corresponding prover request
    std::unordered_map<std::string, ProverAggregationRequest *>::iterator it = prover.requestsMap.find(uuid);

    // If UUID is not found, return the proper error
    if (it == prover.requestsMap.end())
    {
        zklog.error("MultichainClient::GetProof() invalid uuid:" + uuid);
        getProofResponse.set_result(multichain::v1::GetProofResponse_Result_RESULT_ERROR);
        getProofResponse.set_result_string("invalid UUID");
    }
    else
    {
        ProverAggregationRequest * pProverAggregationRequest = it->second;

        // If request is not completed, return the proper result
        if (!pProverAggregationRequest->bCompleted)
        {
            //zklog.error("ZKProverServiceImpl::GetProof() not completed uuid=" + uuid);
            getProofResponse.set_result(multichain::v1::GetProofResponse_Result_RESULT_PENDING);
            getProofResponse.set_result_string("pending");
        }
        // If request is completed, return the proof
        else
        {
            // Request is completed
            getProofResponse.set_id(uuid);
            if (pProverAggregationRequest->result != ZKR_SUCCESS)
            {
                getProofResponse.set_result(multichain::v1::GetProofResponse_Result_RESULT_COMPLETED_ERROR);
                getProofResponse.set_result_string("completed_error");
            }
            else
            {
                getProofResponse.set_result(multichain::v1::GetProofResponse_Result_RESULT_COMPLETED_OK);
                getProofResponse.set_result_string("completed");
            }

            switch (pProverAggregationRequest->type)
            {
                case prt_genPrepareMultichainProof:
                {
                    multichain::v1::PrepareProof * pPrepareProof = new multichain::v1::PrepareProof();
                    zkassert(pPrepareProof != NULL);

                    pPrepareProof->set_proof(pProverAggregationRequest->multichainPrepProofOutput.dump());
                    pPrepareProof->set_hash_info(pProverAggregationRequest->multichainPrepHashOutput.dump());

                    getProofResponse.set_allocated_prepare_proof(pPrepareProof);
                    break;
                }
                case prt_genAggregatedMultichainProof:
                {
                    string recursiveProof = pProverAggregationRequest->aggregatedMultichainProofOutput.dump();
                    getProofResponse.set_multichain_proof(recursiveProof);
                    break;
                }
                case prt_genFinalMultichainProof:
                {
                    string finalProof = pProverAggregationRequest->proof.getStringProof();
                    getProofResponse.set_multichain_proof(finalProof);
                    break;
                }
                default:
                {
                    zklog.error("MultichainClient::GetProof() invalid pProverAggregationRequest->type=" + to_string(pProverAggregationRequest->type));
                    exitProcess();
                }
            }
        }
    }

    prover.unlock();

#ifdef LOG_SERVICE
    zklog.info("MultichainClient::GetProof() sends response: " + getProofResponse.DebugString());
#endif
    return true;
}

void* multichainClientThread(void* arg)
{
    zklog.info("multichainClientThread() started");
    string uuid;
    MultichainClient *pMultichainClient = (MultichainClient *)arg;
    Watchdog watchdog;
    uint64_t numberOfStreams = 0;

    while (true)
    {
        // Control the number of streams does not exceed the maximum
        if ((pMultichainClient->config.multichainClientMaxStreams > 0) && (numberOfStreams >= pMultichainClient->config.multichainClientMaxStreams))
        {
            zklog.info("multichainClientThread() killing process since we reached the maximum number of streams=" + to_string(pMultichainClient->config.multichainClientMaxStreams));
            exit(0);
        }
        numberOfStreams++;

        ::grpc::ClientContext context;

        std::unique_ptr<grpc::ClientReaderWriter<multichain::v1::ProverMessage, multichain::v1::MultichainMessage>> readerWriter;
        readerWriter = pMultichainClient->stub->Channel(&context);
        watchdog.start(pMultichainClient->config.multichainClientWatchdogTimeout);
        bool bResult;
        while (true)
        {
            ::multichain::v1::MultichainMessage multichainMessage;
            ::multichain::v1::ProverMessage proverMessage;

            // Read a new multichain message
            watchdog.restart();
            bResult = readerWriter->Read(&multichainMessage);
            if (!bResult)
            {
                zklog.error("multichainClientThread() failed calling readerWriter->Read(&multichainMessage)");
                break;
            }
            watchdog.restart();
            
            switch (multichainMessage.request_case())
            {
                case multichain::v1::MultichainMessage::RequestCase::kGetProofRequest:
                    break;
                case multichain::v1::MultichainMessage::RequestCase::kGetStatusRequest:
                case multichain::v1::MultichainMessage::RequestCase::kCancelRequest:
                    zklog.info("multichainClientThread() got: " + multichainMessage.ShortDebugString());
                    break;
                case multichain::v1::MultichainMessage::RequestCase::kGenPrepareMultichainProofRequest:
                    zklog.info("multichainClientThread() got genPrepareMultichainProof() request");
                    break;
                case multichain::v1::MultichainMessage::RequestCase::kGenAggregatedMultichainProofRequest:
                    zklog.info("multichainClientThread() got genAggregatedMultichainProof() request");
                    break;
                case multichain::v1::MultichainMessage::RequestCase::kGenFinalMultichainProofRequest:
                    zklog.info("multichainClientThread() got genFinalMultichainProof() request");
                    break;
                case multichain::v1::MultichainMessage::RequestCase::kCalculateSha256Request:
                    zklog.info("multichainClientThread() got calculateSha256Request() request");
                    break;
                default:
                    break;
            }

            // We return the same ID we got in the multichain message
            proverMessage.set_id(multichainMessage.id());

            string filePrefix = pMultichainClient->config.outputPath + "/" + getTimestamp() + "_" + multichainMessage.id() + ".";

            if (pMultichainClient->config.saveRequestToFile)
            {
                string2file(multichainMessage.DebugString(), filePrefix + "multichain_request.txt");
            }

            switch (multichainMessage.request_case())
            {
                case multichain::v1::MultichainMessage::RequestCase::kGetStatusRequest:
                {
                    // Allocate a new get status response
                    multichain::v1::GetStatusResponse * pGetStatusResponse = new multichain::v1::GetStatusResponse();
                    zkassert(pGetStatusResponse != NULL);

                    // Call GetStatus
                    pMultichainClient->GetStatus(*pGetStatusResponse);

                    // Set the get status response
                    proverMessage.set_allocated_get_status_response(pGetStatusResponse);
                    break;
                }

                case multichain::v1::MultichainMessage::RequestCase::kGenPrepareMultichainProofRequest:
                {
                    // Allocate a new gen prepare multichain proof response
                    multichain::v1::GenPrepareMultichainProofResponse * pGenPrepareMultichainProofResponse = new multichain::v1::GenPrepareMultichainProofResponse();
                    zkassert(pGenPrepareMultichainProofResponse != NULL);

                    // Call GenPrepareMultichainProof
                    pMultichainClient->GenPrepareMultichainProof(multichainMessage.gen_prepare_multichain_proof_request(), *pGenPrepareMultichainProofResponse);

                    // Set the gen prepare multichain response
                    proverMessage.set_allocated_gen_prepare_multichain_proof_response(pGenPrepareMultichainProofResponse);
                    break;
                }

                case multichain::v1::MultichainMessage::RequestCase::kGenAggregatedMultichainProofRequest:
                {
                    // Allocate a new gen aggregated multichain proof response
                    multichain::v1::GenAggregatedMultichainProofResponse * pGenAggregatedMultichainProofResponse = new multichain::v1::GenAggregatedMultichainProofResponse();
                    zkassert(pGenAggregatedMultichainProofResponse != NULL);

                    // Call GenAggregatedMultichainProof
                    pMultichainClient->GenAggregatedMultichainProof(multichainMessage.gen_aggregated_multichain_proof_request(), *pGenAggregatedMultichainProofResponse);

                    // Set the gen aggregated multichain proof response
                    proverMessage.set_allocated_gen_aggregated_multichain_proof_response(pGenAggregatedMultichainProofResponse);
                    break;
                }

                case multichain::v1::MultichainMessage::RequestCase::kGenFinalMultichainProofRequest:
                {
                    // Allocate a new gen final multichain proof response
                    multichain::v1::GenFinalMultichainProofResponse * pGenFinalMultichainProofResponse = new multichain::v1::GenFinalMultichainProofResponse();
                    zkassert(pGenFinalMultichainProofResponse != NULL);

                    // Call GenFinalMultichainProof
                    pMultichainClient->GenFinalMultichainProof(multichainMessage.gen_final_multichain_proof_request(), *pGenFinalMultichainProofResponse);

                    // Set the gen final proof response
                    proverMessage.set_allocated_gen_final_multichain_proof_response(pGenFinalMultichainProofResponse);
                    break;
                }

                case multichain::v1::MultichainMessage::RequestCase::kCancelRequest:
                {
                    // Allocate a new cancel response
                    multichain::v1::CancelResponse * pCancelResponse = new multichain::v1::CancelResponse();
                    zkassert(pCancelResponse != NULL);

                    // Call Cancel
                    pMultichainClient->Cancel(multichainMessage.cancel_request(), *pCancelResponse);

                    // Set the cancel response
                    proverMessage.set_allocated_cancel_response(pCancelResponse);
                    break;
                }

                case multichain::v1::MultichainMessage::RequestCase::kCalculateSha256Request:
                {
                    // Allocate a new calculateSha256 response
                    multichain::v1::CalculateSha256Response * pCalculateSha256Response = new multichain::v1::CalculateSha256Response();
                    zkassert(pCalculateSha256Response != NULL);

                    // Call CalculateSha256
                    pMultichainClient->CalculateSha256(multichainMessage.calculate_sha256_request(), *pCalculateSha256Response);

                    // Set the get proof response
                    proverMessage.set_allocated_calculate_sha256_response(pCalculateSha256Response);
                    break;
                }

                case multichain::v1::MultichainMessage::RequestCase::kGetProofRequest:
                {
                    // Allocate a new cancel response
                    multichain::v1::GetProofResponse * pGetProofResponse = new multichain::v1::GetProofResponse();
                    zkassert(pGetProofResponse != NULL);

                    // Call GetProof
                    pMultichainClient->GetProof(multichainMessage.get_proof_request(), *pGetProofResponse);

                    // Set the get proof response
                    proverMessage.set_allocated_get_proof_response(pGetProofResponse);
                    break;
                }

                default:
                {
                    zklog.error("multichainClientThread() received an invalid type=" + to_string(multichainMessage.request_case()));
                    break;
                }
            }

            // Write the prover message
            watchdog.restart();
            bResult = readerWriter->Write(proverMessage);
            if (!bResult)
            {
                zklog.error("multichainClientThread() failed calling readerWriter->Write(proverMessage)");
                break;
            }
            watchdog.restart();
            
            switch (multichainMessage.request_case())
            {
                case multichain::v1::MultichainMessage::RequestCase::kGetStatusRequest:
                case multichain::v1::MultichainMessage::RequestCase::kGenPrepareMultichainProofRequest:
                case multichain::v1::MultichainMessage::RequestCase::kGenAggregatedMultichainProofRequest:
                case multichain::v1::MultichainMessage::RequestCase::kGenFinalMultichainProofRequest:
                case multichain::v1::MultichainMessage::RequestCase::kCalculateSha256Request:
                case multichain::v1::MultichainMessage::RequestCase::kCancelRequest:
                    zklog.info("multichainClientThread() sent: " + proverMessage.ShortDebugString());
                    break;
                case multichain::v1::MultichainMessage::RequestCase::kGetProofRequest:
                    if (proverMessage.get_proof_response().result() != multichain::v1::GetProofResponse_Result_RESULT_PENDING)
                        zklog.info("multichainClientThread() getProof() response sent; result=" + proverMessage.get_proof_response().result_string());
                    break;
                default:
                    break;
            }

            if (pMultichainClient->config.saveResponseToFile)
            {
                string2file(proverMessage.DebugString(), filePrefix + "multichain_response.txt");
            }
        }
        watchdog.stop();
        zklog.info("multichainClientThread() channel broken; will retry in 5 seconds");
        sleep(5);
    }
    return NULL;
}
