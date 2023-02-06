
#include <nlohmann/json.hpp>
#include "aggregator_client_mock.hpp"

using namespace std;
using json = nlohmann::json;

struct timeval lastAggregatorGenProof = {0, 0};
tProverRequestType requestType;
string lastAggregatorUUID;

AggregatorClientMock::AggregatorClientMock (Goldilocks &fr, const Config &config) :
    fr(fr),
    config(config)
{
    // Create channel
    std::shared_ptr<grpc_impl::Channel> channel = ::grpc::CreateChannel(config.aggregatorClientHost + ":" + to_string(config.aggregatorClientPort), grpc::InsecureChannelCredentials());

    // Create stub (i.e. client)
    stub = new aggregator::v1::AggregatorService::Stub(channel);
}

void AggregatorClientMock::runThread (void)
{
    cout << "AggregatorClientMock::runThread() creating aggregatorClientThread" << endl;
    pthread_create(&t, NULL, aggregatorClientMockThread, this);
}

void AggregatorClientMock::waitForThread (void)
{
    pthread_join(t, NULL);
}

bool AggregatorClientMock::GetStatus (::aggregator::v1::GetStatusResponse &getStatusResponse)
{
    bool bComputing = (TimeDiff(lastAggregatorGenProof) < config.aggregatorClientMockTimeout); //·

    // Set last computed request data
    getStatusResponse.set_last_computed_request_id(bComputing ? getUUID() : lastAggregatorUUID);
    getStatusResponse.set_last_computed_end_time(time(NULL));

    // If computing, set the current request data
    getStatusResponse.set_status(bComputing ? aggregator::v1::GetStatusResponse_Status_STATUS_COMPUTING : aggregator::v1::GetStatusResponse_Status_STATUS_IDLE);
    getStatusResponse.set_current_computing_request_id(bComputing ? lastAggregatorUUID : "");
    getStatusResponse.set_current_computing_start_time(time(NULL));

    // Set the versions
    getStatusResponse.set_version_proto("v0_0_1");
    getStatusResponse.set_version_server("0.0.1");

    // Set the list of pending requests uuids
    getStatusResponse.add_pending_request_queue_ids(getUUID());
    getStatusResponse.add_pending_request_queue_ids(getUUID());
    getStatusResponse.add_pending_request_queue_ids(getUUID());

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

#ifdef LOG_SERVICE
    cout << "AggregatorClientMock::GetStatus() returns: " << getStatusResponse.DebugString() << endl;
#endif
    return true;
}

bool AggregatorClientMock::GenBatchProof (const aggregator::v1::GenBatchProofRequest &genBatchProofRequest, aggregator::v1::GenBatchProofResponse &genBatchProofResponse)
{
#ifdef LOG_SERVICE
    cout << "AggregatorClientMock::GenBatchProof() called with request: " << genBatchProofRequest.DebugString() << endl;
#endif
    requestType = prt_genBatchProof;
    // Build the response as Ok, returning the UUID assigned by the prover to this request
    genBatchProofResponse.set_result(aggregator::v1::Result::RESULT_OK);
    lastAggregatorUUID = getUUID();
    genBatchProofResponse.set_id(lastAggregatorUUID);
    gettimeofday(&lastAggregatorGenProof,NULL);

#ifdef LOG_SERVICE
    cout << "AggregatorClientMock::GenBatchProof() returns: " << genBatchProofResponse.DebugString() << endl;
#endif
    return true;
}

bool AggregatorClientMock::GenAggregatedProof (const aggregator::v1::GenAggregatedProofRequest &genAggregatedProofRequest, aggregator::v1::GenAggregatedProofResponse &genAggregatedProofResponse)
{
#ifdef LOG_SERVICE
    cout << "AggregatorClientMock::GenAggregatedProof() called with request: " << genAggregatedProofRequest.DebugString() << endl;
#endif
    requestType = prt_genAggregatedProof;

    // Build the response as Ok, returning the UUID assigned by the prover to this request
    genAggregatedProofResponse.set_result(aggregator::v1::Result::RESULT_OK);
    lastAggregatorUUID = getUUID();
    genAggregatedProofResponse.set_id(lastAggregatorUUID);
    gettimeofday(&lastAggregatorGenProof,NULL);

#ifdef LOG_SERVICE
    cout << "AggregatorClientMock::GenAggregatedProof() returns: " << genAggregatedProofResponse.DebugString() << endl;
#endif
    return true;
}

bool AggregatorClientMock::GenFinalProof (const aggregator::v1::GenFinalProofRequest &genFinalProofRequest, aggregator::v1::GenFinalProofResponse &genFinalProofResponse)
{
#ifdef LOG_SERVICE
    cout << "AggregatorClientMock::GenFinalProof() called with request: " << genFinalProofRequest.DebugString() << endl;
#endif

    requestType = prt_genFinalProof;

    // Build the response as Ok, returning the UUID assigned by the prover to this request
    genFinalProofResponse.set_result(aggregator::v1::Result::RESULT_OK);
    lastAggregatorUUID = getUUID();
    genFinalProofResponse.set_id(lastAggregatorUUID);
    gettimeofday(&lastAggregatorGenProof,NULL);

#ifdef LOG_SERVICE
    cout << "AggregatorClientMock::GenFinalProof() returns: " << genFinalProofResponse.DebugString() << endl;
#endif
    return true;
}

bool AggregatorClientMock::Cancel (const aggregator::v1::CancelRequest &cancelRequest, aggregator::v1::CancelResponse &cancelResponse)
{
    bool bComputing = (TimeDiff(lastAggregatorGenProof) < config.aggregatorClientMockTimeout);

    if (bComputing && (cancelRequest.id() == lastAggregatorUUID ))
    {
        cancelResponse.set_result(aggregator::v1::Result::RESULT_OK);
        lastAggregatorGenProof = {0,0};
    }
    else
    {
        cancelResponse.set_result(aggregator::v1::Result::RESULT_ERROR);
    }

#ifdef LOG_SERVICE
    cout << "AggregatorClientMock::Cancel() returns: " << cancelResponse.DebugString() << endl;
#endif
    return true;
}

bool AggregatorClientMock::GetProof (const aggregator::v1::GetProofRequest &getProofRequest, aggregator::v1::GetProofResponse &getProofResponse)
{
#ifdef LOG_SERVICE
    cout << "AggregatorClientMock::GetProof() received request: " << getProofRequest.DebugString();
#endif

    bool bComputing = (TimeDiff(lastAggregatorGenProof) < config.aggregatorClientMockTimeout);

    // Get the prover request UUID from the request
    string uuid = getProofRequest.id();

    if (!bComputing && (uuid == lastAggregatorUUID))
    {
        // Request is completed
        getProofResponse.set_id(uuid);
        getProofResponse.set_result(aggregator::v1::GetProofResponse_Result_RESULT_COMPLETED_OK);
        getProofResponse.set_result_string("completed");
        switch (requestType)
        {
            case prt_genFinalProof:
            {
                // Convert the returned Proof to aggregator::Proof
                aggregator::v1::Proof * pProofProver = new aggregator::v1::Proof();
                aggregator::v1::FinalProof * pFinalProof = new aggregator::v1::FinalProof();

                pProofProver->add_proof_a("13661670604050723159190639550237390237901487387303122609079617855313706601738");
                pProofProver->add_proof_a("318870292909531730706266902424471322193388970015138106363857068613648741679");
                pProofProver->add_proof_a("1");

                aggregator::v1::ProofB *pProofB = pProofProver->add_proof_b();
                pProofB->add_proofs("697129936138216869261087581911668981951894602632341950972818743762373194907");
                pProofB->add_proofs("8382255061406857865565510718293473646307698289010939169090474571110768554297");
                pProofB = pProofProver->add_proof_b();
                pProofB->add_proofs("15430920731683674465693779067364347784717314152940718599921771157730150217435");
                pProofB->add_proofs("9973632244944366583831174453935477607483467152902406810554814671794600888188");
                pProofB = pProofProver->add_proof_b();
                pProofB->add_proofs("1");
                pProofB->add_proofs("0");
                pProofProver->add_proof_c("19319469652444706345294120534164146052521965213898291140974711293816652378032");
                pProofProver->add_proof_c("20960565072144725955004735885836324119094967998861346319897532045008317265851");
                pProofProver->add_proof_c("1");
                pFinalProof->set_allocated_proof(pProofProver);

                // Set public inputs extended
                aggregator::v1::PublicInputs* pPublicInputs = new(aggregator::v1::PublicInputs);
                pPublicInputs->set_old_state_root(string2ba("0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9"));
                pPublicInputs->set_old_acc_input_hash(string2ba("0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9"));
                pPublicInputs->set_old_batch_num(1);
                pPublicInputs->set_chain_id(1000);
                pPublicInputs->set_batch_l2_data(string2ba("0x617b3a3528F9cDd6630fd3301B9c8911F7Bf063D"));
                pPublicInputs->set_global_exit_root(string2ba("0x617b3a3528F9cDd6630fd3301B9c8911F7Bf063D"));
                pPublicInputs->set_eth_timestamp(1000000);
                pPublicInputs->set_sequencer_addr("0x617b3a3528F9cDd6630fd3301B9c8911F7Bf063D");
                pPublicInputs->set_aggregator_addr("0x617b3a3528F9cDd6630fd3301B9c8911F7Bf063D");
                aggregator::v1::PublicInputsExtended* pPublicInputsExtended = new(aggregator::v1::PublicInputsExtended);
                pPublicInputsExtended->set_allocated_public_inputs(pPublicInputs);
                pPublicInputsExtended->set_new_state_root("0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9");
                pPublicInputsExtended->set_new_acc_input_hash("0x1afd6eaf13538380d99a245c2acc4a25481b54556ae080cf07d1facc0638cd8e");
                pPublicInputsExtended->set_new_local_exit_root("0x17c04c3760510b48c6012742c540a81aba4bca2f78b9d14bfd2f123e2e53ea3e");
                pPublicInputsExtended->set_new_batch_num(2);
                pFinalProof->set_allocated_public_(pPublicInputsExtended);
                getProofResponse.set_allocated_final_proof(pFinalProof);
                break; 
            }
            case prt_genBatchProof:
            {
                getProofResponse.set_recursive_proof("88888670604050723159190639550237390237901487387303122609079617855313706601738");
                break;
            }
            case prt_genAggregatedProof:
            {
                getProofResponse.set_recursive_proof("99999670604050723159190639550237390237901487387303122609079617855313706601738");
                break;
            }
            default:
            {
                cerr << "AggregatorClient::GetProof() invalid pProverRequest->type=" << requestType << endl;
                exitProcess();
            }
        }
    }
    else if (bComputing && (uuid == lastAggregatorUUID))
    {
        // Request is being computed
        getProofResponse.set_id(uuid);
        getProofResponse.set_result(aggregator::v1::GetProofResponse_Result_RESULT_PENDING);
        getProofResponse.set_result_string("pending");
    }
    else
    {
        // Request is being computed
        getProofResponse.set_id(uuid);
        getProofResponse.set_result(aggregator::v1::GetProofResponse_Result_RESULT_ERROR);
        getProofResponse.set_result_string("pending");
    }

#ifdef LOG_SERVICE
    cout << "AggregatorClientMock::GetProof() sends response: " << getProofResponse.DebugString();
#endif
    return true;
}

void* aggregatorClientMockThread(void* arg)
{
    cout << "aggregatorClientMockThread() started" << endl;
    string uuid;
    AggregatorClientMock *pAggregatorClientMock = (AggregatorClientMock *)arg;

    while (true)
    {
        ::grpc::ClientContext context;
        std::unique_ptr<grpc::ClientReaderWriter<aggregator::v1::ProverMessage, aggregator::v1::AggregatorMessage>> readerWriter;
        readerWriter = pAggregatorClientMock->stub->Channel(&context);
        bool bResult;
        while (true)
        {
            ::aggregator::v1::AggregatorMessage aggregatorMessage;
            ::aggregator::v1::ProverMessage proverMessage;

            // Read a new aggregator message
            bResult = readerWriter->Read(&aggregatorMessage);
            if (!bResult)
            {
                cerr << "aggregatorClientMockThread() failed calling readerWriter->Read(&aggregatorMessage)" << endl;
                break;
            }
            cout << "aggregatorClientMockThread() got: " << aggregatorMessage.ShortDebugString() << endl;

            // We return the same ID we got in the aggregator message
            proverMessage.set_id(aggregatorMessage.id());

            string filePrefix = pAggregatorClientMock->config.outputPath + "/" + getTimestamp() + "_" + aggregatorMessage.id() + ".";

            if (pAggregatorClientMock->config.saveRequestToFile)
            {
                string2File(aggregatorMessage.DebugString(), filePrefix + "aggregator_request.txt");
            }

            switch (aggregatorMessage.request_case())
            {
                case aggregator::v1::AggregatorMessage::RequestCase::kGetStatusRequest:
                {
                    // Allocate a new get status response
                    aggregator::v1::GetStatusResponse * pGetStatusResponse = new aggregator::v1::GetStatusResponse();
                    zkassertpermanent(pGetStatusResponse != NULL);

                    // Call GetStatus
                    pAggregatorClientMock->GetStatus(*pGetStatusResponse);

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
                    pAggregatorClientMock->GenBatchProof(aggregatorMessage.gen_batch_proof_request(), *pGenBatchProofResponse);

                    // Set the gen batch proof response
                    proverMessage.set_allocated_gen_batch_proof_response(pGenBatchProofResponse);
                    break;
                }

                case aggregator::v1::AggregatorMessage::RequestCase::kGenAggregatedProofRequest:
                {
                    // Allocate a new gen aggregated proof response
                    aggregator::v1::GenAggregatedProofResponse * pGenAggregatedProofResponse = new aggregator::v1::GenAggregatedProofResponse();
                    zkassertpermanent(pGenAggregatedProofResponse != NULL);

                    // Call GenAggregatedProof
                    pAggregatorClientMock->GenAggregatedProof(aggregatorMessage.gen_aggregated_proof_request(), *pGenAggregatedProofResponse);

                    // Set the gen aggregated proof response
                    proverMessage.set_allocated_gen_aggregated_proof_response(pGenAggregatedProofResponse);
                    break;
                }

                case aggregator::v1::AggregatorMessage::RequestCase::kGenFinalProofRequest:
                {
                    // Allocate a new gen final proof response
                    aggregator::v1::GenFinalProofResponse * pGenFinalProofResponse = new aggregator::v1::GenFinalProofResponse();
                    zkassertpermanent(pGenFinalProofResponse != NULL);

                    // Call GenFinalProof
                    pAggregatorClientMock->GenFinalProof(aggregatorMessage.gen_final_proof_request(), *pGenFinalProofResponse);

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
                    pAggregatorClientMock->Cancel(aggregatorMessage.cancel_request(), *pCancelResponse);

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
                    pAggregatorClientMock->GetProof(aggregatorMessage.get_proof_request(), *pGetProofResponse);

                    // Set the get proof response
                    proverMessage.set_allocated_get_proof_response(pGetProofResponse);
                    break;
                }

                default:
                {
                    cerr << "aggregatorClientMockThread() received an invalid type=" << aggregatorMessage.request_case() << endl;
                    break;
                }
            }

            // Write the prover message
            bResult = readerWriter->Write(proverMessage);
            if (!bResult)
            {
                cerr << "aggregatorClientMockThread() failed calling readerWriter->Write(proverMessage)" << endl;
                break;
            }
            cout << "aggregatorClientMockThread() sent: " << proverMessage.ShortDebugString() << endl;
            
            if (pAggregatorClientMock->config.saveResponseToFile)
            {
                string2File(proverMessage.DebugString(), filePrefix + "aggregator_response.txt");
            }
        }
        cout << "aggregatorClientMockThread() channel broken; will retry in 5 seconds" << endl;
        sleep(5);
    }
    return NULL;
}