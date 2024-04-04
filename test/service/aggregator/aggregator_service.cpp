#include "config.hpp"
#include "aggregator_service.hpp"
#include "input.hpp"
#include "proof_fflonk.hpp"
#include "definitions.hpp"
#include <grpcpp/grpcpp.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

#define AGGREGATOR_SERVER_NUMBER_OF_LOOPS 1

#define AGGREGATOR_SERVER_RETRY_SLEEP 10
#define AGGREGATOR_SERVER_NUMBER_OF_GET_PROOF_RETRIES 600  // 600 retries every 10 seconds = 6000 seconds = 100 minutes

::grpc::Status AggregatorServiceImpl::Channel(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream)
{
#ifdef LOG_SERVICE
    cout << "AggregatorServiceImpl::Channel() stream starts" << endl;
#endif
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    aggregator::v1::Result result;
    string uuid;
    ::grpc::Status grpcStatus;
    string requestID;
    string proof;

    //batch0
    const string inputBatchFile0  = "testvectors/e2e/" + string(PROVER_FORK_NAMESPACE_STRING) + "/input-0-batch-0-0.json";
    const string outputBatchFile0 = "testvectors/aggregatedBatchProof/recursive1.zkin.proof_0.json";

    //batch1
    const string inputBatchFile1  = "testvectors/e2e/" + string(PROVER_FORK_NAMESPACE_STRING) + "/input-0-batch-0-1.json";
    const string outputBatchFile1 = "testvectors/aggregatedBatchProof/recursive1.zkin.proof_1.json";

    //aggregate batches 01
    const string inputBatchFile01a = outputBatchFile0;
    const string inputBatchFile01b = outputBatchFile1;
    const string outputBatchFile01 = "testvectors/aggregatedBatchProof/recursive2.zkin.proof_01.json";

    //batch2
    const string inputBatchFile2  = "testvectors/e2e/" + string(PROVER_FORK_NAMESPACE_STRING) + "/input-0-batch-0-2.json";
    const string outputBatchFile2 = "testvectors/aggregatedBatchProof/recursive1.zkin.proof_2.json";
    
    //aggregate batches 02
    const string inputBatchFile02a = outputBatchFile01;
    const string inputBatchFile02b = outputBatchFile2;
    const string outputBatchFile02 = "testvectors/blobOuterProof/recursive2.zkin.proof_02.json";

    //blob inner for batches 02
    const string inputBlobInnerFile02  = "testvectors/e2e/" + string(PROVER_FORK_NAMESPACE_STRING) + "/input-0-blob-inner-0.json";
    const string outputBlobInnerFile02 = "testvectors/blobOuterProof/blob_inner_recursive1.zkin.proof_02.json";

    //blob outer for batches 02
    const string inputBlobOuterFile02a  = outputBatchFile02;
    const string inputBlobOuterFile02b  = outputBlobInnerFile02;
    const string outputBlobOuterFile02  = "testvectors/aggregatedBlobOuterProof/blob_outer.zkin.proof_02.json"; 

    //batch3
    const string inputBatchFile3  = "testvectors/e2e/" + string(PROVER_FORK_NAMESPACE_STRING) + "/input-0-batch-1-0.json";
    const string outputBatchFile3 = "testvectors/blobOuterProof/recursive1.zkin.proof_3.json";

    //blob inner for batches 33
    const string inputBlobInnerFile33  = "testvectors/e2e/" + string(PROVER_FORK_NAMESPACE_STRING) + "/input-0-blob-inner-1.json";
    const string outputBlobInnerFile33 = "testvectors/blobOuterProof/blob_inner_recursive1.zkin.proof_33.json";

    //blob outer for batches 33
    const string inputBlobOuterFile33a  = outputBatchFile3;
    const string inputBlobOuterFile33b  = outputBlobInnerFile33;
    const string outputBlobOuterFile33  = "testvectors/aggregatedBlobOuterProof/blob_outer.zkin.proof_33.json"; 

    //batch4
    const string inputBatchFile4  = "testvectors/e2e/" + string(PROVER_FORK_NAMESPACE_STRING) + "/input-0-batch-2-0.json";
    const string outputBatchFile4 = "testvectors/aggregatedBatchProof/recursive1.zkin.proof_4.json";

    //batch5
    const string inputBatchFile5  = "testvectors/e2e/" + string(PROVER_FORK_NAMESPACE_STRING) + "/input-0-batch-2-1.json";
    const string outputBatchFile5 = "testvectors/aggregatedBatchProof/recursive1.zkin.proof_5.json";

    //aggregate batches 45
    const string inputBatchFile45a = outputBatchFile4;
    const string inputBatchFile45b = outputBatchFile5;
    const string outputBatchFile45 = "testvectors/blobOuterProof/recursive2.zkin.proof_45.json";

    //blob inner 45
    const string inputBlobInnerFile45  = "testvectors/e2e/" + string(PROVER_FORK_NAMESPACE_STRING) + "/input-0-blob-inner-2.json";
    const string outputBlobInnerFile45 = "testvectors/blobOuterProof/blob_inner_recursive1.zkin.proof_45.json";

    //blob outer for batches 45
    const string inputBlobOuterFile45a  = outputBatchFile45;
    const string inputBlobOuterFile45b  = outputBlobInnerFile45;
    const string outputBlobOuterFile45  = "testvectors/aggregatedBlobOuterProof/blob_outer.zkin.proof_45.json"; 

    //aggregate blob outer 02 and blob outer 33
    const string inputBlobOuterFile03a  = outputBlobOuterFile02;
    const string inputBlobOuterFile03b  = outputBlobOuterFile33;
    const string outputBlobOuterFile03  = "testvectors/aggregatedBlobOuterProof/blob_outer_recursive2.zkin.proof_03.json";

    //aggregate blob outer 03 and blob outer 45
    const string inputBlobOuterFile05a  = outputBlobOuterFile03;
    const string inputBlobOuterFile05b  = outputBlobOuterFile45;
    const string outputBlobOuterFile05  = "testvectors/finalProof/blob_outer_recursive2.zkin.proof_05.json";

    //final proof
    const string inputFileFinal  = outputBlobOuterFile05;
    const string outputFileFinal = "testvectors/finalProof/proof.json";

    // Get status
    grpcStatus = GetStatus(context, stream);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }

    // Cancel an invalid request ID and check result
    grpcStatus = Cancel(context, stream, "invalid_id", result);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    if (result != aggregator::v1::Result::RESULT_ERROR)
    {
        cerr << "Error: AggregatorServiceImpl::Channel() got cancel result=" << result << " instead of RESULT_CANCEL_ERROR" << endl;
        return Status::CANCELLED;
    }

    for ( uint64_t loop=0; loop<AGGREGATOR_SERVER_NUMBER_OF_LOOPS; loop++ )
    {
        // Generate batch proof 0
        grpcStatus = GenAndGetBatchProof(context, stream, inputBatchFile0, outputBatchFile0);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetBatchProof(" << inputBatchFile0 << ", " << outputBatchFile0 << ")" << endl;

        // Generate batch proof 1
        grpcStatus = GenAndGetBatchProof(context, stream, inputBatchFile1, outputBatchFile1);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetBatchProof(" << inputBatchFile1 << ", " << outputBatchFile1 << ")" << endl;

        // Generate aggregated proof 01
        grpcStatus = GenAndGetAggregatedBatchProof(context, stream, inputBatchFile01a, inputBatchFile01b, outputBatchFile01);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetAggregatedProof(" << inputBatchFile01a << ", " << inputBatchFile01b << ", " << outputBatchFile01 << ")" << endl;


        // Generate batch proof 2
        grpcStatus = GenAndGetBatchProof(context, stream, inputBatchFile2, outputBatchFile2);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetBatchProof(" << inputBatchFile2 << ", " << outputBatchFile2 << ")" << endl;

        // Generate aggregated proof 02
        grpcStatus = GenAndGetAggregatedBatchProof(context, stream, inputBatchFile02a, inputBatchFile02b, outputBatchFile02);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetAggregatedProof(" << inputBatchFile02a << ", " << inputBatchFile02b << ", " << outputBatchFile02 << ")" << endl;

        // Generate blob inner proof 02
        grpcStatus = GenAndGetBlobInnerProof(context, stream, inputBlobInnerFile02, outputBlobInnerFile02);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetBlobInnerProof(" << inputBlobInnerFile02 << ", " << outputBlobInnerFile02 << ")" << endl;

        // Generate blob outer proof 02
        grpcStatus = GenAndGetBlobOuterProof(context, stream, outputBatchFile02, outputBlobInnerFile02, outputBlobOuterFile02);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetBlobOuterProof(" << outputBatchFile02 << ", " << outputBlobInnerFile02 << ", " << outputBlobOuterFile02 << ")" << endl;

        // Generate batch proof 3
        grpcStatus = GenAndGetBatchProof(context, stream, inputBatchFile3, outputBatchFile3);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetBatchProof(" << inputBatchFile3 << ", " << outputBatchFile3 << ")" << endl;

        // Generate blob inner proof 33
        grpcStatus = GenAndGetBlobInnerProof(context, stream, inputBlobInnerFile33, outputBlobInnerFile33);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetBlobInnerProof(" << inputBlobInnerFile33 << ", " << outputBlobInnerFile33 << ")" << endl;

        // Generate blob outer proof 33
        grpcStatus = GenAndGetBlobOuterProof(context, stream, outputBatchFile3, outputBlobInnerFile33, outputBlobOuterFile33);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetBlobOuterProof(" << outputBatchFile3 << ", " << outputBlobInnerFile33 << ", " << outputBlobOuterFile33 << ")" << endl;

        // Generate batch proof 4
        grpcStatus = GenAndGetBatchProof(context, stream, inputBatchFile4, outputBatchFile4);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetBatchProof(" << inputBatchFile4 << ", " << outputBatchFile4 << ")" << endl;

        // Generate batch proof 5
        grpcStatus = GenAndGetBatchProof(context, stream, inputBatchFile5, outputBatchFile5);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetBatchProof(" << inputBatchFile5 << ", " << outputBatchFile5 << ")" << endl;

        // Generate aggregated proof 45
        grpcStatus = GenAndGetAggregatedBatchProof(context, stream, inputBatchFile45a, inputBatchFile45b, outputBatchFile45);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetAggregatedProof(" << inputBatchFile45a << ", " << inputBatchFile45b << ", " << outputBatchFile45 << ")" << endl;

        // Generate blob inner proof 45
        grpcStatus = GenAndGetBlobInnerProof(context, stream, inputBlobInnerFile45, outputBlobInnerFile45);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetBlobInnerProof(" << inputBlobInnerFile45 << ", " << outputBlobInnerFile45 << ")" << endl;

        // Generate blob outer proof 45
        grpcStatus = GenAndGetBlobOuterProof(context, stream, outputBatchFile45, outputBlobInnerFile45, outputBlobOuterFile45);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetBlobOuterProof(" << outputBatchFile45 << ", " << outputBlobInnerFile45 << ", " << outputBlobOuterFile45 << ")" << endl;

        // Generate aggregated blob outer proof 03
        grpcStatus = GenAndGetAggregatedBlobOuterProof(context, stream, inputBlobOuterFile03a, inputBlobOuterFile03b, outputBlobOuterFile03);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAggregatedBlobOuterProof(" << inputBlobOuterFile03a << ", " << inputBlobOuterFile03b << ", " << outputBlobOuterFile03 << ")" << endl;

        // Generate aggregated blob outer proof 05
        grpcStatus = GenAndGetAggregatedBlobOuterProof(context, stream, inputBlobOuterFile05a, inputBlobOuterFile05b, outputBlobOuterFile05);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAggregatedBlobOuterProof(" << inputBlobOuterFile05a << ", " << inputBlobOuterFile05b << ", " << outputBlobOuterFile05 << ")" << endl;
        
        // Generate final proof
        grpcStatus = GenAndGetFinalProof(context, stream, inputFileFinal, outputFileFinal);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetFinalProof(" << inputFileFinal << ", " << outputFileFinal << ")" << endl;
    }

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::GetStatus(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream)
{
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;

    // Send a get status request message
    aggregatorMessage.Clear();
    aggregator::v1::GetStatusRequest * pGetStatusRequest = new aggregator::v1::GetStatusRequest();
    zkassertpermanent(pGetStatusRequest != NULL);
    aggregatorMessage.set_allocated_get_status_request(pGetStatusRequest);
    messageId++;
    aggregatorMessage.set_id(to_string(messageId));
    bResult = stream->Write(aggregatorMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GetStatus() failed calling stream->Write(aggregatorMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get status response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GetStatus() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGetStatusResponse)
    {
        cerr << "Error: AggregatorServiceImpl::GetStatus() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GET_STATUS_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != aggregatorMessage.id())
    {
        cerr << "Error: AggregatorServiceImpl::GetStatus() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
        return Status::CANCELLED;
    }

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::Cancel(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string &requestID, aggregator::v1::Result &result)
{
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;

    // Send a cancel request message
    aggregatorMessage.Clear();
    messageId++;
    aggregatorMessage.set_id(to_string(messageId));
    aggregator::v1::CancelRequest * pCancelRequest = new aggregator::v1::CancelRequest();
    zkassertpermanent(pCancelRequest != NULL);
    pCancelRequest->set_id(requestID);
    aggregatorMessage.set_allocated_cancel_request(pCancelRequest);
    bResult = stream->Write(aggregatorMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Write(aggregatorMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding cancel response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kCancelResponse)
    {
        cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.response_case=" << proverMessage.response_case() << " instead of CANCEL_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != aggregatorMessage.id())
    {
        cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
        return Status::CANCELLED;
    }

    // Check cancel result
    result = proverMessage.cancel_response().result();

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::GenBatchProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string &inputFile, string &requestID)
{
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;

    if (inputFile.size() == 0)
    {
        cerr << "Error: AggregatorServiceImpl::GenBatchProof() found inputFile empty" << endl;
        exitProcess();
    }

    aggregator::v1::InputProver *pInputProver = new aggregator::v1::InputProver();
    zkassertpermanent(pInputProver != NULL);
    Input input(fr);
    json inputJson;
    file2json(inputFile, inputJson);
    zkresult zkResult = input.load(inputJson);
    if (zkResult != ZKR_SUCCESS)
    {
        cerr << "Error: AggregatorServiceImpl::GenBatchProof() failed calling input.load() zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
        exitProcess();
    }

    // Parse public inputs from file
    aggregator::v1::PublicInputs * pPublicInputs = new aggregator::v1::PublicInputs();
    pPublicInputs->set_old_state_root(scalar2ba(input.publicInputsExtended.publicInputs.oldStateRoot));
    pPublicInputs->set_old_batch_acc_input_hash(scalar2ba(input.publicInputsExtended.publicInputs.oldAccInputHash));
    pPublicInputs->set_previous_l1_info_tree_root(scalar2ba(input.publicInputsExtended.publicInputs.previousL1InfoTreeRoot));   
    pPublicInputs->set_previous_l1_info_tree_index(input.publicInputsExtended.publicInputs.previousL1InfoTreeIndex); 
    pPublicInputs->set_chain_id(input.publicInputsExtended.publicInputs.chainID);
    pPublicInputs->set_fork_id(input.publicInputsExtended.publicInputs.forkID);
    pPublicInputs->set_batch_l2_data(string2ba(input.publicInputsExtended.publicInputs.batchL2Data));
    pPublicInputs->set_sequencer_addr(Add0xIfMissing(input.publicInputsExtended.publicInputs.sequencerAddr.get_str(16)));
    pPublicInputs->set_forced_hash_data(scalar2ba(input.publicInputsExtended.publicInputs.forcedHashData));
    // Forced data
    aggregator::v1::ForcedData * pForcedData = new aggregator::v1::ForcedData();
    pForcedData->set_global_exit_root(scalar2ba(input.publicInputsExtended.publicInputs.forcedData.globalExitRoot));
    pForcedData->set_block_hash_l1(scalar2ba(input.publicInputsExtended.publicInputs.forcedData.blockHashL1));
    pForcedData->set_min_timestamp(input.publicInputsExtended.publicInputs.forcedData.minTimestamp);
    pPublicInputs->set_allocated_forced_data(pForcedData);
    pPublicInputs->set_aggregator_addr(Add0xIfMissing(input.publicInputsExtended.publicInputs.aggregatorAddress.get_str(16)));
    // Parse L1 data
    unordered_map<uint64_t, L1Data>::const_iterator itL1Data;
    for (itL1Data = input.l1InfoTreeData.begin(); itL1Data != input.l1InfoTreeData.end(); itL1Data++)
    {
        aggregator::v1::L1Data l1Data;
        l1Data.set_global_exit_root(string2ba(itL1Data->second.globalExitRoot.get_str(16)));
        l1Data.set_block_hash_l1(string2ba(itL1Data->second.blockHashL1.get_str(16)));
        l1Data.set_min_timestamp(itL1Data->second.minTimestamp);
        l1Data.set_initial_historic_root(scalar2ba(itL1Data->second.initialHistoricRoot));
        for (uint64_t i=0; i<itL1Data->second.smtProof.size(); i++)
        {
            l1Data.add_smt_proof_previous_index(string2ba(itL1Data->second.smtProofPreviousIndex[i].get_str(16)));
        }
        (*pInputProver->mutable_public_inputs()->mutable_l1_info_tree_data())[itL1Data->first] = l1Data;
    }
    pInputProver->set_allocated_public_inputs(pPublicInputs);

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
        (*pInputProver->mutable_db())[key] = value;
    }

    // Parse contracts data
    DatabaseMap::ProgramMap::const_iterator itc;
    for (itc=input.contractsBytecode.begin(); itc!=input.contractsBytecode.end(); itc++)
    {
        string key = NormalizeToNFormat(itc->first, 64);
        string value;
        vector<uint8_t> contractValue = itc->second;
        for (uint64_t i=0; i<contractValue.size(); i++)
        {
            value += byte2string(contractValue[i]);
        }
        (*pInputProver->mutable_contracts_bytecode())[key] = value;
    }

    // Allocate the gen batch request
    aggregator::v1::GenBatchProofRequest *pGenBatchProofRequest = new aggregator::v1::GenBatchProofRequest();
    zkassertpermanent(pGenBatchProofRequest != NULL );
    pGenBatchProofRequest->set_allocated_input(pInputProver);

    // Send the gen proof request
    aggregatorMessage.Clear();
    messageId++;
    aggregatorMessage.set_id(to_string(messageId));
    aggregatorMessage.set_allocated_gen_batch_proof_request(pGenBatchProofRequest);
    bResult = stream->Write(aggregatorMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenBatchProof() failed calling stream->Write(aggregatorMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get proof response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenBatchProof() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGenBatchProofResponse)
    {
        cerr << "Error: AggregatorServiceImpl::GenBatchProof() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GEN_BATCH_PROOF_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != aggregatorMessage.id())
    {
        cerr << "Error: AggregatorServiceImpl::GenBatchProof() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
        return Status::CANCELLED;
    }

    requestID = proverMessage.gen_batch_proof_response().id();

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::GenBlobInnerProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFile, string &requestID)
{
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;

    if(inputFile.size() == 0){
        cerr<<" Error: AggregatorServiceImpl::GenBlobInnerProof() found inputFile empty" << endl;
    }
    aggregator::v1::InputBlobInnerProver *pInputBlobInnerProver = new aggregator::v1::InputBlobInnerProver();
    zkassertpermanent(pInputBlobInnerProver != NULL);
    Input input(fr);
    json inputJson;
    file2json(inputFile, inputJson);
    zkresult zkResult = input.load(inputJson);
    if(zkResult != ZKR_SUCCESS){
        cerr << "Error: AggregatorServiceImpl::GenBlobInnerProof() failed calling input.load() zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
        exitProcess();
    }

    // Parse public inputs
    aggregator::v1::PublicBlobInnerInputs * pPublicInputs = new aggregator::v1::PublicBlobInnerInputs();
    pPublicInputs->set_old_blob_state_root(scalar2ba(input.publicInputsExtended.publicInputs.oldStateRoot));
    pPublicInputs->set_old_blob_acc_input_hash(scalar2ba(input.publicInputsExtended.publicInputs.oldBlobAccInputHash));
    pPublicInputs->set_old_num_blob(input.publicInputsExtended.publicInputs.oldBlobNum);
    pPublicInputs->set_old_state_root(scalar2ba(input.publicInputsExtended.publicInputs.oldStateRoot));
    pPublicInputs->set_fork_id(input.publicInputsExtended.publicInputs.forkID);
    pPublicInputs->set_last_l1_info_tree_index(input.publicInputsExtended.publicInputs.lastL1InfoTreeIndex);
    pPublicInputs->set_last_l1_info_tree_root(scalar2ba(input.publicInputsExtended.publicInputs.lastL1InfoTreeRoot));
    pPublicInputs->set_sequencer_addr(Add0xIfMissing(input.publicInputsExtended.publicInputs.sequencerAddr.get_str(16)));
    pPublicInputs->set_timestamp_limit(input.publicInputsExtended.publicInputs.timestampLimit);
    pPublicInputs->set_zk_gas_limit(input.publicInputsExtended.publicInputs.zkGasLimit);
    pPublicInputs->set_blob_type(input.publicInputsExtended.publicInputs.blobType);
    pPublicInputs->set_point_z(scalar2ba(input.publicInputsExtended.publicInputs.pointZ));
    pPublicInputs->set_point_y(scalar2ba(input.publicInputsExtended.publicInputs.pointY));
    pPublicInputs->set_blob_data(string2ba(input.publicInputsExtended.publicInputs.blobData));
    pPublicInputs->set_forced_hash_data(scalar2ba(input.publicInputsExtended.publicInputs.forcedHashData));

    pInputBlobInnerProver->set_allocated_public_inputs(pPublicInputs);

    //Allocate the gen blob inner proof request
    aggregator::v1::GenBlobInnerProofRequest *pGenBlobInnerProofRequest = new aggregator::v1::GenBlobInnerProofRequest();
    zkassertpermanent(pGenBlobInnerProofRequest != NULL);
    pGenBlobInnerProofRequest->set_allocated_input(pInputBlobInnerProver);

     // Send the gen proof request
    aggregatorMessage.Clear();
    messageId++;
    aggregatorMessage.set_id(to_string(messageId));
    aggregatorMessage.set_allocated_gen_blob_inner_proof_request(pGenBlobInnerProofRequest);
    bResult = stream->Write(aggregatorMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenBlobInnerProof() failed calling stream->Write(aggregatorMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get proof response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenBlobInnerProof() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGenBlobInnerProofResponse)
    {
        cerr << "Error: AggregatorServiceImpl::GenBlobInnerProof() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GEN_BLOB_INNER_PROOF_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != aggregatorMessage.id())
    {
        cerr << "Error: AggregatorServiceImpl::GenBlobInnerProof() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
        return Status::CANCELLED;
    }

    requestID = proverMessage.gen_blob_inner_proof_response().id();
        
    return Status::OK;
}

::grpc::Status  AggregatorServiceImpl::GenBlobOuterProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFileBatch, const string & inputFileBlobInner, string &requestID)
{
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;
    string inputFileBatchContent;
    string inputFileBlobInnerContent;

    if (inputFileBatch.size() == 0)
    {
        cerr << "Error: AggregatorServiceImpl::GenBlobOuterProof() found inputFileBatch empty" << endl;
        exitProcess();
    }
    file2string(inputFileBatch, inputFileBatchContent);

    if (inputFileBlobInner.size() == 0)
    {
        cerr << "Error: AggregatorServiceImpl::GenBlobOuterProof() found inputFileBlobInner empty" << endl;
        exitProcess();
    }
    file2string(inputFileBlobInner, inputFileBlobInnerContent);

    // Allocate the blob outer proof request
    aggregator::v1::GenBlobOuterProofRequest *pGenBlobOuterProofRequest = new aggregator::v1::GenBlobOuterProofRequest();
    zkassertpermanent(pGenBlobOuterProofRequest != NULL );
    pGenBlobOuterProofRequest->set_batch_proof(inputFileBatchContent);
    pGenBlobOuterProofRequest->set_blob_inner_proof(inputFileBlobInnerContent);

    // Send the gen proof request
    aggregatorMessage.Clear();
    messageId++;
    aggregatorMessage.set_id(to_string(messageId));
    aggregatorMessage.set_allocated_gen_blob_outer_proof_request(pGenBlobOuterProofRequest);
    bResult = stream->Write(aggregatorMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenBlobOuterProof() failed calling stream->Write(aggregatorMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get proof response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenBlobOuterProof() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }

    // Check type
    if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGenBlobOuterProofResponse)
    {
        cerr << "Error: AggregatorServiceImpl::GenBlobOuterProof() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GEN_BLOB_OUTER_PROOF_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != aggregatorMessage.id())
    {
        cerr << "Error: AggregatorServiceImpl::GenBlobOuterProof() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
        return Status::CANCELLED;
    }

    requestID = proverMessage.gen_blob_outer_proof_response().id();
    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::GenAggregatedBatchProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string &inputFileA, const string &inputFileB, string &requestID)
{
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;
    string inputFileAContent;
    string inputFileBContent;

    if (inputFileA.size() == 0)
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedBatchProof() found inputFileA empty" << endl;
        exitProcess();
    }
    file2string(inputFileA, inputFileAContent);

    if (inputFileB.size() == 0)
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedBatchProof() found inputFileB empty" << endl;
        exitProcess();
    }
    file2string(inputFileB, inputFileBContent);

    // Allocate the aggregated batch request
    aggregator::v1::GenAggregatedBatchProofRequest *pGenAggregatedBatchProofRequest = new aggregator::v1::GenAggregatedBatchProofRequest();
    zkassertpermanent(pGenAggregatedBatchProofRequest != NULL );
    pGenAggregatedBatchProofRequest->set_recursive_proof_1(inputFileAContent);
    pGenAggregatedBatchProofRequest->set_recursive_proof_2(inputFileBContent);

    // Send the gen proof request
    aggregatorMessage.Clear();
    messageId++;
    aggregatorMessage.set_id(to_string(messageId));
    aggregatorMessage.set_allocated_gen_aggregated_batch_proof_request(pGenAggregatedBatchProofRequest);
    bResult = stream->Write(aggregatorMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedBatchProof() failed calling stream->Write(aggregatorMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get proof response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedBatchProof() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGenAggregatedBatchProofResponse)
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedBatchProof() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GEN_AGGREGATED_BATCH_PROOF_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != aggregatorMessage.id())
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedBatchProof() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
        return Status::CANCELLED;
    }

    requestID = proverMessage.gen_aggregated_batch_proof_response().id();

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::GenAggregatedBlobOuterProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFileA, const string & inputFileB, string &requestID)
{
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;
    string inputFileAContent;
    string inputFileBContent;

    if (inputFileA.size() == 0)
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedBlobOuterProof() found inputFileA empty" << endl;
        exitProcess();
    }
    file2string(inputFileA, inputFileAContent);

    if (inputFileB.size() == 0)
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedBlobOuterProof() found inputFileB empty" << endl;
        exitProcess();
    }
    file2string(inputFileB, inputFileBContent);

    // Allocate the aggregated blob outer request
    aggregator::v1::GenAggregatedBlobOuterProofRequest *pGenAggregatedBlobOuterProofRequest = new aggregator::v1::GenAggregatedBlobOuterProofRequest();
    zkassertpermanent(pGenAggregatedBlobOuterProofRequest != NULL );
    pGenAggregatedBlobOuterProofRequest->set_recursive_proof_1(inputFileAContent);
    pGenAggregatedBlobOuterProofRequest->set_recursive_proof_2(inputFileBContent);

    // Send the gen proof request
    aggregatorMessage.Clear();
    messageId++;
    aggregatorMessage.set_id(to_string(messageId));
    aggregatorMessage.set_allocated_gen_aggregated_blob_outer_proof_request(pGenAggregatedBlobOuterProofRequest);
    bResult = stream->Write(aggregatorMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedBlobOuterProof() failed calling stream->Write(aggregatorMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get proof response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedBlobOuterProof() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGenAggregatedBlobOuterProofResponse)
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedBlobOuterProof() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GEN_AGGREGATED_BLOB_OUTER_PROOF_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != aggregatorMessage.id())
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedBlobOuterProof() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
        return Status::CANCELLED;
    }

    requestID = proverMessage.gen_aggregated_blob_outer_proof_response().id();

    return Status::OK;}

::grpc::Status AggregatorServiceImpl::GenFinalProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string &inputFile, string &requestID)
{
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;
    string inputFileContent;

    if (inputFile.size() == 0)
    {
        cerr << "Error: AggregatorServiceImpl::GenFinalProof() found inputFile empty" << endl;
        exitProcess();
    }
    file2string(inputFile, inputFileContent);

    // Allocate the final proof request
    aggregator::v1::GenFinalProofRequest *pGenFinalProofRequest = new aggregator::v1::GenFinalProofRequest();
    zkassertpermanent(pGenFinalProofRequest != NULL );
    pGenFinalProofRequest->set_recursive_proof(inputFileContent);

    // Send the gen proof request
    aggregatorMessage.Clear();
    messageId++;
    aggregatorMessage.set_id(to_string(messageId));
    aggregatorMessage.set_allocated_gen_final_proof_request(pGenFinalProofRequest);
    bResult = stream->Write(aggregatorMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenFinalProof() failed calling stream->Write(aggregatorMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get proof response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenFinalProof() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGenFinalProofResponse)
    {
        cerr << "Error: AggregatorServiceImpl::GenFinalProof() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GEN_AGGREGATED_PROOF_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != aggregatorMessage.id())
    {
        cerr << "Error: AggregatorServiceImpl::GenFinalProof() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
        return Status::CANCELLED;
    }

    requestID = proverMessage.gen_final_proof_response().id();

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::GetProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string &requestID, aggregator::v1::GetProofResponse_Result &result, string &proof)
{
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;

    // Send a get proof request message
    aggregatorMessage.Clear();
    messageId++;
    aggregatorMessage.set_id(to_string(messageId));
    aggregator::v1::GetProofRequest * pGetProofRequest = new aggregator::v1::GetProofRequest();
    zkassertpermanent(pGetProofRequest != NULL);
    pGetProofRequest->set_id(requestID);
    aggregatorMessage.set_allocated_get_proof_request(pGetProofRequest);
    bResult = stream->Write(aggregatorMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Write(aggregatorMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get proof response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGetProofResponse)
    {
        cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GET_PROOF_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != aggregatorMessage.id())
    {
        cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
        return Status::CANCELLED;
    }

    // Copy get proof result
    result = proverMessage.get_proof_response().result();
    if ( proverMessage.get_proof_response().has_final_proof() )
    {
        proof = proverMessage.get_proof_response().final_proof().proof();
    }
    else
    {
        proof = proverMessage.get_proof_response().recursive_proof();
    }

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::GenAndGetBatchProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFile, const string &outputFile)
{
    ::grpc::Status grpcStatus;
    string requestID;

    // Generate batch proof 0
    grpcStatus = GenBatchProof(context, stream, inputFile, requestID);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    cout << "AggregatorServiceImpl::GenAndGetBatchProof() called GenBatchProof() and got requestID=" << requestID << endl;
    return waitProof(context,stream,"BatchProof",requestID, outputFile);
}

::grpc::Status AggregatorServiceImpl::GenAndGetBlobInnerProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFile, const string &outputFile)
{
    ::grpc::Status grpcStatus;
    string requestID;

    // Generate batch proof 0
    grpcStatus = GenBlobInnerProof(context, stream, inputFile, requestID);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    cout << "AggregatorServiceImpl::GenAndGetBlobInnerProof() called GenBlobInnerProof() and got requestID=" << requestID << endl;
    return waitProof(context,stream,"BlobInnerProof",requestID, outputFile);
}

::grpc::Status AggregatorServiceImpl::GenAndGetBlobOuterProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFileBatch, const string & inputFileBlob, const string &outputFile)
{
    ::grpc::Status grpcStatus;
    string requestID;

    grpcStatus = GenBlobOuterProof(context, stream, inputFileBatch, inputFileBlob, requestID);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    cout << "AggregatorServiceImpl::GenAndGetBlobOuterProof() called GenBlobOuterProof() and got requestID=" << requestID << endl;
    return waitProof(context,stream,"BlobOuterProof",requestID, outputFile);

}

::grpc::Status AggregatorServiceImpl::GenAndGetAggregatedBatchProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFileA, const string & inputFileB, const string &outputFile)
{
    ::grpc::Status grpcStatus;
    string requestID;

    grpcStatus = GenAggregatedBatchProof(context, stream, inputFileA, inputFileB, requestID);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    cout << "AggregatorServiceImpl::GenAndGetAggregatedProof() called GenAggregatedBatchProof() and got requestID=" << requestID << endl;

    return waitProof(context,stream,"AggregatedBatchProof",requestID, outputFile);
}

::grpc::Status AggregatorServiceImpl::GenAndGetAggregatedBlobOuterProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFileA, const string & inputFileB, const string &outputFile)
 {
    ::grpc::Status grpcStatus;
    string requestID;

     grpcStatus = GenAggregatedBlobOuterProof(context, stream, inputFileA, inputFileB, requestID);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    cout << "AggregatorServiceImpl::GenAndGetAggregatedBlobOuterProof() called GenAggregatedBlobOuterProof() and got requestID=" << requestID << endl;

    return waitProof(context,stream,"AggregatedBlobOuterProof",requestID, outputFile);
 } 

::grpc::Status AggregatorServiceImpl::GenAndGetFinalProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFile, const string &outputFile)
{
    ::grpc::Status grpcStatus;
    string requestID;
    
    grpcStatus = GenFinalProof(context, stream, inputFile, requestID);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    cout << "AggregatorServiceImpl::GenAndGetFinalProof() called GenFinalProof() and got requestID=" << requestID << endl;

   return waitProof(context,stream,"FinalProof",requestID, outputFile);

}

::grpc::Status AggregatorServiceImpl::waitProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string& proverName,  const string requestID, const string &outputFile)
{
    string proof;
    ::grpc::Status grpcStatus;
    uint64_t i;

     
    for ( i=0; i<AGGREGATOR_SERVER_NUMBER_OF_GET_PROOF_RETRIES; i++)
    {
        sleep(AGGREGATOR_SERVER_RETRY_SLEEP);

        aggregator::v1::GetProofResponse_Result getProofResponseResult;
        grpcStatus = GetProof(context, stream, requestID, getProofResponseResult, proof);        
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }

        if (getProofResponseResult == aggregator::v1::GetProofResponse_Result_RESULT_COMPLETED_OK)
        {
            break;
        }        
        if (getProofResponseResult == aggregator::v1::GetProofResponse_Result_RESULT_PENDING)
        {
            continue;
        }
        cerr << "Error: AggregatorServiceImpl::GenAndGet" + proverName + "() got getProofResponseResult=" << getProofResponseResult << " instead of RESULT_PENDING or RESULT_COMPLETED_OK" << endl;
        return Status::CANCELLED;
    }
    if (  i == AGGREGATOR_SERVER_NUMBER_OF_GET_PROOF_RETRIES)
    {
        cerr << "Error: AggregatorServiceImpl::GenAndGet"+ proverName + "() timed out waiting for batch proof" << endl;
        return Status::CANCELLED;
    }
    if (proof.size() == 0)
    {
        cerr << "Error: AggregatorServiceImpl::GenAndGet" + proverName + "() got an empty batch proof" << endl;
        return Status::CANCELLED;
    }
    string2file(proof, outputFile);
    return Status::OK;
}
    
::grpc::Status AggregatorServiceImpl::ChannelOld(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream)
{
#ifdef LOG_SERVICE
    cout << "AggregatorServiceImpl::Channel() stream starts" << endl;
#endif
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;

    //while (true)
    {
        // CALL GET STATUS

        // Send a get status request message
        aggregatorMessage.Clear();
        aggregator::v1::GetStatusRequest * pGetStatusRequest = new aggregator::v1::GetStatusRequest();
        zkassertpermanent(pGetStatusRequest != NULL);
        aggregatorMessage.set_allocated_get_status_request(pGetStatusRequest);
        messageId++;
        aggregatorMessage.set_id(to_string(messageId));
        bResult = stream->Write(aggregatorMessage);
        if (!bResult)
        {
            cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Write(aggregatorMessage)" << endl;
            return Status::CANCELLED;
        }

        // Receive the corresponding get status response message
        proverMessage.Clear();
        bResult = stream->Read(&proverMessage);
        if (!bResult)
        {
            cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Read(proverMessage)" << endl;
            return Status::CANCELLED;
        }
        
        // Check type
        if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGetStatusResponse)
        {
            cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GET_STATUS_RESPONSE" << endl;
            return Status::CANCELLED;
        }

        // Check id
        if (proverMessage.id() != aggregatorMessage.id())
        {
            cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
            return Status::CANCELLED;
        }

        sleep(1);

        // CALL CANCEL (it should return an error)

        // Send a cancel request message
        aggregatorMessage.Clear();
        messageId++;
        aggregatorMessage.set_id(to_string(messageId));
        aggregator::v1::CancelRequest * pCancelRequest = new aggregator::v1::CancelRequest();
        zkassertpermanent(pCancelRequest != NULL);
        pCancelRequest->set_id("invalid_id");
        aggregatorMessage.set_allocated_cancel_request(pCancelRequest);
        bResult = stream->Write(aggregatorMessage);
        if (!bResult)
        {
            cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Write(aggregatorMessage)" << endl;
            return Status::CANCELLED;
        }

        // Receive the corresponding cancel response message
        proverMessage.Clear();
        bResult = stream->Read(&proverMessage);
        if (!bResult)
        {
            cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Read(proverMessage)" << endl;
            return Status::CANCELLED;
        }
        
        // Check type
        if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kCancelResponse)
        {
            cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.response_case=" << proverMessage.response_case() << " instead of CANCEL_RESPONSE" << endl;
            return Status::CANCELLED;
        }

        // Check id
        if (proverMessage.id() != aggregatorMessage.id())
        {
            cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
            return Status::CANCELLED;
        }

        // Check cancel result
        if (proverMessage.cancel_response().result() != aggregator::v1::Result::RESULT_ERROR)
        {
            cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.cancel_response().result()=" << proverMessage.cancel_response().result() << " instead of RESULT_CANCEL_ERROR" << endl;
            return Status::CANCELLED;
        }

        sleep(1);

        // Call GEN PROOF

        if (config.inputFile.size() == 0)
        {
            cerr << "Error: AggregatorServiceImpl::Channel() found config.inputFile empty" << endl;
            exitProcess();
        }
    //::grpc::ClientContext context;
        aggregator::v1::InputProver *pInputProver = new aggregator::v1::InputProver();
        zkassertpermanent(pInputProver != NULL);
        Input input(fr);
        json inputJson;
        file2json(config.inputFile, inputJson);
        zkresult zkResult = input.load(inputJson);
        if (zkResult != ZKR_SUCCESS)
        {
            cerr << "Error: AggregatorServiceImpl::Channel() failed calling input.load() zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
            exitProcess();
        }

        // Parse public inputs
        aggregator::v1::PublicInputs * pPublicInputs = new aggregator::v1::PublicInputs();
        pPublicInputs->set_old_state_root(scalar2ba(input.publicInputsExtended.publicInputs.oldStateRoot));
        //pPublicInputs->set_old_acc_input_hash(scalar2ba(input.publicInputsExtended.publicInputs.oldAccInputHash));
        //pPublicInputs->set_old_batch_num(input.publicInputsExtended.publicInputs.oldBatchNum);
        pPublicInputs->set_chain_id(input.publicInputsExtended.publicInputs.chainID);
        pPublicInputs->set_fork_id(input.publicInputsExtended.publicInputs.forkID);
        pPublicInputs->set_batch_l2_data(input.publicInputsExtended.publicInputs.batchL2Data);
        //pPublicInputs->set_l1_info_root(scalar2ba(input.publicInputsExtended.publicInputs.l1InfoRoot));
        //pPublicInputs->set_timestamp_limit(input.publicInputsExtended.publicInputs.timestampLimit);
        //pPublicInputs->set_forced_blockhash_l1(scalar2ba(input.publicInputsExtended.publicInputs.forcedBlockHashL1));
        pPublicInputs->set_sequencer_addr(Add0xIfMissing(input.publicInputsExtended.publicInputs.sequencerAddr.get_str(16)));
        pPublicInputs->set_aggregator_addr(Add0xIfMissing(input.publicInputsExtended.publicInputs.aggregatorAddress.get_str(16)));
        pInputProver->set_allocated_public_inputs(pPublicInputs);

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
            (*pInputProver->mutable_db())[key] = value;
        }

        // Parse contracts data
        DatabaseMap::ProgramMap::const_iterator itc;
        for (itc=input.contractsBytecode.begin(); itc!=input.contractsBytecode.end(); itc++)
        {
            string key = NormalizeToNFormat(itc->first, 64);
            string value;
            vector<uint8_t> contractValue = itc->second;
            for (uint64_t i=0; i<contractValue.size(); i++)
            {
                value += byte2string(contractValue[i]);
            }
            (*pInputProver->mutable_contracts_bytecode())[key] = value;
        }

        // Allocate the gen batch request
        aggregator::v1::GenBatchProofRequest *pGenBatchProofRequest = new aggregator::v1::GenBatchProofRequest();
        zkassertpermanent(pGenBatchProofRequest != NULL );
        pGenBatchProofRequest->set_allocated_input(pInputProver);

        // Send the gen proof request
        aggregatorMessage.Clear();
        messageId++;
        aggregatorMessage.set_id(to_string(messageId));
        aggregatorMessage.set_allocated_gen_batch_proof_request(pGenBatchProofRequest);
        bResult = stream->Write(aggregatorMessage);
        if (!bResult)
        {
            cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Write(aggregatorMessage)" << endl;
            return Status::CANCELLED;
        }

        // Receive the corresponding get proof response message
        proverMessage.Clear();
        bResult = stream->Read(&proverMessage);
        if (!bResult)
        {
            cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Read(proverMessage)" << endl;
            return Status::CANCELLED;
        }
        
        // Check type
        if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGenBatchProofResponse)
        {
            cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GEN_BATCH_PROOF_RESPONSE" << endl;
            return Status::CANCELLED;
        }

        // Check id
        if (proverMessage.id() != aggregatorMessage.id())
        {
            cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
            return Status::CANCELLED;
        }

        uuid = proverMessage.gen_batch_proof_response().id();

        // CALL GET PROOF AND CHECK IT IS PENDING

        for (uint64_t i=0; i<5; i++)
        {
            // Send a get proof request message
            aggregatorMessage.Clear();
            messageId++;
            aggregatorMessage.set_id(to_string(messageId));
            aggregator::v1::GetProofRequest * pGetProofRequest = new aggregator::v1::GetProofRequest();
            zkassertpermanent(pGetProofRequest != NULL);
            pGetProofRequest->set_id(uuid);
            aggregatorMessage.set_allocated_get_proof_request(pGetProofRequest);
            bResult = stream->Write(aggregatorMessage);
            if (!bResult)
            {
                cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Write(aggregatorMessage)" << endl;
                return Status::CANCELLED;
            }

            // Receive the corresponding get proof response message
            proverMessage.Clear();
            bResult = stream->Read(&proverMessage);
            if (!bResult)
            {
                cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Read(proverMessage)" << endl;
                return Status::CANCELLED;
            }
            
            // Check type
            if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGetProofResponse)
            {
                cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GET_PROOF_RESPONSE" << endl;
                return Status::CANCELLED;
            }

            // Check id
            if (proverMessage.id() != aggregatorMessage.id())
            {
                cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
                return Status::CANCELLED;
            }

            // Check get proof result
            if (proverMessage.get_proof_response().result() != aggregator::v1::GetProofResponse_Result_RESULT_PENDING)
            {
                cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.get_proof_response().result()=" << proverMessage.get_proof_response().result() << " instead of RESULT_GET_PROOF_PENDING" << endl;
                return Status::CANCELLED;
            }

            sleep(5);
        }
    }

#ifdef LOG_SERVICE
    cout << "AggregatorServiceImpl::Channel() stream done" << endl;
#endif

    return Status::OK;
}