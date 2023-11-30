#include "config.hpp"
#include "multichain_service.hpp"
#include "input.hpp"
#include "proof_fflonk.hpp"
#include <grpcpp/grpcpp.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

#define MULTICHAIN_SERVER_NUMBER_OF_LOOPS 1

#define MULTICHAIN_SERVER_RETRY_SLEEP 10
#define MULTICHAIN_SERVER_NUMBER_OF_GET_PROOF_RETRIES 600  // 600 retries every 10 seconds = 6000 seconds = 100 minutes

::grpc::Status MultichainServiceImpl::Channel(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream)
{
#ifdef LOG_SERVICE
    cout << "MultichainServiceImpl::Channel() stream starts" << endl;
#endif
    multichain::v1::MultichainMessage multichainMessage;
    multichain::v1::ProverMessage proverMessage;
    multichain::v1::Result result;
    string uuid;
    ::grpc::Status grpcStatus;
    string requestID;
    string proof;

    const string publicsFile0 = "testvectors/batchProof/input_executor_0.json";
    const string inputFile0  = "testvectors/aggregatedProof/recursive1.zkin.proof_0.json";
    const string outputFile0 = "testvectors/aggregationLayer/multichainPrepProof/multichainPrep.zkin.proof_0.json";
    const string outHashInfoFile0 = "testvectors/aggregationLayer/previousHash/previous_hash_1.json";

    const string publicsFile1 = "testvectors/batchProof/input_executor_1.json";
    const string inputFile1  = "testvectors/aggregatedProof/recursive1.zkin.proof_1.json";
    const string outputFile1 = "testvectors/aggregationLayer/multichainPrepProof/multichainPrep.zkin.proof_1.json";
    const string outHashInfoFile1 = "testvectors/aggregationLayer/previousHash/previous_hash_2.json";

    const string inputFile01a = outputFile0;
    const string inputFile01b = outputFile1;
    const string outputFile01 = "testvectors/aggregationLayer/multichainAggProof/multichainAgg.zkin.proof_01.json";

    const string publicsFile2 = "testvectors/batchProof/input_executor_2.json";
    const string inputFile2  = "testvectors/aggregatedProof/recursive1.zkin.proof_2.json";
    const string outputFile2 = "testvectors/aggregationLayer/multichainPrepProof/multichainPrep.zkin.proof_2.json";
    const string outHashInfoFile2 = "testvectors/aggregationLayer/previousHash/previous_hash_3.json";

    const string publicsFile3 = "testvectors/batchProof/input_executor_3.json";
    const string inputFile3  = "testvectors/aggregatedProof/recursive1.zkin.proof_3.json";
    const string outputFile3 = "testvectors/aggregationLayer/multichainPrepProof/multichainPrep.zkin.proof_3.json";
    const string outHashInfoFile3 = "testvectors/aggregationLayer/previousHash/previous_hash_4.json";

    const string inputFile23a = outputFile2;
    const string inputFile23b = outputFile3;
    const string outputFile23 = "testvectors/aggregationLayer/multichainAggProof/multichainAgg.zkin.proof_23.json";

    const string inputFile0123a = outputFile01;
    const string inputFile0123b = outputFile23;
    const string outputFile0123 = "testvectors/aggregationLayer/multichainAggProof/multichainAgg.zkin.proof_0123.json";

    const string inputFileFinal  = outputFile0123;
    const string outputFileFinal = "testvectors/aggregationLayer/multichainFinalProof/proof.json";


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
    if (result != multichain::v1::Result::RESULT_ERROR)
    {
        cerr << "Error: MultichainServiceImpl::Channel() got cancel result=" << result << " instead of RESULT_CANCEL_ERROR" << endl;
        return Status::CANCELLED;
    }

    //Calculate out hash 0
    json outHash0;
    grpcStatus = CalculateSha256Publics(context, stream, publicsFile0, "", outHashInfoFile0);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    file2json(outHashInfoFile0, outHash0);
    cout << "MultichainServiceImpl::Channel() called CalculateSha256Publics(" << publicsFile0 << ", " << "" << ", " << ", " << outHashInfoFile0 << ")" << endl;

    //Calculate out hash 1
    json outHash1;
    grpcStatus = CalculateSha256Publics(context, stream, publicsFile1, outHashInfoFile0, outHashInfoFile1);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    file2json(outHashInfoFile1, outHash1);
    cout << "MultichainServiceImpl::Channel() called CalculateSha256Publics(" << publicsFile1 << ", " << outHashInfoFile0 << ", " << ", " << outHashInfoFile1 << ")" << endl;

    //Calculate out hash 2
    json outHash2;
    grpcStatus = CalculateSha256Publics(context, stream, publicsFile2, outHashInfoFile1, outHashInfoFile2);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    file2json(outHashInfoFile2, outHash2);
    cout << "MultichainServiceImpl::Channel() called CalculateSha256Publics(" << publicsFile2 << ", " << outHashInfoFile1 << ", " << ", " << outHashInfoFile2 << ")" << endl;

    //Calculate out hash 3
    json outHash3;
    grpcStatus = CalculateSha256Publics(context, stream, publicsFile3, outHashInfoFile2, outHashInfoFile3);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    file2json(outHashInfoFile3, outHash3);
    cout << "MultichainServiceImpl::Channel() called CalculateSha256Publics(" << publicsFile3 << ", " << outHashInfoFile2 << ", " << ", " << outHashInfoFile3 << ")" << endl;

    cout << "Number of loops: " << MULTICHAIN_SERVER_NUMBER_OF_LOOPS << endl;
    for ( uint64_t loop=0; loop<MULTICHAIN_SERVER_NUMBER_OF_LOOPS; loop++ )
    {
        cout << "Starting loop: " << loop << endl;

        // Generate prepare multichain proof 0
        grpcStatus = GenAndGetPrepareMultichainProof(context, stream, inputFile0, "", outputFile0, outHashInfoFile0);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "MultichainServiceImpl::Channel() called GenAndGetPrepareMultichainProof(" << inputFile0 << ", " << "" << ", " << outputFile0<< ", " << outHashInfoFile0 << ")" << endl;
        
        // Compare the output hash with the one calculated using calculateSha256 rpc
        json newOutHash0;
        file2json(outHashInfoFile0, newOutHash0);
        for( uint64_t i = 0; i < 8; ++i) {
            if (newOutHash0["prevHash"][i] != outHash0["prevHash"][i])
            {
                cerr << "Error: MultichainServiceImpl::Channel() prevHash does not match" << endl;
                return Status::CANCELLED;
            }
        }
        if (newOutHash0["nPrevBlocks"] != outHash0["nPrevBlocks"])
        {
            cerr << "Error: MultichainServiceImpl::Channel() nPrevBlocks does not match" << endl;
            return Status::CANCELLED;
        }
        // Generate prepare multichain proof 1
        grpcStatus = GenAndGetPrepareMultichainProof(context, stream, inputFile1, outHashInfoFile0, outputFile1, outHashInfoFile1);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "MultichainServiceImpl::Channel() called GenAndGetPrepareMultichainProof(" << inputFile1 << ", " << outHashInfoFile0 << ", " << outputFile1 << ", " << outHashInfoFile1 << ")" << endl;

        // Compare the output hash with the one calculated using calculateSha256 rpc
        json newOutHash1;
        file2json(outHashInfoFile1, newOutHash1);
        for( uint64_t i = 0; i < 8; ++i) {
            if (newOutHash1["prevHash"][i] != outHash1["prevHash"][i])
            {
                cerr << "Error: MultichainServiceImpl::Channel() prevHash does not match" << endl;
                return Status::CANCELLED;
            }
        }
        if (newOutHash1["nPrevBlocks"] != outHash1["nPrevBlocks"])
        {
            cerr << "Error: MultichainServiceImpl::Channel() nPrevBlocks does not match" << endl;
            return Status::CANCELLED;
        }

        // Generate aggregated multichain proof 01
        grpcStatus = GenAndGetAggregatedMultichainProof(context, stream, inputFile01a, inputFile01b, outputFile01);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "MultichainServiceImpl::Channel() called GenAndGetAggregatedMultichainProof(" << inputFile01a << ", " << inputFile01b << ", " << outputFile01 << ")" << endl;


        // Generate prepare multichain proof 2
        grpcStatus = GenAndGetPrepareMultichainProof(context, stream, inputFile2, outHashInfoFile1, outputFile2, outHashInfoFile2);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "MultichainServiceImpl::Channel() called GenAndGetPrepareMultichainProof(" << inputFile2 << ", " << outHashInfoFile1 << ", " << outputFile2 << ", " << outHashInfoFile2 << ")" << endl;

        // Compare the output hash with the one calculated using calculateSha256 rpc
        json newOutHash2;
        file2json(outHashInfoFile2, newOutHash2);
        for( uint64_t i = 0; i < 8; ++i) {
            if (newOutHash2["prevHash"][i] != outHash2["prevHash"][i])
            {
                cerr << "Error: MultichainServiceImpl::Channel() prevHash does not match" << endl;
                return Status::CANCELLED;
            }
        }
        if (newOutHash2["nPrevBlocks"] != outHash2["nPrevBlocks"])
        {
            cerr << "Error: MultichainServiceImpl::Channel() nPrevBlocks does not match" << endl;
            return Status::CANCELLED;
        }

        // Generate prepare multichain proof 3
        grpcStatus = GenAndGetPrepareMultichainProof(context, stream, inputFile3, outHashInfoFile2, outputFile3, outHashInfoFile3);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "MultichainServiceImpl::Channel() called GenAndGetPrepareMultichainProof(" << inputFile3 << ", " << outHashInfoFile2 << ", " << outputFile3 << ", " << outHashInfoFile3 << ")" << endl;

        // Compare the output hash with the one calculated using calculateSha256 rpc
        json newOutHash3;
        file2json(outHashInfoFile3, newOutHash3);
        for( uint64_t i = 0; i < 8; ++i) {
            if (newOutHash3["prevHash"][i] != outHash3["prevHash"][i])
            {
                cerr << "Error: MultichainServiceImpl::Channel() prevHash does not match" << endl;
                return Status::CANCELLED;
            }
        }
        if (newOutHash3["nPrevBlocks"] != outHash3["nPrevBlocks"])
        {
            cerr << "Error: MultichainServiceImpl::Channel() nPrevBlocks does not match" << endl;
            return Status::CANCELLED;
        }

        // Generate aggregated multichain proof 23
        grpcStatus = GenAndGetAggregatedMultichainProof(context, stream, inputFile23a, inputFile23b, outputFile23);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "MultichainServiceImpl::Channel() called GenAndGetAggregatedMultichainProof(" << inputFile23a << ", " << inputFile23b << ", " << outputFile23 << ")" << endl;


        // Generate aggregated proof 0123
        grpcStatus = GenAndGetAggregatedMultichainProof(context, stream, inputFile0123a, inputFile0123b, outputFile0123);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "MultichainServiceImpl::Channel() called GenAndGetAggregatedMultichainProof(" << inputFile0123a << ", " << inputFile0123b << ", " << outputFile0123 << ")" << endl;

        // Generate final proof
        grpcStatus = GenAndGetFinalMultichainProof(context, stream, inputFileFinal, outputFileFinal);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "MultichainServiceImpl::Channel() called GenAndGetFinalMultichainProof(" << inputFileFinal << ", " << outputFileFinal << ")" << endl;

        cout << "Ending loop: " << loop << endl;
    }

    return Status::OK;
}

::grpc::Status MultichainServiceImpl::GetStatus(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream)
{
    multichain::v1::MultichainMessage multichainMessage;
    multichain::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;

    // Send a get status request message
    multichainMessage.Clear();
    multichain::v1::GetStatusRequest * pGetStatusRequest = new multichain::v1::GetStatusRequest();
    zkassertpermanent(pGetStatusRequest != NULL);
    multichainMessage.set_allocated_get_status_request(pGetStatusRequest);
    messageId++;
    multichainMessage.set_id(to_string(messageId));
    bResult = stream->Write(multichainMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::GetStatus() failed calling stream->Write(multichainMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get status response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::GetStatus() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != multichain::v1::ProverMessage::ResponseCase::kGetStatusResponse)
    {
        cerr << "Error: MultichainServiceImpl::GetStatus() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GET_STATUS_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != multichainMessage.id())
    {
        cerr << "Error: MultichainServiceImpl::GetStatus() got proverMessage.id=" << proverMessage.id() << " instead of multichainMessage.id=" << multichainMessage.id() << endl;
        return Status::CANCELLED;
    }

    return Status::OK;
}

::grpc::Status MultichainServiceImpl::Cancel(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string &requestID, multichain::v1::Result &result)
{
    multichain::v1::MultichainMessage multichainMessage;
    multichain::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;

    // Send a cancel request message
    multichainMessage.Clear();
    messageId++;
    multichainMessage.set_id(to_string(messageId));
    multichain::v1::CancelRequest * pCancelRequest = new multichain::v1::CancelRequest();
    zkassertpermanent(pCancelRequest != NULL);
    pCancelRequest->set_id(requestID);
    multichainMessage.set_allocated_cancel_request(pCancelRequest);
    bResult = stream->Write(multichainMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::Cancel() failed calling stream->Write(multichainMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding cancel response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::Cancel() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != multichain::v1::ProverMessage::ResponseCase::kCancelResponse)
    {
        cerr << "Error: MultichainServiceImpl::Cancel() got proverMessage.response_case=" << proverMessage.response_case() << " instead of CANCEL_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != multichainMessage.id())
    {
        cerr << "Error: MultichainServiceImpl::Cancel() got proverMessage.id=" << proverMessage.id() << " instead of multichainMessage.id=" << multichainMessage.id() << endl;
        return Status::CANCELLED;
    }

    // Check cancel result
    result = proverMessage.cancel_response().result();

    return Status::OK;
}

::grpc::Status MultichainServiceImpl::CalculateSha256Publics(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string & publicsFile, const string & previousHashFile, const string &outputHashFile)
{
    multichain::v1::MultichainMessage multichainMessage;
    multichain::v1::ProverMessage proverMessage;

    bool bResult;
    string publicsFileContent;
    string previousHashContent;

    if (publicsFile.size() == 0)
    {
        cerr << "Error: MultichainServiceImpl::CalculateSha256Publics() found publicsFile empty" << endl;
        exitProcess();
    }
    file2string(publicsFile, publicsFileContent);

    if(previousHashFile.size() != 0) {
        file2string(previousHashFile, previousHashContent);
    }

    // Allocate the prepare proof request
    multichain::v1::CalculateSha256Request *pCalculateSha256RequestRequest = new multichain::v1::CalculateSha256Request();
    zkassertpermanent(pCalculateSha256RequestRequest != NULL);
    pCalculateSha256RequestRequest->set_publics(publicsFileContent);
    pCalculateSha256RequestRequest->set_previous_hash(previousHashContent);

    // Send the calculate sha256 request
    multichainMessage.Clear();
    messageId++;
    multichainMessage.set_id(to_string(messageId));
    multichainMessage.set_allocated_calculate_sha256_request(pCalculateSha256RequestRequest);
    bResult = stream->Write(multichainMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::CalculateSha256Publics() failed calling stream->Write(multichainMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding CalculateSha256Publics response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::CalculateSha256Publics() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != multichain::v1::ProverMessage::ResponseCase::kCalculateSha256Response)
    {
        cerr << "Error: MultichainServiceImpl::CalculateSha256Publics() got proverMessage.response_case=" << proverMessage.response_case() << " instead of CALCULATE_SHA256_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != multichainMessage.id())
    {
        cerr << "Error: MultichainServiceImpl::CalculateSha256Publics() got proverMessage.id=" << proverMessage.id() << " instead of multichainMessage.id=" << multichainMessage.id() << endl;
        return Status::CANCELLED;
    }

    string2file(proverMessage.calculate_sha256_response().out_hash(), outputHashFile);

    return Status::OK;

}

::grpc::Status MultichainServiceImpl::GenPrepareMultichainProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string &inputFile, const string &previousHashFile, string &requestID)
{
    multichain::v1::MultichainMessage multichainMessage;
    multichain::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;
    string inputFileContent;
    string previousHashContent;

    if (inputFile.size() == 0)
    {
        cerr << "Error: MultichainServiceImpl::GenPrepareMultichainProof() found inputFile empty" << endl;
        exitProcess();
    }
    file2string(inputFile, inputFileContent);

    if(previousHashFile.size() != 0) {
        file2string(previousHashFile, previousHashContent);
    }

    // Allocate the prepare proof request
    multichain::v1::GenPrepareMultichainProofRequest *pGenPrepareMultichainProofRequest = new multichain::v1::GenPrepareMultichainProofRequest();
    zkassertpermanent(pGenPrepareMultichainProofRequest != NULL);
    pGenPrepareMultichainProofRequest->set_recursive_proof(inputFileContent);
    pGenPrepareMultichainProofRequest->set_previous_hash(previousHashContent);


    // Send the gen proof request
    multichainMessage.Clear();
    messageId++;
    multichainMessage.set_id(to_string(messageId));
    multichainMessage.set_allocated_gen_prepare_multichain_proof_request(pGenPrepareMultichainProofRequest);
    bResult = stream->Write(multichainMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::GenPrepareMultichainProof() failed calling stream->Write(multichainMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get proof response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::GenPrepareMultichainProof() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != multichain::v1::ProverMessage::ResponseCase::kGenPrepareMultichainProofResponse)
    {
        cerr << "Error: MultichainServiceImpl::GenPrepareMultichainProof() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GEN_FINAL_MULTICHAIN_PROOF_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != multichainMessage.id())
    {
        cerr << "Error: MultichainServiceImpl::GenPrepareMultichainProof() got proverMessage.id=" << proverMessage.id() << " instead of multichainMessage.id=" << multichainMessage.id() << endl;
        return Status::CANCELLED;
    }

    requestID = proverMessage.gen_prepare_multichain_proof_response().id();

    return Status::OK;
}

::grpc::Status MultichainServiceImpl::GenAggregatedMultichainProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string &inputFileA, const string &inputFileB, string &requestID)
{
    multichain::v1::MultichainMessage multichainMessage;
    multichain::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;
    string inputFileAContent;
    string inputFileBContent;

    if (inputFileA.size() == 0)
    {
        cerr << "Error: MultichainServiceImpl::GenAggregatedMultichainProof() found inputFileA empty" << endl;
        exitProcess();
    }
    file2string(inputFileA, inputFileAContent);

    if (inputFileB.size() == 0)
    {
        cerr << "Error: MultichainServiceImpl::GenAggregatedMultichainProof() found inputFileB empty" << endl;
        exitProcess();
    }
    file2string(inputFileB, inputFileBContent);

    // Allocate the aggregated multichain proof request
    multichain::v1::GenAggregatedMultichainProofRequest *pGenAggregatedMultichainProofRequest = new multichain::v1::GenAggregatedMultichainProofRequest();
    zkassertpermanent(pGenAggregatedMultichainProofRequest != NULL );
    pGenAggregatedMultichainProofRequest->set_multichain_proof_1(inputFileAContent);
    pGenAggregatedMultichainProofRequest->set_multichain_proof_2(inputFileBContent);

    // Send the gen proof request
    multichainMessage.Clear();
    messageId++;
    multichainMessage.set_id(to_string(messageId));
    multichainMessage.set_allocated_gen_aggregated_multichain_proof_request(pGenAggregatedMultichainProofRequest);
    bResult = stream->Write(multichainMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::GenAggregatedMultichainProof() failed calling stream->Write(multichainMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get proof response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::GenAggregatedMultichainProof() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != multichain::v1::ProverMessage::ResponseCase::kGenAggregatedMultichainProofResponse)
    {
        cerr << "Error: MultichainServiceImpl::GenAggregatedMultichainProof() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GEN_AGGREGATED_MULTICHAIN_PROOF_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != multichainMessage.id())
    {
        cerr << "Error: MultichainServiceImpl::GenAggregatedMultichainProof() got proverMessage.id=" << proverMessage.id() << " instead of multichainMessage.id=" << multichainMessage.id() << endl;
        return Status::CANCELLED;
    }

    requestID = proverMessage.gen_aggregated_multichain_proof_response().id();

    return Status::OK;
}

::grpc::Status MultichainServiceImpl::GenFinalMultichainProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string &inputFile, string &requestID)
{
    multichain::v1::MultichainMessage multichainMessage;
    multichain::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;
    string inputFileContent;

    if (inputFile.size() == 0)
    {
        cerr << "Error: MultichainServiceImpl::GenFinalMultichainProof() found inputFile empty" << endl;
        exitProcess();
    }
    file2string(inputFile, inputFileContent);

    // Allocate the final multichain proof request
    multichain::v1::GenFinalMultichainProofRequest *pGenFinalMultichainProofRequest = new multichain::v1::GenFinalMultichainProofRequest();
    zkassertpermanent(pGenFinalMultichainProofRequest != NULL );
    pGenFinalMultichainProofRequest->set_multichain_proof(inputFileContent);

    // Send the gen proof request
    multichainMessage.Clear();
    messageId++;
    multichainMessage.set_id(to_string(messageId));
    multichainMessage.set_allocated_gen_final_multichain_proof_request(pGenFinalMultichainProofRequest);
    bResult = stream->Write(multichainMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::GenFinalMultichainProof() failed calling stream->Write(multichainMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get proof response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::GenFinalMultichainProof() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != multichain::v1::ProverMessage::ResponseCase::kGenFinalMultichainProofResponse)
    {
        cerr << "Error: MultichainServiceImpl::GenFinalMultichainProof() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GEN_FINAL_MULTICHAIN_PROOF_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != multichainMessage.id())
    {
        cerr << "Error: MultichainServiceImpl::GenFinalMultichainProof() got proverMessage.id=" << proverMessage.id() << " instead of multichainMessage.id=" << multichainMessage.id() << endl;
        return Status::CANCELLED;
    }

    requestID = proverMessage.gen_final_multichain_proof_response().id();

    return Status::OK;
}

::grpc::Status MultichainServiceImpl::GetProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string &requestID, multichain::v1::GetProofResponse_Result &result, string &proof)
{
    multichain::v1::MultichainMessage multichainMessage;
    multichain::v1::ProverMessage proverMessage;
    bool bResult;

    // Send a get proof request message
    multichainMessage.Clear();
    messageId++;
    multichainMessage.set_id(to_string(messageId));
    multichain::v1::GetProofRequest * pGetProofRequest = new multichain::v1::GetProofRequest();
    zkassertpermanent(pGetProofRequest != NULL);
    pGetProofRequest->set_id(requestID);
    multichainMessage.set_allocated_get_proof_request(pGetProofRequest);
    bResult = stream->Write(multichainMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::GetProof() failed calling stream->Write(multichainMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get proof response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::GetProof() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != multichain::v1::ProverMessage::ResponseCase::kGetProofResponse)
    {
        cerr << "Error: MultichainServiceImpl::GetProof() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GET_PROOF_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != multichainMessage.id())
    {
        cerr << "Error: MultichainServiceImpl::GetProof() got proverMessage.id=" << proverMessage.id() << " instead of multichainMessage.id=" << multichainMessage.id() << endl;
        return Status::CANCELLED;
    }

    // Copy get proof result
    result = proverMessage.get_proof_response().result();
    proof = proverMessage.get_proof_response().multichain_proof();

    return Status::OK;
}

::grpc::Status MultichainServiceImpl::GetProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string &requestID, multichain::v1::GetProofResponse_Result &result, string &proof, string &hashInfo)
{
    multichain::v1::MultichainMessage multichainMessage;
    multichain::v1::ProverMessage proverMessage;
    bool bResult;

    // Send a get proof request message
    multichainMessage.Clear();
    messageId++;
    multichainMessage.set_id(to_string(messageId));
    multichain::v1::GetProofRequest * pGetProofRequest = new multichain::v1::GetProofRequest();
    zkassertpermanent(pGetProofRequest != NULL);
    pGetProofRequest->set_id(requestID);
    multichainMessage.set_allocated_get_proof_request(pGetProofRequest);
    bResult = stream->Write(multichainMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::GetProof() failed calling stream->Write(multichainMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get proof response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: MultichainServiceImpl::GetProof() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != multichain::v1::ProverMessage::ResponseCase::kGetProofResponse)
    {
        cerr << "Error: MultichainServiceImpl::GetProof() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GET_PROOF_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != multichainMessage.id())
    {
        cerr << "Error: MultichainServiceImpl::GetProof() got proverMessage.id=" << proverMessage.id() << " instead of multichainMessage.id=" << multichainMessage.id() << endl;
        return Status::CANCELLED;
    }

    // Copy get proof result
    result = proverMessage.get_proof_response().result();
    proof = proverMessage.get_proof_response().prepare_proof().proof();
    hashInfo = proverMessage.get_proof_response().prepare_proof().hash_info();

    return Status::OK;
}


::grpc::Status MultichainServiceImpl::GenAndGetPrepareMultichainProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string & inputFile, const string & previousHashFile, const string &outputFile, const string &hashInfoFile)
{
    ::grpc::Status grpcStatus;
    string requestID;
    string proof;
    string hashInfo;
    uint64_t i;

    // Generate prepare multichain proof
    grpcStatus = GenPrepareMultichainProof(context, stream, inputFile, previousHashFile, requestID);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    cout << "MultichainServiceImpl::GenAndGetPrepareMultichainProof() called GenPrepareMultichainProof() and got requestID=" << requestID << endl;

    // Get prepare multichain proof
    for (i=0; i<MULTICHAIN_SERVER_NUMBER_OF_GET_PROOF_RETRIES; i++)
    {
        sleep(MULTICHAIN_SERVER_RETRY_SLEEP);

        multichain::v1::GetProofResponse_Result getProofResponseResult;
        grpcStatus = GetProof(context, stream, requestID, getProofResponseResult, proof, hashInfo);        
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }

        if (getProofResponseResult == multichain::v1::GetProofResponse_Result_RESULT_COMPLETED_OK)
        {
            break;
        }        
        if (getProofResponseResult == multichain::v1::GetProofResponse_Result_RESULT_PENDING)
        {
            continue;
        }
        cerr << "Error: MultichainServiceImpl::GenAndGetPrepareMultichainProof() got getProofResponseResult=" << getProofResponseResult << " instead of RESULT_PENDING or RESULT_COMPLETED_OK" << endl;
        return Status::CANCELLED;
    }
    if (i == MULTICHAIN_SERVER_NUMBER_OF_GET_PROOF_RETRIES)
    {
        cerr << "Error: MultichainServiceImpl::GenAndGetPrepareMultichainProof() timed out waiting for recursive proof" << endl;
        return Status::CANCELLED;
    }
    if (proof.size() == 0)
    {
        cerr << "Error: MultichainServiceImpl::GenAndGetPrepareMultichainProof() got an empty recursive proof" << endl;
        return Status::CANCELLED;
    }
    if (hashInfo.size() == 0)
    {
        cerr << "Error: MultichainServiceImpl::GenAndGetPrepareMultichainProof() got an empty last hash info" << endl;
        return Status::CANCELLED;
    }
    string2file(proof, outputFile);
    string2file(hashInfo, hashInfoFile);

    return Status::OK;
}

::grpc::Status MultichainServiceImpl::GenAndGetAggregatedMultichainProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string & inputFileA, const string & inputFileB, const string &outputFile)
{
    ::grpc::Status grpcStatus;
    string requestID;
    string proof;
    uint64_t i;

    // Generate aggregated multichain proof
    grpcStatus = GenAggregatedMultichainProof(context, stream, inputFileA, inputFileB, requestID);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    cout << "MultichainServiceImpl::GenAndGetAggregatedMultichainProof() called GenAggregatedMultichainProof() and got requestID=" << requestID << endl;

    // Get aggregated multichain proof
    for (i=0; i<MULTICHAIN_SERVER_NUMBER_OF_GET_PROOF_RETRIES; i++)
    {
        sleep(MULTICHAIN_SERVER_RETRY_SLEEP);

        multichain::v1::GetProofResponse_Result getProofResponseResult;
        grpcStatus = GetProof(context, stream, requestID, getProofResponseResult, proof);        
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }

        if (getProofResponseResult == multichain::v1::GetProofResponse_Result_RESULT_COMPLETED_OK)
        {
            break;
        }        
        if (getProofResponseResult == multichain::v1::GetProofResponse_Result_RESULT_PENDING)
        {
            continue;
        }
        cerr << "Error: MultichainServiceImpl::GenAndGetAggregatedMultichainProof() got getProofResponseResult=" << getProofResponseResult << " instead of RESULT_PENDING or RESULT_COMPLETED_OK" << endl;
        return Status::CANCELLED;
    }
    if (i == MULTICHAIN_SERVER_NUMBER_OF_GET_PROOF_RETRIES)
    {
        cerr << "Error: MultichainServiceImpl::GenAndGetAggregatedMultichainProof() timed out waiting for multichain proof" << endl;
        return Status::CANCELLED;
    }
    if (proof.size() == 0)
    {
        cerr << "Error: MultichainServiceImpl::GenAndGetAggregatedMultichainProof() got an empty multichain proof" << endl;
        return Status::CANCELLED;
    }
    string2file(proof, outputFile);

    return Status::OK;
}

::grpc::Status MultichainServiceImpl::GenAndGetFinalMultichainProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::multichain::v1::MultichainMessage, ::multichain::v1::ProverMessage>* stream, const string & inputFile, const string &outputFile)
{
    ::grpc::Status grpcStatus;
    string requestID;
    string proof;
    uint64_t i;

    grpcStatus = GenFinalMultichainProof(context, stream, inputFile, requestID);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    cout << "MultichainServiceImpl::GenAndGetFinalMultichainProof() called GenFinalMultichainProof() and got requestID=" << requestID << endl;

    for (i=0; i<MULTICHAIN_SERVER_NUMBER_OF_GET_PROOF_RETRIES; i++)
    {
        sleep(MULTICHAIN_SERVER_RETRY_SLEEP);

        multichain::v1::GetProofResponse_Result getProofResponseResult;
        grpcStatus = GetProof(context, stream, requestID, getProofResponseResult, proof);        
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }

        if (getProofResponseResult == multichain::v1::GetProofResponse_Result_RESULT_COMPLETED_OK)
        {
            break;
        }        
        if (getProofResponseResult == multichain::v1::GetProofResponse_Result_RESULT_PENDING)
        {
            continue;
        }
        cerr << "Error: MultichainServiceImpl::GenAndGetFinalMultichainProof() got getProofResponseResult=" << getProofResponseResult << " instead of RESULT_PENDING or RESULT_COMPLETED_OK" << endl;
        return Status::CANCELLED;
    }
    if (i == MULTICHAIN_SERVER_NUMBER_OF_GET_PROOF_RETRIES)
    {
        cerr << "Error: MultichainServiceImpl::GenAndGetFinalMultichainProof() timed out waiting for final multichain proof" << endl;
        return Status::CANCELLED;
    }
    if (proof.size() == 0)
    {
        cerr << "Error: MultichainServiceImpl::GenAndGetFinalMultichainProof() got an final multichain proof" << endl;
        return Status::CANCELLED;
    }
    string2file(proof, outputFile);

    return Status::OK;
}