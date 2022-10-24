
#include <nlohmann/json.hpp>
#include "prover_client.hpp"

using namespace std;
using json = nlohmann::json;

ProverClient::ProverClient (Goldilocks &fr, const Config &config) :
    fr(fr),
    config(config)
{
    // Create channel
    std::shared_ptr<grpc_impl::Channel> channel = ::grpc::CreateChannel(config.proverClientHost + ":" + to_string(config.proverClientPort), grpc::InsecureChannelCredentials());

    // Create stub (i.e. client)
    stub = new zkprover::v1::ZKProverService::Stub(channel);
}

void ProverClient::runThread (void)
{
    cout << "ProverClient::runThread() creating clientThread" << endl;
    pthread_create(&t, NULL, clientThread, this);
}

void ProverClient::waitForThread (void)
{
    pthread_join(t, NULL);
}

void ProverClient::GetStatus (void)
{
    ::grpc::ClientContext context;
    ::zkprover::v1::GetStatusRequest request;
    ::zkprover::v1::GetStatusResponse response;
    stub->GetStatus(&context, request, &response);
    cout << "ProverClient::GetStatus() got: " << response.ShortDebugString() << endl;
}

string ProverClient::GenProof (void)
{
    if (config.inputFile.size() == 0)
    {
        cerr << "Error: ProverClient::GenProof() found config.inputFile empty" << endl;
        exit(-1);
    }
    ::grpc::ClientContext context;
    ::zkprover::v1::InputProver *pInputProver = new ::zkprover::v1::InputProver();
    Input input(fr);
    json inputJson;
    file2json(config.inputFile, inputJson);
    zkresult zkResult = input.load(inputJson);
    if (zkResult != ZKR_SUCCESS)
    {
        cerr << "Error: ProverClient::GenProof() failed calling input.load() zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
        exit(-1);
    }

    // Parse public inputs
    zkprover::v1::PublicInputs * pPublicInputs = new zkprover::v1::PublicInputs();
    pPublicInputs->set_old_state_root(input.publicInputs.oldStateRoot);
    pPublicInputs->set_old_local_exit_root(input.publicInputs.oldStateRoot);
    pPublicInputs->set_new_state_root(input.publicInputs.newStateRoot);
    pPublicInputs->set_new_local_exit_root(input.publicInputs.newLocalExitRoot);
    pPublicInputs->set_sequencer_addr(input.publicInputs.sequencerAddr);
    pPublicInputs->set_batch_hash_data(input.publicInputs.batchHashData);
    pPublicInputs->set_batch_num(input.publicInputs.batchNum);
    pPublicInputs->set_chain_id(input.publicInputs.chainId);
    pPublicInputs->set_eth_timestamp(input.publicInputs.timestamp);
    pInputProver->set_allocated_public_inputs(pPublicInputs);

    // Parse global exit root
    pInputProver->set_global_exit_root(input.globalExitRoot);

    // Parse batch L2 data
    pInputProver->set_batch_l2_data(input.batchL2Data);

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

    ::zkprover::v1::GenProofRequest request;
    request.set_allocated_input(pInputProver);
    ::zkprover::v1::GenProofResponse response;
    stub->GenProof(&context, request, &response);
    cout << "Client::GenProof() got: " << response.ShortDebugString() << endl;
    return response.id();
}

bool ProverClient::GetProof (const string &uuid)
{
    if (uuid.size() == 0)
    {
        cerr << "Error: ProverClient::GetProof() called with uuid empty" << endl;
        exit(-1);
    }
    ::grpc::ClientContext context;
    ::zkprover::v1::GetProofRequest request;
    request.set_id(uuid);
    ::zkprover::v1::GetProofResponse response;
    //stub->GetProof(&context, request, &response);
    std::unique_ptr<grpc::ClientReaderWriter<zkprover::v1::GetProofRequest, zkprover::v1::GetProofResponse>> readerWriter;
    readerWriter = stub->GetProof(&context);
    readerWriter->Write(request);
    readerWriter->Read(&response);
    cout << "Client::GetProof() got: " << response.ShortDebugString() << endl;
    if (response.result() == zkprover::v1::GetProofResponse_ResultGetProof_RESULT_GET_PROOF_PENDING)
    {
        return false;
    }
    return true;
}

bool ProverClient::Cancel (const string &uuid)
{
    if (uuid.size() == 0)
    {
        cerr << "Error: ProverClient::Cancel() called with uuid empty" << endl;
        exit(-1);
    }
    ::grpc::ClientContext context;
    ::zkprover::v1::CancelRequest request;
    request.set_id(uuid);
    ::zkprover::v1::CancelResponse response;
    stub->Cancel(&context, request, &response);
    cout << "Client::Cancel() got: " << response.ShortDebugString() << endl;
    if (response.result() == zkprover::v1::CancelResponse_ResultCancel_RESULT_CANCEL_OK)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void* clientThread(void* arg)
{
    cout << "clientThread() started" << endl;
    string uuid;
    ProverClient *pClient = (ProverClient *)arg;
    sleep(2);

    // Get server status
    cout << "clientThread() calling GetStatus()" << endl;
    pClient->GetStatus();

    // Generate a proof, call get proof up to 100 times (x5sec) until completed
    cout << "clientThread() calling GenProof()" << endl;
    uuid = pClient->GenProof();
    uint64_t i = 0;
    for (i=0; i<200; i++)
    {
        sleep(5);
        if (pClient->GetProof(uuid)) break;
    }
    if (i == 100)
    {
        cerr << "Error: clientThread() GetProof() polling failed" << endl;
        exit(-1);
    }

    // Cancelling an alreay completed request should fail
    cout << "clientThread() calling Cancel()" << endl;
    if (pClient->Cancel(uuid))
    {
        cerr << "Error: clientThread() Cancel() of completed request did not fail" << endl;
        exit(-1);
    }

    return NULL;
}