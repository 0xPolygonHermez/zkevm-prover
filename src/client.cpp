
#include <nlohmann/json.hpp>
#include "client.hpp"
#include "utils.hpp"

using namespace std;
using json = nlohmann::json;

Client::Client (RawFr &fr, const Config &config) :
    fr(fr),
    config(config)
{
    // Create channel
    std::shared_ptr<grpc_impl::Channel> channel = ::grpc::CreateChannel("localhost:" + to_string(config.clientPort), grpc::InsecureChannelCredentials());

    // Create stub (i.e. client)
    stub = new zkprover::ZKProver::Stub(channel);
}

void Client::runThread (void)
{
    pthread_create(&t, NULL, clientThread, this);
}

void Client::GetStatus (void)
{
    ::grpc::ClientContext context;
    ::zkprover::NoParams request;
    ::zkprover::ResGetStatus response;
    stub->GetStatus(&context, request, &response);
    cout << "Client::GetStatus() got: " << response.DebugString() << endl;
}

string Client::GenProof (void)
{
    if (config.inputFile.size() == 0)
    {
        cerr << "Error: Client::GenProof() found config.inputFile empty" << endl;
        exit(-1);
    }
    ::grpc::ClientContext context;
    ::zkprover::InputProver request;
    Input input(fr);
    json inputJson;
    file2json(config.inputFile, inputJson);
    input.load(inputJson);
    input.preprocessTxs();
    input2InputProver(fr, input, request);
    ::zkprover::ResGenProof response;
    stub->GenProof(&context, request, &response);
    cout << "Client::GenProof() got: " << response.DebugString() << endl;
    return response.id();
}

bool Client::GetProof (const string &uuid)
{
    if (uuid.size() == 0)
    {
        cerr << "Error: Client::GetProof() called with uuid empty" << endl;
        exit(-1);
    }
    ::grpc::ClientContext context;
    ::zkprover::RequestId request;
    request.set_id(uuid);
    ::zkprover::ResGetProof response;
    stub->GetProof(&context, request, &response);
    cout << "Client::GetProof() got: " << response.DebugString() << endl;
    if (response.result() == zkprover::ResGetProof_ResultGetProof_PENDING)
    {
        return false;
    }
    return true;
}

bool Client::Cancel (const string &uuid)
{
    if (uuid.size() == 0)
    {
        cerr << "Error: Client::Cancel() called with uuid empty" << endl;
        exit(-1);
    }
    ::grpc::ClientContext context;
    ::zkprover::RequestId request;
    request.set_id(uuid);
    ::zkprover::ResCancel response;
    stub->Cancel(&context, request, &response);
    cout << "Client::Cancel() got: " << response.DebugString() << endl;
    if (response.result() == zkprover::ResCancel_ResultCancel_OK)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool Client::Execute (void)
{
    if (config.inputFile.size() == 0)
    {
        cerr << "Error: Client::Execute() found config.clientInputFile empty" << endl;
        exit(-1);
    }
    ::grpc::ClientContext context;
    ::zkprover::InputProver request;
    Input input(fr);
    json inputJson;
    file2json(config.inputFile, inputJson);
    input.load(inputJson);
    input.preprocessTxs();
    input2InputProver(fr, input, request);
    ::zkprover::ResExecute response;
    std::unique_ptr<grpc::ClientReaderWriter<zkprover::InputProver, zkprover::ResExecute>> readerWriter;
    readerWriter = stub->Execute(&context);
    readerWriter->Write(request);
    readerWriter->Read(&response);
    cout << "Client::Execute() got: " << response.DebugString() << endl;
    return true; // TODO: return result, when available
}

void* clientThread(void* arg)
{
    cout << "clientThread() started" << endl;
    string uuid;
    Client *pClient = (Client *)arg;
    sleep(10);

    // Get server status
    cout << "clientThread() calling GetStatus()" << endl;
    pClient->GetStatus();

    // Generate a proof, call get proof up to 100 times (x5sec) until completed
    cout << "clientThread() calling GenProof()" << endl;
    uuid = pClient->GenProof();
    uint64_t i = 0;
    for (i=0; i<100; i++)
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

    // Execute should block and succeed
    cout << "clientThread() calling Execute()" << endl;
    pClient->Execute();
    return NULL;
}