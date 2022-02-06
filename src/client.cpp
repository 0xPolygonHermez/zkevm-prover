
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
    std::shared_ptr<grpc_impl::Channel> channel = ::grpc::CreateChannel("localhost:" + to_string(config.gRPCServerPort), grpc::InsecureChannelCredentials());

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
    if (config.clientInputFile.size() == 0)
    {
        cerr << "Error: Client::GenProof() found config.clientInputFile empty" << endl;
        exit(-1);
    }
    ::grpc::ClientContext context;
    ::zkprover::InputProver request;
    Input input(fr);
    json inputJson;
    file2json(config.clientInputFile, inputJson);
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

void Client::Cancel (const string &uuid)
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

}

void Client::Execute (void)
{
    if (config.clientInputFile.size() == 0)
    {
        cerr << "Error: Client::Execute() found config.clientInputFile empty" << endl;
        exit(-1);
    }
    ::grpc::ClientContext context;
    ::zkprover::InputProver request;
    Input input(fr);
    json inputJson;
    file2json(config.clientInputFile, inputJson);
    input.load(inputJson);
    input.preprocessTxs();
    input2InputProver(fr, input, request);
    ::zkprover::ResExecute response;
    std::unique_ptr<grpc::ClientReaderWriter<zkprover::InputProver, zkprover::ResExecute>> readerWriter;
    readerWriter = stub->Execute(&context);
    readerWriter->Write(request);
    readerWriter->Read(&response);
    cout << "Client::Execute() got: " << response.DebugString() << endl;
}

void* clientThread(void* arg)
{
    string uuid;
    Client *pClient = (Client *)arg;
    sleep(5);
    pClient->GetStatus();
    uuid = pClient->GenProof();
    for (uint64_t i=0; i<100; i++)
    {
        sleep(5);
        if (pClient->GetProof(uuid)) break;
    }
    pClient->Cancel(uuid);
    pClient->Execute();
    return NULL;
}