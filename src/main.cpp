#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include <filesystem>
#include "goldilocks_base_field.hpp"
#include "sm/main/main_executor.hpp"
#include "utils.hpp"
#include "config.hpp"
#include "version.hpp"
#include "proof2zkin.hpp"
#include "calcwit.hpp"
#include "circom.hpp"
#include "zkevm_verifier_cpp/main.hpp"
#include "prover.hpp"
#include "service/prover/prover_server.hpp"
#include "service/prover/prover_server_mock.hpp"
#include "service/prover/prover_client.hpp"
#include "service/executor/executor_server.hpp"
#include "service/executor/executor_client.hpp"
#include "sm/keccak_f/keccak.hpp"
#include "sm/keccak_f/keccak_executor_test.hpp"
#include "sm/storage/storage_executor.hpp"
#include "sm/storage/storage_test.hpp"
#include "sm/binary/binary_test.hpp"
#include "sm/mem_align/mem_align_test.hpp"
#include "starkpil/test/stark_test.hpp"
#include "timer.hpp"
#include "statedb/statedb_server.hpp"
#include "service/statedb/statedb_test.hpp"

using namespace std;
using namespace std::filesystem;
using json = nlohmann::json;

/*
    Prover (available via GRPC service)
    |\
    | Executor (available via GRPC service)
    | |\
    | | Main State Machine
    | | Byte4 State Machine
    | | Binary State Machine
    | | Memory State Machine
    | | Mem Align State Machine
    | | Arithmetic State Machine
    | | Storage State Machine------\
    | |                             |--> Poseidon G State Machine
    | | Padding PG State Machine---/
    | | Padding KK SM -> Padding KK Bit -> Nine To One SM -> Keccak-f SM -> Norm Gate 9 SM
    |  \
    |   State DB (available via GRPC service)
    |   |\
    |   | SMT
    |    \
    |     Database
    |\
    | Stark
    |\
    | Circom
*/

void runFile (Prover& prover, ProverRequest& proverRequest, string file)
{
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    if (file.size() > 0)
    {
        json inputJson;
        file2json(file, inputJson);
        proverRequest.input.load(inputJson);
    }
    TimerStopAndLog(INPUT_LOAD);

    // Call the prover
    TimerStart(PROVE);
    Proof proof;
    prover.prove(&proverRequest);
    TimerStopAndLog(PROVE);
}

int main(int argc, char **argv)
{
    // Always print the version
    cout << "Version: " << string(ZKEVM_PROVER_VERSION) << endl;

    if (argc==2)
    {
        if ( (strcmp(argv[1], "-v") == 0) || (strcmp(argv[1], "--version") == 0) )
        {
            // If requested to only print the version, then exit the program
            return 0;
        }
    }

    TimerStart(WHOLE_PROCESS);

    // Parse the name of the configuration file
    char * pConfigFile = (char *)"config.json";
    if (argc==3)
    {
        if ( (strcmp(argv[1], "-c") == 0) || (strcmp(argv[1], "--config") == 0) )
        {
            pConfigFile = argv[2];
        }
    }

    // Create one instance of Config based on the contents of the file config.json
    TimerStart(LOAD_CONFIG_JSON);
    json configJson;
    file2json(pConfigFile, configJson);
    Config config;
    config.load(configJson);
    config.print();
    TimerStopAndLog(LOAD_CONFIG_JSON);

    // Create one instance of the Goldilocks finite field instance
    Goldilocks fr;

    // Create one instance of the Poseidon hash library
    PoseidonGoldilocks poseidon;

    /* TOOLS */

    // Generate Keccak SM script
    if ( config.runKeccakScriptGenerator )
    {
        KeccakGenerateScript(config);
    }

    /* TESTS */

    // Test STARK
    if ( config.runStarkTest )
    {
        StarkTest();
    }

    // Test Keccak SM
    if ( config.runKeccakTest )
    {
        //Keccak2Test();
        KeccakSMTest();
        KeccakSMExecutorTest(fr, config);
    }

    // Test Storage SM
    if ( config.runStorageSMTest )
    {
        StorageSMTest(fr, poseidon, config);
    }

    // Test Binary SM
    if ( config.runBinarySMTest )
    {
        BinarySMTest(fr, config);
    }

    // Test MemAlign SM
    if ( config.runMemAlignSMTest )
    {
        MemAlignSMTest(fr, config);
    }

    // If there is nothing else to run, exit normally
    if (!config.runProverServer && !config.runProverServerMock && !config.runProverClient &&
        !config.runExecutorServer && !config.runExecutorClient &&
        !config.runFile && !config.runFileFast && !config.runStateDBServer && !config.runStateDBTest)
    {
        exit(0);
    }

#if 0
    BatchMachineExecutor::batchInverseTest(fr);
#endif

    // Creat output3 directory, if specified; otherwise, current working directory will be used to store output files
    if (config.outputPath.size()>0)
    {
        string command = "[ -d " + config.outputPath + " ] && echo \"Output directory already exists\" || mkdir -p " + config.outputPath;
        int iResult = system(command.c_str());
        if (iResult!=0)
        {
            cerr << "main() system() returned: " << to_string(iResult) << endl;
        }
    }

    // Create an instace of the Prover
    TimerStart(PROVER_CONSTRUCTOR);
    Prover prover( fr,
                   poseidon,
                   config );
    TimerStopAndLog(PROVER_CONSTRUCTOR);

    /* SERVERS */

    // Create the StateDB server and run it, if configured
    StateDBServer stateDBServer (fr, config);
    if (config.runStateDBServer)
    {
        cout << "Launching StateDB server thread..." << endl;
        stateDBServer.runThread();
    }

    // Create the prover server and run it, if configured
    ZkServer proverServer(fr, prover, config);
    if (config.runProverServer)
    {
        cout << "Launching prover server thread..." << endl;
        proverServer.runThread();
    }

    // Create the prover server mock and run it, if configured
    ZkServerMock proverServerMock(fr, prover, config);
    if (config.runProverServerMock)
    {
        cout << "Launching prover mock server thread..." << endl;
        proverServerMock.runThread();
    }

    // Create the executor server and run it, if configured
    ExecutorServer executorServer(fr, prover, config);
    if (config.runExecutorServer)
    {
        cout << "Launching executor server thread..." << endl;
        executorServer.runThread();
    }

    /* FILE-BASED INPUT */

    // Generate a proof from the input file
    if (config.runFile)
    {
        // Create an empty prover request
        ProverRequest proverRequest(fr);

        if (config.inputFile.back() == '/') // Process all input files in the folder
        {
            // Get the files in the folder
            vector<string> vfiles;
            for (directory_entry p: directory_iterator(config.inputFile))
            {
                vfiles.push_back(p.path().filename());
            }
            // Sort files alphabetically
            sort(vfiles.begin(),vfiles.end());
            // Process each input file in order
            for (vector<string>::const_iterator it(vfiles.begin()), it_end(vfiles.end()); it!=it_end; it++) 
            {
                cout << "runFile inputFile=" << *it << endl;
                // Init proverRequest
                proverRequest.init(config, *it);
                // Call the prover
                runFile (prover, proverRequest, config.inputFile+*it);
            }
        } else {
            // Init proverRequest
            proverRequest.init(config);
            // Call the prover
            runFile (prover, proverRequest, config.inputFile);
        }
    }

    // Execute (no proof generation) the input file
    if (config.runFileFast)
    {
        // Create and init an empty prover request
        ProverRequest proverRequest(fr);
        proverRequest.init(config);

        // Load and parse input JSON file
        TimerStart(INPUT_LOAD);
        if (config.inputFile.size() > 0)
        {
            json inputJson;
            file2json(config.inputFile, inputJson);
            proverRequest.input.load(inputJson);
        }
        TimerStopAndLog(INPUT_LOAD);

        ProverRequest proverRequest2(proverRequest);

        // Call the prover
        TimerStart(PROVE_EXECUTE_FAST);
        prover.processBatch(&proverRequest);
        TimerStopAndLog(PROVE_EXECUTE_FAST);

        // Call the prover, again, since the first time there is some setup work involved
        TimerStart(PROVE_EXECUTE_FAST_2);
        prover.processBatch(&proverRequest2);
        TimerStopAndLog(PROVE_EXECUTE_FAST_2);
    }

    /* CLIENTS */

    // Create the prover client and run it, if configured
    ProverClient proverClient(fr, config);
    if (config.runProverClient)
    {
        cout << "Launching client thread..." << endl;
        proverClient.runThread();
    }

    // Create the executor client and run it, if configured
    ExecutorClient executorClient(fr, config);
    if (config.runExecutorClient)
    {
        cout << "Launching executor client thread..." << endl;
        executorClient.runThread();
    }

    // Run the stateDB test, if configured
    if (config.runStateDBTest)
    {
        cout << "Launching StateDB test thread..." << endl;
        runStateDBTest(config);
    }

    /* THREADS COMPETION */

    // Wait for the executor client thread to end
    if (config.runExecutorClient)
    {
        executorClient.waitForThread();
        sleep(1);
        exit(0);
    }

    // Wait for the prover client thread to end
    if (config.runProverClient)
    {
        proverClient.waitForThread();
        sleep(1);
        exit(0);
    }

    // Wait for the prover server thread to end
    if (config.runProverServer)
    {
        proverServer.waitForThread();
    }

    // Wait for the prover mock server thread to end
    if (config.runProverServerMock)
    {
        proverServerMock.waitForThread();
    }

    // Wait for the executor server thread to end
    if (config.runExecutorServer)
    {
        executorServer.waitForThread();
    }

    // Wait for StateDBServer thread to end
    if (config.runStateDBServer)
    {
        stateDBServer.waitForThread();
    }

    TimerStopAndLog(WHOLE_PROCESS);

    cout << "Done" << endl;
}