#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
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

void runFile (Prover &prover, ProverRequest &proverRequest, Config &config)
{
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    if (config.inputFile.size() > 0)
    {
        json inputJson;
        file2json(config.inputFile, inputJson);
        zkresult zkResult = proverRequest.input.load(inputJson);
        if (zkResult != ZKR_SUCCESS)
        {
            cerr << "Error: runFile() failed calling proverRequest.input.load() zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
            exit(-1);
        }
    }
    TimerStopAndLog(INPUT_LOAD);

    // Call the prover
    TimerStart(PROVE);
    Proof proof;
    prover.prove(&proverRequest);
    TimerStopAndLog(PROVE);
}

void runFileFast (Prover &prover, ProverRequest &proverRequest, Config &config) 
{
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    if (config.inputFile.size() > 0)
    {
        json inputJson;
        file2json(config.inputFile, inputJson);
        zkresult zkResult = proverRequest.input.load(inputJson);
        if (zkResult != ZKR_SUCCESS)
        {
            cerr << "Error: runFileFast() failed calling proverRequest.input.load() zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
            exit(-1);
        }
    }
    TimerStopAndLog(INPUT_LOAD);

    TimerStart(PROVE_EXECUTE_FAST);
    prover.processBatch(&proverRequest);
    TimerStopAndLog(PROVE_EXECUTE_FAST);
}

class RunFileThreadArguments
{
public:
    Goldilocks &fr;
    Prover &prover;
    Config &config;
    RunFileThreadArguments(Goldilocks &fr, Prover &prover, Config &config) : fr(fr), prover(prover), config(config) {};
};

#define RUN_FILE_MULTITHREAD_N_THREADS  100
#define RUN_FILE_MULTITHREAD_N_FILES 100

void * runFileThread(void *arg)
{
    RunFileThreadArguments *pArgs = (RunFileThreadArguments *)arg;

    // For all files
    for (uint64_t i=0; i<RUN_FILE_MULTITHREAD_N_FILES; i++)
    {
        // Create and init an empty prover request
        ProverRequest proverRequest(pArgs->fr);
        proverRequest.init(pArgs->config);
        runFileFast(pArgs->prover, proverRequest, pArgs->config);
    }

    return NULL;
}

void runFileFastMultithread (Goldilocks &fr, Prover &prover, Config &config) 
{
    RunFileThreadArguments args(fr, prover, config);

    pthread_t threads[RUN_FILE_MULTITHREAD_N_THREADS];

    // Launch all threads
    for (uint64_t i=0; i<RUN_FILE_MULTITHREAD_N_THREADS; i++)
    {
        pthread_create(&threads[i], NULL, runFileThread, &args);
    }

    // Wait for all threads to complete
    for (uint64_t i=0; i<RUN_FILE_MULTITHREAD_N_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
}

int main(int argc, char **argv)
{
    /* CONFIG */

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

    // Check required files presence
    ensureFileExists(config.romFile);
    if (config.generateProof())
    {
        bool bError = false;
        if (!ensureFileExists(config.constPolsFile)) bError = true;
        if (!ensureFileExists(config.constPolsC12aFile)) bError = true;
        if (!ensureFileExists(config.constPolsC12bFile)) bError = true;
        if (!ensureFileExists(config.constantsTreeFile)) bError = true;
        if (!ensureFileExists(config.constantsTreeC12aFile)) bError = true;
        if (!ensureFileExists(config.constantsTreeC12bFile)) bError = true;
        if (!ensureFileExists(config.verifierFile)) bError = true;
        if (!ensureFileExists(config.verifierFileC12a)) bError = true;
        if (!ensureFileExists(config.verifierFileC12b)) bError = true;
        if (!ensureFileExists(config.starkVerifierFile)) bError = true;
        if (!ensureFileExists(config.storageRomFile)) bError = true;
        if (!ensureFileExists(config.starkInfoFile)) bError = true;
        if (!ensureFileExists(config.starkInfoC12aFile)) bError = true;
        if (!ensureFileExists(config.starkInfoC12bFile)) bError = true;
        if (!ensureFileExists(config.execC12aFile)) bError = true;
        if (!ensureFileExists(config.execC12bFile)) bError = true;
        if (bError) exitProcess();
    }

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
        !config.runExecutorServer && !config.runExecutorClient && !config.runExecutorClientMultithread &&
        !config.runFile && !config.runFileFast && !config.runFileFastMultithread && !config.runStateDBServer && !config.runStateDBTest)
    {
        exit(0);
    }

#if 0
    BatchMachineExecutor::batchInverseTest(fr);
#endif

    // Create output directory, if specified; otherwise, current working directory will be used to store output files
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
            Config tmpConfig = config;
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile,true);
            // Process each input file in order
            for (size_t i=0; i<files.size(); i++)
            {
                tmpConfig.inputFile = config.inputFile + files[i];
                cout << "runFile inputFile=" << tmpConfig.inputFile << endl;
                // Init proverRequest
                proverRequest.init(tmpConfig);
                // Call the prover
                runFile (prover, proverRequest, tmpConfig);
            }
        } else {
            // Init proverRequest
            proverRequest.init(config);
            // Call the prover
            runFile (prover, proverRequest, config);
        }
    }

    // Execute (no proof generation) the input file
    if (config.runFileFast)
    {
        // Create and init an empty prover request
        ProverRequest proverRequest(fr);

        if (config.inputFile.back() == '/') {
            Config tmpConfig = config;
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile,true);
            // Process each input file in order
            for (size_t i=0; i<files.size(); i++)
            {
                tmpConfig.inputFile = config.inputFile + files[i];
                cout << "runFileFast inputFile=" << tmpConfig.inputFile << endl;
                // Init proverRequest
                proverRequest.init(tmpConfig);
                // Call the prover
                runFileFast (prover, proverRequest, tmpConfig);
            }
        } else {
            proverRequest.init(config);
            runFileFast(prover, proverRequest, config);
        }
    }

    // Execute (no proof generation) the input file, in a multithread way
    if (config.runFileFastMultithread)
    {
        runFileFastMultithread(fr, prover, config);
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

    // Run the executor client multithread, if configured
    if (config.runExecutorClientMultithread)
    {
        cout << "Launching executor client threads..." << endl;
        executorClient.runThreads();
    }

    // Run the stateDB test, if configured
    if (config.runStateDBTest)
    {
        cout << "Launching StateDB test thread..." << endl;
        runStateDBTest(config);
    }

    /* WAIT FOR CLIENT THREADS COMPETION */

    // Wait for the executor client thread to end
    if (config.runExecutorClient)
    {
        executorClient.waitForThread();
        sleep(1);
        exit(0);
    }

    // Wait for the executor client thread to end
    if (config.runExecutorClientMultithread)
    {
        executorClient.waitForThreads();
        cout << "All executor client threads have completed" << endl;
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