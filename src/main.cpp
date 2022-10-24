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
#include "service/statedb/statedb.hpp"

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

void runFileGenProof (Goldilocks fr, Prover &prover, Config &config)
{
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    // Create and init an empty prover request
    ProverRequest proverRequest(fr);
    proverRequest.type = prt_genProof;
    proverRequest.init(config, false);
    if (config.inputFile.size() > 0)
    {
        json inputJson;
        file2json(config.inputFile, inputJson);
        zkresult zkResult = proverRequest.input.load(inputJson);
        if (zkResult != ZKR_SUCCESS)
        {
            cerr << "Error: runFileGenProof() failed calling proverRequest.input.load() zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
            exit(-1);
        }
    }
    TimerStopAndLog(INPUT_LOAD);

    // Call the prover
    prover.genProof(&proverRequest);
}

void runFileGenBatchProof (Goldilocks fr, Prover &prover, Config &config)
{
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    // Create and init an empty prover request
    ProverRequest proverRequest(fr);
    proverRequest.type = prt_genBatchProof;
    proverRequest.init(config, false);
    if (config.inputFile.size() > 0)
    {
        json inputJson;
        file2json(config.inputFile, inputJson);
        zkresult zkResult = proverRequest.input.load(inputJson);
        if (zkResult != ZKR_SUCCESS)
        {
            cerr << "Error: runFileGenBatchProof() failed calling proverRequest.input.load() zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
            exit(-1);
        }
    }
    TimerStopAndLog(INPUT_LOAD);

    // Call the prover
    prover.genBatchProof(&proverRequest);
}

void runFileGenAggregatedProof (Goldilocks fr, Prover &prover, Config &config)
{
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    // Create and init an empty prover request
    ProverRequest proverRequest(fr);
    proverRequest.type = prt_genAggregatedProof;
    proverRequest.init(config, false);
    if (config.inputFile.size() > 0)
    {
        file2json(config.inputFile, proverRequest.aggregatedProofInput);
    }
    TimerStopAndLog(INPUT_LOAD);

    // Call the prover
    prover.genAggregatedProof(&proverRequest);
}

void runFileGenFinalProof (Goldilocks fr, Prover &prover, Config &config)
{
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    // Create and init an empty prover request
    ProverRequest proverRequest(fr);
    proverRequest.type = prt_genFinalProof;
    proverRequest.init(config, false);
    if (config.inputFile.size() > 0)
    {
        file2json(config.inputFile, proverRequest.finalProofInput);
    }
    TimerStopAndLog(INPUT_LOAD);

    // Call the prover
    prover.genFinalProof(&proverRequest);
}

void runFileProcessBatch (Goldilocks fr, Prover &prover, Config &config)
{
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    // Create and init an empty prover request
    ProverRequest proverRequest(fr);
    proverRequest.type = prt_processBatch;
    proverRequest.init(config, true);
    if (config.inputFile.size() > 0)
    {
        json inputJson;
        file2json(config.inputFile, inputJson);
        zkresult zkResult = proverRequest.input.load(inputJson);
        if (zkResult != ZKR_SUCCESS)
        {
            cerr << "Error: runFileProcessBatch() failed calling proverRequest.input.load() zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
            exit(-1);
        }
    }
    TimerStopAndLog(INPUT_LOAD);

    // Call the prover
    prover.processBatch(&proverRequest);
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

void * runFileProcessBatchThread(void *arg)
{
    RunFileThreadArguments *pArgs = (RunFileThreadArguments *)arg;

    // For all files
    for (uint64_t i=0; i<RUN_FILE_MULTITHREAD_N_FILES; i++)
    {
        runFileProcessBatch(pArgs->fr, pArgs->prover, pArgs->config);
    }

    return NULL;
}

void runFileProcessBatchMultithread (Goldilocks &fr, Prover &prover, Config &config)
{
    RunFileThreadArguments args(fr, prover, config);

    pthread_t threads[RUN_FILE_MULTITHREAD_N_THREADS];

    // Launch all threads
    for (uint64_t i=0; i<RUN_FILE_MULTITHREAD_N_THREADS; i++)
    {
        pthread_create(&threads[i], NULL, runFileProcessBatchThread, &args);
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

    // Print the zkProver version
    cout << "Version: " << string(ZKEVM_PROVER_VERSION) << endl;

    // Print the number of cores
    cout << "Number of cores=" << getNumberOfCores() << endl;

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
    bool bError = false;
    if (!fileExists(config.romFile))
    {
        cerr << "Error: required file config.constPolsFile=" << config.constPolsFile << " does not exist" << endl;
        bError = true;
    }
    if (config.generateProof())
    {
        if (!fileExists(config.constPolsFile))
        {
            cerr << "Error: required file config.constPolsFile=" << config.constPolsFile << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.constPolsC12aFile))
        {
            cerr << "Error: required file config.constPolsC12aFile=" << config.constPolsC12aFile << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.constPolsC12bFile))
        {
            cerr << "Error: required file config.constPolsC12bFile=" << config.constPolsC12bFile << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.constantsTreeFile))
        {
            cerr << "Error: required file config.constantsTreeFile=" << config.constantsTreeFile << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.constantsTreeC12aFile))
        {
            cerr << "Error: required file config.constantsTreeC12aFile=" << config.constantsTreeC12aFile << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.constantsTreeC12bFile))
        {
            cerr << "Error: required file config.constantsTreeC12bFile=" << config.constantsTreeC12bFile << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.verifierFile))
        {
            cerr << "Error: required file config.verifierFile=" << config.verifierFile << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.verifierFileC12a))
        {
            cerr << "Error: required file config.verifierFileC12a=" << config.verifierFileC12a << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.verifierFileC12b))
        {
            cerr << "Error: required file config.verifierFileC12b=" << config.verifierFileC12b << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.starkVerifierFile))
        {
            cerr << "Error: required file config.starkVerifierFile=" << config.starkVerifierFile << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.storageRomFile))
        {
            cerr << "Error: required file config.storageRomFile=" << config.storageRomFile << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.starkInfoFile))
        {
            cerr << "Error: required file config.starkInfoFile=" << config.starkInfoFile << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.starkInfoC12aFile))
        {
            cerr << "Error: required file config.starkInfoC12aFile=" << config.starkInfoC12aFile << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.starkInfoC12bFile))
        {
            cerr << "Error: required file config.starkInfoC12bFile=" << config.starkInfoC12bFile << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.execC12aFile))
        {
            cerr << "Error: required file config.execC12aFile=" << config.execC12aFile << " does not exist" << endl;
            bError = true;
        }
        if (!fileExists(config.execC12bFile))
        {
            cerr << "Error: required file config.execC12bFile=" << config.execC12bFile << " does not exist" << endl;
            bError = true;
        }
    }
    if (bError) exitProcess();

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
        !config.runStateDBServer && !config.runStateDBTest &&
        !config.runFileGenProof && !config.runFileGenBatchProof && !config.runFileGenAggregatedProof && !config.runFileGenFinalProof &&
        !config.runFileProcessBatch && !config.runFileProcessBatchMultithread)
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

    /* INIT DB CACHE */
    if (config.loadDBToMemCache && (config.runProverServer || config.runExecutorServer || config.runStateDBServer))
    {
        StateDB stateDB(fr, config);
        stateDB.loadDB2MemCache();
    }

    /* SERVERS */

    // Create the StateDB server and run it, if configured
    StateDBServer * pStateDBServer = NULL;
    if (config.runStateDBServer)
    {
        pStateDBServer = new StateDBServer(fr, config);
        zkassert(pStateDBServer != NULL);
        cout << "Launching StateDB server thread..." << endl;
        pStateDBServer->runThread();
    }

    // Create the prover server and run it, if configured
    ZkServer * pProverServer = NULL;
    if (config.runProverServer)
    {
        pProverServer = new ZkServer(fr, prover, config);
        zkassert(pProverServer != NULL);
        cout << "Launching prover server thread..." << endl;
        pProverServer->runThread();
    }

    // Create the prover server mock and run it, if configured
    ZkServerMock * pProverServerMock = NULL;
    if (config.runProverServerMock)
    {
        pProverServerMock = new ZkServerMock(fr, prover, config);
        zkassert(pProverServerMock != NULL);
        cout << "Launching prover mock server thread..." << endl;
        pProverServerMock->runThread();
    }

    // Create the executor server and run it, if configured
    ExecutorServer * pExecutorServer = NULL;
    if (config.runExecutorServer)
    {
        pExecutorServer = new ExecutorServer(fr, prover, config);
        zkassert(pExecutorServer != NULL);
        cout << "Launching executor server thread..." << endl;
        pExecutorServer->runThread();
    }

    /* FILE-BASED INPUT */

    // Generate a proof from the input file
    if (config.runFileGenProof)
    {
        if (config.inputFile.back() == '/') // Process all input files in the folder
        {
            Config tmpConfig = config;
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile,true);
            // Process each input file in order
            for (size_t i=0; i<files.size(); i++)
            {
                tmpConfig.inputFile = config.inputFile + files[i];
                cout << "runFileGenProof inputFile=" << tmpConfig.inputFile << endl;
                // Call the prover
                runFileGenProof (fr, prover, tmpConfig);
            }
        } else {
            // Call the prover
            runFileGenProof (fr, prover, config);
        }
    }

    // Generate a batch proof from the input file
    if (config.runFileGenBatchProof)
    {
        if (config.inputFile.back() == '/') // Process all input files in the folder
        {
            Config tmpConfig = config;
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile,true);
            // Process each input file in order
            for (size_t i=0; i<files.size(); i++)
            {
                tmpConfig.inputFile = config.inputFile + files[i];
                cout << "runFileGenBatchProof inputFile=" << tmpConfig.inputFile << endl;
                // Call the prover
                runFileGenBatchProof (fr, prover, tmpConfig);
            }
        } else {
            // Call the prover
            runFileGenBatchProof (fr, prover, config);
        }
    }

    // Generate an aggregated proof from the input file
    if (config.runFileGenAggregatedProof)
    {
        if (config.inputFile.back() == '/') // Process all input files in the folder
        {
            Config tmpConfig = config;
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile,true);
            // Process each input file in order
            for (size_t i=0; i<files.size(); i++)
            {
                tmpConfig.inputFile = config.inputFile + files[i];
                cout << "runFileGenAggregatedProof inputFile=" << tmpConfig.inputFile << endl;
                // Call the prover
                runFileGenAggregatedProof (fr, prover, tmpConfig);
            }
        } else {
            // Call the prover
            runFileGenAggregatedProof (fr, prover, config);
        }
    }

    // Generate a final proof from the input file
    if (config.runFileGenFinalProof)
    {
        if (config.inputFile.back() == '/') // Process all input files in the folder
        {
            Config tmpConfig = config;
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile,true);
            // Process each input file in order
            for (size_t i=0; i<files.size(); i++)
            {
                tmpConfig.inputFile = config.inputFile + files[i];
                cout << "runFileGenFinalProof inputFile=" << tmpConfig.inputFile << endl;
                // Call the prover
                runFileGenFinalProof (fr, prover, tmpConfig);
            }
        } else {
            // Call the prover
            runFileGenFinalProof (fr, prover, config);
        }
    }

    // Execute (no proof generation) the input file
    if (config.runFileProcessBatch)
    {
        if (config.inputFile.back() == '/') {
            Config tmpConfig = config;
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile,true);
            // Process each input file in order
            for (size_t i=0; i<files.size(); i++)
            {
                tmpConfig.inputFile = config.inputFile + files[i];
                cout << "runFileProcessBatch inputFile=" << tmpConfig.inputFile << endl;
                // Call the prover
                runFileProcessBatch (fr, prover, tmpConfig);
            }
        } else {
            runFileProcessBatch(fr, prover, config);
        }
    }

    // Execute (no proof generation) the input file, in a multithread way
    if (config.runFileProcessBatchMultithread)
    {
        runFileProcessBatchMultithread(fr, prover, config);
    }

    /* CLIENTS */

    // Create the prover client and run it, if configured
    ProverClient * pProverClient = NULL;
    if (config.runProverClient)
    {
        pProverClient = new ProverClient(fr, config);
        zkassert(pProverClient != NULL);
        cout << "Launching client thread..." << endl;
        pProverClient->runThread();
    }

    // Create the executor client and run it, if configured
    ExecutorClient * pExecutorClient = NULL;
    if (config.runExecutorClient)
    {
        pExecutorClient = new ExecutorClient(fr, config);
        zkassert(pExecutorClient != NULL);
        cout << "Launching executor client thread..." << endl;
        pExecutorClient->runThread();
    }

    // Run the executor client multithread, if configured
    if (config.runExecutorClientMultithread)
    {
        if (pExecutorClient == NULL)
        {
            pExecutorClient = new ExecutorClient(fr, config);
            zkassert(pExecutorClient != NULL);
        }
        cout << "Launching executor client threads..." << endl;
        pExecutorClient->runThreads();
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
        zkassert(pExecutorClient != NULL);
        pExecutorClient->waitForThread();
        sleep(1);
        exit(0);
    }

    // Wait for the executor client thread to end
    if (config.runExecutorClientMultithread)
    {
        zkassert(pExecutorClient != NULL);
        pExecutorClient->waitForThreads();
        cout << "All executor client threads have completed" << endl;
        sleep(1);
        exit(0);
    }

    // Wait for the prover client thread to end
    if (config.runProverClient)
    {
        zkassert(pProverClient != NULL);
        pProverClient->waitForThread();
        sleep(1);
        exit(0);
    }

    // Wait for the prover server thread to end
    if (config.runProverServer)
    {
        zkassert(pProverServer != NULL);
        pProverServer->waitForThread();
    }

    // Wait for the prover mock server thread to end
    if (config.runProverServerMock)
    {
        zkassert(pProverServerMock != NULL);
        pProverServerMock->waitForThread();
    }

    // Wait for the executor server thread to end
    if (config.runExecutorServer)
    {
        zkassert(pExecutorServer != NULL);
        pExecutorServer->waitForThread();
    }

    // Wait for StateDBServer thread to end
    if (config.runStateDBServer)
    {
        zkassert(pStateDBServer != NULL);
        pStateDBServer->waitForThread();
    }

    // Clean up
    if (pExecutorClient != NULL)
    {
        delete pExecutorClient;
        pExecutorClient = NULL;
    }
    if (pProverClient != NULL)
    {
        delete pProverClient;
        pProverClient = NULL;
    }
    if (pProverServer != NULL)
    {
        delete pProverServer;
        pProverServer = NULL;
    }
    if (pProverServerMock != NULL)
    {
        delete pProverServerMock;
        pProverServerMock = NULL;
    }
    if (pExecutorServer != NULL)
    {
        delete pExecutorServer;
        pExecutorServer = NULL;
    }
    if (pStateDBServer != NULL)
    {
        delete pStateDBServer;
        pStateDBServer = NULL;
    }

    TimerStopAndLog(WHOLE_PROCESS);

    cout << "Done" << endl;
}