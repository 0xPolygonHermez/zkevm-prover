#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include "goldilocks/goldilocks_base_field.hpp"
#include "sm/main/main_executor.hpp"
#include "utils.hpp"
#include "config.hpp"
#include "proof2zkin.hpp"
#include "calcwit.hpp"
#include "circom.hpp"
#include "verifier_cpp/main.hpp"
#include "prover.hpp"
#include "service/zkprover/server.hpp"
#include "service/zkprover/server_mock.hpp"
#include "service/zkprover/client.hpp"
#include "service/executor/executor_server.hpp"
#include "service/executor/executor_client.hpp"
#include "keccak2/keccak2.hpp"
#include "sm/keccak_f/keccak.hpp"
#include "sm/keccak_f/keccak_executor_test.hpp"
#include "sm/storage/storage_executor.hpp"
#include "sm/storage/storage_test.hpp"
#include "sm/binary/binary_test.hpp"
#include "sm/mem_align/mem_align_test.hpp"
#include "starkpil/test/stark_test.hpp"
#include "timer.hpp"
#include "statedb/statedb_server.hpp"
#include "statedb/statedb_client.hpp"
#include "statedb/statedb_test.hpp"

using namespace std;
using json = nlohmann::json;

int main(int argc, char **argv)
{    
    TimerStart(WHOLE_PROCESS);
    TimerStart(PARSE_JSON_FILES);

    // Load configuration file into a json object, and then into a Config instance
    json configJson;
    file2json("config.json", configJson);
    Config config;
    config.load(configJson);

    // Goldilocks finite field instance
    Goldilocks fr;

    // Test finite field
    if ( config.runFiniteFieldTest )
    {
        //fr.test();
    }

    // Test STARK

    if ( config.runStarkTest )
    {
        StarkTest();
    }

    // Poseidon instance
    Poseidon_goldilocks poseidon;

    // Generate Keccak SM script
    if ( config.runKeccakScriptGenerator )
    {
        KeccakGenerateScript(config);
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
        !config.runExecutorServer && !config.runExecutorServerMock && !config.runExecutorClient &&
        !config.runFile && !config.runFileFast && !config.runStateDBServer && !config.runStateDBClient &&
        !config.runStateDBLoad)
    {
        exit(0);
    }

    // Log parsed arguments and/or default file names
    cout << "Input file=" << config.inputFile << endl;
    cout << "Output path=" << config.outputPath << endl;
    cout << "ROM file=" << config.romFile << endl;
    cout << "PIL file=" << config.pilFile << endl;
    cout << "Output file=" << config.cmPolsFile << endl;
    cout << "Constants file=" << config.constPolsFile << endl;
    cout << "Constants tree file=" << config.constantsTreeFile << endl;
    cout << "Script file=" << config.scriptFile << endl;
    cout << "STARK file=" << config.starkFile << endl;
    cout << "Verifier file=" << config.verifierFile << endl;
    cout << "Witness file=" << config.witnessFile << endl;
    cout << "STARK verifier file=" << config.starkVerifierFile << endl;
    cout << "Public file=" << config.publicFile << endl;
    cout << "Proof file=" << config.proofFile << endl;

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

    TimerStopAndLog(PARSE_JSON_FILES);

    // Allocate an area of memory, mapped to file, to read all the constant polynomials,
    // and create them using the allocated address
    TimerStart(LOAD_CONST_POLS_TO_MEMORY);
    void * pAddress = NULL;
    if (config.constPolsFile.size() == 0)
    {
        cerr << "Error: main() received an empty cofnig.constPolsFile" << endl;
        exit(-1);
    }
    pAddress = mapFile(config.constPolsFile, ConstantPols::size(), false);
    cout << "Prover::prove() successfully mapped " << ConstantPols::size() << " bytes from constant file " << config.constPolsFile << endl;
    ConstantPols constPols(pAddress);
    TimerStopAndLog(LOAD_CONST_POLS_TO_MEMORY);

    // Create the prover
    TimerStart(PROVER_CONSTRUCTOR);
    Prover prover(  fr,
                    poseidon,
                    constPols,
                    config );
    TimerStopAndLog(PROVER_CONSTRUCTOR);

    // Create the StateDB server and run it if configured
    StateDB stateDB (fr, config, true, false);
    StateDBServer stateDBServer (fr, config, stateDB);
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

    // Create the executor server mock and run it, if configured
    /*ExecutorServerMock executorServerMock(fr, prover, config);
    if (config.runExecutorServerMock)
    {
        cout << "Launching executor mock server thread..." << endl;
        executorServerMock.runThread();
    }*/

    // Generate a proof from the input file
    if (config.runFile)
    {
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

        // Call the prover
        TimerStart(PROVE);
        Proof proof;
        prover.prove(&proverRequest);
        TimerStopAndLog(PROVE);
    }

    // Execute (no proof generation) the input file
    if (config.runFileFast)
    {
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

        // Call the prover
        TimerStart(PROVE1);
        prover.execute(&proverRequest);
        TimerStopAndLog(PROVE1);

        // Call the prover, again
        TimerStart(PROVE2);
        prover.execute(&proverRequest);
        TimerStopAndLog(PROVE2);
    }

    // Create the prover client and run it, if configured
    Client proverClient(fr, config);
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

    // Create the StateDBclient and run the test it if configured
    StateDBClient stateDBClient(fr, config);
    if (config.runStateDBClient)
    {
        cout << "Launching StateDB test client thread..." << endl;
        runStateDBTest(&stateDBClient);
    }

    // Run the load database process it if configured
    if (config.runStateDBLoad)
    {
        cout << "Launching StateDB load database thread..." << endl;
        runStateDBLoad(config);
    }    

    /* Wait for threads to complete */

    if (config.runExecutorClient)
    {
        executorClient.waitForThread();
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

    // Wait for the executor mock server thread to end
    /*if (config.runExecutorServerMock)
    {
        executorServerMock.waitForThread();
    }*/

    TimerStopAndLog(WHOLE_PROCESS);

    cout << "Done" << endl;
}