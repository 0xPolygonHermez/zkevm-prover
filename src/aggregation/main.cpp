#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include "goldilocks_base_field.hpp"
#include "utils.hpp"
#include "config.hpp"
#include "version.hpp"
#include "calcwit.hpp"
#include "circom.hpp"
#include "prover_aggregation.hpp"
#include "multichain_server.hpp"
#include "multichain_client.hpp"
#include "timer.hpp"
#include "zklog.hpp"
#include "zkglobals.hpp"

using namespace std;
using json = nlohmann::json;

/*
    ProverAggregation (available via GRPC service)
    |\
    | Stark
    |\
    | Circom
*/

void runFileCalculateHash(Goldilocks fr, ProverAggregation &proverAggregation, Config &config) {
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    // Create and init an empty prover request
    ProverAggregationRequest proverAggregationRequest(fr, config, prt_calculateHash);
    if (config.inputFile.size() > 0)
    {
        file2json(config.inputFile, proverAggregationRequest.chainPublicsInput);
    }
    if (config.inputFile2.size() > 0)
    {
        file2json(config.inputFile2, proverAggregationRequest.prevHashInput);
    }
    TimerStopAndLog(INPUT_LOAD);

    // Call the prover
    proverAggregation.calculateHash(&proverAggregationRequest);
}

void runFileGenPrepareMultichainProof(Goldilocks fr, ProverAggregation &proverAggregation, Config &config) {
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    // Create and init an empty prover request
    ProverAggregationRequest proverAggregationRequest(fr, config, prt_genPrepareMultichainProof);
    if (config.inputFile.size() > 0)
    {
        file2json(config.inputFile, proverAggregationRequest.multichainPrepProofInput);
    }
    if (config.inputFile2.size() > 0)
    {
        file2json(config.inputFile2, proverAggregationRequest.multichainPrepPrevHashInput);
    }
    TimerStopAndLog(INPUT_LOAD);

    // Call the prover
    proverAggregation.genPrepareMultichainProof(&proverAggregationRequest);
}

void runFileGenAggregatedMultichainProof(Goldilocks fr, ProverAggregation &proverAggregation, Config &config) {
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    // Create and init an empty prover request
    ProverAggregationRequest proverAggregationRequest(fr, config, prt_genAggregatedMultichainProof);
    if (config.inputFile.size() > 0)
    {
        file2json(config.inputFile, proverAggregationRequest.aggregatedMultichainProofInput1);
    }
    if (config.inputFile2.size() > 0)
    {
        file2json(config.inputFile2, proverAggregationRequest.aggregatedMultichainProofInput2);
    }
    TimerStopAndLog(INPUT_LOAD);

    // Call the prover
    proverAggregation.genAggregatedMultichainProof(&proverAggregationRequest);
}

void runFileGenFinalMultichainProof(Goldilocks fr, ProverAggregation &proverAggregation, Config &config) {
    // Load and parse input JSON file
    TimerStart(INPUT_LOAD);
    // Create and init an empty prover request
    ProverAggregationRequest proverAggregationRequest(fr, config, prt_genFinalMultichainProof);
    if (config.inputFile.size() > 0)
    {
        file2json(config.inputFile, proverAggregationRequest.finalMultichainProofInput);
    }
    if(config.aggregatorAddress.size() > 0) {
        proverAggregationRequest.aggregatorAddress = config.aggregatorAddress;
    }
    TimerStopAndLog(INPUT_LOAD);

    // Call the prover
    proverAggregation.genFinalMultichainProof(&proverAggregationRequest);
}

int main(int argc, char **argv)
{
    /* CONFIG */

    if (argc == 2)
    {
        if ((strcmp(argv[1], "-v") == 0) || (strcmp(argv[1], "--version") == 0))
        {
            // If requested to only print the version, then exit the program
            return 0;
        }
    }

    // Parse the name of the configuration file
    char *pConfigFile = (char *)"config/config.json";
    if (argc == 3)
    {
        if ((strcmp(argv[1], "-c") == 0) || (strcmp(argv[1], "--config") == 0))
        {
            pConfigFile = argv[2];
        }
    }

    // Create one instance of Config based on the contents of the file config.json
    json configJson;
    file2json(pConfigFile, configJson);
    config.load(configJson);
    zklog.setJsonLogs(config.jsonLogs);
    zklog.setPID(config.proverID.substr(0, 7)); // Set the logs prefix

    // Print the zkProver version
    zklog.info("Version: " + string(ZKEVM_PROVER_VERSION));

    // Test that stderr is properly logged
    cerr << "Checking error channel; ignore this trace\n";
    zklog.warning("Checking warning channel; ignore this trace");

    // Print the configuration file name
    string configFileName = pConfigFile;
    zklog.info("Config file: " + configFileName);

    // Print the number of cores
    zklog.info("Number of cores=" + to_string(getNumberOfCores()));

    // Print the hostname and the IP address
    string ipAddress;
    getIPAddress(ipAddress);
    zklog.info("IP address=" + ipAddress);

#ifdef DEBUG
    zklog.info("DEBUG defined");
#endif

    config.print();

    TimerStart(WHOLE_PROCESS);

    if (config.check())
    {
        zklog.error("main() failed calling config.check()");
        exitProcess();
    }

    // Create one instance of the Goldilocks finite field instance
    Goldilocks fr;

    // Create one instance of the Poseidon hash library
    PoseidonGoldilocks poseidon;
   
    // If there is nothing else to run, exit normally
    if (!config.runMultichainServer && !config.runMultichainClient && !config.runFileCalculateHash &&
        !config.runFileGenPrepareMultichainProof && !config.runFileGenAggregatedMultichainProof && !config.runFileGenFinalMultichainProof)
    {
        return 0;
    }

    // Create output directory, if specified; otherwise, current working directory will be used to store output files
    if (config.outputPath.size() > 0)
    {
        ensureDirectoryExists(config.outputPath);
    }

    // Create an instace of the ProverAggregation
    TimerStart(PROVER_AGGREGATION_CONSTRUCTOR);
    ProverAggregation proverAggregation(fr,
                  poseidon,
                  config);
    TimerStopAndLog(PROVER_AGGREGATION_CONSTRUCTOR);

    /* SERVERS */

    // Create the multichain server and run it, if configured
    MultichainServer *pMultichainServer = NULL;
    if (config.runMultichainServer)
    {
        pMultichainServer = new MultichainServer(fr, config);
        zkassert(pMultichainServer != NULL);
        zklog.info("Launching multichain server thread...");
        pMultichainServer->runThread();
        sleep(5);
    }

    // Calculate last hash from a prevHash file and a publics file
    if (config.runFileCalculateHash)
    {
        if (config.inputFile.back() == '/') // Process all input files in the folder
        {
            Config tmpConfig = config;
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile, true);
            // Process each input file in order
            for (size_t i = 0; i < files.size(); i++)
            {
                tmpConfig.inputFile = config.inputFile + files[i];
                zklog.info("runFileCalculateHash inputFile=" + tmpConfig.inputFile);
                // Call the prover
                runFileCalculateHash(fr, proverAggregation, tmpConfig);
            }
        }
        else
        {
            // Call the prover
            runFileCalculateHash(fr, proverAggregation, config);
        }
    }
    
    // Generate a multichain prepare proof from the input file
    if (config.runFileGenPrepareMultichainProof)
    {
        if (config.inputFile.back() == '/') // Process all input files in the folder
        {
            Config tmpConfig = config;
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile, true);
            // Process each input file in order
            for (size_t i = 0; i < files.size(); i++)
            {
                tmpConfig.inputFile = config.inputFile + files[i];
                zklog.info("runFileGenPrepareMultichainProof inputFile=" + tmpConfig.inputFile);
                // Call the prover
                runFileGenPrepareMultichainProof(fr, proverAggregation, tmpConfig);
            }
        }
        else
        {
            // Call the prover
            runFileGenPrepareMultichainProof(fr, proverAggregation, config);
        }
    }

    // Generate an aggregated multichain proof from the inputs file
    if (config.runFileGenAggregatedMultichainProof)
    {
        if (config.inputFile.back() == '/') // Process all input files in the folder
        {
            Config tmpConfig = config;
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile, true);
            // Process each input file in order
            for (size_t i = 0; i < files.size(); i++)
            {
                tmpConfig.inputFile = config.inputFile + files[i];
                zklog.info("runFileGenAggregatedMultichainProof inputFile=" + tmpConfig.inputFile);
                // Call the prover
                runFileGenAggregatedMultichainProof(fr, proverAggregation, tmpConfig);
            }
        }
        else
        {
            // Call the prover
            runFileGenAggregatedMultichainProof(fr, proverAggregation, config);
        }
    }

    // Generate a final multichain proof from the input file
    if (config.runFileGenFinalMultichainProof)
    {
        if (config.inputFile.back() == '/') // Process all input files in the folder
        {
            Config tmpConfig = config;
            // Get files sorted alphabetically from the folder
            vector<string> files = getFolderFiles(config.inputFile, true);
            // Process each input file in order
            for (size_t i = 0; i < files.size(); i++)
            {
                tmpConfig.inputFile = config.inputFile + files[i];
                zklog.info("runFileGenFinalMultichainProof inputFile=" + tmpConfig.inputFile);
                // Call the prover
                runFileGenFinalMultichainProof(fr, proverAggregation, tmpConfig);
            }
        }
        else
        {
            // Call the prover
            runFileGenFinalMultichainProof(fr, proverAggregation, config);
        }
    }

    /* CLIENTS */

    // Create the multichain client and run it, if configured
    MultichainClient *pMultichainClient = NULL;
    if (config.runMultichainClient)
    {
        pMultichainClient = new MultichainClient(fr, config, proverAggregation);
        zkassert(pMultichainClient != NULL);
        zklog.info("Launching multichain client thread...");
        pMultichainClient->runThread();
    }

    // Wait for the multichain client thread to end
    if (config.runMultichainClient)
    {
        zkassert(pMultichainClient != NULL);
        pMultichainClient->waitForThread();
        sleep(1);
        return 0;
    }

    // Wait for the multichain server thread to end
    if (config.runMultichainServer)
    {
        zkassert(pMultichainServer != NULL);
        pMultichainServer->waitForThread();
    }

    // Clean up
    if (pMultichainClient != NULL)
    {
        delete pMultichainClient;
        pMultichainClient = NULL;
    }

    if (pMultichainServer != NULL)
    {
        delete pMultichainServer;
        pMultichainServer = NULL;
    }

    TimerStopAndLog(WHOLE_PROCESS);

    zklog.info("Done");
}