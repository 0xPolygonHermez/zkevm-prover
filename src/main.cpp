#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include "ffiasm/fr.hpp"
#include "executor.hpp"
#include "utils.hpp"
#include "config.hpp"
#include "stark_struct.hpp"
#include "pil.hpp"
#include "script.hpp"
#include "mem.hpp"
#include "batchmachine_executor.hpp"
#include "proof2zkin.hpp"
#include "calcwit.hpp"
#include "circom.hpp"
#include "verifier_cpp/main.hpp"
#include "prover.hpp"
#include "server.hpp"

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

    // This raw FR library has been compiled to implement the curve BN128
    RawFr fr;

#if 0
    BatchMachineExecutor::batchInverseTest(fr);
#endif

    // Creat output directory, if specified
    if (config.outputPath.size()>0)
    {
        string command = "mkdir -p " + config.outputPath;
        system(command.c_str());
    }

    // Load and parse ROM JSON file
    TimerStart(ROM_LOAD);
    Rom romData;
    if (config.romFile.size()==0)
    {
        cerr << "Error: ROM file name is empty" << endl;
        exit(-1);
    }
    else
    {
        json romJson;
        file2json(config.romFile, romJson);
        romData.load(fr, romJson);
    }
    TimerStopAndLog(ROM_LOAD);

    // Load and parse PIL JSON file
    TimerStart(PIL_LOAD);
    Pil pil;
    if (config.pilFile.size()==0)
    {
        cerr << "Error: PIL file name is empty" << endl;
        exit(-1);
    }
    else
    {
        json pilJson;
        file2json(config.pilFile, pilJson);
        pil.parse(pilJson);
    }
    TimerStopAndLog(PIL_LOAD);

    // Load and parse script JSON file
    TimerStart(SCRIPT_LOAD);
    Script script(fr);
    if (config.scriptFile.size()==0)
    {
        cerr << "Error: script file name is empty" << endl;
        exit(-1);
    }
    else
    {
        json scriptJson;
        file2json(config.scriptFile, scriptJson);
        script.parse(scriptJson);
    }
    TimerStopAndLog(SCRIPT_LOAD);

    TimerStopAndLog(PARSE_JSON_FILES);

    // Load constant polynomials into memory, and map them to an existing input file containing their values
    TimerStart(LOAD_CONST_POLS_TO_MEMORY);
    Pols constPols;
    constPols.load(pil.constPols);
    constPols.mapToInputFile(config.constPolsFile);
    TimerStopAndLog(LOAD_CONST_POLS_TO_MEMORY);

    // Create the prover
    TimerStart(PROVER_CONSTRUCTOR);
    Prover prover(  fr,
                    romData,
                    script,
                    pil,
                    constPols,
                    config );
    TimerStopAndLog(PROVER_CONSTRUCTOR);
    if (config.bServer)
    {
        // Create server instance, passing all constant data
        ZkServer server(fr, prover, config);

        // Run the server
        server.run(); // Internally, it calls prover.prove() for every input data received, in order to generate the proof and return it to the client
    }
    else
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

    // Unload the ROM data
    TimerStart(ROM_UNLOAD);
    romData.unload();
    TimerStopAndLog(ROM_UNLOAD);

    TimerStopAndLog(WHOLE_PROCESS);

    cout << "Done" << endl;
}