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
#include "server_mock.hpp"
#include "client.hpp"
#include "eth_opcodes.hpp"
#include "opcode_address.hpp"
#include "keccak/keccak2.hpp"

using namespace std;
using json = nlohmann::json;

int main(int argc, char **argv)
{
    //KeccakTest();
    
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

    /* Load and parse ROM JSON file */

    TimerStart(ROM_LOAD);

    // Check rom file name
    if (config.romFile.size()==0)
    {
        cerr << "Error: ROM file name is empty" << endl;
        exit(-1);
    }

    // Load file contents into a json instance
    json romJson;
    file2json(config.romFile, romJson);

    // Load program array in Rom instance
    if (!romJson.contains("program") ||
        !romJson["program"].is_array() )
    {
        cerr << "Error: ROM file does not contain a program array at root level" << endl;
        exit(-1);
    }
    Rom romData;
    romData.load(fr, romJson["program"]);

    // Initialize the Ethereum opcode list: opcode=array position, operation=position content
    ethOpcodeInit();

    // Use the rom labels object to map every opcode to a ROM address
    if (!romJson.contains("labels") ||
        !romJson["labels"].is_object() )
    {
        cerr << "Error: ROM file does not contain a labels object at root level" << endl;
        exit(-1);
    }
    opcodeAddressInit(romJson["labels"]);

    TimerStopAndLog(ROM_LOAD);

    // Load and parse PIL JSON file
    TimerStart(PIL_LOAD);
    Pil pil;
    if (config.pilFile.size()==0)
    {
        cerr << "Error: PIL file name is empty" << endl;
        exit(-1);
    }
    json pilJson;
    file2json(config.pilFile, pilJson);
    pil.parse(pilJson);
    TimerStopAndLog(PIL_LOAD);

    // Load and parse script JSON file
    TimerStart(SCRIPT_LOAD);
    Script script(fr);
    if (config.scriptFile.size()==0)
    {
        cerr << "Error: script file name is empty" << endl;
        exit(-1);
    }
    json scriptJson;
    file2json(config.scriptFile, scriptJson);
    script.parse(scriptJson);
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

    // Create the server and run it if configured
    ZkServer server(fr, prover, config);
    if (config.runServer)
    {
        server.runThread();
    }

    // Create the server mock and run it if configured
    ZkServerMock serverMock(fr, prover, config);
    if (config.runServerMock)
    {
        serverMock.runThread();
    }

    if (!config.runServer)
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

    // Create the client and run it if configured
    Client client(fr, config);
    if (config.runClient)
    {
        client.runThread();
    }

    // Wait for the server thread to end
    if (config.runServer)
    {
        server.waitForThread();
    }

    // Unload the ROM data
    TimerStart(ROM_UNLOAD);
    romData.unload();
    TimerStopAndLog(ROM_UNLOAD);

    TimerStopAndLog(WHOLE_PROCESS);

    cout << "Done" << endl;
}