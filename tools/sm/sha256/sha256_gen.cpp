
#include "sha256.hpp"
#include "timer.hpp"
#include "sha256_input.hpp"

int potato() {
    return 1;
}

/*
void SHA256Gen (
    const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput,
    string scriptFile, string polsFile, string connectionsFile)
{

    SHA256Input input;
    input.init(pInput, inputSize);
    SHA256State S;
    
    uint8_t r[1088];
    while (input.getNextBits(r))
    {

        KeccakF(S);
    }

    if (scriptFile.size() > 0)
    {
        json j;
        S.saveScriptToJson(j);
        cout << "Generating SHA256 script file: " << scriptFile << endl;
        json2file(j, scriptFile);
        cout << "Generated SHA256 script file: " << scriptFile << endl;
        scriptFile = "";
    }

    if (polsFile.size() > 0)
    {
        json j;
        S.savePolsToJson(j);
        cout << "Generating SHA256 polynomials file: " << polsFile << endl;
        json2file(j, polsFile);
        cout << "Generated SHA256 polynomials file: " << polsFile << endl;
        polsFile = "";
    }

    if (connectionsFile.size() > 0)
    {
        json j;
        S.saveConnectionsToJson(j);
        cout << "Generating SHA256 connections file: " << connectionsFile << endl;
        json2file(j, connectionsFile);
        cout << "Generated SHA256 connections file: " << connectionsFile << endl;
        connectionsFile = "";
    }
}

void KeccakGenerateScript (const Config & config)
{
    TimerStart(SHA256_SM_GENERATE_SCRIPT);
    uint8_t hash[32];
    SHA256Gen(NULL, 0, hash, config.keccakScriptFile, config.keccakPolsFile, config.keccakConnectionsFile);
    TimerStopAndLog(SHA256_SM_GENERATE_SCRIPT);
}
*/