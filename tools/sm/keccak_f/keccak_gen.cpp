#include "keccak.hpp"
#include "timer.hpp"
#include "keccak_input.hpp"

void KeccakGen (
    const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput,
    string scriptFile, string polsFile, string connectionsFile)
{
    KeccakInput input;
    input.init(pInput, inputSize);
    GateState S(KeccakGateConfig);

    uint8_t r[1088];
    while (input.getNextBits(r))
    {
        for (uint64_t i=0; i<1088; i++)
        {
            S.gate[S.gateConfig.sinRef0 + i*44].pin[pin_a].bit ^= r[i];
        }
        KeccakF(S);
        S.printCounters();

        // Generate the script file only after the first keccak-f round
        if (scriptFile.size() > 0)
        {
            json j;
            S.saveScriptToJson(j);
            cout << "Generating Keccak script file: " << scriptFile << endl;
            json2file(j, scriptFile);
            cout << "Generated Keccak script file: " << scriptFile << endl;
            scriptFile = "";
        }

        // Generate the polynomials file only after the first keccak-f round
        if (polsFile.size() > 0)
        {
            json j;
            S.savePolsToJson(j);
            cout << "Generating Keccak polynomials file: " << polsFile << endl;
            json2file(j, polsFile);
            cout << "Generated Keccak polynomials file: " << polsFile << endl;
            polsFile = "";
        }

        // Generate the connections file only after the first keccak-f round
        if (connectionsFile.size() > 0)
        {
            json j;
            S.saveConnectionsToJson(j);
            cout << "Generating Keccak connections file: " << connectionsFile << endl;
            json2file(j, connectionsFile);
            cout << "Generated Keccak connections file: " << connectionsFile << endl;
            connectionsFile = "";
        }
        
        S.copySoutToSinAndResetRefs();
    }
    S.getOutput(pOutput);
}

void KeccakGenerateScript (const Config & config)
{
    TimerStart(KECCAK_SM_GENERATE_SCRIPT);
    uint8_t hash[32];
    KeccakGen(NULL, 0, hash, config.keccakScriptFile, config.keccakPolsFile, config.keccakConnectionsFile);
    TimerStopAndLog(KECCAK_SM_GENERATE_SCRIPT);
}