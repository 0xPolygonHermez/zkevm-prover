#include "sha256_gate.hpp"
#include "timer.hpp"
#include "sha256_input.hpp"
#include "utils.hpp"

void SHA256Gen (
    const uint8_t * pInput, uint64_t inputSize, string &hash,
    string scriptFile="", string polsFile="", string connectionsFile="")
{
    // TODO: Export the generation part from SHA256 and put it here, just like
    // what was done with keccak_gen.cpp.
    SHA256Gate(pInput, inputSize, hash, scriptFile, polsFile, connectionsFile);
}

void SHA256GenerateScript (const Config & config)
{
    TimerStart(SHA256_SM_GENERATE_SCRIPT);
    std::string hash = "";
    SHA256Gen(NULL, 0, hash, config.sha256ScriptFile, config.sha256PolsFile, config.sha256ConnectionsFile);
    TimerStopAndLog(SHA256_SM_GENERATE_SCRIPT);
}