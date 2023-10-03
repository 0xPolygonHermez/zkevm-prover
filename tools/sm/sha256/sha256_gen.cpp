#include "sha256_gate.hpp"
#include "timer.hpp"
#include "sha256_input.hpp"
#include "utils.hpp"
#include "sha256_config.hpp"

void SHA256Gen (string scriptFile="", string polsFile="", string connectionsFile="")
{
    // TODO: Export the generation part from SHA256 and put it here, just like
    // what was done with keccak_gen.cpp.
    uint8_t randomTestVector[32];
    for (size_t i = 0; i < 32; ++i)
    {
        //randomTestVector[i] = static_cast<uint8_t>(dis(gen));
        randomTestVector[i] = 0;
    }
    uint64_t paddedSizeInBitsMin = 32*8 + 1 + 64;
    uint64_t paddedSizeInBits = ((paddedSizeInBitsMin / 512) + 1)*512;
    uint64_t paddedSize = paddedSizeInBits / 8;
    uint64_t paddedZeros = (paddedSizeInBits - paddedSizeInBitsMin)/8;

    // Create the padding data buffer
    uint8_t padding[64] = {0};
    u642bytes(32*8, &padding[56], true);
    uint64_t onePosition = 64 - 8 - paddedZeros - 1;
    padding[onePosition] = 0x80;

    uint8_t randomTestVectorPadded[paddedSize];
    for (uint64_t i=0; i<32; i++)
    {
        randomTestVectorPadded[i] = randomTestVector[i];
    }
    for (uint64_t i=0; i<(paddedSize-32); i++)
    {
        randomTestVectorPadded[32+i] = padding[onePosition+i];
    }
    GateState S(SHA256GateConfig);
    SHA256Gate(S, randomTestVectorPadded, scriptFile, polsFile, connectionsFile);
}

void SHA256GenerateScript (const Config & config)
{
    TimerStart(SHA256_SM_GENERATE_SCRIPT);
    SHA256Gen(config.sha256ScriptFile, config.sha256PolsFile, config.sha256ConnectionsFile);
    TimerStopAndLog(SHA256_SM_GENERATE_SCRIPT);
}
