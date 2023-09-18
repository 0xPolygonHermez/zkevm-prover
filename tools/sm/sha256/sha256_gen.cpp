#include "sha256_gate.hpp"
#include "timer.hpp"
#include "sha256_input.hpp"
#include "utils.hpp"
#include "sha256_config.hpp"

void SHA256Gen (string scriptFile="", string polsFile="", string connectionsFile="")
{
    uint8_t thirtyTwoZeroBytes[32];
    for (size_t i = 0; i < 32; ++i)
    {
        thirtyTwoZeroBytes[i] = 0;
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

    uint8_t thirtyTwoZeroBytesPadded[paddedSize];
    for (uint64_t i=0; i<32; i++)
    {
        thirtyTwoZeroBytesPadded[i] = thirtyTwoZeroBytes[i];
    }
    for (uint64_t i=0; i<(paddedSize-32); i++)
    {
        thirtyTwoZeroBytesPadded[32+i] = padding[onePosition+i];
    }
    SHA256Gate(GateState(SHA256GateConfig), thirtyTwoZeroBytesPadded, scriptFile, polsFile, connectionsFile);
}

void SHA256GenerateScript (const Config & config)
{
    TimerStart(SHA256_SM_GENERATE_SCRIPT);
    SHA256Gen(config.sha256ScriptFile, config.sha256PolsFile, config.sha256ConnectionsFile);
    TimerStopAndLog(SHA256_SM_GENERATE_SCRIPT);
}
