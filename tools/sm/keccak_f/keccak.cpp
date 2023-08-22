#include "keccak.hpp"
#include "timer.hpp"
#include "keccak_input.hpp"

/*
    Input is a buffer of any length, including 0
    Output is 256 bits long buffer containing the 32B keccak hash of the input
*/
void Keccak (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput)
{
    KeccakInput input;
    input.init(pInput, inputSize);
    KeccakState S;

    uint8_t r[1088];
    while (input.getNextBits(r))
    {
        for (uint64_t i=0; i<1088; i++)
        {
            S.gate[SinRef0 + i*44].pin[pin_a].bit ^= r[i];
        }
        KeccakF(S);
        S.printCounters();
        
        S.copySoutToSinAndResetRefs();
    }
    S.getOutput(pOutput);
}