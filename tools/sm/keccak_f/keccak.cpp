#include "keccak.hpp"
#include "timer.hpp"
#include "keccak_input.hpp"
#include "gate_config.hpp"

GateConfig KeccakGateConfig = GateConfig(
	uint64_t(0),
	uint64_t(155286),
	uint64_t(160000),
	uint64_t(1),
	uint64_t(45),
	uint64_t(1600),
	uint64_t(44),
	uint64_t(45+(1600*44)),
	uint64_t(1600),
	uint64_t(44),
    uint64_t(1<<23)
);

/*
    Input is a buffer of any length, including 0
    Output is 256 bits long buffer containing the 32B keccak hash of the input
*/
void Keccak (const uint8_t * pInput, uint64_t inputSize, uint8_t * pOutput)
{
    KeccakInput input;
    input.init(pInput, inputSize);
    GateState S(KeccakGateConfig);

    uint8_t r[1088];
    while (input.getNextBits(r))
    {
        for (uint64_t i=0; i<1088; i++)
        {
            S.gate[KeccakGateConfig.sinRef0 + i*44].pin[pin_a].bit ^= r[i];
        }
        KeccakF(S);
        S.printCounters();
        
        S.copySoutToSinAndResetRefs();
    }
    S.getOutput(pOutput);
}