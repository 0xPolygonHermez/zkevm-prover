#include "padding_kkbit_executor.hpp"
#include "sm/keccak_f/keccak.hpp"
#include "timer.hpp"
#include "definitions.hpp"
#include "Keccak-more-compact.hpp"
#include "zkmax.hpp"
#include "zklog.hpp"

inline uint64_t getStateBit ( const uint8_t (&state)[200], uint64_t i )
{
    return (state[i/8] >> (i%8)) & 1;
}

inline void setStateBit (uint8_t (&state)[200], uint64_t i, uint64_t b )
{
    state[i/8] ^= (b << (i%8));
}

inline void callKeccakF (const uint8_t (&inputState)[200], uint8_t (&outputState)[200])
{
    /* Copy inputState into OuputState */
    memcpy(outputState, inputState, sizeof(outputState));

    /* Call keccak-f (one single round) to update outputState */
    KeccakF1600(outputState);
}

void PaddingKKBitExecutor::execute (vector<PaddingKKBitExecutorInput> &input, PaddingKKBitCommitPols &pols, vector<Bits2FieldExecutorInput> &required)
{
#ifdef LOG_TIME_STATISTICS
    struct timeval t;
    uint64_t keccakTime=0, keccakTimes=0;
#endif
    // Check that input size does not exeed the number of slots
    if (input.size() > nSlots)
    {
        zklog.error("PaddingKKBitExecutor::execute() Too many entries input.size()=" + to_string(input.size()) + " > nSlots=" + to_string(nSlots));
        exitProcess();
    }

    uint64_t curInput = 0;
    uint64_t p = 0;
    uint64_t pDone = 0;

    // Convert pols.sOutX to and array, for programming convenience
    CommitPol sOut[8] = { pols.sOut0, pols.sOut1, pols.sOut2, pols.sOut3, pols.sOut4, pols.sOut5, pols.sOut6, pols.sOut7 };

    uint8_t currentState[200];
    bool bCurStateWritten = false;

    for (uint64_t i=0; i<nSlots; i++)
    {
        bool connected = true;

        uint8_t stateWithR[200];

        // If this is the first block of a message, or we run out of them, start from a reset state
        if ((curInput>=input.size()) || (input[curInput].connected == false))
        {
            connected = false;
            memset(stateWithR, 0, sizeof(stateWithR));
        }
        else
        {
            // Copy: stateWithR = currentState;
            memcpy(&stateWithR, &currentState, sizeof(stateWithR));
        }

        for (uint64_t j=0; j<136; j++)
        {
            uint8_t byte = (curInput < input.size()) ? input[curInput].data[j] : 0;
            pols.r8[p] = fr.zero();
            for (uint64_t k=0; k<8; k++)
            {
                uint64_t bit = (byte >> k) & 1;
                setStateBit(stateWithR, j*8+k, bit);
                pols.rBit[p] = fr.fromU64(bit);
                pols.r8[p+1] = fr.fromU64( fr.toU64(pols.r8[p]) | ((uint64_t(bit) << k)) );
                if (bCurStateWritten) pols.sOutBit[p] = fr.fromU64( getStateBit(currentState, j*8 + k) );
                if (connected) pols.connected[p] = fr.one();
                p++;
            }

            if (connected) pols.connected[p] = fr.one();
            p++;
        }
        
        for (uint64_t j=0; j<512; j++)
        {
            if (bCurStateWritten) pols.sOutBit[p] = fr.fromU64( getStateBit(currentState, 136*8 + j) );
            if (connected) pols.connected[p] = fr.one();
            p++;
        }
#ifdef LOG_TIME_STATISTICS
        gettimeofday(&t, NULL);
#endif
        callKeccakF(stateWithR, currentState);
        bCurStateWritten = true;
#ifdef LOG_TIME_STATISTICS
        keccakTime += TimeDiff(t);
        keccakTimes+=1;
#endif
        Bits2FieldExecutorInput bits2FieldExecutorInput;
        // Copy: bits2FieldExecutorInput.inputState = stateWithR
        memcpy(&bits2FieldExecutorInput.inputState, stateWithR, sizeof(bits2FieldExecutorInput.inputState));
        // Copy: bits2FieldExecutorInput.outputState = currentState
        memcpy(&bits2FieldExecutorInput.outputState, currentState, sizeof(bits2FieldExecutorInput.outputState));

        required.push_back(bits2FieldExecutorInput);

        for (uint64_t j=0; j<256; j++)
        {
            pols.sOutBit[p] = fr.fromU64( getStateBit(currentState, j) );
            if (connected) pols.connected[p] = fr.one();

            uint64_t bit = j%8;
            uint64_t byte = j/8;
            uint64_t chunk = 7 - byte/4;
            uint64_t byteInChunk = 3 - byte%4;

            for (uint64_t k=0; k<8; k++)
            {
                if ( k == chunk) {
                    sOut[k][p+1] = fr.fromU64( fr.toU64(sOut[k][p]) | (fr.toU64(pols.sOutBit[p]) << ( byteInChunk*8 + bit)) );
                } else {
                    sOut[k][p+1] = sOut[k][p];
                }
            }
            p += 1;
        }

        if (connected) pols.connected[p] = fr.one();
        p++;

        curInput++;
    }

    pDone = p;

    // Connect the last state with the first
    uint64_t pp = 0;
    for (uint64_t j=0; j<136; j++)
    {
        for (uint64_t k=0; k<8; k++)
        {
            pols.sOutBit[pp] = fr.fromU64( getStateBit(currentState, j*8 + k) );
            pp += 1;
        }
        pols.sOutBit[pp] = fr.zero();
        pp++;
    }

    for (uint64_t j=0; j<512; j++)
    {
        pols.sOutBit[pp] = fr.fromU64( getStateBit(currentState, 136*8 + j) );
        pp++;
    }

    zklog.info("PaddingKKBitExecutor successfully processed " + to_string(input.size()) + " Keccak actions p=" + to_string(p) + " pDone=" + to_string(pDone) + " (" + to_string((double(pDone)*100)/N) + "%)");
#ifdef LOG_TIME_STATISTICS
    zklog.info("TIMER STATISTICS: PaddingKKBitExecutor: Keccak time: " + to_string(double(keccakTime)/1000) + " ms, called " + to_string(keccakTimes) + " times, so " + to_string(keccakTime/zkmax(keccakTimes,(uint64_t)1)) + " us/time");
#endif
}

