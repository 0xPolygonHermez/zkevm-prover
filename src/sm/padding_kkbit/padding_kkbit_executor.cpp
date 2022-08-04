#include "padding_kkbit_executor.hpp"
#include "sm/keccak_f/keccak.hpp"


uint64_t bitFromState (const uint64_t (&st)[5][5][2], uint64_t i)
{
    uint64_t y = i / 320;
    uint64_t x = (i % 320) / 64;
    uint64_t z = i % 64;
    uint64_t z1 = z / 32;
    uint64_t z2 = z%32;

    return (st[x][y][z1] >> z2) & 1;
}

void setStateBit (uint64_t (&st)[5][5][2], uint64_t i, uint64_t b)
{
    uint64_t y = i/320;
    uint64_t x = (i%320)/64;
    uint64_t z = i%64;
    uint64_t z1 = z/32;
    uint64_t z2 = z%32;

    st[x][y][z1] ^=  (b << z2);
}

void callKeccakF (const uint64_t (&input)[5][5][2], uint64_t (&output)[5][5][2])
{
    // Copy input into state S
    KeccakState S;
    for (uint64_t i=0; i<1600; i++)
    {
        uint8_t bit = bitFromState(input, i);
        S.gate[SinRef0 + i*9].pin[pin_a].bit = bit;
    }

    // Call Keccak-f
    KeccakF(S);
    S.copySoutToSinAndResetRefs();
    
    // Reset output
    memset(output, 0, sizeof(output));

    // Add the corresponding bits
    for (uint64_t i=0; i<1600; i++)
    {
        uint8_t bit = (S.gate[SinRef0 + i*9].pin[pin_a].bit & 1);
        if (bit == 1) setStateBit(output, i, bit);
    }
}

void PaddingKKBitExecutor::execute (vector<PaddingKKBitExecutorInput> &input, PaddingKKBitCommitPols &pols, vector<Nine2OneExecutorInput> &required)
{
    uint64_t curInput = 0;
    uint64_t p = 0;
    //uint64_t v = 0;

    // Convert pols.sOutX to and array, for programming convenience
    CommitPol sOut[8] = { pols.sOut0, pols.sOut1, pols.sOut2, pols.sOut3, pols.sOut4, pols.sOut5, pols.sOut6, pols.sOut7 };

    uint64_t curState[5][5][2];
    bool bCurStateWritten = false;

    for (uint64_t i=0; i<nSlots; i++)
    {
        bool connected = true;

        uint64_t stateWithR[5][5][2];
        if ((curInput>=input.size()) || (input[curInput].connected == false))
        {
            connected = false;
            memset(stateWithR, 0, sizeof(stateWithR));
        }
        else
        {
            // Copy: stateWithR = curState;
            memcpy(&stateWithR, &curState, sizeof(stateWithR));
        }

        for (uint64_t j=0; j<136; j++)
        {
            uint8_t byte = (curInput < input.size()) ? input[curInput].r[j] : 0;
            pols.r8[p] = fr.zero();
            for (uint64_t k=0; k<8; k++)
            {
                uint64_t bit = (byte >> k) & 1;
                setStateBit(stateWithR, j*8+k, bit);
                pols.rBit[p] = fr.fromU64(bit);
                pols.r8[p+1] = fr.fromU64( fr.toU64(pols.r8[p]) | ((uint64_t(bit) << k)) );
                if (bCurStateWritten) pols.sOutBit[p] = fr.fromU64( bitFromState(curState, j*8 + k) );
                if (connected) pols.connected[p] = fr.one();
                p++;
            }

            if (connected) pols.connected[p] = fr.one();
            p++;
        }
        
        for (uint64_t j=0; j<512; j++)
        {
            if (bCurStateWritten) pols.sOutBit[p] = fr.fromU64( bitFromState(curState, 136*8 + j) );
            if (connected) pols.connected[p] = fr.one();
            p++;
        }

        callKeccakF(stateWithR, curState);
        bCurStateWritten = true;

        Nine2OneExecutorInput nine2OneExecutorInput;
        // Copy: nine2OneExecutorInput.st[0] = stateWithR
        memcpy(&nine2OneExecutorInput.st[0], stateWithR, sizeof(nine2OneExecutorInput.st[0]));
        // Copy: nine2OneExecutorInput.st[1] = curState
        memcpy(&nine2OneExecutorInput.st[1], curState, sizeof(nine2OneExecutorInput.st[1]));

        required.push_back(nine2OneExecutorInput);

        for (uint64_t j=0; j<256; j++)
        {
            pols.sOutBit[p] = fr.fromU64( bitFromState(curState, j) );
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

    // Connect the last state with the first
    uint64_t pp = 0;
    for (uint64_t j=0; j<136; j++)
    {
        for (uint64_t k=0; k<8; k++)
        {
            pols.sOutBit[pp] = fr.fromU64( bitFromState(curState, j*8 + k) );
            pp += 1;
        }
        pols.sOutBit[pp] = fr.zero();
        pp++;
    }

    for (uint64_t j=0; j<512; j++)
    {
        pols.sOutBit[pp] = fr.fromU64( bitFromState(curState, 136*8 + j) );
        pp++;
    }

    cout << "PaddingKKBitExecutor successfully processed " << input.size() << " Keccak actions" << endl;
}

