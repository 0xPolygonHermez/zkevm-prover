#include "padding_kkbit_executor.hpp"


uint64_t bitFromState (uint64_t (&st)[5][5][2], uint64_t i)
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

void PaddingKKBitExecutor::execute (vector<PaddingKKBitExecutorInput> &input, PaddingKKBitCommitPols &pols, vector<Nine2OneExecutorInput> &required)
{
    uint64_t N = pols.degree();
    uint64_t nSlots = 9*((N-1)/slotSize);

    uint64_t curInput = 0;
    uint64_t p = 0;
    //uint64_t v = 0;

    // Convert pols.sOutX to and array, for programming convenience
    FieldElement * sOut[8];
    sOut[0] = pols.sOut0;
    sOut[1] = pols.sOut1;
    sOut[2] = pols.sOut2;
    sOut[3] = pols.sOut3;
    sOut[4] = pols.sOut4;
    sOut[5] = pols.sOut5;
    sOut[6] = pols.sOut6;
    sOut[7] = pols.sOut7;

    uint64_t curState[5][5][2];

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
            for (uint64_t x=0; x<5; x++)
                for (uint64_t y=0; y<5; y++)
                    for (uint64_t z=0; z<2; z++)
                        stateWithR[x][y][z] = curState[x][y][z];
        }

        for (uint64_t j=0; j<136; j++)
        {
            uint8_t byte = (curInput < input.size()) ? input[curInput].r[j] : 0;
            pols.r8[p] = 0;
            for (uint64_t k=0; k<8; k++)
            {
                uint64_t bit = (byte >> k) & 1;
                setStateBit(stateWithR, j*8+k, bit);
                pols.rBit[p] = bit;
                pols.r8[p+1] = pols.r8[p] | ((uint64_t(bit) << k));
                if (curState) pols.sOutBit[p] = bitFromState(curState, j*8 + k);
                for (uint64_t r=0; r<8; r++) sOut[r][p] = 0;
                pols.connected[p] = connected ? 1 : 0;
    
                p++;
            }

            pols.rBit[p] = 0;
            if (curState) pols.sOutBit[p] = 0;
            for (uint64_t k=0; k<8; k++) sOut[k][p] = 0;
            pols.connected[p] = connected ? 1 : 0;

            p++;
        }
        
        for (uint64_t j=0; j<512; j++)
        {
            pols.rBit[p] = 0;
            pols.r8[p] = 0;
            if (curState) pols.sOutBit[p] = bitFromState(curState, 136*8 + j);
            for (uint64_t r=0; r<8; r++) sOut[r][p] = 0;
            pols.connected[p] = connected ? 1 : 0;

            p++;
        }

        //curState = keccakF(stateWithR); TODO: Migrate this code

        Nine2OneExecutorInput nine2OneExecutorInput;
        for (uint64_t x=0; x<5; x++)
            for (uint64_t y=0; y<5; y++)
                for (uint64_t z=0; z<2; z++)
                    nine2OneExecutorInput.st[0][x][y][z] = stateWithR[x][y][z];

        for (uint64_t x=0; x<5; x++)
            for (uint64_t y=0; y<5; y++)
                for (uint64_t z=0; z<2; z++)
                    nine2OneExecutorInput.st[1][x][y][z] = curState[x][y][z];

        required.push_back(nine2OneExecutorInput);

        for (uint64_t k=0; k<8; k++) sOut[k][p] = 0;
        for (uint64_t j=0; j<256; j++)
        {
            pols.rBit[p] = 0;
            pols.r8[p] = 0;
            pols.sOutBit[p] = bitFromState(curState, j);
            pols.connected[p] = connected ? 1 : 0;

            uint64_t bit = j%8;
            uint64_t byte = j/8;
            uint64_t chunk = 7 - byte/4;
            uint64_t byteInChunk = 3 - byte%4;

            for (uint64_t k=0; k<8; k++)
            {
                if ( k == chunk) {
                    sOut[k][p+1] = sOut[k][p] | (pols.sOutBit[p] << ( byteInChunk*8 + bit));
                } else {
                    sOut[k][p+1] = sOut[k][p];
                }
            }
            p += 1;
        }

        // 0x52b3f53ff196a28e7d2d01283ef9427070bda64128fb5630b97b6ab17a8ff0a8

        pols.rBit[p] = 0;
        pols.r8[p] = 0;
        pols.sOutBit[p] = 0;
        pols.connected[p] = connected ? 1 : 0;
        p++;

        curInput++;
    }

    uint64_t pp = 0;
    // Connect the last state with the first
    for (uint64_t j=0; j<136; j++)
    {
        for (uint64_t k=0; k<8; k++)
        {
            pols.sOutBit[pp] = bitFromState(curState, j*8 + k);    
            pp += 1;
        }
        pols.sOutBit[pp] = 0;
        pp++;
    }

    for (uint64_t j=0; j<512; j++)
    {
        pols.sOutBit[pp] = bitFromState(curState, 136*8 + j);

        pp++;
    }

    /*while (p<N) {
        pols.rBit[p] = 0;
        pols.r8[p] = 0;
        pols.sOutBit[p] = 0;
        for (uint64_t r=0; r<8; r++) sOut[r][p] = 0;
        pols.connected[p] = 0;

        p++;
    }*/

    cout << "PaddingKKBitExecutor successfully processed " << input.size() << " Keccak actions" << endl;
}

