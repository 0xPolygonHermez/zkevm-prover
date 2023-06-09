#include "keccak_f_executor.hpp"
#include "keccak_executor_test.hpp"
#include "timer.hpp"

#if 0
void KeccakSMTest1 (KeccakFExecutor &executor)
{
    /* Use a well-known input */
    uint8_t input[188] = {
        0x09, 0x0B, 0xCA, 0xF7, 0x34, 0xC4, 0xF0, 0x6C, 0x93, 0x95,
        0x4A, 0x82, 0x7B, 0x45, 0xA6, 0xE8, 0xC6, 0x7B, 0x8E, 0x0F, 
        0xD1, 0xE0, 0xA3, 0x5A, 0x1C, 0x59, 0x82, 0xD6, 0x96, 0x18, 
        0x28, 0xF9, 0x09, 0x0B, 0xCA, 0xF7, 0x34, 0xC4, 0xF0, 0x6C, 
        0x93, 0x95, 0x4A, 0x82, 0x7B, 0x45, 0xA6, 0xE8, 0xC6, 0x7B, 

        0x8E, 0x0F, 0xD1, 0xE0, 0xA3, 0x5A, 0x1C, 0x59, 0x82, 0xD6, 
        0x96, 0x18, 0x28, 0xF9, 0x09, 0x0B, 0xCA, 0xF7, 0x34, 0xC4, 
        0xF0, 0x6C, 0x93, 0x95, 0x4A, 0x82, 0x7B, 0x45, 0xA6, 0xE8, 
        0xC6, 0x7B, 0x8E, 0x0F, 0xD1, 0xE0, 0xA3, 0x5A, 0x1C, 0x59, 
        0x82, 0xD6, 0x96, 0x18, 0x28, 0xF9, 0x17, 0xC0, 0x4C, 0x37, 

        0x60, 0x51, 0x0B, 0x48, 0xC6, 0x01, 0x27, 0x42, 0xC5, 0x40, 
        0xA8, 0x1A, 0xBA, 0x4B, 0xCA, 0x2F, 0x78, 0xB9, 0xD1, 0x4B, 
        0xFD, 0x2F, 0x12, 0x3E, 0x2E, 0x53, 0xEA, 0x3E, 0x61, 0x7B, 
        0x3A, 0x35, 0x28, 0xF9, 0xCD, 0xD6,   0x63, 0x0F, 0xD3, 0x30, 
        0x1B, 0x9C, 0x89, 0x11, 0xF7, 0xBF, 0x06, 0x3D, 0x29, 0x90,

        0x27, 0xCC, 0x1E, 0xE6, 0x56, 0x7E, 0x0F, 0xE5, 0xD6, 0x64, 
        0x87, 0x11, 0x82, 0xE4, 0xC6, 0xEA, 0xDA, 0xE6, 0x1A, 0x17, 
        0x06, 0xD8, 0x6D, 0x27, 0x32, 0x1A, 0xC3, 0x24, 0x6F, 0x98, 
        0x00, 0x00, 0x03, 0xE9, 0x00, 0x00, 0x00, 0x01};

    uint64_t inputSize = 188; // 188

    /* Call Keccak to get the hash of the input */
    TimerStart(KECCAK_SM_EXECUTOR);
    uint8_t hash[32];
    executor.Keccak(input, inputSize, hash);
    TimerStopAndLog(KECCAK_SM_EXECUTOR);
    printBa(hash, 32, "hash");    // Expected result: hash:0x1AFD6EAF13538380D99A245C2ACC4A25481B54556AE080CF07D1FACC0638CD8E

    /* Call the current Keccak to compare */
    TimerStart(CURRENT_KECCAK);
    string aux = keccak256(input, inputSize);
    TimerStopAndLog(CURRENT_KECCAK);
    cout << "Current Keccak: " << aux << endl;
}
#endif

void KeccakSMTest2 (KeccakFExecutor &executor)
{
    cout << "Starting 54-slots testing..." << endl;
    uint8_t Sin[Keccak_NumberOfSlots][136];
    uint8_t hashSout[Keccak_NumberOfSlots][32];
    string hashString[Keccak_NumberOfSlots];

    // Init Sin and hashString
    for (uint64_t slot=0; slot<Keccak_NumberOfSlots; slot++)
    {
        for (uint64_t i=0; i<136; i++)
        {
            Sin[slot][i] = rand();
        }
        Sin[slot][135] = 0b10000001;
        hashString[slot] = keccak256(&Sin[slot][0], 135);
    }

    uint8_t *bit;
    bit = (uint8_t *)malloc(Keccak_PolLength);
    if (bit==NULL)
    {
        cerr << "ERROR: KeccakSMExecutorTest() failed calling malloc of length:" << Keccak_PolLength << endl;
        exit(-1);
    }
    memset(bit, 0, Keccak_PolLength);
    for (uint64_t slot=0; slot<Keccak_NumberOfSlots; slot++)
    {
        for (uint64_t i=0; i<136; i++)
        {
            uint8_t aux[8];
            byte2bits(Sin[slot][i], aux);
            for (uint64_t j=0; j<8; j++)
            {
                bit[relRef2AbsRef(SinRef0 + (i*8 + j)*9, slot)] = aux[j];
            }
        }
    }
    TimerStart(KECCAK_SM_EXECUTOR_54);
    executor.execute(bit);
    TimerStopAndLog(KECCAK_SM_EXECUTOR_54);
    for (uint64_t slot=0; slot<Keccak_NumberOfSlots; slot++)
    {
        for (uint64_t i=0; i<32; i++)
        {
            uint8_t aux[8];
            for (uint64_t j=0; j<8; j++)
            {
                aux[j] = bit[relRef2AbsRef(SoutRef0 + (i*8 + j)*9, slot)];
            }

            bits2byte(aux, hashSout[slot][i]);
        }
        string aux;
        ba2string(aux, &hashSout[slot][0], 32);
        aux = "0x" + aux;
        //cout << "Sout" << slot << " = " << aux << endl;
        if (aux != hashString[slot])
        {
            cerr << "Error: slot=" << slot << " Sout:" << aux << " does not match hash:" << hashString[slot] << endl;
        }
        //printBa(&hashSout[slot][0], 32, "Sout"+to_string(slot));
        //cout << "Hash" << slot << " = " << hashString[slot] << endl;
    }

    free(bit);
}

void KeccakSMTest3 (KeccakFExecutor &executor)
{
    cout << "Starting 54x9 slots test..." << endl;
    KeccakFExecuteInput * pInput;
    pInput = new KeccakFExecuteInput();
    string hash[Keccak_NumberOfSlots][9];
    for (uint64_t slot=0; slot<Keccak_NumberOfSlots; slot++)
    {
        for (uint64_t row=0; row<9; row++)
        {
            // Fill 135 bytes with random data
            for (uint64_t i=0; i<1080; i++)
            {
                pInput->Sin[slot][row][i] = rand()%2;
            }

            // Last byte is for padding, i.e. 10000001
            pInput->Sin[slot][row][1080] = 1;
            pInput->Sin[slot][row][1087] = 1;
            uint8_t aux[135];
            for (uint64_t i=0; i<135; i++)
            {
                bits2byte(&(pInput->Sin[slot][row][i*8]), aux[i]);
            }
            hash[slot][row] = keccak256(aux, 135);
        }
    }
    KeccakFExecuteOutput * pOutput;
    pOutput = new KeccakFExecuteOutput();
    TimerStart(KECCAK_SM_EXECUTOR_54_9);
    executor.execute(*pInput, *pOutput);
    TimerStopAndLog(KECCAK_SM_EXECUTOR_54_9);

    for (uint64_t slot=0; slot<Keccak_NumberOfSlots; slot++)
    {
        for (uint64_t row=0; row<9; row++)
        {
            uint8_t aux[256];
            for (uint64_t i=0; i<256; i++)
            {
                if ( ( pOutput->pol[pin_a][relRef2AbsRef(SoutRef0+i*9, slot)] & (~Keccak_Mask) ) != 0 )
                {
                    cerr << "Error: output pin a is not normalized at slot=" << slot << " bit=" << i << endl;
                }
                if ( ( pOutput->pol[pin_a][relRef2AbsRef(SoutRef0+i*9, slot)] & (uint64_t(1)<<(row*7)) ) == 0)
                {
                    aux[i] = 0;
                }
                else
                {
                    aux[i] = 1;
                }
            }
            uint8_t aux2[32];
            for (uint64_t i=0; i<32; i++)
            {
                bits2byte(&aux[i*8], aux2[i]);
            }
            string aux3;
            ba2string(aux3, aux2, 32);
            aux3 = "0x" + aux3;
            if (aux3 != hash[slot][row])
            {
                cerr << "Error: slot=" << slot << " bit=" << row << " Sout=" << aux3 << " does not match hash=" << hash[slot][row] << endl;
            }
            //printBits(aux, 256, "slot" + to_string(slot) + "row" + to_string(row));
            //cout << "hash-" << slot << "-" << row << " = " << hash[slot][row] << endl;
        }
    }
    delete pInput;
    delete pOutput;
}

void KeccakSMTest4 (Goldilocks &fr, const Config &config, KeccakFExecutor &executor)
{    
    void * pAddress = malloc(CommitPols::pilSize());
    if (pAddress == NULL)
    {
        zklog.error("KeccakSMTest4() failed calling malloc() of size=" + to_string(CommitPols::pilSize()));
        exitProcess();
    }
    CommitPols cmPols(pAddress, CommitPols::pilDegree());

    uint64_t numberOfSlots = (cmPols.KeccakF.degree()-1)/158418;

    cout << "Starting FE " << numberOfSlots << "x9 slots test..." << endl;

    uint64_t inputLength = numberOfSlots*1600;
    uint64_t inputSize = inputLength*sizeof(uint64_t);
    Goldilocks::Element * pInput;
    pInput = (Goldilocks::Element *)malloc(inputSize);
    memset(pInput, 0, inputSize);

    string * pHash = new string[numberOfSlots*9];

    for (uint64_t slot=0; slot<numberOfSlots; slot++)
    {
        for (uint64_t row=0; row<9; row++)
        {
            uint8_t bits[1080];

            // Fill 135 bytes with random data
            for (uint64_t i=0; i<1080; i++)
            {
                bits[i] = (rand()%2);
                pInput[slot*1600 + i] = fr.fromU64( fr.toU64(pInput[slot*1600 + i]) | uint64_t(bits[i])<<(row*7) );
                /*if (slot==0 && row==1 && i<128)
                    cout << "i=" << i << " bit=" << uint64_t(bits[i]) << " input=" << pInput[slot*1600 + i] << endl;*/
            }

            // Last byte is for padding, i.e. 10000001
            pInput[slot*1600 + 1080] = fr.fromU64( fr.toU64(pInput[slot*1600 + 1080]) | Keccak_Mask );
            pInput[slot*1600 + 1087] = fr.fromU64( fr.toU64(pInput[slot*1600 + 1087]) | Keccak_Mask );

            // Get a byte array
            uint8_t bytes[135];
            for (uint64_t i=0; i<135; i++)
            {
                bits2byte(&(bits[i*8]), bytes[i]);
            }
            
            // Calculate and store the hash
            pHash[slot*9 + row] = keccak256(bytes, 135);
        }
    }

    // Call the Keccak SM executor
    TimerStart(KECCAK_SM_EXECUTOR_FE);
    executor.execute(pInput, inputLength, cmPols.KeccakF);
    TimerStopAndLog(KECCAK_SM_EXECUTOR_FE);

    for (uint64_t slot=0; slot<numberOfSlots; slot++)
    {
        for (uint64_t row=0; row<9; row++)
        {
            uint8_t aux[256];
            for (uint64_t i=0; i<256; i++)
            {
                if ( ( executor.getPol(cmPols.KeccakF.a, relRef2AbsRef(SoutRef0+i*44, slot)) & (~Keccak_Mask) ) != 0 )
                {
                    cerr << "Error: output pin a is not normalized at slot=" << slot << " bit=" << i << endl;
                }
                if ( ( executor.getPol(cmPols.KeccakF.a, relRef2AbsRef(SoutRef0+i*44, slot)) & (uint64_t(1)<<row) ) == 0)
                {
                    aux[i] = 0;
                }
                else
                {
                    aux[i] = 1;
                }
            }
            uint8_t aux2[32];
            for (uint64_t i=0; i<32; i++)
            {
                bits2byte(&aux[i*8], aux2[i]);
            }
            string aux3;
            ba2string(aux3, aux2, 32);
            aux3 = "0x" + aux3;
            if (aux3 != pHash[slot*9 + row])
            {
                cerr << "Error: slot=" << slot << " bit=" << row << " Sout=" << aux3 << " does not match hash=" << pHash[slot*9 + row] << endl;
                if (slot>1) break;
            }
            //printBits(aux, 256, "slot" + to_string(slot) + "row" + to_string(row));
            //cout << "hash-" << slot << "-" << row << " = " << hash[slot][row] << endl;
        }
    }

    free(pInput);
    delete[] pHash;
    free(pAddress);
}

uint64_t KeccakSMExecutorTest (Goldilocks &fr, const Config &config)
{
    cout << "KeccakSMExecutorTest() starting" << endl;

    KeccakFExecutor executor(fr, config);
    //KeccakSMTest1(executor);
    KeccakSMTest2(executor);
    KeccakSMTest3(executor);
    KeccakSMTest4(fr, config, executor);

    cout << "KeccakSMExecutorTest() done" << endl;
    return 0;
}
