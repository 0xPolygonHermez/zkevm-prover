#include "keccak_f_executor.hpp"
#include "keccak_executor_test.hpp"
#include "timer.hpp"

void KeccakTest4(Goldilocks &fr, const Config &config, KeccakFExecutor &executor)
{
    void *pAddress = malloc(CommitPols::pilSize());
    if (pAddress == NULL)
    {
        zklog.error("KeccakTest4() failed calling malloc() of size=" + to_string(CommitPols::pilSize()));
        exitProcess();
    }
    CommitPols cmPols(pAddress, CommitPols::pilDegree());

    const uint64_t numberOfSlots = ((KeccakGateConfig.polLength - 1) / KeccakGateConfig.slotSize);
    const uint64_t keccakMask = 0xFFFFFFFFFFF;

    cout << "Starting FE " << numberOfSlots << "x9 slots test..." << endl;

    const uint64_t individualInputLength = 1600;
    const uint64_t inputLength = numberOfSlots * individualInputLength;
    const uint64_t inputSize = inputLength * sizeof(uint64_t);
    const uint64_t randomByteCount = 135;
    const uint64_t randomBitCount = randomByteCount * 8;
    const uint64_t hashCount = 9;

    Goldilocks::Element *pInput;
    pInput = (Goldilocks::Element *)malloc(inputSize);
    memset(pInput, 0, inputSize);

    string *pHash = new string[numberOfSlots * hashCount];

    for (uint64_t slot = 0; slot < numberOfSlots; slot++)
    {
        for (uint64_t row = 0; row < hashCount; row++)
        {
            uint8_t bits[randomBitCount];

            // Fill randomByteCount bytes with random data
            for (uint64_t i = 0; i < randomBitCount; i++)
            {
                bits[i] = (rand() % 2);
                pInput[slot * individualInputLength + i] = fr.fromU64(fr.toU64(pInput[slot * individualInputLength + i]) | uint64_t(bits[i]) << (row * 7));
                /*if (slot==0 && row==1 && i<128)
                    cout << "i=" << i << " bit=" << uint64_t(bits[i]) << " input=" << pInput[slot*individualInputLength + i] << endl;*/
            }

            // Last byte is for padding, i.e. 10000001
            pInput[slot * individualInputLength + randomBitCount] = fr.fromU64(fr.toU64(pInput[slot * individualInputLength + randomBitCount]) | keccakMask);
            pInput[slot * individualInputLength + randomBitCount + 7] = fr.fromU64(fr.toU64(pInput[slot * individualInputLength + randomBitCount + 7]) | keccakMask);

            // Get a byte array
            uint8_t bytes[randomByteCount];
            for (uint64_t i = 0; i < randomByteCount; i++)
            {
                bits2byte(&(bits[i * 8]), bytes[i]);
            }

            // Calculate and store the hash
            pHash[slot * hashCount + row] = keccak256(bytes, randomByteCount);
        }
    }

    // Call the Keccak SM executor
    TimerStart(KECCAK_SM_EXECUTOR_FE);
    executor.execute(pInput, inputLength, cmPols.KeccakF);
    TimerStopAndLog(KECCAK_SM_EXECUTOR_FE);

    for (uint64_t slot = 0; slot < numberOfSlots; slot++)
    {
        for (uint64_t row = 0; row < hashCount; row++)
        {
            uint8_t aux[256];
            for (uint64_t i = 0; i < 256; i++)
            {
                if ((executor.getPol(cmPols.KeccakF.a, KeccakGateConfig.relRef2AbsRef(KeccakGateConfig.soutRef0 + i * 44, slot)) & (~keccakMask)) != 0)
                {
                    cerr << "Error: output pin a is not normalized at slot=" << slot << " bit=" << i << endl;
                }
                if ((executor.getPol(cmPols.KeccakF.a, KeccakGateConfig.relRef2AbsRef(KeccakGateConfig.soutRef0 + i * 44, slot)) & (uint64_t(1) << row)) == 0)
                {
                    aux[i] = 0;
                }
                else
                {
                    aux[i] = 1;
                }
            }
            uint8_t aux2[32];
            for (uint64_t i = 0; i < 32; i++)
            {
                bits2byte(&aux[i * 8], aux2[i]);
            }
            string aux3;
            ba2string(aux3, aux2, 32);
            aux3 = "0x" + aux3;
            if (aux3 != pHash[slot * hashCount + row])
            {
                cerr << "Error: slot=" << slot << " bit=" << row << " Sout=" << aux3 << " does not match hash=" << pHash[slot * 9 + row] << endl;
                if (slot > 1)
                    break;
            }
            // printBits(aux, 256, "slot" + to_string(slot) + "row" + to_string(row));
            // cout << "hash-" << slot << "-" << row << " = " << hash[slot][row] << endl;
        }
    }

    free(pInput);
    delete[] pHash;
    free(pAddress);
}

uint64_t KeccakSMExecutorTest(Goldilocks &fr, const Config &config)
{
    cout << "KeccakSMExecutorTest() starting" << endl;

    KeccakFExecutor executor(fr, config);
    KeccakTest4(fr, config, executor);

    cout << "KeccakSMExecutorTest() done" << endl;
    return 0;
}
