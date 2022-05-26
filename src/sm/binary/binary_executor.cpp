
#include <nlohmann/json.hpp>
#include "binary_executor.hpp"
#include "binary_action_bytes.hpp"
#include "binary_defines.hpp"
#include "utils.hpp"
#include "scalar.hpp"

using json = nlohmann::json;

/*  =========
    FACTORS
    =========
    FACTOR0 => 0x1  0x100   0x10000 0x01000000  0x0  0x0    0x0     0x0         ... 0x0  0x0    0x0     0x0         0x1 0x100   0x10000 0x01000000  0x0  ...
    FACTOR1 => 0x0  0x0     0x0     0x0         0x1  0x100  0x10000 0x01000000  ... 0x0  0x0    0x0     0x0         0x0 0x0     0x0     0x0         0x0  ...     
    ...
    FACTOR7 => 0x0  0x0     0x0     0x0         0x0  0x0     0x0     0x0        ... 0x1  0x100  0x10000 0x01000000  0x0 0x0     0x0     0x0         0x0  ...
*/
void BinaryExecutor::buildFactors (void)
{
    // The REGISTERS_NUM is equal to the number of factors
    for (uint64_t i = 0; i < REGISTERS_NUM; i++)
    {
        vector<uint64_t> aux;
        FACTOR.push_back(aux);
        for (uint64_t j = 0; j < N; j += BYTES_PER_REGISTER)
        {
            for (uint64_t k = 0; k < BYTES_PER_REGISTER; k++)
            {
                uint64_t factor = (uint64_t(1)<<(8*k)) * (((j % (REGISTERS_NUM * BYTES_PER_REGISTER)) / BYTES_PER_REGISTER ) == i);
                FACTOR[i].push_back(factor);
            }
        }
    }
}

/*  =========
    RESET
    =========
    1 0 0 ... { REGISTERS_NUM * BYTES_PER_REGISTER } ... 0 1 0 ... { REGISTERS_NUM * BYTES_PER_REGISTER } 0
    1 0 0 ... { REGISTERS_NUM * BYTES_PER_REGISTER } ... 0 1 0 ... { REGISTERS_NUM * BYTES_PER_REGISTER } 0
    ...
    1 0 0 ... { REGISTERS_NUM * BYTES_PER_REGISTER } ... 0 1 0 ... { REGISTERS_NUM * BYTES_PER_REGISTER } 0
*/
void BinaryExecutor::buildReset (void)
{
    for (uint64_t i = 0; i < N; i++)
    {
        RESET.push_back((i % (REGISTERS_NUM * BYTES_PER_REGISTER)) == 0);
    }
}

void BinaryExecutor::execute (vector<BinaryAction> &action, BinaryCommitPols &pols)
{
    // Split actions into bytes
    vector<BinaryActionBytes> input;
    for (uint64_t i=0; i<action.size(); i++)
    {
        BinaryActionBytes actionBytes;
        scalar2bytes(action[i].a, actionBytes.a_bytes);
        scalar2bytes(action[i].b, actionBytes.b_bytes);
        scalar2bytes(action[i].c, actionBytes.c_bytes);
        actionBytes.opcode = action[i].opcode;
        input.push_back(actionBytes);
    }

    // Local array of N uint32
    uint32_t * c0Temp;
    c0Temp = (uint32_t *)malloc(N*sizeof(uint32_t));
    if (c0Temp == NULL)
    {
        cerr << "Error: BinaryExecutor::execute() failed calling malloc() for c0Temp" << endl;
        exit(-1);
    }
    memset(c0Temp, 0, N*sizeof(uint32_t));

    // Utility pointers
    uint32_t * a[8];
    a[0] = pols.a0;
    a[1] = pols.a1;
    a[2] = pols.a2;
    a[3] = pols.a3;
    a[4] = pols.a4;
    a[5] = pols.a5;
    a[6] = pols.a6;
    a[7] = pols.a7;

    uint32_t * b[8];
    b[0] = pols.b0;
    b[1] = pols.b1;
    b[2] = pols.b2;
    b[3] = pols.b3;
    b[4] = pols.b4;
    b[5] = pols.b5;
    b[6] = pols.b6;
    b[7] = pols.b7;

    uint32_t * c[8];
    c[0] = pols.c0;
    c[1] = pols.c1;
    c[2] = pols.c2;
    c[3] = pols.c3;
    c[4] = pols.c4;
    c[5] = pols.c5;
    c[6] = pols.c6;
    c[7] = pols.c7;

    // Process all the inputs
    for (uint64_t i = 0; i < input.size(); i++)
    {
#ifdef LOG_BINARY_EXECUTOR
        if (i%10000 == 0)
        {
            cout << "Computing binary pols " << i << "/" << input.size() << endl;
        }
#endif

        for (uint64_t j = 0; j < LATCH_SIZE; j++)
        {
            // TODO: Ask Jordi/Edu how to deal with BigInt() assignments to pols

            pols.opcode[i*LATCH_SIZE + j] = input[i].opcode;
            pols.freeInA[i*LATCH_SIZE + j] = input[i].a_bytes[j];
            pols.freeInB[i*LATCH_SIZE + j] = input[i].b_bytes[j];
            pols.freeInC[i*LATCH_SIZE + j] = input[i].c_bytes[j];

            if (j == LATCH_SIZE - 1)
            {
                pols.last[i*LATCH_SIZE + j] = 1;
            }
            else
            {
                pols.last[i*LATCH_SIZE + j] = 0;
            }

            uint64_t cout;
            switch (input[i].opcode)
            {
                // ADD   (OPCODE = 0)
                case 0:
                {
                    uint64_t sum = input[i].a_bytes[j] + input[i].b_bytes[j];
                    pols.cOut[i*LATCH_SIZE + j] = sum>>8;
                    break;
                }
                // SUB   (OPCODE = 1)
                case 1:
                {
                    if (input[i].a_bytes[j] - pols.cIn[i*LATCH_SIZE + j] >= input[i].b_bytes[j])
                    {
                        pols.cOut[i*LATCH_SIZE + j] = 0;
                    }
                    else
                    {
                        pols.cOut[i*LATCH_SIZE + j] = 1;
                    }
                    break;
                }
                // LT    (OPCODE = 2)
                case 2:
                {

                    if ( (input[i].a_bytes[j] <= input[i].b_bytes[j]) && pols.cIn[i*LATCH_SIZE + j] )
                    {
                        cout = 1;
                    }
                    else
                    {
                        cout = 0;
                    }

                    pols.cOut[i*LATCH_SIZE + j] = cout;
                    if (pols.last[i*LATCH_SIZE + j] == 1)
                    {
                        pols.freeInC[i*LATCH_SIZE + j] = cout;
                        pols.useCarry[i*LATCH_SIZE + j] = 1;
                    }
                    else
                    {
                        pols.freeInC[i*LATCH_SIZE + j] = 0;
                        pols.useCarry[i*LATCH_SIZE + j] = 0;
                    }
                    break;
                }
                // SLT    (OPCODE = 3)
                case 3:
                {
                    if (RESET[i*LATCH_SIZE + j])
                    {
                        pols.cIn[i*LATCH_SIZE + j] = 1;
                    }

                    pols.last[i*LATCH_SIZE + j] ? pols.useCarry[i*LATCH_SIZE + j] = 1 : pols.useCarry[i*LATCH_SIZE + j] = 0;

                    if (pols.last[i*LATCH_SIZE + j])
                    {
                        uint8_t sig_a = input[i].a_bytes[j] >> 7;
                        uint8_t sig_b = input[i].b_bytes[j] >> 7;
                        // A Negative ; B Positive
                        if (sig_a > sig_b)
                        {
                            cout = 1;
                        }
                        // A Positive ; B Negative
                        else if (sig_a < sig_b)
                        {
                            cout = 0;
                        }
                        // A and B equals
                        else
                        {
                            if (sig_a == 1)
                            {
                                if ( (input[i].a_bytes[j] > input[i].b_bytes[j]) && pols.cIn[i*LATCH_SIZE + j] )
                                {
                                    cout = 1;
                                }
                                else
                                {
                                    cout = 0;
                                }
                            }
                            else
                            {
                                if ( (input[i].a_bytes[j] < input[i].b_bytes[j]) && pols.cIn[i*LATCH_SIZE + j] )
                                {
                                    cout = 1;
                                }
                                else
                                {
                                    cout = 0;
                                }
                            }
                        }
                        pols.freeInC[i*LATCH_SIZE + j] = cout;
                    }
                    else
                    {
                        if ( (input[i].a_bytes[j] <= input[i].b_bytes[j]) && pols.cIn[i*LATCH_SIZE + j] )
                        {
                            cout = 1;
                        }
                        else
                        {
                            cout = 0;
                        }
                        pols.freeInC[i*LATCH_SIZE + j] = 0;
                    }
                    pols.cOut[i*LATCH_SIZE + j] = cout;
                    break;
                }
                // EQ    (OPCODE = 4)
                case 4:
                {
                    if (RESET[i*LATCH_SIZE + j])
                    {
                        pols.cIn[i*LATCH_SIZE + j] = 1;
                    }

                    if ( (input[i].a_bytes[j] == input[i].b_bytes[j]) && (pols.cIn[i*LATCH_SIZE + j] == 1) )
                    {
                        cout = 1;
                    }
                    else
                    {
                        cout = 0;
                    }
                    pols.cOut[i*LATCH_SIZE + j] = cout;
                    if (pols.last[i*LATCH_SIZE + j] == 1)
                    {
                        pols.freeInC[i*LATCH_SIZE + j] = cout;
                        pols.useCarry[i*LATCH_SIZE + j] = 1;
                    }
                    else
                    {
                        pols.freeInC[i*LATCH_SIZE + j] = 0;
                        pols.useCarry[i*LATCH_SIZE + j] = 0;
                    }
                    break;
                }
                default:
                {
                    pols.cIn[i*LATCH_SIZE + j] = 0;
                    pols.cOut[i*LATCH_SIZE + j] = 0;
                    break;
                }
            }
            // We can set the cIn and the LCin when RESET =1
            if (RESET[(i*LATCH_SIZE + j + 1) % N])
            {
                pols.cIn[(i*LATCH_SIZE + j + 1) % N] = 0;
            }
            else
            {
                pols.cIn[(i*LATCH_SIZE + j + 1) % N] = pols.cOut[i*LATCH_SIZE + j];
            }
            pols.lCout[(i*LATCH_SIZE + j + 1) % N] = pols.cOut[i*LATCH_SIZE + j];
            pols.lOpcode[(i*LATCH_SIZE + j + 1) % N] = pols.opcode[i*LATCH_SIZE + j];

            pols.a0[(i*LATCH_SIZE + j + 1) % N] = pols.a0[(i*LATCH_SIZE + j) % N] * (1 - RESET[(i*LATCH_SIZE + j) % N]) + pols.freeInA[(i*LATCH_SIZE + j) % N] * FACTOR[0][(i*LATCH_SIZE + j) % N];
            pols.b0[(i*LATCH_SIZE + j + 1) % N] = pols.b0[(i*LATCH_SIZE + j) % N] * (1 - RESET[(i*LATCH_SIZE + j) % N]) + pols.freeInB[(i*LATCH_SIZE + j) % N] * FACTOR[0][(i*LATCH_SIZE + j) % N];

            c0Temp[(i*LATCH_SIZE + j) % N] = pols.c0[(i*LATCH_SIZE + j) % N] * (1 - RESET[(i*LATCH_SIZE + j) % N]) + pols.freeInC[(i*LATCH_SIZE + j) % N] * FACTOR[0][(i*LATCH_SIZE + j) % N];
            pols.c0[(i*LATCH_SIZE + j + 1) % N] = pols.useCarry[(i*LATCH_SIZE + j) % N] * (pols.cOut[(i*LATCH_SIZE + j) % N] - c0Temp[(i*LATCH_SIZE + j) % N]) + c0Temp[(i*LATCH_SIZE + j) % N];

            //if ((i*LATCH_SIZE + j) % 10000 === 0) console.log(`Computing final binary pols ${(i * LATCH_SIZE + j)}/${N}`);

            for (uint64_t k = 1; k < REGISTERS_NUM; k++)
            {
                a[k][(i*LATCH_SIZE + j + 1) % N] = a[k][(i*LATCH_SIZE + j) % N] * (1 - RESET[(i*LATCH_SIZE + j) % N]) + pols.freeInA[(i*LATCH_SIZE + j) % N] * FACTOR[k][(i*LATCH_SIZE + j) % N];
                b[k][(i*LATCH_SIZE + j + 1) % N] = b[k][(i*LATCH_SIZE + j) % N] * (1 - RESET[(i*LATCH_SIZE + j) % N]) + pols.freeInB[(i*LATCH_SIZE + j) % N] * FACTOR[k][(i*LATCH_SIZE + j) % N];
                if (pols.last[i*LATCH_SIZE + j] && pols.useCarry[i*LATCH_SIZE + j])
                {
                    c[k][(i*LATCH_SIZE + j + 1) % N] = 0;
                }
                else
                {
                    c[k][(i*LATCH_SIZE + j + 1) % N] = c[k][(i*LATCH_SIZE + j) % N] * (1 - RESET[(i*LATCH_SIZE + j) % N]) + pols.freeInC[(i*LATCH_SIZE + j) % N] * FACTOR[k][(i*LATCH_SIZE + j) % N];
                }
            }
        }
    }

    for (uint64_t i = input.size()*LATCH_SIZE; i < N; i++)
    {
        //if (i % 10000 === 0) console.log(`Computing final binary pols ${i}/${N}`);
        pols.a0[(i + 1) % N] = pols.a0[i] * (1 - RESET[i]) + pols.freeInA[i] * FACTOR[0][i];
        pols.b0[(i + 1) % N] = pols.b0[i] * (1 - RESET[i]) + pols.freeInB[i] * FACTOR[0][i];

        c0Temp[i] = pols.c0[i] * (1 - RESET[i]) + pols.freeInC[i] * FACTOR[0][i];
        pols.c0[(i + 1) % N] = pols.useCarry[i] * (pols.cOut[i] - c0Temp[i]) + c0Temp[i];

        for (uint64_t j = 1; j < REGISTERS_NUM; j++)
        {
            a[j][(i + 1) % N] = a[j][i] * (1 - RESET[i]) + pols.freeInA[i] * FACTOR[j][i];
            b[j][(i + 1) % N] = b[j][i] * (1 - RESET[i]) + pols.freeInB[i] * FACTOR[j][i];
            c[j][(i + 1) % N] = c[j][i] * (1 - RESET[i]) + pols.freeInC[i] * FACTOR[j][i];
        }
    }

    cout << "BinaryExecutor successfully processed " << action.size() << " binary actions" << endl;
}

// To be used only for testing, since it allocates a lot of memory
void BinaryExecutor::execute (vector<BinaryAction> &action)
{
    void * pAddress = mapFile(config.cmPolsFile, CommitPols::size(), true);
    CommitPols cmPols(pAddress);
    execute(action, cmPols.Binary);
    unmapFile(pAddress, CommitPols::size());
}