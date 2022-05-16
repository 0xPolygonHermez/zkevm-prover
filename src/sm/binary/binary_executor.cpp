
#include <nlohmann/json.hpp>
#include "binary_executor.hpp"
#include "binary_pols.hpp"
#include "binary_action_bytes.hpp"
#include "binary_defines.hpp"
#include "utils.hpp"
#include "scalar.hpp"

using json = nlohmann::json;

void BinaryExecutor::execute (vector<BinaryAction> &action)
{
    // Allocate polynomials
    BinaryPols pols(config);
    pols.alloc(polSize, pilJson);

    // Split actions into bytes
    vector<BinaryActionBytes> input;
    for (uint64_t i=0; i<action.size(); i++)
    {
        uint64_t dataSize;
        BinaryActionBytes actionBytes;
        dataSize = 32;
        scalar2ba(actionBytes.a_bytes, dataSize, action[i].a);
        dataSize = 32;
        scalar2ba(actionBytes.b_bytes, dataSize, action[i].b);
        dataSize = 32;
        scalar2ba(actionBytes.c_bytes, dataSize, action[i].c);
        actionBytes.opcode = action[i].opcode;
        input.push_back(actionBytes);
    }

    uint64_t N = polSize;

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

            pols.opcode[i*LATCH_SIZE + 1 + j] = input[i].opcode;
            pols.freeInA[i*LATCH_SIZE + 1 + j] = input[i].a_bytes[j];
            pols.freeInB[i*LATCH_SIZE + 1 + j] = input[i].b_bytes[j];
            pols.freeInC[i*LATCH_SIZE + 1 + j] = input[i].c_bytes[j];

            if (j == LATCH_SIZE - 1)
            {
                pols.last[i*LATCH_SIZE + 1 + j] = 1;
            }
            else
            {
                pols.last[i*LATCH_SIZE + 1 + j] = 0;
            }

            // We can set the cIn and the LCin when RESET =1
            if (constPols.RESET[i*LATCH_SIZE + j])
            {
                pols.cIn[i*LATCH_SIZE + 1 + j] = 0;
            }
            else
            {
                pols.cIn[i*LATCH_SIZE + 1 + j] = pols.cOut[i*LATCH_SIZE + j];
            }

            uint64_t cout;
            switch (input[i].opcode)
            {
                // ADD   (OPCODE = 1)
                case 1:
                {
                    uint64_t sum = input[i].a_bytes[j] + input[i].b_bytes[j];
                    pols.cOut[i*LATCH_SIZE + 1 + j] = sum>>8;
                    break;
                }
                // SUB   (OPCODE = 2)
                case 2:
                {
                    if (input[i].a_bytes[j] - pols.cIn[i*LATCH_SIZE + 1 + j] >= input[i].b_bytes[j])
                    {
                        pols.cOut[i*LATCH_SIZE + 1 + j] = 0;
                    }
                    else
                    {
                        pols.cOut[i*LATCH_SIZE + 1 + j] = 1;
                    }
                    break;
                }
                // LT    (OPCODE = 3)
                case 3:
                {
                    pols.useCarry[i*LATCH_SIZE + 1 + j] = 1;

                    if (input[i].a_bytes[j] < input[i].b_bytes[j])
                    {
                        cout = 1;
                    }
                    else if (input[i].a_bytes[j] == input[i].b_bytes[j])
                    {
                        cout = pols.cIn[i*LATCH_SIZE + 1 + j];
                    }
                    else
                    {
                        cout = 0;
                    }

                    pols.cOut[i*LATCH_SIZE + 1 + j] = cout;
                    if (pols.last[i*LATCH_SIZE + 1 + j] == 1)
                    {
                        pols.freeInC[i*LATCH_SIZE + 1 + j] = input[i].c_bytes[0];
                    }
                    else
                    {
                        pols.freeInC[i*LATCH_SIZE + 1 + j] = pols.cOut[i*LATCH_SIZE + 1 + j];
                    }
                    break;
                }
                // GT    (OPCODE = 4)
                case 4:
                {
                    pols.useCarry[i*LATCH_SIZE + 1 + j] = 1;

                    if (input[i].a_bytes[j] > input[i].b_bytes[j])
                    {
                        cout = 1;
                    }
                    else if (input[i].a_bytes[j] == input[i].b_bytes[j])
                    {
                        cout = pols.cIn[i*LATCH_SIZE + 1 + j];
                    }
                    else
                    {
                        cout = 0;
                    }
                    pols.cOut[i*LATCH_SIZE + 1 + j] = cout;
                    if (pols.last[i*LATCH_SIZE + 1 + j] == 1)
                    {
                        pols.freeInC[i*LATCH_SIZE + 1 + j] = input[i].c_bytes[0];
                    }
                    else
                    {
                        pols.freeInC[i*LATCH_SIZE + 1 + j] = pols.cOut[i*LATCH_SIZE + 1 + j];
                    }
                    break;
                }
                // SLT    (OPCODE = 5)
                case 5:
                {
                    pols.useCarry[i*LATCH_SIZE + 1 + j] = 1;

                    if (pols.last[i*LATCH_SIZE + 1 + j])
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
                                if (input[i].a_bytes[j] > input[i].b_bytes[j])
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
                                if (input[i].a_bytes[j] < input[i].b_bytes[j])
                                {
                                    cout = 1;
                                }
                                else
                                {
                                    cout = 0;
                                }
                            }
                        }
                    }
                    else
                    {
                        if (input[i].a_bytes[j] < input[i].b_bytes[j])
                        {
                            cout = 1;
                        }
                        else if (input[i].a_bytes[j] == input[i].b_bytes[j])
                        {
                            cout = pols.cIn[i*LATCH_SIZE + 1 + j];
                        }
                        else
                        {
                            cout = 0;
                        }
                    }
                    pols.cOut[i*LATCH_SIZE + 1 + j] = cout;
                    if (pols.last[i*LATCH_SIZE + 1 + j] == 1)
                    {
                        pols.freeInC[i*LATCH_SIZE + 1 + j] = input[i].c_bytes[0];
                    }
                    else
                    {
                        pols.freeInC[i*LATCH_SIZE + 1 + j] = pols.cOut[i*LATCH_SIZE + 1 + j];
                    }
                    break;
                }
                // SGT    (OPCODE = 6)
                case 6:
                {
                    pols.useCarry[i*LATCH_SIZE + 1 + j] = 1;

                    if (pols.last[i*LATCH_SIZE + 1 + j])
                    {
                        uint8_t sig_a = input[i].a_bytes[j] >> 7;
                        uint8_t sig_b = input[i].b_bytes[j] >> 7;

                        // A Negative ; B Positive
                        if (sig_a > sig_b)
                        {
                            cout = 0;
                        }
                        // A Positive ; B Negative
                        else if (sig_a < sig_b)
                        {
                            cout = 1;
                        }
                        // A and B equals
                        else
                        {
                            if (sig_a == 1)
                            {
                                if (input[i].a_bytes[j] < input[i].b_bytes[j])
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
                                if (input[i].a_bytes[j] > input[i].b_bytes[j])
                                {
                                    cout = 1;
                                }
                                else
                                {
                                    cout = 0;
                                }
                            }
                        }
                    }
                    else
                    {
                        if (input[i].a_bytes[j] > input[i].b_bytes[j])
                        {
                            cout = 1;
                        }
                        else if (input[i].a_bytes[j] == input[i].b_bytes[j])
                        {
                            cout = pols.cIn[i*LATCH_SIZE + 1 + j];
                        }
                        else
                        {
                            cout = 0;
                        }
                    }
                    pols.cOut[i*LATCH_SIZE + 1 + j] = cout;
                    if (pols.last[i*LATCH_SIZE + 1 + j] == 1)
                    {
                        pols.freeInC[i*LATCH_SIZE + 1 + j] = input[i].c_bytes[0];
                    }
                    else
                    {
                        pols.freeInC[i*LATCH_SIZE + 1 + j] = pols.cOut[i*LATCH_SIZE + 1 + j];
                    }
                    break;
                }
                // EQ    (OPCODE = 7)
                case 7:
                {
                    if (constPols.RESET[i*LATCH_SIZE + j])
                    {
                        pols.cIn[i*LATCH_SIZE + 1 + j] = 1;
                    }
                    pols.useCarry[i*LATCH_SIZE + 1 + j] = 1;

                    if ( (input[i].a_bytes[j] == input[i].b_bytes[j]) && (pols.cIn[i*LATCH_SIZE + 1 + j] == 1) )
                    {
                        cout = 1;
                    }
                    else
                    {
                        cout = 0;
                    }
                    pols.cOut[i*LATCH_SIZE + 1 + j] = cout;
                    if (pols.last[i*LATCH_SIZE + 1 + j] == 1)
                    {
                        pols.freeInC[i*LATCH_SIZE + 1 + j] = input[i].c_bytes[0];
                    }
                    else
                    {
                        pols.freeInC[i*LATCH_SIZE + 1 + j] = pols.cOut[i*LATCH_SIZE + 1 + j];
                    }
                    break;
                }
                // ISZERO (OPCODE = 8)
                case 8:
                {
                    if (constPols.RESET[i*LATCH_SIZE + j])
                    {
                        pols.cIn[i*LATCH_SIZE + 1 + j] = 1;
                    }
                    pols.useCarry[i*LATCH_SIZE + 1 + j] = 1;
                    if ( (input[i].a_bytes[j] == 0) && (pols.cIn[i*LATCH_SIZE + 1 + j] == 1) )
                    {
                        cout = 1;
                    }
                    else
                    {
                        cout = 0;
                    }
                    pols.cOut[i*LATCH_SIZE + 1 + j] = cout;
                    if (pols.last[i*LATCH_SIZE + 1 + j] == 1)
                    {
                        pols.freeInC[i*LATCH_SIZE + 1 + j] = input[i].c_bytes[0];
                    }
                    else
                    {
                        pols.freeInC[i*LATCH_SIZE + 1 + j] = pols.cOut[i*LATCH_SIZE + 1 + j];
                    }
                    break;
                }
                default:
                {
                    pols.cIn[i*LATCH_SIZE + 1 + j] = 0;
                    pols.cOut[i*LATCH_SIZE + 1 + j] = 0;
                    break;
                }
            }
        }
    }

    // Generate registers
    for (uint64_t i=0; i<N; i++)
    {
#ifdef LOG_BINARY_EXECUTOR
        if (i%10000 == 0)
        {
            cout << "Computing final binary pols " << i << "/" << N << endl;
        }
#endif

        pols.a0[(i + 1) % N] = pols.a0[i] * (1 - constPols.RESET[i]) + pols.freeInA[i] * constPols.FACTOR[0][i];
        pols.b0[(i + 1) % N] = pols.b0[i] * (1 - constPols.RESET[i]) + pols.freeInB[i] * constPols.FACTOR[0][i];

        pols.c0Temp[(i + 1) % N] = pols.c0Temp[i] * (1 - constPols.RESET[i]) + pols.freeInC[i] * constPols.FACTOR[0][i];
        pols.c0[(i + 1) % N] = pols.useCarry[i] * (pols.cOut[i] - pols.c0Temp[i]) + pols.c0Temp[i];

        pols.a1[(i + 1) % N] = pols.a1[i] * (1 - constPols.RESET[i]) + pols.freeInA[i] * constPols.FACTOR[1][i];
        pols.b1[(i + 1) % N] = pols.b1[i] * (1 - constPols.RESET[i]) + pols.freeInB[i] * constPols.FACTOR[1][i];
        pols.c1[(i + 1) % N] = pols.c1[i] * (1 - constPols.RESET[i]) + pols.freeInC[i] * constPols.FACTOR[1][i];

        pols.a2[(i + 1) % N] = pols.a2[i] * (1 - constPols.RESET[i]) + pols.freeInA[i] * constPols.FACTOR[2][i];
        pols.b2[(i + 1) % N] = pols.b2[i] * (1 - constPols.RESET[i]) + pols.freeInB[i] * constPols.FACTOR[2][i];
        pols.c2[(i + 1) % N] = pols.c2[i] * (1 - constPols.RESET[i]) + pols.freeInC[i] * constPols.FACTOR[2][i];

        pols.a3[(i + 1) % N] = pols.a3[i] * (1 - constPols.RESET[i]) + pols.freeInA[i] * constPols.FACTOR[3][i];
        pols.b3[(i + 1) % N] = pols.b3[i] * (1 - constPols.RESET[i]) + pols.freeInB[i] * constPols.FACTOR[3][i];
        pols.c3[(i + 1) % N] = pols.c3[i] * (1 - constPols.RESET[i]) + pols.freeInC[i] * constPols.FACTOR[3][i];

        pols.a4[(i + 1) % N] = pols.a4[i] * (1 - constPols.RESET[i]) + pols.freeInA[i] * constPols.FACTOR[4][i];
        pols.b4[(i + 1) % N] = pols.b4[i] * (1 - constPols.RESET[i]) + pols.freeInB[i] * constPols.FACTOR[4][i];
        pols.c4[(i + 1) % N] = pols.c4[i] * (1 - constPols.RESET[i]) + pols.freeInC[i] * constPols.FACTOR[4][i];

        pols.a5[(i + 1) % N] = pols.a5[i] * (1 - constPols.RESET[i]) + pols.freeInA[i] * constPols.FACTOR[5][i];
        pols.b5[(i + 1) % N] = pols.b5[i] * (1 - constPols.RESET[i]) + pols.freeInB[i] * constPols.FACTOR[5][i];
        pols.c5[(i + 1) % N] = pols.c5[i] * (1 - constPols.RESET[i]) + pols.freeInC[i] * constPols.FACTOR[5][i];

        pols.a6[(i + 1) % N] = pols.a6[i] * (1 - constPols.RESET[i]) + pols.freeInA[i] * constPols.FACTOR[6][i];
        pols.b6[(i + 1) % N] = pols.b6[i] * (1 - constPols.RESET[i]) + pols.freeInB[i] * constPols.FACTOR[6][i];
        pols.c6[(i + 1) % N] = pols.c6[i] * (1 - constPols.RESET[i]) + pols.freeInC[i] * constPols.FACTOR[6][i];

        pols.a7[(i + 1) % N] = pols.a7[i] * (1 - constPols.RESET[i]) + pols.freeInA[i] * constPols.FACTOR[7][i];
        pols.b7[(i + 1) % N] = pols.b7[i] * (1 - constPols.RESET[i]) + pols.freeInB[i] * constPols.FACTOR[7][i];
        pols.c7[(i + 1) % N] = pols.c7[i] * (1 - constPols.RESET[i]) + pols.freeInC[i] * constPols.FACTOR[7][i];
    }
}