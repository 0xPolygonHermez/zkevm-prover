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
    // Check that we have enough room in polynomials  TODO: Do this check in JS
    if (action.size()*LATCH_SIZE > N)
    {
        cerr << "Error: Too many Binary entries" << endl;
        exitProcess();
    }

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
        exitProcess();
    }
    memset(c0Temp, 0, N*sizeof(uint32_t));

    // Utility pointers
    CommitPol a[8] = { pols.a0, pols.a1, pols.a2, pols.a3, pols.a4, pols.a5, pols.a6, pols.a7 };

    CommitPol b[8] = { pols.b0, pols.b1, pols.b2, pols.b3, pols.b4, pols.b5, pols.b6, pols.b7 };

    CommitPol c[8] = { pols.c0, pols.c1, pols.c2, pols.c3, pols.c4, pols.c5, pols.c6, pols.c7 };

    // Process all the inputs
//#pragma omp parallel for // TODO: Disabled since OMP decreases performance, probably due to cache invalidations
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

            pols.opcode[i*LATCH_SIZE + j] = fr.fromU64(input[i].opcode);
            pols.freeInA[i*LATCH_SIZE + j] = fr.fromU64(input[i].a_bytes[j]);
            pols.freeInB[i*LATCH_SIZE + j] = fr.fromU64(input[i].b_bytes[j]);
            pols.freeInC[i*LATCH_SIZE + j] = fr.fromU64(input[i].c_bytes[j]);

            if (j == LATCH_SIZE - 1)
            {
                pols.last[i*LATCH_SIZE + j] = fr.one();
            }
            else
            {
                pols.last[i*LATCH_SIZE + j] = fr.zero(); // TODO: Should we comment this out?
            }

            uint64_t cout;
            switch (input[i].opcode)
            {
                // ADD   (OPCODE = 0)
                case 0:
                {
                    uint64_t sum = input[i].a_bytes[j] + input[i].b_bytes[j] + fr.toU64(pols.cIn[i*LATCH_SIZE + j]);
                    pols.cOut[i*LATCH_SIZE + j] = fr.fromU64(sum>>8);
                    break;
                }
                // SUB   (OPCODE = 1)
                case 1:
                {
                    if (input[i].a_bytes[j] - fr.toU64(pols.cIn[i*LATCH_SIZE + j]) >= input[i].b_bytes[j])
                    {
                        pols.cOut[i*LATCH_SIZE + j] = fr.zero(); // TODO: Comment out?
                    }
                    else
                    {
                        pols.cOut[i*LATCH_SIZE + j] = fr.one();
                    }
                    break;
                }
                // LT    (OPCODE = 2)
                case 2:
                {
                    if (RESET[i*LATCH_SIZE + j])
                    {
                        pols.freeInC[i*LATCH_SIZE + j] = fr.fromU64(input[i].c_bytes[LATCH_SIZE - 1]); // Only change the freeInC when reset or Last
                    }
                    if ( input[i].a_bytes[j] < input[i].b_bytes[j] )
                    {
                        cout = 1;
                    }
                    else if ( input[i].a_bytes[j] == input[i].b_bytes[j] )
                    {
                        cout = fr.toU64(pols.cIn[i*LATCH_SIZE + j]);
                    }
                    else
                    {
                        cout = 0;
                    }

                    pols.cOut[i*LATCH_SIZE + j] = fr.fromU64(cout);
                    if (fr.isOne(pols.last[i*LATCH_SIZE + j]))
                    {
                        pols.freeInC[i*LATCH_SIZE + j] = fr.fromU64(input[i].c_bytes[0]);
                        pols.useCarry[i*LATCH_SIZE + j] = fr.one();
                    }
                    else
                    {
                        pols.useCarry[i*LATCH_SIZE + j] = fr.zero(); // TODO: Comment out?
                    }
                    break;
                }
                // SLT    (OPCODE = 3)
                case 3:
                {
                    (!fr.isZero(pols.last[i*LATCH_SIZE + j])) ? pols.useCarry[i*LATCH_SIZE + j] = fr.one() : pols.useCarry[i*LATCH_SIZE + j] = fr.zero(); // TODO: Comment out?
                    if (RESET[i*LATCH_SIZE + j])
                    {
                        pols.freeInC[i*LATCH_SIZE + j] = fr.fromU64(input[i].c_bytes[LATCH_SIZE - 1]); // TODO: Comment out? Only change the freeInC when reset or Last
                    }

                    if (!fr.isZero(pols.last[i*LATCH_SIZE + j]))
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
                            if ( input[i].a_bytes[j] < input[i].b_bytes[j] )
                            {
                                cout = 1;
                            }
                            else if ( input[i].a_bytes[j] == input[i].b_bytes[j] )
                            {
                                cout = fr.toU64(pols.cIn[i*LATCH_SIZE + j]);
                            }
                            else
                            {
                                cout = 0;
                            }
                        }
                        pols.freeInC[i*LATCH_SIZE + j] = fr.fromU64(input[i].c_bytes[0]); // Only change the freeInC when reset or Lastcout;
                    }
                    else
                    {
                        if ( input[i].a_bytes[j] <= input[i].b_bytes[j] )
                        {
                            cout = 1;
                        }
                        else if ( input[i].a_bytes[j] == input[i].b_bytes[j] )
                        {
                            cout = fr.toU64(pols.cIn[i*LATCH_SIZE + j]);
                        }
                        else
                        {
                            cout = 0;
                        }
                    }
                    pols.cOut[i*LATCH_SIZE + j] = fr.fromU64(cout);
                    break;
                }
                // EQ    (OPCODE = 4)
                case 4:
                {
                    if (RESET[i*LATCH_SIZE + j])
                    {
                        pols.cIn[i*LATCH_SIZE + j] = fr.one();
                        pols.freeInC[i*LATCH_SIZE + j] = fr.fromU64(input[i].c_bytes[LATCH_SIZE - 1]); // TODO: Comment out? Only change the freeInC when reset or Last
                    }

                    if ( (input[i].a_bytes[j] == input[i].b_bytes[j]) && fr.isOne(pols.cIn[i*LATCH_SIZE + j]) )
                    {
                        cout = 1;
                    }
                    else
                    {
                        cout = 0;
                    }
                    pols.cOut[i*LATCH_SIZE + j] = fr.fromU64(cout);

                    if (fr.isOne(pols.last[i*LATCH_SIZE + j]))
                    {
                        pols.freeInC[i*LATCH_SIZE + j] = fr.fromU64(input[i].c_bytes[0]); // Only change the freeInC when reset or Last;
                        pols.useCarry[i*LATCH_SIZE + j] = fr.one();
                    }
                    else
                    {
                        pols.useCarry[i*LATCH_SIZE + j] = fr.zero(); // TODO: Comment out?
                    }
                    break;
                }
                default:
                {
                    pols.cIn[i*LATCH_SIZE + j] = fr.zero(); // TODO: Comment out?
                    pols.cOut[i*LATCH_SIZE + j] = fr.zero(); // TODO: Comment out?
                    break;
                }
            }
            // We can set the cIn and the LCin when RESET =1
            if (RESET[(i*LATCH_SIZE + j + 1) % N])
            {
                pols.cIn[(i*LATCH_SIZE + j + 1) % N] = fr.zero(); // TODO: Comment out?
            }
            else
            {
                pols.cIn[(i*LATCH_SIZE + j + 1) % N] = pols.cOut[i*LATCH_SIZE + j];
            }
            pols.lCout[(i*LATCH_SIZE + j + 1) % N] = pols.cOut[i*LATCH_SIZE + j];
            pols.lOpcode[(i*LATCH_SIZE + j + 1) % N] = pols.opcode[i*LATCH_SIZE + j];

            pols.a0[(i*LATCH_SIZE + j + 1) % N] = fr.fromU64( fr.toU64(pols.a0[(i*LATCH_SIZE + j) % N]) * (1 - RESET[(i*LATCH_SIZE + j) % N]) + fr.toU64(pols.freeInA[(i*LATCH_SIZE + j) % N]) * FACTOR[0][(i*LATCH_SIZE + j) % N] );
            pols.b0[(i*LATCH_SIZE + j + 1) % N] = fr.fromU64( fr.toU64(pols.b0[(i*LATCH_SIZE + j) % N]) * (1 - RESET[(i*LATCH_SIZE + j) % N]) + fr.toU64(pols.freeInB[(i*LATCH_SIZE + j) % N]) * FACTOR[0][(i*LATCH_SIZE + j) % N] );

            c0Temp[(i*LATCH_SIZE + j) % N] = fr.toU64(pols.c0[(i*LATCH_SIZE + j) % N]) * (1 - RESET[(i*LATCH_SIZE + j) % N]) + fr.toU64(pols.freeInC[(i*LATCH_SIZE + j) % N]) * FACTOR[0][(i*LATCH_SIZE + j) % N];
            pols.c0[(i*LATCH_SIZE + j + 1) % N] = fr.fromU64( fr.toU64(pols.useCarry[(i*LATCH_SIZE + j) % N]) * (fr.toU64(pols.cOut[(i*LATCH_SIZE + j) % N]) - c0Temp[(i*LATCH_SIZE + j) % N]) + c0Temp[(i*LATCH_SIZE + j) % N] );

            //if ((i*LATCH_SIZE + j) % 10000 === 0) console.log(`Computing final binary pols ${(i * LATCH_SIZE + j)}/${N}`);

            for (uint64_t k = 1; k < REGISTERS_NUM; k++)
            {
                a[k][(i*LATCH_SIZE + j + 1) % N] = fr.fromU64( fr.toU64(a[k][(i*LATCH_SIZE + j) % N]) * (1 - RESET[(i*LATCH_SIZE + j) % N]) + fr.toU64(pols.freeInA[(i*LATCH_SIZE + j) % N]) * FACTOR[k][(i*LATCH_SIZE + j) % N] );
                b[k][(i*LATCH_SIZE + j + 1) % N] = fr.fromU64( fr.toU64(b[k][(i*LATCH_SIZE + j) % N]) * (1 - RESET[(i*LATCH_SIZE + j) % N]) + fr.toU64(pols.freeInB[(i*LATCH_SIZE + j) % N]) * FACTOR[k][(i*LATCH_SIZE + j) % N] );
                if (!fr.isZero(pols.last[i*LATCH_SIZE + j]) && !fr.isZero(pols.useCarry[i*LATCH_SIZE + j]))
                {
                    c[k][(i*LATCH_SIZE + j + 1) % N] = fr.zero(); // TODO: Comment out?
                }
                else
                {
                    c[k][(i*LATCH_SIZE + j + 1) % N] = fr.fromU64( fr.toU64(c[k][(i*LATCH_SIZE + j) % N]) * (1 - RESET[(i*LATCH_SIZE + j) % N]) + fr.toU64(pols.freeInC[(i*LATCH_SIZE + j) % N]) * FACTOR[k][(i*LATCH_SIZE + j) % N] );
                }
            }
        }
    }

    for (uint64_t i = input.size()*LATCH_SIZE; i < N; i++)
    {
        //if (i % 10000 === 0) console.log(`Computing final binary pols ${i}/${N}`);
        pols.a0[(i + 1) % N] = fr.fromU64( fr.toU64(pols.a0[i]) * (1 - RESET[i]) + fr.toU64(pols.freeInA[i]) * FACTOR[0][i] );
        pols.b0[(i + 1) % N] = fr.fromU64( fr.toU64(pols.b0[i]) * (1 - RESET[i]) + fr.toU64(pols.freeInB[i]) * FACTOR[0][i] );

        c0Temp[i] = fr.toU64(pols.c0[i]) * (1 - RESET[i]) + fr.toU64(pols.freeInC[i]) * FACTOR[0][i];
        pols.c0[(i + 1) % N] = fr.fromU64( fr.toU64(pols.useCarry[i]) * (fr.toU64(pols.cOut[i]) - c0Temp[i]) + c0Temp[i] );

        for (uint64_t j = 1; j < REGISTERS_NUM; j++)
        {
            a[j][(i + 1) % N] = fr.fromU64( fr.toU64(a[j][i]) * (1 - RESET[i]) + fr.toU64(pols.freeInA[i]) * FACTOR[j][i] );
            b[j][(i + 1) % N] = fr.fromU64( fr.toU64(b[j][i]) * (1 - RESET[i]) + fr.toU64(pols.freeInB[i]) * FACTOR[j][i] );
            c[j][(i + 1) % N] = fr.fromU64( fr.toU64(c[j][i]) * (1 - RESET[i]) + fr.toU64(pols.freeInC[i]) * FACTOR[j][i] );
        }
    }

    free(c0Temp);

    cout << "BinaryExecutor successfully processed " << action.size() << " binary actions" << endl;
}

// To be used only for testing, since it allocates a lot of memory
void BinaryExecutor::execute (vector<BinaryAction> &action)
{
    void * pAddress = mapFile(config.cmPolsFile, CommitPols::pilSize(), true);
    CommitPols cmPols(pAddress, CommitPols::pilDegree());
    execute(action, cmPols.Binary);
    unmapFile(pAddress, CommitPols::pilSize());
}