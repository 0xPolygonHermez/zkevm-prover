#include <nlohmann/json.hpp>
#include "binary_executor.hpp"
#include "binary_action_bytes.hpp"
#include "binary_defines.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "timer.hpp"
#include "zklog.hpp"

using json = nlohmann::json;

BinaryExecutor::BinaryExecutor (Goldilocks &fr, const Config &config) :
    fr(fr),
    config(config),
    N(BinaryCommitPols::pilDegree())
{
    TimerStart(BINARY_EXECUTOR);

    buildFactors();
    buildReset();

    TimerStopAndLog(BINARY_EXECUTOR);
}

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
    TimerStart(BINARY_BUILD_FACTORS);

    for (uint64_t j = 0; j < REGISTERS_NUM; j++)
    {
        vector<uint64_t> aux;
        FACTOR.push_back(aux);
    }

#pragma omp parallel for
    for (uint64_t j = 0; j < REGISTERS_NUM; j++)
    {
        for (uint64_t index = 0; index < N; index++)
        {
            uint64_t k = (index / STEPS_PER_REGISTER) % REGISTERS_NUM;
            if (j == k)
            {
                FACTOR[j].push_back(((index % 2) == 0) ? 1 : uint64_t(1)<<16);
            }
            else
            {
                FACTOR[j].push_back(0);
            }
        }
    }

    TimerStopAndLog(BINARY_BUILD_FACTORS);
}

/*  =========
    RESET
    =========
    1 0 0 ... { STEPS } ... 0 1 0 ... { STEPS } 0
    1 0 0 ... { STEPS } ... 0 1 0 ... { STEPS } 0
    1 0 0 ... { STEPS } ... 0 1 0 ... { STEPS } 0
*/
void BinaryExecutor::buildReset (void)
{
    TimerStart(BINARY_BUILD_RESET);

    for (uint64_t i = 0; i < N; i++)
    {
        RESET.push_back((i % STEPS) == 0);
    }

    TimerStopAndLog(BINARY_BUILD_RESET);
}

void BinaryExecutor::execute (vector<BinaryAction> &action, BinaryCommitPols &pols)
{
    // Check that we have enough room in polynomials  TODO: Do this check in JS
    if (action.size()*LATCH_SIZE > N)
    {
        zklog.error("BinaryExecutor::execute() Too many Binary entries=" + to_string(action.size()) + " > N/LATCH_SIZE=" + to_string(N/LATCH_SIZE));
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
        actionBytes.type = action[i].type;
        input.push_back(actionBytes);
    }

    // Local array of N uint32
    uint32_t * c0Temp = (uint32_t *)calloc(N*sizeof(uint32_t),1);
    if (c0Temp == NULL)
    {
        zklog.error("BinaryExecutor::execute() failed calling malloc() for c0Temp");
        exitProcess();
    }

    // Process all the inputs
//#pragma omp parallel for // TODO: Disabled since OMP decreases performance, probably due to cache invalidations
    for (uint64_t i = 0; i < input.size(); i++)
    {
#ifdef LOG_BINARY_EXECUTOR
        if (i%10000 == 0)
        {
            zklog.info("Computing binary pols " + to_string(i) + "/" + to_string(input.size()));
        }
#endif

        for (uint64_t j = 0; j < STEPS; j++)
        {
            bool last = (j == (STEPS - 1)) ? true : false;
            uint64_t index = i*STEPS + j;
            pols.opcode[index] = fr.fromU64(input[i].opcode);

            Goldilocks::Element cIn = fr.zero();
            Goldilocks::Element cOut = fr.zero();
            bool reset = (j == 0) ? true : false;
            bool useCarry = false;

            for (uint64_t k = 0; k < 2; k++)
            {
                cIn = (k == 0) ? pols.cIn[index] : cOut;

                uint64_t byteA = input[i].a_bytes[j*2 + k];
                uint64_t byteB = input[i].b_bytes[j*2 + k];
                uint64_t byteC = input[i].c_bytes[j*2 + k];
                bool resetByte = reset && (k == 0);
                bool lastByte = last && (k == 1);
                pols.freeInA[k][index] = fr.fromU64(byteA);
                pols.freeInB[k][index] = fr.fromU64(byteB);
                pols.freeInC[k][index] = fr.fromU64(byteC);

                // carry management

                switch (input[i].opcode)
                {
                    // ADD   (OPCODE = 0)
                    case 0:
                    {
                        uint64_t sum = byteA + byteB + fr.toU64(cIn);
                        cOut = fr.fromU64(sum >> 8);
                        break;
                    }
                    // SUB   (OPCODE = 1)
                    case 1:
                    {
                        if ((int64_t)byteA - (int64_t)fr.toU64(cIn) >= (int64_t)byteB)
                        {
                            cOut = fr.zero();
                        }
                        else
                        {
                            cOut = fr.one();
                        }
                        break;
                    }
                    // LT    (OPCODE = 2)
                    case 2:
                    {
                        if (resetByte)
                        {
                            pols.freeInC[0][index] = fr.fromU64(input[i].c_bytes[STEPS-1]); // Only change the freeInC when reset or Last
                        }
                        
                        if (byteA < byteB)
                        {
                            cOut = fr.one();
                        }
                        else if (byteA == byteB)
                        {
                            cOut = cIn;
                        }
                        else
                        {
                            cOut = fr.zero();
                        }

                        if (lastByte)
                        {
                            useCarry = true;
                            pols.freeInC[1][index] = fr.fromU64(input[i].c_bytes[0]);
                        }
                        break;
                    }
                    // SLT    (OPCODE = 3)
                    case 3:
                    {
                        useCarry = last;
                        if (resetByte)
                        {
                            pols.freeInC[0][index] = fr.fromU64(input[i].c_bytes[STEPS-1]);  // Only change the freeInC when reset or Last
                        }
                        if (lastByte)
                        {
                            uint64_t sig_a = byteA >> 7;
                            uint64_t sig_b = byteB >> 7;
                            // A Negative ; B Positive
                            if (sig_a > sig_b)
                            {
                                cOut = fr.one();
                            }
                            // A Positive ; B Negative
                            else if (sig_a < sig_b)
                            {
                                cOut = fr.zero();
                            }
                            // A and B equals
                            else
                            {
                                if (byteA < byteB)
                                {
                                    cOut = fr.one();
                                }
                                else if (byteA == byteB)
                                {
                                    cOut = cIn;
                                }
                                else
                                {
                                    cOut = fr.zero();
                                }
                            }
                            pols.freeInC[k][index] = fr.fromU64(input[i].c_bytes[0]); // Only change the freeInC when reset or Last
                        }
                        else
                        {
                            if (byteA < byteB)
                            {
                                cOut = fr.one();
                            }
                            else if (byteA == byteB)
                            {
                                cOut = cIn;
                            }
                            else
                            {
                                cOut = fr.zero();
                            }
                        }
                        break;
                    }
                    // EQ    (OPCODE = 4)
                    case 4:
                    {
                        if (resetByte)
                        {
                            // cIn = 1n
                            // pols.cIn[index] = 1n;
                            pols.freeInC[k][index] = fr.fromU64(input[i].c_bytes[STEPS-1]);
                        }

                        if ( (byteA == byteB) && fr.isZero(cIn) )
                        {
                            cOut = fr.zero();
                        }
                        else
                        {
                            cOut = fr.one();
                        }

                        if (lastByte)
                        {
                            useCarry = true;
                            cOut = fr.isZero(cOut)? fr.one() : fr.zero();
                            pols.freeInC[k][index] = fr.fromU64(input[i].c_bytes[0]); // Only change the freeInC when reset or Last
                        }
                        
                        break;
                    }
                    // AND    (OPCODE = 5)
                    case 5:
                    {
                        // setting carry if result of AND was non zero
                        if ( (byteC == 0) && fr.isZero(cIn) )
                        {
                            cOut = fr.zero();
                        }
                        else
                        {
                            cOut = fr.one();
                        }
                        break;
                    }
                    default:
                    {
                        cIn = fr.zero();
                        cOut = fr.zero();
                        break;
                    }
                }

                // setting carries
                if (k == 0)
                {
                    pols.cMiddle[index] = cOut;
                }
                else
                {
                    pols.cOut[index] = cOut;
                }
            }

            pols.useCarry[index] = useCarry ? fr.one() : fr.zero();

            uint64_t nextIndex = (index + 1) % N;
            bool nextReset = (nextIndex % STEPS) == 0 ? true : false;

            // We can set the cIn and the LCin when RESET =1
            if (nextReset)
            {
                pols.cIn[nextIndex] = fr.zero();
            }
            else
            {
                pols.cIn[nextIndex] = pols.cOut[index];
            }
            pols.lCout[nextIndex] = pols.cOut[index];
            pols.lOpcode[nextIndex] = pols.opcode[index];

            pols.a[0][nextIndex] = fr.fromU64( fr.toU64(pols.a[0][index])*(reset ? 0 : 1) + fr.toU64(pols.freeInA[0][index])*FACTOR[0][index] + 256*fr.toU64(pols.freeInA[1][index])*FACTOR[0][index] );
            pols.b[0][nextIndex] = fr.fromU64( fr.toU64(pols.b[0][index])*(reset ? 0 : 1) + fr.toU64(pols.freeInB[0][index])*FACTOR[0][index] + 256*fr.toU64(pols.freeInB[1][index])*FACTOR[0][index] );

            c0Temp[index] = fr.toU64(pols.c[0][index])*(reset ? 0 : 1) + fr.toU64(pols.freeInC[0][index])*FACTOR[0][index] + 256*fr.toU64(pols.freeInC[1][index])*FACTOR[0][index];
            pols.c[0][nextIndex] = (!fr.isZero(pols.useCarry[index])) ? pols.cOut[index] : fr.fromU64(c0Temp[index]);

            for (uint64_t k = 1; k < REGISTERS_NUM; k++)
            {
                pols.a[k][nextIndex] = fr.fromU64( fr.toU64(pols.a[k][index])*(reset ? 0 : 1) + fr.toU64(pols.freeInA[0][index])*FACTOR[k][index] + 256*fr.toU64(pols.freeInA[1][index])*FACTOR[k][index] );
                pols.b[k][nextIndex] = fr.fromU64( fr.toU64(pols.b[k][index])*(reset ? 0 : 1) + fr.toU64(pols.freeInB[0][index])*FACTOR[k][index] + 256*fr.toU64(pols.freeInB[1][index])*FACTOR[k][index] );
                if (last && useCarry)
                {
                    pols.c[k][nextIndex] = fr.zero();
                }
                else
                {
                    pols.c[k][nextIndex] = fr.fromU64( fr.toU64(pols.c[k][index])*(reset ? 0 : 1) + fr.toU64(pols.freeInC[0][index])*FACTOR[k][index] + 256*fr.toU64(pols.freeInC[1][index])*FACTOR[k][index] );
                }
            }
        }

        if (input[i].type == 1)
        {
            pols.resultBinOp[((i+1) * STEPS)%N] = fr.one();
        }
        if (input[i].type == 2)
        {
            pols.resultValidRange [((i+1) * STEPS)%N] = fr.one();
        }
    }

    for (uint64_t index = input.size()*STEPS; index < N; index++)
    {
        uint64_t nextIndex = (index + 1) % N;
        bool reset = (index % STEPS) == 0 ? true : false;
        pols.a[0][nextIndex] = fr.fromU64( fr.toU64(pols.a[0][index]) * (reset ? 0 : 1) + fr.toU64(pols.freeInA[0][index]) * FACTOR[0][index] + 256 * fr.toU64(pols.freeInA[1][index]) * FACTOR[0][index] );
        pols.b[0][nextIndex] = fr.fromU64( fr.toU64(pols.b[0][index]) * (reset ? 0 : 1) + fr.toU64(pols.freeInB[0][index]) * FACTOR[0][index] + 256 * fr.toU64(pols.freeInB[1][index]) * FACTOR[0][index] );

        c0Temp[index] = fr.toU64(pols.c[0][index]) * (reset ? 0 : 1) + fr.toU64(pols.freeInC[0][index]) * FACTOR[0][index] + 256 * fr.toU64(pols.freeInC[1][index]) * FACTOR[0][index];
        pols.c[0][nextIndex] = fr.fromU64( fr.toU64(pols.useCarry[index]) * (fr.toU64(pols.cOut[index]) - c0Temp[index]) + c0Temp[index] );

        for (uint64_t j = 1; j < REGISTERS_NUM; j++)
        {
            pols.a[j][nextIndex] = fr.fromU64( fr.toU64(pols.a[j][index])*(reset ? 0 : 1) + fr.toU64(pols.freeInA[0][index]) * FACTOR[j][index] + 256 * fr.toU64(pols.freeInA[1][index]) * FACTOR[j][index] );
            pols.b[j][nextIndex] = fr.fromU64( fr.toU64(pols.b[j][index])*(reset ? 0 : 1) + fr.toU64(pols.freeInB[0][index]) * FACTOR[j][index] + 256 * fr.toU64(pols.freeInB[1][index]) * FACTOR[j][index] );
            pols.c[j][nextIndex] = fr.fromU64( fr.toU64(pols.c[j][index])*(reset ? 0 : 1) + fr.toU64(pols.freeInC[0][index]) * FACTOR[j][index] + 256 * fr.toU64(pols.freeInC[1][index]) * FACTOR[j][index] );
        }
    }

    free(c0Temp);

    zklog.info("BinaryExecutor successfully processed " + to_string(action.size()) + " binary actions (" + to_string((double(action.size())*LATCH_SIZE*100)/N) + "%)");
}

// To be used only for testing, since it allocates a lot of memory
void BinaryExecutor::execute (vector<BinaryAction> &action)
{
    void * pAddress = malloc(CommitPols::pilSize());
    if (pAddress == NULL)
    {
        zklog.error("BinaryExecutor::execute() failed calling malloc() of size=" + to_string(CommitPols::pilSize()));
        exitProcess();
    }
    CommitPols cmPols(pAddress, CommitPols::pilDegree());
    execute(action, cmPols.Binary);
    free(pAddress);
}