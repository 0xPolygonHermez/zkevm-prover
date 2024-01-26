#include <nlohmann/json.hpp>
#include "binary_executor.hpp"
#include "binary_action_bytes.hpp"
#include "binary_defines.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "timer.hpp"
#include "zklog.hpp"
#include <fstream>
#ifdef __ZKEVM_SM__
#include "zkevm_sm.h"
#endif
#include "poseidon_goldilocks.hpp"

using json = nlohmann::json;

BinaryExecutor::BinaryExecutor (Goldilocks &fr, const Config &config) :
    fr(fr),
    config(config),
    N(BinaryCommitPols::pilDegree())
{
    TimerStart(BINARY_EXECUTOR);
#ifdef __ZKEVM_SM__
    ZkevmSMBinaryPtr = sm_binary_new(N);
#else
    buildFactors();
    buildReset();
#endif
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
#ifdef __ZKEVM_SM__
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

    sm_binary_execute(ZkevmSMBinaryPtr, (void *)input.data(), input.size(), (void *)pols.address(), 1880, 751 * 8, N);
#else
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
        
        const uint64_t opcode = input[i].opcode;
        uint64_t reset4 = opcode == 8 ? 1 : 0;
        Goldilocks::Element previousAreLt4 = fr.zero();

        for (uint64_t j = 0; j < STEPS; j++)
        {
            bool last = (j == (STEPS - 1)) ? true : false;
            uint64_t index = i*STEPS + j;
            pols.opcode[index] = fr.fromU64(opcode);

            Goldilocks::Element cIn = fr.zero();
            Goldilocks::Element cOut = fr.zero();
            bool reset = (j == 0) ? true : false;
            bool useCarry = false;
            uint64_t usePreviousAreLt4 = 0;

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

                // ONLY forcarry, ge4 management
                switch (opcode)
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
                    // LT4   (OPTCODE = 8)
                    case 8:
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
                            if(opcode == 2 || cOut == fr.zero()){
                                useCarry = true;
                                pols.freeInC[1][index] = fr.fromU64(input[i].c_bytes[0]);
                            } else {
                                usePreviousAreLt4 = 1;
                                // SPECIAL CASE: using a runtime value previousAreLt4, but lookup table was static, means in
                                // this case put expected value, because when rebuild c using correctly previousAreLt4 no freeInC.
                                pols.freeInC[1][index] = cOut;
                            }
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
            if( j % 16 == 3){
                previousAreLt4 = cOut;
            } else if ( j % 16 ==7 || j % 16 == 11){
                previousAreLt4 = previousAreLt4 * cOut;
            }

            pols.useCarry[index] = useCarry ? fr.one() : fr.zero();
            pols.usePreviousAreLt4[index] = fr.fromU64(usePreviousAreLt4);
            pols.reset4[index] = fr.fromU64(reset4);

            uint64_t nextIndex = (index + 1) % N;
            bool nextReset = (nextIndex % STEPS) == 0 ? true : false;

            pols.previousAreLt4[nextIndex] = previousAreLt4;

            // We can set the cIn and the LCin when RESET =1
            if (nextReset)
            {
                pols.cIn[nextIndex] = fr.zero();
            }
            else
            {
                pols.cIn[nextIndex] = (reset4 == 1 && (index % 4) == 3) ? fr.zero() : pols.cOut[index];
            }
            pols.lCout[nextIndex] = usePreviousAreLt4 ? previousAreLt4 : pols.cOut[index];
            pols.lOpcode[nextIndex] = pols.opcode[index];

            pols.a[0][nextIndex] = fr.fromU64( fr.toU64(pols.a[0][index])*(reset ? 0 : 1) + fr.toU64(pols.freeInA[0][index])*FACTOR[0][index] + 256*fr.toU64(pols.freeInA[1][index])*FACTOR[0][index] );
            pols.b[0][nextIndex] = fr.fromU64( fr.toU64(pols.b[0][index])*(reset ? 0 : 1) + fr.toU64(pols.freeInB[0][index])*FACTOR[0][index] + 256*fr.toU64(pols.freeInB[1][index])*FACTOR[0][index] );

            c0Temp[index] = fr.toU64(pols.c[0][index])*(reset ? 0 : 1) + fr.toU64(pols.freeInC[0][index])*FACTOR[0][index] + 256*fr.toU64(pols.freeInC[1][index])*FACTOR[0][index];
            pols.c[0][nextIndex] = (!fr.isZero(pols.useCarry[index])) ? pols.cOut[index] : (pols.usePreviousAreLt4[index] == fr.one() ? pols.previousAreLt4[index] : fr.fromU64(c0Temp[index]));

            for (uint64_t k = 1; k < REGISTERS_NUM; k++)
            {
                pols.a[k][nextIndex] = fr.fromU64( fr.toU64(pols.a[k][index])*(reset ? 0 : 1) + fr.toU64(pols.freeInA[0][index])*FACTOR[k][index] + 256*fr.toU64(pols.freeInA[1][index])*FACTOR[k][index] );
                pols.b[k][nextIndex] = fr.fromU64( fr.toU64(pols.b[k][index])*(reset ? 0 : 1) + fr.toU64(pols.freeInB[0][index])*FACTOR[k][index] + 256*fr.toU64(pols.freeInB[1][index])*FACTOR[k][index] );
                if (last && (useCarry || usePreviousAreLt4))
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
    zklog.info("BinaryExecutor successfully processed " + to_string(action.size()) + " binary actions (" + to_string((double(action.size()) * LATCH_SIZE * 100) / N) + "%)");
#endif
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

    double startTime = omp_get_wtime();
    execute(action, cmPols.Binary);
    cout << "Execution time: " << omp_get_wtime() - startTime << "\n";


    // // Sum all the vlaues for each polynomial in cmPols.Binary
    // Goldilocks::Element sum_pol_opcode = fr.zero();
    // Goldilocks::Element sum_pol_a0 = fr.zero();
    // Goldilocks::Element sum_pol_a1 = fr.zero();
    // Goldilocks::Element sum_pol_a2 = fr.zero();
    // Goldilocks::Element sum_pol_a3 = fr.zero();
    // Goldilocks::Element sum_pol_a4 = fr.zero();
    // Goldilocks::Element sum_pol_a5 = fr.zero();
    // Goldilocks::Element sum_pol_a6 = fr.zero();
    // Goldilocks::Element sum_pol_a7 = fr.zero();
    // Goldilocks::Element sum_pol_b0 = fr.zero();
    // Goldilocks::Element sum_pol_b1 = fr.zero();
    // Goldilocks::Element sum_pol_b2 = fr.zero();
    // Goldilocks::Element sum_pol_b3 = fr.zero();
    // Goldilocks::Element sum_pol_b4 = fr.zero();
    // Goldilocks::Element sum_pol_b5 = fr.zero();
    // Goldilocks::Element sum_pol_b6 = fr.zero();
    // Goldilocks::Element sum_pol_b7 = fr.zero();
    // Goldilocks::Element sum_pol_c0 = fr.zero();
    // Goldilocks::Element sum_pol_c1 = fr.zero();
    // Goldilocks::Element sum_pol_c2 = fr.zero();
    // Goldilocks::Element sum_pol_c3 = fr.zero();
    // Goldilocks::Element sum_pol_c4 = fr.zero();
    // Goldilocks::Element sum_pol_c5 = fr.zero();
    // Goldilocks::Element sum_pol_c6 = fr.zero();
    // Goldilocks::Element sum_pol_c7 = fr.zero();
    // Goldilocks::Element sum_pol_freein_a0 = fr.zero();
    // Goldilocks::Element sum_pol_freein_a1 = fr.zero();
    // Goldilocks::Element sum_pol_freein_b0 = fr.zero();
    // Goldilocks::Element sum_pol_freein_b1 = fr.zero();
    // Goldilocks::Element sum_pol_freein_c0 = fr.zero();
    // Goldilocks::Element sum_pol_freein_c1 = fr.zero();
    // Goldilocks::Element sum_pol_cin = fr.zero();
    // Goldilocks::Element sum_pol_cmiddle = fr.zero();
    // Goldilocks::Element sum_pol_cout = fr.zero();
    // Goldilocks::Element sum_pol_l_cout = fr.zero();
    // Goldilocks::Element sum_pol_l_opcode = fr.zero();
    // Goldilocks::Element sum_pol_previous_are_lt4 = fr.zero();
    // Goldilocks::Element sum_pol_use_previous_are_lt4 = fr.zero();
    // Goldilocks::Element sum_pol_reset4 = fr.zero();
    // Goldilocks::Element sum_pol_use_carry = fr.zero();
    // Goldilocks::Element sum_pol_result_bin_op = fr.zero();
    // Goldilocks::Element sum_pol_result_valid_range = fr.zero();

    // for (uint64_t i = 0; i < CommitPols::pilDegree(); ++i) {
    //     sum_pol_opcode = sum_pol_opcode + cmPols.Binary.opcode[i];
    //     sum_pol_a0 = sum_pol_a0 + cmPols.Binary.a[0][i];
    //     sum_pol_a1 = sum_pol_a1 + cmPols.Binary.a[1][i];
    //     sum_pol_a2 = sum_pol_a2 + cmPols.Binary.a[2][i];
    //     sum_pol_a3 = sum_pol_a3 + cmPols.Binary.a[3][i];
    //     sum_pol_a4 = sum_pol_a4 + cmPols.Binary.a[4][i];
    //     sum_pol_a5 = sum_pol_a5 + cmPols.Binary.a[5][i];
    //     sum_pol_a6 = sum_pol_a6 + cmPols.Binary.a[6][i];
    //     sum_pol_a7 = sum_pol_a7 + cmPols.Binary.a[7][i];
    //     sum_pol_b0 = sum_pol_b0 + cmPols.Binary.b[0][i];
    //     sum_pol_b1 = sum_pol_b1 + cmPols.Binary.b[1][i];
    //     sum_pol_b2 = sum_pol_b2 + cmPols.Binary.b[2][i];
    //     sum_pol_b3 = sum_pol_b3 + cmPols.Binary.b[3][i];
    //     sum_pol_b4 = sum_pol_b4 + cmPols.Binary.b[4][i];
    //     sum_pol_b5 = sum_pol_b5 + cmPols.Binary.b[5][i];
    //     sum_pol_b6 = sum_pol_b6 + cmPols.Binary.b[6][i];
    //     sum_pol_b7 = sum_pol_b7 + cmPols.Binary.b[7][i];
    //     sum_pol_c0 = sum_pol_c0 + cmPols.Binary.c[0][i];
    //     sum_pol_c1 = sum_pol_c1 + cmPols.Binary.c[1][i];
    //     sum_pol_c2 = sum_pol_c2 + cmPols.Binary.c[2][i];
    //     sum_pol_c3 = sum_pol_c3 + cmPols.Binary.c[3][i];
    //     sum_pol_c4 = sum_pol_c4 + cmPols.Binary.c[4][i];
    //     sum_pol_c5 = sum_pol_c5 + cmPols.Binary.c[5][i];
    //     sum_pol_c6 = sum_pol_c6 + cmPols.Binary.c[6][i];
    //     sum_pol_c7 = sum_pol_c7 + cmPols.Binary.c[7][i];
    //     sum_pol_freein_a0 = sum_pol_freein_a0 + cmPols.Binary.freeInA[0][i];
    //     sum_pol_freein_a1 = sum_pol_freein_a1 + cmPols.Binary.freeInA[1][i];
    //     sum_pol_freein_b0 = sum_pol_freein_b0 + cmPols.Binary.freeInB[0][i];
    //     sum_pol_freein_b1 = sum_pol_freein_b1 + cmPols.Binary.freeInB[1][i];
    //     sum_pol_freein_c0 = sum_pol_freein_c0 + cmPols.Binary.freeInC[0][i];
    //     sum_pol_freein_c1 = sum_pol_freein_c1 + cmPols.Binary.freeInC[1][i];
    //     sum_pol_cin = sum_pol_cin + cmPols.Binary.cIn[i];
    //     sum_pol_cmiddle = sum_pol_cmiddle + cmPols.Binary.cMiddle[i];
    //     sum_pol_cout = sum_pol_cout + cmPols.Binary.cOut[i];
    //     sum_pol_l_cout = sum_pol_l_cout + cmPols.Binary.lCout[i];
    //     sum_pol_l_opcode = sum_pol_l_opcode + cmPols.Binary.lOpcode[i];
    //     sum_pol_previous_are_lt4 = sum_pol_previous_are_lt4 + cmPols.Binary.previousAreLt4[i];
    //     sum_pol_use_previous_are_lt4 = sum_pol_use_previous_are_lt4 + cmPols.Binary.usePreviousAreLt4[i];
    //     sum_pol_reset4 = sum_pol_reset4 + cmPols.Binary.reset4[i];
    //     sum_pol_use_carry = sum_pol_use_carry + cmPols.Binary.useCarry[i];
    //     sum_pol_result_bin_op = sum_pol_result_bin_op + cmPols.Binary.resultBinOp[i];
    //     sum_pol_result_valid_range = sum_pol_result_valid_range + cmPols.Binary.resultValidRange[i];
    // }

    // std::cout << "sum_pol_opcode = " << fr.toString(sum_pol_opcode) << std::endl;
    // std::cout << "sum_pol_a0 = " << fr.toString(sum_pol_a0) << std::endl;
    // std::cout << "sum_pol_a1 = " << fr.toString(sum_pol_a1) << std::endl;
    // std::cout << "sum_pol_a2 = " << fr.toString(sum_pol_a2) << std::endl;
    // std::cout << "sum_pol_a3 = " << fr.toString(sum_pol_a3) << std::endl;
    // std::cout << "sum_pol_a4 = " << fr.toString(sum_pol_a4) << std::endl;
    // std::cout << "sum_pol_a5 = " << fr.toString(sum_pol_a5) << std::endl;
    // std::cout << "sum_pol_a6 = " << fr.toString(sum_pol_a6) << std::endl;
    // std::cout << "sum_pol_a7 = " << fr.toString(sum_pol_a7) << std::endl;
    // std::cout << "sum_pol_b0 = " << fr.toString(sum_pol_b0) << std::endl;
    // std::cout << "sum_pol_b1 = " << fr.toString(sum_pol_b1) << std::endl;
    // std::cout << "sum_pol_b2 = " << fr.toString(sum_pol_b2) << std::endl;
    // std::cout << "sum_pol_b3 = " << fr.toString(sum_pol_b3) << std::endl;
    // std::cout << "sum_pol_b4 = " << fr.toString(sum_pol_b4) << std::endl;
    // std::cout << "sum_pol_b5 = " << fr.toString(sum_pol_b5) << std::endl;
    // std::cout << "sum_pol_b6 = " << fr.toString(sum_pol_b6) << std::endl;
    // std::cout << "sum_pol_b7 = " << fr.toString(sum_pol_b7) << std::endl;
    // std::cout << "sum_pol_c0 = " << fr.toString(sum_pol_c0) << std::endl;
    // std::cout << "sum_pol_c1 = " << fr.toString(sum_pol_c1) << std::endl;
    // std::cout << "sum_pol_c2 = " << fr.toString(sum_pol_c2) << std::endl;
    // std::cout << "sum_pol_c3 = " << fr.toString(sum_pol_c3) << std::endl;
    // std::cout << "sum_pol_c4 = " << fr.toString(sum_pol_c4) << std::endl;
    // std::cout << "sum_pol_c5 = " << fr.toString(sum_pol_c5) << std::endl;
    // std::cout << "sum_pol_c6 = " << fr.toString(sum_pol_c6) << std::endl;
    // std::cout << "sum_pol_c7 = " << fr.toString(sum_pol_c7) << std::endl;
    // std::cout << "sum_pol_freein_a0 = " << fr.toString(sum_pol_freein_a0) << std::endl;
    // std::cout << "sum_pol_freein_a1 = " << fr.toString(sum_pol_freein_a1) << std::endl;
    // std::cout << "sum_pol_freein_b0 = " << fr.toString(sum_pol_freein_b0) << std::endl;
    // std::cout << "sum_pol_freein_b1 = " << fr.toString(sum_pol_freein_b1) << std::endl;
    // std::cout << "sum_pol_freein_c0 = " << fr.toString(sum_pol_freein_c0) << std::endl;
    // std::cout << "sum_pol_freein_c1 = " << fr.toString(sum_pol_freein_c1) << std::endl;
    // std::cout << "sum_pol_cin = " << fr.toString(sum_pol_cin) << std::endl;
    // std::cout << "sum_pol_cmiddle = " << fr.toString(sum_pol_cmiddle) << std::endl;
    // std::cout << "sum_pol_cout = " << fr.toString(sum_pol_cout) << std::endl;
    // std::cout << "sum_pol_l_cout = " << fr.toString(sum_pol_l_cout) << std::endl;
    // std::cout << "sum_pol_l_opcode = " << fr.toString(sum_pol_l_opcode) << std::endl;
    // std::cout << "sum_pol_previous_are_lt4 = " << fr.toString(sum_pol_previous_are_lt4) << std::endl;
    // std::cout << "sum_pol_use_previous_are_lt4 = " << fr.toString(sum_pol_use_previous_are_lt4) << std::endl;
    // std::cout << "sum_pol_reset4 = " << fr.toString(sum_pol_reset4) << std::endl;
    // std::cout << "sum_pol_use_carry = " << fr.toString(sum_pol_use_carry) << std::endl;
    // std::cout << "sum_pol_result_bin_op = " << fr.toString(sum_pol_result_bin_op) << std::endl;
    // std::cout << "sum_pol_result_valid_range = " << fr.toString(sum_pol_result_valid_range) << std::endl;

    // // Sum all sum_pol variables
    // Goldilocks::Element sum_all = sum_pol_opcode + sum_pol_a0 + sum_pol_a1 + sum_pol_a2 + sum_pol_a3 + sum_pol_a4 + sum_pol_a5 + sum_pol_a6 + sum_pol_a7 + sum_pol_b0 + sum_pol_b1 + sum_pol_b2 + sum_pol_b3 + sum_pol_b4 + sum_pol_b5 + sum_pol_b6 + sum_pol_b7 + sum_pol_c0 + sum_pol_c1 + sum_pol_c2 + sum_pol_c3 + sum_pol_c4 + sum_pol_c5 + sum_pol_c6 + sum_pol_c7 + sum_pol_freein_a0 + sum_pol_freein_a1 + sum_pol_freein_b0 + sum_pol_freein_b1 + sum_pol_freein_c0 + sum_pol_freein_c1 + sum_pol_cin + sum_pol_cmiddle + sum_pol_cout + sum_pol_l_cout + sum_pol_l_opcode + sum_pol_previous_are_lt4 + sum_pol_use_previous_are_lt4 + sum_pol_reset4 + sum_pol_use_carry + sum_pol_result_bin_op + sum_pol_result_valid_range;

    // std::cout << "sum_all = " << fr.toString(sum_all) << std::endl;

    // copy all the thrace in a single array
    /*uint64_t size = CommitPols::pilDegree() * cmPols.Binary.numPols();
    Goldilocks::Element *trace = (Goldilocks::Element *)calloc(size, sizeof(Goldilocks::Element));
    for (int i = 0; i < CommitPols::pilDegree(); i++)
    {
        uint64_t index = i * 751 + 235;
        for (int j = 0; j < cmPols.Binary.numPols(); j++)
        {
            trace[i * cmPols.Binary.numPols() + j] = ((Goldilocks::Element *)cmPols.Binary.address())[index + 8];
        }
    }
    Goldilocks::Element hash[4];
    PoseidonGoldilocks::linear_hash_seq(hash, trace, size);

    std::cout<<"hash: "<<std::endl;
    for(int i=0; i<4; i++){
        std::cout<<Goldilocks::toU64(hash[i])<<std::endl;
    }*/
    free(pAddress);
}