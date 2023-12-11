#include <nlohmann/json.hpp>
#include "arith_executor.hpp"
#include "arith_action_bytes.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "zklog.hpp"
#include "goldilocks_precomputed.hpp"
#include "zkglobals.hpp"

using json = nlohmann::json;

int64_t eq0 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq1 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq2 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq3 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq4 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq5 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq6 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq7 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq8 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq9 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq10 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);

const uint16_t chunksPrimeSecp256k1[16] = { 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                                            0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFE, 0xFFFF, 0xFC2F };
const uint16_t chunksPrimeBN254[16] = { 0x3064, 0x4E72, 0xE131, 0xA029, 0xB850, 0x45B6, 0x8181, 0x585D, 
                                        0x9781, 0x6A91, 0x6871, 0xCA8D, 0x3C20, 0x8C16, 0xD87C, 0xFD47 };

void ArithExecutor::execute (vector<ArithAction> &action, ArithCommitPols &pols)
{
    // Get a scalar with the bn254 prime
    mpz_class pBN254;
    pBN254.set_str(fq.toString(fq.negOne(), 16), 16);
    pBN254++;

    // Check that we have enough room in polynomials  TODO: Do this check in JS
    if (action.size()*32 > N)
    {
        zklog.error("ArithExecutor::execute() Too many Arith entries=" + to_string(action.size()) + " > N/32=" + to_string(N/32));
        exitProcess();
    }

    // Split actions into bytes
    vector<ArithActionBytes> input;
    for (uint64_t i=0; i<action.size(); i++)
    {
        uint64_t dataSize;
        ArithActionBytes actionBytes;

        actionBytes.x1 = action[i].x1;
        actionBytes.y1 = action[i].y1;
        actionBytes.x2 = action[i].x2;
        actionBytes.y2 = action[i].y2;
        actionBytes.x3 = action[i].x3;
        actionBytes.y3 = action[i].y3;
        actionBytes.selEq0 = action[i].selEq0;
        actionBytes.selEq1 = action[i].selEq1;
        actionBytes.selEq2 = action[i].selEq2;
        actionBytes.selEq3 = action[i].selEq3;
        actionBytes.selEq4 = action[i].selEq4;
        actionBytes.selEq5 = action[i].selEq5;
        actionBytes.selEq6 = action[i].selEq6;

        dataSize = 16;
        scalar2ba16(actionBytes._x1, dataSize, action[i].x1);
        dataSize = 16;
        scalar2ba16(actionBytes._y1, dataSize, action[i].y1);
        dataSize = 16;
        scalar2ba16(actionBytes._x2, dataSize, action[i].x2);
        dataSize = 16;
        scalar2ba16(actionBytes._y2, dataSize, action[i].y2);
        dataSize = 16;
        scalar2ba16(actionBytes._x3, dataSize, action[i].x3);
        dataSize = 16;
        scalar2ba16(actionBytes._y3, dataSize, action[i].y3);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq0, dataSize, action[i].selEq0);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq1, dataSize, action[i].selEq1);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq2, dataSize, action[i].selEq2);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq3, dataSize, action[i].selEq3);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq4, dataSize, action[i].selEq4);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq5, dataSize, action[i].selEq5);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq6, dataSize, action[i].selEq6);

        memset(actionBytes._s, 0, sizeof(actionBytes._s));
        memset(actionBytes._q0, 0, sizeof(actionBytes._q0));
        memset(actionBytes._q1, 0, sizeof(actionBytes._q1));
        memset(actionBytes._q2, 0, sizeof(actionBytes._q2));

        input.push_back(actionBytes);
    }

    RawFec::Element s;
    RawFec::Element aux1, aux2;
    mpz_class q0, q1, q2;

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
        // TODO: if not have x1, need to componse it

        RawFec::Element x1;
        RawFec::Element y1;
        RawFec::Element x2;
        RawFec::Element y2;
        RawFec::Element x3;
        RawFec::Element y3;
        scalar2fec(fec, x1, input[i].x1);
        scalar2fec(fec, y1, input[i].y1);
        scalar2fec(fec, x2, input[i].x2);
        scalar2fec(fec, y2, input[i].y2);
        scalar2fec(fec, x3, input[i].x3);
        scalar2fec(fec, y3, input[i].y3);

        // In the following, recall that we can only work with unsiged integers of 256 bits.
        // Therefore, as the quotient needs to be represented in our VM, we need to know
        // the worst negative case and add an offset so that the resulting name is never negative.
        // Then, this offset is also added in the PIL constraint to ensure the equality.
        // Note1: Since we can choose whether the quotient is positive or negative, we choose it so
        //        that the added offset is the lowest.
        // Note2: x1,x2,y1,y2 can be assumed to be alias free, as this is the pre condition in the Arith SM.
        //        I.e, x1,x2,y1,y2 ∈ [0, 2^256-1].
        if (input[i].selEq1 == 1)
        {
            // s=(y2-y1)/(x2-x1)
            fec.sub(aux1, y2, y1);
            fec.sub(aux2, x2, x1);
            if (fec.isZero(aux2))
            {
                zklog.error("ArithExecutor::execute() divide by zero calculating S for input " + to_string(i));
                exitProcess();
            }
            fec.div(s, aux1, aux2);

            // Get s as a scalar
            mpz_class sScalar;
            fec2scalar(fec, s, sScalar);

            // Check
            mpz_class pq0;
            pq0 = sScalar*input[i].x2 - sScalar*input[i].x1 - input[i].y2 + input[i].y1; // Worst values are {-2^256*(2^256-1),2^256*(2^256-1)}
            q0 = pq0/pFec;
            if ((pq0 - pFec*q0) != 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " with the calculated q0 the residual is not zero (diff point)");
                exitProcess();
            } 
            q0 += ScalarTwoTo257;
            if(q0 < 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " the q0 with offset is negative (diff point). Actual value: " + q0.get_str(16));
                exitProcess();
            }
        }
        else if (input[i].selEq2 == 1)
        {
            // s = 3*x1*x1/(y1+y1
            fec.mul(aux1, x1, x1);
            fec.fromUI(aux2, 3);
            fec.mul(aux1, aux1, aux2);
            fec.add(aux2, y1, y1);
            fec.div(s, aux1, aux2);

            // Get s as a scalar
            mpz_class sScalar;
            fec2scalar(fec, s, sScalar);

            // Check
            mpz_class pq0;
            pq0 = sScalar*2*input[i].y1 - 3*input[i].x1*input[i].x1; // Worst values are {-3*(2^256-1)**2,2*(2^256-1)**2}
                                                                     // with |-3*(2^256-1)**2| > 2*(2^256-1)**2
            q0 = -(pq0/pFec);
            if ((pq0 + pFec*q0) != 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " with the calculated q0 the residual is not zero (same point)");
                exitProcess();
            } 
            q0 += ScalarTwoTo258;
            if(q0 < 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " the q0 with offset is negative (same point). Actual value: " + q0.get_str(16));
                exitProcess();
            }
        }
        else
        {
            fec.fromUI(s, 0);
            q0 = 0;
        }

        if (input[i].selEq3 == 1)
        {
            // Get s as a scalar
            mpz_class sScalar;
            fec2scalar(fec, s, sScalar);

            // Check q1
            mpz_class pq1;
            pq1 = sScalar*sScalar - input[i].x1 - input[i].x2 - input[i].x3; /// Worst values are {-3*(2^256-1),(2^256-1)**2}
                                                                             // with (2^256-1)**2 > |-3*(2^256-1)|
            q1 = pq1/pFec;
            if ((pq1 - pFec*q1) != 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " with the calculated q1 the residual is not zero");
                exitProcess();
            }
            // offset 
            q1 += 4; //2**2
            if(q1 < 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " the q1 with offset is negative (point addition). Actual value: " + q1.get_str(16));
                exitProcess();
            }

            // Check q2
            mpz_class pq2;
            pq2 = sScalar*input[i].x1 - sScalar*input[i].x3 - input[i].y1 - input[i].y3; // Worst values are {-(2^256+1)*(2^256-1),(2^256-1)**2}
                                                                                         // with |-(2^256+1)*(2^256-1)| > (2^256-1)**2
            q2 = -(pq2/pFec);
            if ((pq2 + pFec*q2) != 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " with the calculated q2 the residual is not zero");
                exitProcess();
            }
            //offset 
            q2 += ScalarTwoTo257;
            if(q2 < 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " the q2 with offset is negative (point addition). Actual value: " + q2.get_str(16));
                exitProcess();
            }

        }        
        else if (input[i].selEq4 == 1)
        {
            // Check q1
            mpz_class pq1;
            pq1 = input[i].x1*input[i].x2 - input[i].y1*input[i].y2 - input[i].x3; /// Worst values are {-2^256*(2^256-1),(2^256-1)**2}
                                                                                   // with |-2^256*(2^256-1)| > (2^256-1)**2
            q1 = -(pq1/pBN254);
            if ((pq1 + pBN254*q1) != 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " with the calculated q1 the residual is not zero");
                exitProcess();
            }
            // offset
            q1 += ScalarTwoTo259;
            if(q1 < 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " the q1 with offset is negative (complex mul). Actual value: " + q1.get_str(16));
                exitProcess();
            }

            // Check q2
            mpz_class pq2;
            pq2 = input[i].y1*input[i].x2 + input[i].x1*input[i].y2 - input[i].y3; // Worst values are {-(2^256-1),2*(2^256-1)}
                                                                                   // with 2*(2^256-1) > |-(2^256-1)|
            q2 = pq2/pBN254;
            if ((pq2 - pBN254*q2) != 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " with the calculated q2 the residual is not zero");
                exitProcess();
            }
            // offset
            q2 += 8; //2**3
            if(q2 < 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " the q2 with offset is negative (complex mul). Actual value: " + q2.get_str(16));
                exitProcess();
            }
        }
        else if (input[i].selEq5 == 1)
        {
            // Check q1
            mpz_class pq1;
            pq1 = input[i].x1 + input[i].x2 - input[i].x3; // Worst values are {-(2^256-1),2*(2^256-1)}
                                                           // with 2*(2^256-1) > |-(2^256-1)|
            q1 = pq1/pBN254;
            if ((pq1 - pBN254*q1) != 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " with the calculated q1 the residual is not zero");
                exitProcess();
            }
            //offset
            q1 += 8; //2**3
            if(q1 < 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " the q1 with offset is negative (complex add). Actual value: " + q1.get_str(16));
                exitProcess();
            }

            // Check q2
            mpz_class pq2;
            pq2 = input[i].y1 + input[i].y2 - input[i].y3; // Worst values are {-(2^256-1),2*(2^256-1)}
                                                           // with 2*(2^256-1) > |-(2^256-1)|
            q2 = pq2/pBN254;
            if ((pq2 - pBN254*q2) != 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " with the calculated q2 the residual is not zero");
                exitProcess();
            }
            //offset
            q2 += 8; //2**3
            if(q2 < 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " the q2 with offset is negative (complex add). Actual value: " + q2.get_str(16));
                exitProcess();
            }
        }
        else if (input[i].selEq6 == 1)
        {
            // Check q1
            mpz_class pq1;
            pq1 = input[i].x1 - input[i].x2 - input[i].x3; // Worst values are {-2*(2^256-1),(2^256-1)}
                                                           // with |-2*(2^256-1)| > (2^256-1)
            q1 = -(pq1/pBN254);
            if ((pq1 + pBN254*q1) != 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " with the calculated q1 the residual is not zero");
                exitProcess();
            }
            //offset
            q1 += 8; //2**3
            if(q1 < 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " the q1 with offset is negative (complex sub). Actual value: " + q1.get_str(16));
                exitProcess();
            }
            // Check q2
            mpz_class pq2;
            pq2 = input[i].y1 - input[i].y2 - input[i].y3; // Worst values are {-2*(2^256-1),(2^256-1)}
                                                           // with |-2*(2^256-1)| > (2^256-1)
                                                           
            q2 = -(pq2/pBN254);
            if ((pq2 + pBN254*q2) != 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " with the calculated q2 the residual is not zero");
                exitProcess();
            }
            //offset
            q2 += 8; //2**3
            if(q2 < 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " the q2 with offset is negative (complex sub). Actual value: " + q2.get_str(16));
                exitProcess();
            }
        }
        else
        {
            q1 = 0;
            q2 = 0;
        }

        // Get s as a scalar
        mpz_class sScalar;
        fec2scalar(fec, s, sScalar);

        uint64_t dataSize;
        dataSize = 16;
        scalar2ba16(input[i]._s, dataSize, sScalar);
        dataSize = 16;
        scalar2ba16(input[i]._q0, dataSize, q0);
        dataSize = 16;
        scalar2ba16(input[i]._q1, dataSize, q1);
        dataSize = 16;
        scalar2ba16(input[i]._q2, dataSize, q2);
    }
    
    // Process all the inputs
//#pragma omp parallel for // TODO: Disabled since OMP decreases performance, probably due to cache invalidations
    for (uint64_t i = 0; i < input.size(); i++)
    {
        uint64_t offset = i*32;
        bool xAreDifferent = false;
        bool valueLtPrime = false;
        for (uint64_t step=0; step<32; step++)
        {
            uint64_t index = offset + step;
            uint64_t nextIndex = (index + 1) % N;
            uint64_t step16 = step % 16;
            if (step16 == 0)
            {
                valueLtPrime = false;
            }
            for (uint64_t j=0; j<16; j++)
            {
                pols.x1[j][index] = fr.fromU64(input[i]._x1[j]);
                pols.y1[j][index] = fr.fromU64(input[i]._y1[j]);
                pols.x2[j][index] = fr.fromU64(input[i]._x2[j]);
                pols.y2[j][index] = fr.fromU64(input[i]._y2[j]);
                pols.x3[j][index] = fr.fromU64(input[i]._x3[j]);
                pols.y3[j][index] = fr.fromU64(input[i]._y3[j]);
                pols.s[j][index]  = fr.fromU64(input[i]._s[j]);
                pols.q0[j][index] = fr.fromU64(input[i]._q0[j]);
                pols.q1[j][index] = fr.fromU64(input[i]._q1[j]);
                pols.q2[j][index] = fr.fromU64(input[i]._q2[j]);
            }
            pols.selEq[0][index] = fr.fromU64(input[i].selEq0);
            pols.selEq[1][index] = fr.fromU64(input[i].selEq1);
            pols.selEq[2][index] = fr.fromU64(input[i].selEq2);
            pols.selEq[3][index] = fr.fromU64(input[i].selEq3);
            pols.selEq[4][index] = fr.fromU64(input[i].selEq4);
            pols.selEq[5][index] = fr.fromU64(input[i].selEq5);
            pols.selEq[6][index] = fr.fromU64(input[i].selEq6);

            // selEq1 (addition different points) is select need to check that points are diferent
            if (!fr.isZero(pols.selEq[1][index]) && (step < 16))
            {
                if (xAreDifferent == false)
                {
                    Goldilocks::Element delta = fr.sub(pols.x2[step][index], pols.x1[step][index]);
                    pols.xDeltaChunkInverse[index] = fr.isZero(delta) ? fr.zero() : glp.inv(delta);
                    xAreDifferent = fr.isZero(delta) ? false : true;
                }
                pols.xAreDifferent[nextIndex] = xAreDifferent ? fr.one() : fr.zero();
            }

            // If either selEq3,selEq4,selEq5,selEq6 is selected, we need to ensure that x3, y3 is alias free.
            // Recall that selEq3 work over the Secp256k1 curve, and selEq4,selEq5,selEq6 work over the BN254 curve.
            if (!fr.isZero(pols.selEq[3][index]) || !fr.isZero(pols.selEq[4][index]) || !fr.isZero(pols.selEq[5][index]) || !fr.isZero(pols.selEq[6][index]))
            {
                Goldilocks::Element chunkValue = step < 16 ? pols.x3[15 - step16][offset] : pols.y3[15 - step16][offset];
                uint64_t chunkPrime = !fr.isZero(pols.selEq[3][index]) ? chunksPrimeSecp256k1[step16] : chunksPrimeBN254[step16];

                bool chunkLtPrime = valueLtPrime ? false : (fr.toU64(chunkValue) < chunkPrime);
                valueLtPrime = valueLtPrime || chunkLtPrime;
                pols.chunkLtPrime[index] = chunkLtPrime ? fr.one() : fr.zero();
                pols.valueLtPrime[nextIndex] = valueLtPrime ? fr.one() : fr.zero();
            }

            pols.selEq[0][offset + step] = fr.fromU64(input[i].selEq0);
            pols.selEq[1][offset + step] = fr.fromU64(input[i].selEq1);
            pols.selEq[2][offset + step] = fr.fromU64(input[i].selEq2);
            pols.selEq[3][offset + step] = fr.fromU64(input[i].selEq3);
            pols.selEq[4][offset + step] = fr.fromU64(input[i].selEq4);
            pols.selEq[5][offset + step] = fr.fromU64(input[i].selEq5);
            pols.selEq[6][offset + step] = fr.fromU64(input[i].selEq6);
        }

        mpz_class carry[3] = {0, 0, 0};
        uint64_t eqIndexToCarryIndex[11] = {0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2};
        mpz_class eq[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        vector<uint64_t> eqIndexes;
        if (!fr.isZero(pols.selEq[0][offset])) eqIndexes.push_back(0);
        if (!fr.isZero(pols.selEq[1][offset])) eqIndexes.push_back(1);
        if (!fr.isZero(pols.selEq[2][offset])) eqIndexes.push_back(2);
        if (!fr.isZero(pols.selEq[3][offset])) { eqIndexes.push_back(3); eqIndexes.push_back(4); }
        if (!fr.isZero(pols.selEq[4][offset])) { eqIndexes.push_back(5); eqIndexes.push_back(6); }
        if (!fr.isZero(pols.selEq[5][offset])) { eqIndexes.push_back(7); eqIndexes.push_back(8); }
        if (!fr.isZero(pols.selEq[6][offset])) { eqIndexes.push_back(9); eqIndexes.push_back(10); }

        mpz_class auxScalar;
        for (uint64_t step=0; step<32; step++)
        {
            for (uint64_t k=0; k<eqIndexes.size(); k++)
            {
                uint64_t eqIndex = eqIndexes[k];
                uint64_t carryIndex = eqIndexToCarryIndex[eqIndex];
                switch(eqIndex)
                {
                    case 0:  eq[eqIndex] = eq0(fr, pols, step, offset); break;
                    case 1:  eq[eqIndex] = eq1(fr, pols, step, offset); break;
                    case 2:  eq[eqIndex] = eq2(fr, pols, step, offset); break;
                    case 3:  eq[eqIndex] = eq3(fr, pols, step, offset); break;
                    case 4:  eq[eqIndex] = eq4(fr, pols, step, offset); break;
                    case 5:  eq[eqIndex] = eq5(fr, pols, step, offset); break;
                    case 6:  eq[eqIndex] = eq6(fr, pols, step, offset); break;
                    case 7:  eq[eqIndex] = eq7(fr, pols, step, offset); break;
                    case 8:  eq[eqIndex] = eq8(fr, pols, step, offset); break;
                    case 9:  eq[eqIndex] = eq9(fr, pols, step, offset); break;
                    case 10: eq[eqIndex] = eq10(fr, pols, step, offset); break;
                    default:
                        zklog.error("ArithExecutor::execute() invalid eqIndex=" + to_string(eqIndex));
                        exitProcess();
                }
                pols.carry[carryIndex][offset + step] = fr.fromScalar(carry[carryIndex]);
                if (((eq[eqIndex] + carry[carryIndex]) % ScalarTwoTo16) != 0)
                {
                    zklog.error("ArithExecutor::execute() For input " + to_string(i) +
                        " eq[" + to_string(eqIndex) + "]=" + eq[eqIndex].get_str(16) +
                        " and carry[" + to_string(carryIndex) + "]=" + carry[carryIndex].get_str(16) +
                        " do not sum 0 mod 2 to 16");
                    exitProcess();
                }
                carry[carryIndex] = (eq[eqIndex] + carry[carryIndex]) / ScalarTwoTo16;
            }
        }

        if (!fr.isZero(pols.selEq[0][offset]))
        {
            pols.resultEq0[offset + 31] = fr.one();
        }
        if ((!fr.isZero(pols.selEq[1][offset]) && !fr.isZero(pols.selEq[3][offset])) || !fr.isZero(pols.selEq[4][offset]) || !fr.isZero(pols.selEq[5][offset]) || !fr.isZero(pols.selEq[6][offset]))
        {
            pols.resultEq1[offset + 31] = fr.one();
        }
        if (!fr.isZero(pols.selEq[2][offset]) && !fr.isZero(pols.selEq[3][offset]))
        {
            pols.resultEq2[offset + 31] = fr.one();
        }
    }
    
    zklog.info("ArithExecutor successfully processed " + to_string(action.size()) + " arith actions (" + to_string((double(action.size())*32*100)/N) + "%)");
}