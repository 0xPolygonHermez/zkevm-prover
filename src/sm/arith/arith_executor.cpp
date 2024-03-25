#include <nlohmann/json.hpp>
#include "arith_executor.hpp"
#include "arith_action_bytes.hpp"
//#include "arith_defines.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "zklog.hpp"
#include "goldilocks_precomputed.hpp"

using json = nlohmann::json;

Goldilocks::Element eq0 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
Goldilocks::Element eq1 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
Goldilocks::Element eq2 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
Goldilocks::Element eq3 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);
Goldilocks::Element eq4 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o);

const uint16_t chunksPrimeHL[16] = { 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                                     0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFE, 0xFFFF, 0xFC2F };

void ArithExecutor::execute (vector<ArithAction> &action, ArithCommitPols &pols)
{
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
            pq0 = sScalar*input[i].x2 - sScalar*input[i].x1 - input[i].y2 + input[i].y1;
            q0 = -(pq0/pFec);
            if ((pq0 + pFec*q0) != 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " with the calculated q0 the residual is not zero (diff point)");
                exitProcess();
            } 
            q0 += ScalarTwoTo258;
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
            pq0 = sScalar*2*input[i].y1 - 3*input[i].x1*input[i].x1;
            q0 = -(pq0/pFec);
            if ((pq0 + pFec*q0) != 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " with the calculated q0 the residual is not zero (same point)");
                exitProcess();
            } 
            q0 += ScalarTwoTo258;
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
            pq1 = sScalar*sScalar - input[i].x1 - input[i].x2 - input[i].x3;
            q1 = -(pq1/pFec);
            if ((pq1 + pFec*q1) != 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " with the calculated q1 the residual is not zero");
                exitProcess();
            } 
            q1 += ScalarTwoTo258;

            // Check q2
            mpz_class pq2;
            pq2 = sScalar*input[i].x1 - sScalar*input[i].x3 - input[i].y1 - input[i].y3;
            q2 = -(pq2/pFec);
            if ((pq2 + pFec*q2) != 0)
            {
                zklog.error("ArithExecutor::execute() For input " + to_string(i) + " with the calculated q2 the residual is not zero");
                exitProcess();
            } 
            q2 += ScalarTwoTo258;
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

            // selEq3 (addition + doubling points) is select need to check that x3, y3 is alias free.
            if (!fr.isZero(pols.selEq[3][index]))
            {
                Goldilocks::Element chunkValue = step > 15 ? pols.y3[15 - step16][offset] : pols.x3[15 - step16][offset];
                uint64_t chunkPrime = chunksPrimeHL[step16];
                bool chunkLtPrime = valueLtPrime ? false : (fr.toU64(chunkValue) < chunkPrime);
                valueLtPrime = valueLtPrime || chunkLtPrime;
                pols.chunkLtPrime[index] = chunkLtPrime ? fr.one() : fr.zero();
                pols.valueLtPrime[nextIndex] = valueLtPrime ? fr.one() : fr.zero();
            }
        }

        mpz_class carry[3] = {0, 0, 0};
        uint64_t eqIndexToCarryIndex[5] = {0, 0, 0, 1, 2};
        mpz_class eq[5] = {0, 0, 0, 0, 0};

        vector<uint64_t> eqIndexes;
        if (!fr.isZero(pols.selEq[0][offset])) eqIndexes.push_back(0);
        if (!fr.isZero(pols.selEq[1][offset])) eqIndexes.push_back(1);
        if (!fr.isZero(pols.selEq[2][offset])) eqIndexes.push_back(2);
        if (!fr.isZero(pols.selEq[3][offset])) { eqIndexes.push_back(3); eqIndexes.push_back(4); }

        mpz_class auxScalar;
        for (uint64_t step=0; step<32; step++)
        {
            for (uint64_t k=0; k<eqIndexes.size(); k++)
            {
                uint64_t eqIndex = eqIndexes[k];
                uint64_t carryIndex = eqIndexToCarryIndex[eqIndex];
                switch(eqIndex)
                {
                    case 0: eq[eqIndex] = fr.toS64(eq0(fr, pols, step, offset)); break;
                    case 1: eq[eqIndex] = fr.toS64(eq1(fr, pols, step, offset)); break;
                    case 2: eq[eqIndex] = fr.toS64(eq2(fr, pols, step, offset)); break;
                    case 3: eq[eqIndex] = fr.toS64(eq3(fr, pols, step, offset)); break;
                    case 4: eq[eqIndex] = fr.toS64(eq4(fr, pols, step, offset)); break;
                    default:
                        zklog.error("ArithExecutor::execute() invalid eqIndex=" + to_string(eqIndex));
                        exitProcess();
                }
                pols.carry[carryIndex][offset + step] = fr.fromScalar(carry[carryIndex]);
                carry[carryIndex] = (eq[eqIndex] + carry[carryIndex]) / ScalarTwoTo16;
            }
        }

        if (!fr.isZero(pols.selEq[0][offset])) pols.resultEq0[offset + 31] = fr.one();
        if (!fr.isZero(pols.selEq[1][offset])) pols.resultEq1[offset + 31] = fr.one();
        if (!fr.isZero(pols.selEq[2][offset])) pols.resultEq2[offset + 31] = fr.one();
    }
    
    zklog.info("ArithExecutor successfully processed " + to_string(action.size()) + " arith actions (" + to_string((double(action.size())*32*100)/N) + "%)");
}