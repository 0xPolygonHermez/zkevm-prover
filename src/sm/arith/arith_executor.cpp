#include <nlohmann/json.hpp>
#include "arith_executor.hpp"
#include "arith_action_bytes.hpp"
//#include "arith_defines.hpp"
#include "utils.hpp"
#include "scalar.hpp"

using json = nlohmann::json;

int64_t eq0 (ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq1 (ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq2 (ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq3 (ArithCommitPols &p, uint64_t step, uint64_t _o);
int64_t eq4 (ArithCommitPols &p, uint64_t step, uint64_t _o);

void ArithExecutor::execute (vector<ArithAction> &action, ArithCommitPols &pols)
{
    // Check that we have enough room in polynomials  TODO: Do this check in JS
    if (action.size()*32 > N)
    {
        cerr << "Error: Too many Arith entries" << endl;
        exit(-1);
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
        input.push_back(actionBytes);
    }

    RawFec::Element s;
    RawFec::Element aux1, aux2;
    mpz_class q0, q1, q2;

    // Process all the inputs
    for (uint64_t i = 0; i < input.size(); i++)
    {
#ifdef LOG_BINARY_EXECUTOR
        if (i%10000 == 0)
        {
            cout << "Computing binary pols " << i << "/" << input.size() << endl;
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
                cerr << "Error: ArithExecutor::execute() For input " << i << " with the calculated q0 the residual is not zero (diff point)" << endl;
                exit(-1);
            } 
            q0 += TwoTo258;
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
                cerr << "Error: ArithExecutor::execute() For input " << i << " with the calculated q0 the residual is not zero (same point)" << endl;
                exit(-1);
            } 
            q0 += TwoTo258;
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
                cerr << "Error: ArithExecutor::execute() For input " << i << " with the calculated q1 the residual is not zero" << endl;
                exit(-1);
            } 
            q1 += TwoTo258;

            // Check q2
            mpz_class pq2;
            pq2 = sScalar*input[i].x1 - sScalar*input[i].x3 - input[i].y1 - input[i].y3;
            q2 = -(pq2/pFec);
            if ((pq2 + pFec*q2) != 0)
            {
                cerr << "Error: ArithExecutor::execute() For input " << i << " with the calculated q2 the residual is not zero" << endl;
                exit(-1);
            } 
            q2 += TwoTo258;
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
    for (uint64_t i = 0; i < input.size(); i++)
    {
        uint64_t offset = i*32;
        for (uint64_t step=0; step<32; step++)
        {
            for (uint64_t j=0; j<16; j++)
            {
                pols.x1[j][offset + step] = input[i]._x1[j];
                pols.y1[j][offset + step] = input[i]._y1[j];
                pols.x2[j][offset + step] = input[i]._x2[j];
                pols.y2[j][offset + step] = input[i]._y2[j];
                pols.x3[j][offset + step] = input[i]._x3[j];
                pols.y3[j][offset + step] = input[i]._y3[j];
                pols.s[j][offset + step]  = input[i]._s[j];
                pols.q0[j][offset + step] = input[i]._q0[j];
                pols.q1[j][offset + step] = input[i]._q1[j];
                pols.q2[j][offset + step] = input[i]._q2[j];
            }
            pols.selEq[0][offset + step] = input[i].selEq0;
            pols.selEq[1][offset + step] = input[i].selEq1;
            pols.selEq[2][offset + step] = input[i].selEq2;
            pols.selEq[3][offset + step] = input[i].selEq3;
        }

        FieldElement carry[3];
        carry[0] = fr.zero();
        carry[1] = fr.zero();
        carry[2] = fr.zero();
        uint64_t eqIndexToCarryIndex[5] = {0, 0, 0, 1, 2};
        uint64_t eq[5] = {0, 0, 0, 0, 0};

        vector<uint64_t> eqIndexes;
        if (pols.selEq[0][offset]) eqIndexes.push_back(0);
        if (pols.selEq[1][offset]) eqIndexes.push_back(1);
        if (pols.selEq[2][offset]) eqIndexes.push_back(2);
        if (pols.selEq[3][offset]) { eqIndexes.push_back(3); eqIndexes.push_back(4); }

        for (uint64_t step=0; step<32; step++)
        {
            for (uint64_t k=0; k<eqIndexes.size(); k++)
            {
                uint64_t eqIndex = eqIndexes[k];
                uint64_t carryIndex = eqIndexToCarryIndex[eqIndex];
                switch(eqIndex)
                {
                    case 0: eq[eqIndex] = eq0(pols, step, offset); break;
                    case 1: eq[eqIndex] = eq1(pols, step, offset); break;
                    case 2: eq[eqIndex] = eq2(pols, step, offset); break;
                    case 3: eq[eqIndex] = eq3(pols, step, offset); break;
                    case 4: eq[eqIndex] = eq4(pols, step, offset); break;
                    default:
                        cerr << "Error: ArithExecutor::execute() invalid eqIndex=" << eqIndex << endl;
                        exit(-1);
                }
                FieldElement FeTwoAt18(uint64_t(1)<<18);
                FieldElement FeTwoAt16(uint64_t(1)<<16);
                pols.carryL[carryIndex][offset + step] = carry[carryIndex] % FeTwoAt18;
                pols.carryH[carryIndex][offset + step] = carry[carryIndex] / FeTwoAt18;
                carry[carryIndex] = (eq[eqIndex] + carry[carryIndex]) / FeTwoAt16;
            }
        }
    }
    
    cout << "ArithExecutor successfully processed " << action.size() << " arith actions" << endl;
}