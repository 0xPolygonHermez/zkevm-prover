#include "arith_helper.hpp"
#include "zklog.hpp"
#include "definitions.hpp"
#include "main_sm/fork_12/main/eval_command.hpp"

namespace fork_12
{

zkresult Arith_verify ( Context &ctx,
                        Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                        MainExecRequired *required)
{
    zkassert(ctx.pZKPC != NULL);
    uint64_t zkPC = *ctx.pZKPC;
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    // Arith instruction: check that A*B + C = D<<256 + op, using scalars (result can be a big number)
    if (ctx.rom.line[zkPC].arithEq0==1 && ctx.rom.line[zkPC].arithEq1==0 && ctx.rom.line[zkPC].arithEq2==0 && ctx.rom.line[zkPC].arithEq3==0 && ctx.rom.line[zkPC].arithEq4==0 && ctx.rom.line[zkPC].arithEq5==0)
    {
        // Convert to scalar
        mpz_class A, B, C, D, op;
        if (!fea2scalar(fr, A, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(A)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, B, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(B)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, C, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(C)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, D, ctx.pols.D0[i], ctx.pols.D1[i], ctx.pols.D2[i], ctx.pols.D3[i], ctx.pols.D4[i], ctx.pols.D5[i], ctx.pols.D6[i], ctx.pols.D7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(D)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, op, op0, op1, op2, op3, op4, op5, op6, op7))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(op)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }

        // Check the condition
        mpz_class left = (A*B) + C;
        mpz_class right = (D<<256) + op;
        if (left != right)
        {
            zklog.error("Arith_verify() Arithmetic does not match: (A*B) + C = " + left.get_str(16) + ", (D<<256) + op = " + right.get_str(16));
            return ZKR_SM_MAIN_ARITH_MISMATCH;
        }

        // Store the arith action to execute it later with the arith SM
        if (!ctx.bProcessBatch)
        {
            // Copy ROM flags into the polynomials
            ctx.pols.arithEq0[i] = fr.one();
        }
        if (required != NULL)
        {
            ArithAction arithAction;
            arithAction.x1 = A;
            arithAction.y1 = B;
            arithAction.x2 = C;
            arithAction.y2 = D;
            arithAction.x3 = 0;
            arithAction.y3 = op;
            arithAction.selEq0 = 1;
            arithAction.selEq1 = 0;
            arithAction.selEq2 = 0;
            arithAction.selEq3 = 0;
            arithAction.selEq4 = 0;
            arithAction.selEq5 = 0;
            arithAction.selEq6 = 0;
            required->Arith.push_back(arithAction);
        }
    }
    // Arithmetic FP2 multiplication
    else if (ctx.rom.line[zkPC].arithEq0==0 && ctx.rom.line[zkPC].arithEq1==0 && ctx.rom.line[zkPC].arithEq2==0 && ctx.rom.line[zkPC].arithEq3==1 && ctx.rom.line[zkPC].arithEq4==0 && ctx.rom.line[zkPC].arithEq5==0)
    {
        // Convert to scalar
        mpz_class x1, y1, x2, y2, x3, y3;
        if (!fea2scalar(fr, x1, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(A)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, y1, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(B)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, x2, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(C)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, y2, ctx.pols.D0[i], ctx.pols.D1[i], ctx.pols.D2[i], ctx.pols.D3[i], ctx.pols.D4[i], ctx.pols.D5[i], ctx.pols.D6[i], ctx.pols.D7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(D)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, x3, ctx.pols.E0[i], ctx.pols.E1[i], ctx.pols.E2[i], ctx.pols.E3[i], ctx.pols.E4[i], ctx.pols.E5[i], ctx.pols.E6[i], ctx.pols.E7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(E)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, y3, op0, op1, op2, op3, op4, op5, op6, op7))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(op)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }

        // EQ5:  x1 * x2 - y1 * y2 = x3
        // EQ6:  y1 * x2 + x1 * y2 = y3

        RawFq::Element x1fe, y1fe, x2fe, y2fe, x3fe, y3fe;
        fq.fromMpz(x1fe, x1.get_mpz_t());
        fq.fromMpz(y1fe, y1.get_mpz_t());
        fq.fromMpz(x2fe, x2.get_mpz_t());
        fq.fromMpz(y2fe, y2.get_mpz_t());
        fq.fromMpz(x3fe, x3.get_mpz_t());
        fq.fromMpz(y3fe, y3.get_mpz_t());

        RawFq::Element _x3fe, _y3fe;
        _x3fe = fq.sub(fq.mul(x1fe, x2fe), fq.mul(y1fe, y2fe));
        _y3fe = fq.add(fq.mul(y1fe, x2fe), fq.mul(x1fe, y2fe));

        bool x3eq = fq.eq(x3fe, _x3fe);
        bool y3eq = fq.eq(y3fe, _y3fe);

        if (!x3eq || !y3eq)
        {
            zklog.error("Arith_verify() Arithmetic FP2 multiplication point does not match: x3=" + fq.toString(x3fe, 16) + " _x3=" + fq.toString(_x3fe, 16) + " y3=" + fq.toString(y3fe, 16) + " _y3=" + fq.toString(_y3fe, 16));
            return ZKR_SM_MAIN_ARITH_MISMATCH;
        }

        // Store the arith action to execute it later with the arith SM
        if (!ctx.bProcessBatch)
        {
            // Copy ROM flags into the polynomials
            ctx.pols.arithEq3[i] = fr.one();
        }
        if (required != NULL)
        {
            ArithAction arithAction;
            arithAction.x1 = x1;
            arithAction.y1 = y1;
            arithAction.x2 = x2;
            arithAction.y2 = y2;
            arithAction.x3 = x3;
            arithAction.y3 = y3;
            arithAction.selEq0 = 0;
            arithAction.selEq1 = 0;
            arithAction.selEq2 = 0;
            arithAction.selEq3 = 0;
            arithAction.selEq4 = 1;
            arithAction.selEq5 = 0;
            arithAction.selEq6 = 0;
            required->Arith.push_back(arithAction);
        }
    }
    // Arithmetic FP2 addition
    else if (ctx.rom.line[zkPC].arithEq0==0 && ctx.rom.line[zkPC].arithEq1==0 && ctx.rom.line[zkPC].arithEq2==0 && ctx.rom.line[zkPC].arithEq3==0 && ctx.rom.line[zkPC].arithEq4==1 && ctx.rom.line[zkPC].arithEq5==0)
    {
        // Convert to scalar
        mpz_class x1, y1, x2, y2, x3, y3;
        if (!fea2scalar(fr, x1, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(A)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, y1, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(B)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, x2, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(C)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, y2, ctx.pols.D0[i], ctx.pols.D1[i], ctx.pols.D2[i], ctx.pols.D3[i], ctx.pols.D4[i], ctx.pols.D5[i], ctx.pols.D6[i], ctx.pols.D7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(D)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, x3, ctx.pols.E0[i], ctx.pols.E1[i], ctx.pols.E2[i], ctx.pols.E3[i], ctx.pols.E4[i], ctx.pols.E5[i], ctx.pols.E6[i], ctx.pols.E7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(E)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, y3, op0, op1, op2, op3, op4, op5, op6, op7))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(op)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }

        // EQ7:  x1 + x2 = x3
        // EQ8:  y1 + y2 = y3

        RawFq::Element x1fe, y1fe, x2fe, y2fe, x3fe, y3fe;
        fq.fromMpz(x1fe, x1.get_mpz_t());
        fq.fromMpz(y1fe, y1.get_mpz_t());
        fq.fromMpz(x2fe, x2.get_mpz_t());
        fq.fromMpz(y2fe, y2.get_mpz_t());
        fq.fromMpz(x3fe, x3.get_mpz_t());
        fq.fromMpz(y3fe, y3.get_mpz_t());

        RawFq::Element _x3fe, _y3fe;
        _x3fe = fq.add(x1fe, x2fe);
        _y3fe = fq.add(y1fe, y2fe);

        bool x3eq = fq.eq(x3fe, _x3fe);
        bool y3eq = fq.eq(y3fe, _y3fe);

        if (!x3eq || !y3eq)
        {
            zklog.error("Arith_verify() Arithmetic FP2 addition point does not match: x3=" + fq.toString(x3fe, 16) + " _x3=" + fq.toString(_x3fe, 16) + " y3=" + fq.toString(y3fe, 16) + " _y3=" + fq.toString(_y3fe, 16));
            return ZKR_SM_MAIN_ARITH_MISMATCH;
        }

        // Store the arith action to execute it later with the arith SM
        if (!ctx.bProcessBatch)
        {
            // Copy ROM flags into the polynomials
            ctx.pols.arithEq4[i] = fr.one();
        }
        if (required != NULL)
        {
            ArithAction arithAction;
            arithAction.x1 = x1;
            arithAction.y1 = y1;
            arithAction.x2 = x2;
            arithAction.y2 = y2;
            arithAction.x3 = x3;
            arithAction.y3 = y3;
            arithAction.selEq0 = 0;
            arithAction.selEq1 = 0;
            arithAction.selEq2 = 0;
            arithAction.selEq3 = 0;
            arithAction.selEq4 = 0;
            arithAction.selEq5 = 1;
            arithAction.selEq6 = 0;
            required->Arith.push_back(arithAction);
        }
    }
    // Arithmetic FP2 subtraction
    else if (ctx.rom.line[zkPC].arithEq0==0 && ctx.rom.line[zkPC].arithEq1==0 && ctx.rom.line[zkPC].arithEq2==0 && ctx.rom.line[zkPC].arithEq3==0 && ctx.rom.line[zkPC].arithEq4==0 && ctx.rom.line[zkPC].arithEq5==1)
    {
        // Convert to scalar
        mpz_class x1, y1, x2, y2, x3, y3;
        if (!fea2scalar(fr, x1, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(A)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, y1, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(B)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, x2, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(C)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, y2, ctx.pols.D0[i], ctx.pols.D1[i], ctx.pols.D2[i], ctx.pols.D3[i], ctx.pols.D4[i], ctx.pols.D5[i], ctx.pols.D6[i], ctx.pols.D7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(D)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, x3, ctx.pols.E0[i], ctx.pols.E1[i], ctx.pols.E2[i], ctx.pols.E3[i], ctx.pols.E4[i], ctx.pols.E5[i], ctx.pols.E6[i], ctx.pols.E7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(E)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, y3, op0, op1, op2, op3, op4, op5, op6, op7))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(op)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }

        // EQ7:  x1 - x2 = x3
        // EQ8:  y1 - y2 = y3

        RawFq::Element x1fe, y1fe, x2fe, y2fe, x3fe, y3fe;
        fq.fromMpz(x1fe, x1.get_mpz_t());
        fq.fromMpz(y1fe, y1.get_mpz_t());
        fq.fromMpz(x2fe, x2.get_mpz_t());
        fq.fromMpz(y2fe, y2.get_mpz_t());
        fq.fromMpz(x3fe, x3.get_mpz_t());
        fq.fromMpz(y3fe, y3.get_mpz_t());

        RawFq::Element _x3fe, _y3fe;
        _x3fe = fq.sub(x1fe, x2fe);
        _y3fe = fq.sub(y1fe, y2fe);

        bool x3eq = fq.eq(x3fe, _x3fe);
        bool y3eq = fq.eq(y3fe, _y3fe);

        if (!x3eq || !y3eq)
        {
            zklog.error("Arith_verify() Arithmetic FP2 subtraction point does not match: x3=" + fq.toString(x3fe, 16) + " _x3=" + fq.toString(_x3fe, 16) + " y3=" + fq.toString(y3fe, 16) + " _y3=" + fq.toString(_y3fe, 16));
            return ZKR_SM_MAIN_ARITH_MISMATCH;
        }

        // Store the arith action to execute it later with the arith SM
        if (!ctx.bProcessBatch)
        {
            // Copy ROM flags into the polynomials
            ctx.pols.arithEq5[i] = fr.one();
        }
        if (required != NULL)
        {
            ArithAction arithAction;
            arithAction.x1 = x1;
            arithAction.y1 = y1;
            arithAction.x2 = x2;
            arithAction.y2 = y2;
            arithAction.x3 = x3;
            arithAction.y3 = y3;
            arithAction.selEq0 = 0;
            arithAction.selEq1 = 0;
            arithAction.selEq2 = 0;
            arithAction.selEq3 = 0;
            arithAction.selEq4 = 0;
            arithAction.selEq5 = 0;
            arithAction.selEq6 = 1;
            required->Arith.push_back(arithAction);
        }
    }
    // Arith instruction: check curve points
    else
    {
        // Convert to scalar
        mpz_class x1, y1, x2, y2, x3, y3;
        if (!fea2scalar(fr, x1, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(A)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, y1, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(B)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, x2, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(C)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, y2, ctx.pols.D0[i], ctx.pols.D1[i], ctx.pols.D2[i], ctx.pols.D3[i], ctx.pols.D4[i], ctx.pols.D5[i], ctx.pols.D6[i], ctx.pols.D7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(D)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, x3, ctx.pols.E0[i], ctx.pols.E1[i], ctx.pols.E2[i], ctx.pols.E3[i], ctx.pols.E4[i], ctx.pols.E5[i], ctx.pols.E6[i], ctx.pols.E7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(E)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, y3, op0, op1, op2, op3, op4, op5, op6, op7))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(op)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }

        // Convert to RawFec::Element
        RawFec::Element fecX1, fecY1, fecX2, fecY2;
        fec.fromMpz(fecX1, x1.get_mpz_t());
        fec.fromMpz(fecY1, y1.get_mpz_t());
        fec.fromMpz(fecX2, x2.get_mpz_t());
        fec.fromMpz(fecY2, y2.get_mpz_t());

        // Check if this is a double operation
        bool dbl = false;
        if (ctx.rom.line[zkPC].arithEq0==0 && ctx.rom.line[zkPC].arithEq1==1 && ctx.rom.line[zkPC].arithEq2==0 && ctx.rom.line[zkPC].arithEq3==0 && ctx.rom.line[zkPC].arithEq4==0 && ctx.rom.line[zkPC].arithEq5==0)
        {
            dbl = false;
        }
        else if (ctx.rom.line[zkPC].arithEq0==0 && ctx.rom.line[zkPC].arithEq1==0 && ctx.rom.line[zkPC].arithEq2==1 && ctx.rom.line[zkPC].arithEq3==0 && ctx.rom.line[zkPC].arithEq4==0 && ctx.rom.line[zkPC].arithEq5==0)
        {
            dbl = true;
        }
        else
        {
            zklog.error("Arith_verify() Invalid arithmetic op");
            exitProcess();
        }

        // Add the elliptic curve points
        RawFec::Element fecX3, fecY3;
        zkresult r = AddPointEc(ctx, dbl, fecX1, fecY1, dbl?fecX1:fecX2, dbl?fecY1:fecY2, fecX3, fecY3);
        if (r != ZKR_SUCCESS)
        {
            zklog.error("Arith_verify() Failed calling AddPointEc() in arith operation");
            return r;
        }

        // Convert to scalar
        mpz_class _x3, _y3;
        fec.toMpz(_x3.get_mpz_t(), fecX3);
        fec.toMpz(_y3.get_mpz_t(), fecY3);

        // Compare
        bool x3eq = (x3 == _x3);
        bool y3eq = (y3 == _y3);

        if (!x3eq || !y3eq)
        {
            zklog.error(string("Arith_verify() Arithmetic curve ") + (dbl?"dbl":"add") + " point does not match" +
                " x1=" + x1.get_str() +
                " y1=" + y1.get_str() +
                " x2=" + x2.get_str() +
                " y2=" + y2.get_str() +
                " x3=" + x3.get_str() +
                " y3=" + y3.get_str() +
                "_x3=" + _x3.get_str() +
                "_y3=" + _y3.get_str());
            return ZKR_SM_MAIN_ARITH_ECRECOVER_MISMATCH;
        }

        if (!ctx.bProcessBatch)
        {
            ctx.pols.arithEq0[i] = fr.fromU64(ctx.rom.line[zkPC].arithEq0);
            ctx.pols.arithEq1[i] = fr.fromU64(ctx.rom.line[zkPC].arithEq1);
            ctx.pols.arithEq2[i] = fr.fromU64(ctx.rom.line[zkPC].arithEq2);
        }
        if (required != NULL)
        {
            // Store the arith action to execute it later with the arith SM
            ArithAction arithAction;
            arithAction.x1 = x1;
            arithAction.y1 = y1;
            arithAction.x2 = dbl ? x1 : x2;
            arithAction.y2 = dbl ? y1 : y2;
            arithAction.x3 = x3;
            arithAction.y3 = y3;
            arithAction.selEq0 = 0;
            arithAction.selEq1 = dbl ? 0 : 1;
            arithAction.selEq2 = dbl ? 1 : 0;
            arithAction.selEq3 = 1;
            arithAction.selEq4 = 0;
            arithAction.selEq5 = 0;
            arithAction.selEq6 = 0;
            required->Arith.push_back(arithAction);
        }
    }

    return ZKR_SUCCESS;
}

} // namespace