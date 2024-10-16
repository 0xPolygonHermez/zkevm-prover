#include "arith_helper.hpp"
#include "zklog.hpp"
#include "definitions.hpp"
#include "main_sm/fork_13/main/eval_command.hpp"

namespace fork_13
{

zkresult Arith_verify ( Context &ctx,
                        Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                        MainExecRequired *required)
{
    zkassert(ctx.pZKPC != NULL);
    uint64_t zkPC = *ctx.pZKPC;
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;
    //zkassert(ctx.pEvaluation != NULL);
    //uint64_t evaluation = *ctx.pEvaluation;

    uint64_t arithEq = ctx.rom.line[zkPC].arithEq;

    //zklog.info("Arith_verify() arithEq=" + to_string(arithEq));

    // Write polynomials
    if (!ctx.bProcessBatch)
    {
        ctx.pols.arith[i] = fr.one();
        if ((arithEq == 3) || (arithEq == 8))
        {
            ctx.pols.arithSame12[i] = fr.one();
        }
        if (arithEq != 1)
        {
            ctx.pols.arithUseE[i] = fr.one();
        }
        ctx.pols.arithEq[i] = fr.fromU64(arithEq);
    }

    // Arith instruction: check that A*B + C = D<<256 + op, using scalars (result can be a big number)
    if (arithEq == 1)
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
        if (required != NULL)
        {
            ArithAction arithAction;
            arithAction.x1 = A;
            arithAction.y1 = B;
            arithAction.x2 = C;
            arithAction.y2 = D;
            arithAction.x3 = 0;
            arithAction.y3 = op;
            arithAction.arithEq = arithEq;
            required->Arith.push_back(arithAction);
        }
    }
    // Arithmetic FP2 multiplication
    else if (arithEq == 4)
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
        if (required != NULL)
        {
            ArithAction arithAction;
            arithAction.x1 = x1;
            arithAction.y1 = y1;
            arithAction.x2 = x2;
            arithAction.y2 = y2;
            arithAction.x3 = x3;
            arithAction.y3 = y3;
            arithAction.arithEq = arithEq;
            required->Arith.push_back(arithAction);
        }
    }
    // Arithmetic FP2 addition
    else if (arithEq == 5)
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
        if (required != NULL)
        {
            ArithAction arithAction;
            arithAction.x1 = x1;
            arithAction.y1 = y1;
            arithAction.x2 = x2;
            arithAction.y2 = y2;
            arithAction.x3 = x3;
            arithAction.y3 = y3;
            arithAction.arithEq = arithEq;
            required->Arith.push_back(arithAction);
        }
    }
    // Arithmetic FP2 subtraction
    else if (arithEq == 6)
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
        if (required != NULL)
        {
            ArithAction arithAction;
            arithAction.x1 = x1;
            arithAction.y1 = y1;
            arithAction.x2 = x2;
            arithAction.y2 = y2;
            arithAction.x3 = x3;
            arithAction.y3 = y3;
            arithAction.arithEq = arithEq;
            required->Arith.push_back(arithAction);
        }
    }
    // Arith instruction: check curve points
    else if ((arithEq == 2) || (arithEq == 3) || (arithEq == 7) || (arithEq == 8))
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

        // Check if this is a double operation
        bool dbl = false;
        if ((arithEq == 2) || (arithEq == 7))
        {
            dbl = false;
        }
        else if ((arithEq == 3) || (arithEq == 8))
        {
            dbl = true;
        }
        else
        {
            zklog.error("Arith_verify() Invalid arithmetic op arithEq=" + to_string(arithEq));
            exitProcess();
        }

        // Resulting third point coordinates
        mpz_class _x3, _y3;

        // Secp256r1 finite field case
        if ((arithEq == 7) || (arithEq == 8))
        {
            // Get field element versions of the point coordinates
            RawpSecp256r1::Element x1fe;
            pSecp256r1.fromMpz(x1fe, x1.get_mpz_t());
            RawpSecp256r1::Element y1fe;
            pSecp256r1.fromMpz(y1fe, y1.get_mpz_t());
            RawpSecp256r1::Element x2fe;
            if (!dbl) pSecp256r1.fromMpz(x2fe, x2.get_mpz_t());
            RawpSecp256r1::Element y2fe;
            if (!dbl) pSecp256r1.fromMpz(y2fe, y2.get_mpz_t());
            RawpSecp256r1::Element x3fe;
            RawpSecp256r1::Element y3fe;

            RawpSecp256r1::Element s;

            if (dbl)
            {
                // Calculate s divisor
                // Division by zero must be managed by ROM before call ARITH
                RawpSecp256r1::Element divisor;
                divisor = pSecp256r1.add(y1fe, y1fe); // divisor = 2*y1
                if (pSecp256r1.isZero(divisor))
                {
                    zklog.error("Arith_verify() got divisor=0");
                    exitProcess();
                }

                // Calculate s, based on arith equation
                RawpSecp256r1::Element aux1;
                aux1 = pSecp256r1.mul(x1fe, x1fe); // aux1 = x1*x1
                RawpSecp256r1::Element aux2;
                aux2 = pSecp256r1.mul(3, aux1); // aux2 = 3*x1*x1
                RawpSecp256r1::Element aux3;
                aux3 = pSecp256r1.add(aux2, aSecp256r1_fe); // aux3 = 3*x1*x1 + a
                pSecp256r1.div(s, aux3, divisor); // s = (3*x1*x1 + a) / divisor = (3*x1*x1 + a) / 2*y1
            }
            else
            {
                // Calculate s divisor
                // Division by zero must be managed by ROM before call ARITH
                RawpSecp256r1::Element deltaX;
                deltaX = pSecp256r1.sub(x2fe, x1fe); // deltaX = x2 - x1
                if (pSecp256r1.isZero(deltaX))
                {
                    zklog.error("Arith_verify() got deltaX=0");
                    exitProcess();
                }

                // Calculate s
                RawpSecp256r1::Element aux1;
                aux1 = pSecp256r1.sub(y2fe, y1fe); // aux1 = y2 - y1
                pSecp256r1.div(s, aux1, deltaX); // s = (y2 - y1) / deltaX = (y2 - y1) / (x2 - x1)
            }

            // Calculate x3 = s*s - (x1 + x1|x2)
            RawpSecp256r1::Element aux1;
            aux1 = pSecp256r1.add(x1fe, dbl ? x1fe : x2fe); // aux1 = x1 + x1|x2
            RawpSecp256r1::Element aux2;
            aux2 = pSecp256r1.mul(s, s); // aux2 = s*s
            x3fe = pSecp256r1.sub(aux2, aux1); // x3 = s*s - (x1 + x1|x2)
            pSecp256r1.toMpz(_x3.get_mpz_t(), x3fe); // convert x3 to scalar _x3

            // Calculate y3 = s(x1 - x3) - y1
            aux1 = pSecp256r1.sub(x1fe, x3fe); // aux1 = x1 - x3
            aux2 = pSecp256r1.mul(s, aux1); // aux2 = s(x1 - x3)
            y3fe = pSecp256r1.sub(aux2, y1fe); // y3 = s(x1 - x3) - y1
            pSecp256r1.toMpz(_y3.get_mpz_t(), y3fe); // convert y3 to scalar _y3
        }
        // Secp256k1p finite field case
        else
        {
            // Get field element versions of the point coordinates
            RawFec::Element x1fe;
            Secp256k1p.fromMpz(x1fe, x1.get_mpz_t());
            RawFec::Element y1fe;
            Secp256k1p.fromMpz(y1fe, y1.get_mpz_t());
            RawFec::Element x2fe;
            Secp256k1p.fromMpz(x2fe, x2.get_mpz_t());
            RawFec::Element y2fe;
            Secp256k1p.fromMpz(y2fe, y2.get_mpz_t());
            RawFec::Element x3fe;
            RawFec::Element y3fe;

            RawFec::Element s;

            if (dbl)
            {
                // Calculate s divisor
                // Division by zero must be managed by ROM before calling ARITH
                RawFec::Element divisor;
                divisor = Secp256k1p.add(y1fe, y1fe); // divisor = 2*y1
                if (Secp256k1p.isZero(divisor))
                {
                    zklog.error("Arith_verify() got divisor=0");
                    exitProcess();
                }

                // Calculate s, based on arith equation
                RawFec::Element aux1;
                aux1 = Secp256k1p.mul(x1fe, x1fe); // aux1 = x1*x1
                RawFec::Element aux2;
                aux2 = Secp256k1p.mul(3, aux1); // aux2 = 3*x1*x1
                Secp256k1p.div(s, aux2, divisor); // s = 3*x1*x1 / divisor = 3*x1*x1 / 2*y1
            }
            else
            {
                // Calculate s divisor
                // Division by zero must be managed by ROM before call ARITH
                RawFec::Element deltaX;
                deltaX = Secp256k1p.sub(x2fe, x1fe); // deltaX = x2 - x1
                if (Secp256k1p.isZero(deltaX))
                {
                    zklog.error("Arith_verify() got deltaX=0");
                    exitProcess();
                }

                // Calculate s
                RawFec::Element aux1;
                aux1 = Secp256k1p.sub(y2fe, y1fe); // aux1 = y2 - y1
                Secp256k1p.div(s, aux1, deltaX); // s = (y2 - y1) / (x2 - x1)
            }

            // Calculate x3
            RawFec::Element aux1;
            aux1 = Secp256k1p.add(x1fe, dbl ? x1fe : x2fe); // aux1 = x1 + x1|x2
            RawFec::Element aux2;
            aux2 = Secp256k1p.mul(s, s); // aux2 = s*s
            x3fe = Secp256k1p.sub(aux2, aux1); // x3 = s*s - (x1 + x1|x2)
            Secp256k1p.toMpz(_x3.get_mpz_t(), x3fe); // convert x3 to scalar _x3

            // Calculate y3
            aux1 = Secp256k1p.sub(x1fe, x3fe); // aux1 = x1 - x3
            aux2 = Secp256k1p.mul(s, aux1); // aux2 = s(x1 - x3)
            y3fe = Secp256k1p.sub(aux2, y1fe); // y3 = s(x1 - x3) - y1
            Secp256k1p.toMpz(_y3.get_mpz_t(), y3fe); // convert y3 to scalar _y3
        }

        // Compare expected vs. calculated results
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
                " _x3=" + _x3.get_str() +
                " _y3=" + _y3.get_str());
            return ZKR_SM_MAIN_ARITH_ECRECOVER_MISMATCH;
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
            arithAction.arithEq = arithEq;
            required->Arith.push_back(arithAction);
        }
    }

    return ZKR_SUCCESS;
}

} // namespace