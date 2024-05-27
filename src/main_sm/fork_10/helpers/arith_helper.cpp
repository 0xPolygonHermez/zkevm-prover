#include "arith_helper.hpp"
#include "zklog.hpp"
#include "definitions.hpp"

namespace fork_10
{

bool Arith_isFreeInEquation (uint64_t arithEquation)
{
    return (arithEquation == ARITH_MOD) || (arithEquation == ARITH_384_MOD) || (arithEquation == ARITH_256TO384);
}

zkresult Arith_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7)
{
    mpz_class result;

    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].arith == 1);
    uint64_t arithEquation = ctx.rom.line[*ctx.pZKPC].arithEquation;
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;
    mpz_class _a, _b, _c, _d;
    switch(arithEquation)
    {
        case ARITH_MOD:
            if (!fea2scalar(fr, _c, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]))
            {
                zklog.error("Arith_calculate() ARITH_MOD failed calling fea2scalar(C)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (!fea2scalar(fr, _d, ctx.pols.D0[i], ctx.pols.D1[i], ctx.pols.D2[i], ctx.pols.D3[i], ctx.pols.D4[i], ctx.pols.D5[i], ctx.pols.D6[i], ctx.pols.D7[i]))
            {
                zklog.error("Arith_calculate() ARITH_MOD failed calling fea2scalar(D)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
        case ARITH_256TO384:
            if (!fea2scalar(fr, _a, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
            {
                zklog.error("Arith_calculate() ARITH_256TO384/ARITH_MOD failed calling fea2scalar(A)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (!fea2scalar(fr, _b, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
            {
                zklog.error("Arith_calculate() ARITH_256TO384/ARITH_MOD failed calling fea2scalar(A)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            break;
        case ARITH_384_MOD:
            if (!fea384ToScalar(fr, _a, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
            {
                zklog.error("Arith_calculate() ARITH_384_MOD failed calling fea384ToScalar(A)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (!fea384ToScalar(fr, _b, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
            {
                zklog.error("Arith_calculate() ARITH_384_MOD failed calling fea384ToScalar(A)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (!fea384ToScalar(fr, _c, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]))
            {
                zklog.error("Arith_calculate() ARITH_384_MOD failed calling fea384ToScalar(C)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (!fea384ToScalar(fr, _d, ctx.pols.D0[i], ctx.pols.D1[i], ctx.pols.D2[i], ctx.pols.D3[i], ctx.pols.D4[i], ctx.pols.D5[i], ctx.pols.D6[i], ctx.pols.D7[i]))
            {
                zklog.error("Arith_calculate() ARITH_384_MOD failed calling fea384ToScalar(D)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            break;
        default:
            zklog.error("Arith_calculate() invalid arithEquation=" + to_string(arithEquation));
            exitProcess();
    }

    if (arithEquation == ARITH_256TO384)
    {
        if (_b > ScalarMask128)
        {
            zklog.error("Arith_calculate() ARITH_256TO384 invalid b=" + _b.get_str(16) + " > 128 bits");
            return ZKR_SM_MAIN_ARITH_MISMATCH;
        }
        result = _a + (_b << 256);
        scalar2fea384(fr, result, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
        return ZKR_SUCCESS;
    }
    if (_d == 0)
    {
        zklog.error("Arith_calculate() modular arithmetic is undefined when D is zero");
        return ZKR_SM_MAIN_ARITH_MISMATCH;
    }
    result = ((_a * _b) + _c) % _d;
    if (arithEquation == ARITH_MOD)
    {
        scalar2fea(fr, result, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
    }
    else
    {
        scalar2fea384(fr, result, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
    }
    return ZKR_SUCCESS;
}

zkresult Arith_verify ( Context &ctx,
                        Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                        MainExecRequired *required)
{
    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].arith == 1);
    uint64_t arithEquation = ctx.rom.line[*ctx.pZKPC].arithEquation;
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;
    
    uint64_t same12 = 0;
    uint64_t useE = 1;
    uint64_t useCD = 1;
    
    bool is384 = (arithEquation >= ARITH_384_MOD);

    if ((arithEquation == ARITH_BASE) || (arithEquation == ARITH_MOD) || (arithEquation == ARITH_384_MOD))
    {
        useE = 0;

        mpz_class A, B, C, D, op;

        if (is384)
        {
            if (!fea384ToScalar(fr, A, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
            {
                zklog.error("Arith_verify() failed calling fea384ToScalar(A)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (!fea384ToScalar(fr, B, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
            {
                zklog.error("Arith_verify() failed calling fea384ToScalar(A)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (!fea384ToScalar(fr, C, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]))
            {
                zklog.error("Arith_verify() failed calling fea384ToScalar(C)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (!fea384ToScalar(fr, D, ctx.pols.D0[i], ctx.pols.D1[i], ctx.pols.D2[i], ctx.pols.D3[i], ctx.pols.D4[i], ctx.pols.D5[i], ctx.pols.D6[i], ctx.pols.D7[i]))
            {
                zklog.error("Arith_verify() failed calling fea384ToScalar(D)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (!fea384ToScalar(fr, op, op0, op1, op2, op3, op4, op5, op6, op7))
            {
                zklog.error("Arith_verify() failed calling fea384ToScalar(op)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
        }
        else
        {
            if (!fea2scalar(fr, A, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
            {
                zklog.error("Arith_verify() failed calling fea2scalar(A)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (!fea2scalar(fr, B, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
            {
                zklog.error("Arith_verify() failed calling fea2scalar(A)");
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
        }

        mpz_class left, right;
        if (arithEquation == ARITH_BASE)
        {
            left = A*B + C;
            right = (D<<256) + op;
        }
        else
        {
            if (D == 0)
            {
                zklog.error("Arith_verify() Modular arithmetic is undefined when D is zero arithEquation=" + arith2string(arithEquation));
                return ZKR_SM_MAIN_ARITH_MISMATCH;
            }
            left = (A*B + C)%D;
            right = op;
        }

        if (left != right)
        {
            zklog.error("Arith_verify() equation=" + arith2string(arithEquation) + " does not match, A=" + A.get_str(16) + " B=" + B.get_str(16) + " C=" + C.get_str(16) + " D=" + D.get_str(16) + " op=" + op.get_str(16) + " left=" + left.get_str(16) + " right=" + right.get_str(16));
            return ZKR_SM_MAIN_ARITH_MISMATCH;
        }

#ifdef USE_REQUIRED
        if (required != NULL)
        {
            ArithAction arithAction;

            arithAction.x1[0] = ctx.pols.A0[i];
            arithAction.x1[1] = ctx.pols.A1[i];
            arithAction.x1[2] = ctx.pols.A2[i];
            arithAction.x1[3] = ctx.pols.A3[i];
            arithAction.x1[4] = ctx.pols.A4[i];
            arithAction.x1[5] = ctx.pols.A5[i];
            arithAction.x1[6] = ctx.pols.A6[i];
            arithAction.x1[7] = ctx.pols.A7[i];

            arithAction.y1[0] = ctx.pols.B0[i];
            arithAction.y1[1] = ctx.pols.B1[i];
            arithAction.y1[2] = ctx.pols.B2[i];
            arithAction.y1[3] = ctx.pols.B3[i];
            arithAction.y1[4] = ctx.pols.B4[i];
            arithAction.y1[5] = ctx.pols.B5[i];
            arithAction.y1[6] = ctx.pols.B6[i];
            arithAction.y1[7] = ctx.pols.B7[i];

            arithAction.x2[0] = ctx.pols.C0[i];
            arithAction.x2[1] = ctx.pols.C1[i];
            arithAction.x2[2] = ctx.pols.C2[i];
            arithAction.x2[3] = ctx.pols.C3[i];
            arithAction.x2[4] = ctx.pols.C4[i];
            arithAction.x2[5] = ctx.pols.C5[i];
            arithAction.x2[6] = ctx.pols.C6[i];
            arithAction.x2[7] = ctx.pols.C7[i];

            arithAction.y2[0] = ctx.pols.D0[i];
            arithAction.y2[1] = ctx.pols.D1[i];
            arithAction.y2[2] = ctx.pols.D2[i];
            arithAction.y2[3] = ctx.pols.D3[i];
            arithAction.y2[4] = ctx.pols.D4[i];
            arithAction.y2[5] = ctx.pols.D5[i];
            arithAction.y2[6] = ctx.pols.D6[i];
            arithAction.y2[7] = ctx.pols.D7[i];

            arithAction.x3[0] = fr.zero();
            arithAction.x3[1] = fr.zero();
            arithAction.x3[2] = fr.zero();
            arithAction.x3[3] = fr.zero();
            arithAction.x3[4] = fr.zero();
            arithAction.x3[5] = fr.zero();
            arithAction.x3[6] = fr.zero();
            arithAction.x3[7] = fr.zero();

            arithAction.y3[0] = op0;
            arithAction.y3[1] = op1;
            arithAction.y3[2] = op2;
            arithAction.y3[3] = op3;
            arithAction.y3[4] = op4;
            arithAction.y3[5] = op5;
            arithAction.y3[6] = op6;
            arithAction.y3[7] = op7;

            arithAction.equation = arithEquation;

            required->Arith.push_back(arithAction);
        }
#endif

    }
    else if (((arithEquation >= ARITH_ECADD_DIFFERENT) && (arithEquation <= ARITH_BN254_SUBFP2)) ||
             ((arithEquation >= ARITH_BLS12381_MULFP2) && (arithEquation <= ARITH_BLS12381_SUBFP2)))
    {   
        const bool dbl = (arithEquation == ARITH_ECADD_SAME);
        mpz_class x1, y1, x2, y2, x3, y3;
        if (is384)
        {
            if (!fea384ToScalar(fr, x1, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
            {
                zklog.error("Arith_verify() failed calling fea384ToScalar(A)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (!fea384ToScalar(fr, y1, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
            {
                zklog.error("Arith_verify() failed calling fea384ToScalar(A)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (dbl)
            {
                x2 = x1;
                y2 = y1;
            }
            else
            {
                if (!fea384ToScalar(fr, x2, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]))
                {
                    zklog.error("Arith_verify() failed calling fea384ToScalar(C)");
                    return ZKR_SM_MAIN_FEA2SCALAR;
                }
                if (!fea384ToScalar(fr, y2, ctx.pols.D0[i], ctx.pols.D1[i], ctx.pols.D2[i], ctx.pols.D3[i], ctx.pols.D4[i], ctx.pols.D5[i], ctx.pols.D6[i], ctx.pols.D7[i]))
                {
                    zklog.error("Arith_verify() failed calling fea384ToScalar(D)");
                    return ZKR_SM_MAIN_FEA2SCALAR;
                }
            }
            if (!fea384ToScalar(fr, x3, ctx.pols.E0[i], ctx.pols.E1[i], ctx.pols.E2[i], ctx.pols.E3[i], ctx.pols.E4[i], ctx.pols.E5[i], ctx.pols.E6[i], ctx.pols.E7[i]))
            {
                zklog.error("Arith_verify() failed calling fea384ToScalar(A)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (!fea384ToScalar(fr, y3, op0, op1, op2, op3, op4, op5, op6, op7))
            {
                zklog.error("Arith_verify() failed calling fea384ToScalar(op)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
        }
        else
        {
            if (!fea2scalar(fr, x1, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
            {
                zklog.error("Arith_verify() failed calling fea2scalar(A)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (!fea2scalar(fr, y1, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
            {
                zklog.error("Arith_verify() failed calling fea2scalar(A)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (dbl)
            {
                x2 = x1;
                y2 = y1;
            }
            else
            {
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
            }
            if (!fea2scalar(fr, x3, ctx.pols.E0[i], ctx.pols.E1[i], ctx.pols.E2[i], ctx.pols.E3[i], ctx.pols.E4[i], ctx.pols.E5[i], ctx.pols.E6[i], ctx.pols.E7[i]))
            {
                zklog.error("Arith_verify() failed calling fea2scalar(A)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            if (!fea2scalar(fr, y3, op0, op1, op2, op3, op4, op5, op6, op7))
            {
                zklog.error("Arith_verify() failed calling fea2scalar(op)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
        }

        RawFec::Element s;
        if ((arithEquation == ARITH_ECADD_DIFFERENT) || (arithEquation == ARITH_ECADD_SAME))
        {
            if (dbl)
            {
                // Convert to RawFec::Element
                RawFec::Element fecX1, fecY1, fecX2, fecY2;
                fec.fromMpz(fecX1, x1.get_mpz_t());
                fec.fromMpz(fecY1, y1.get_mpz_t());
                fec.fromMpz(fecX2, x2.get_mpz_t());
                fec.fromMpz(fecY2, y2.get_mpz_t());

                // Division by zero must be managed by ROM before call ARITH
                RawFec::Element divisor;
                divisor = fec.add(fecY1, fecY1);

                same12 = 1;
                useCD = 0;

                if (fec.isZero(divisor))
                {
                    zklog.info("Arith_verify() Invalid arithmetic op, DivisionByZero arithEquation=" + arith2string(arithEquation));
                    return ZKR_SM_MAIN_ARITH_ECRECOVER_DIVIDE_BY_ZERO;
                }

                RawFec::Element fecThree;
                fec.fromUI(fecThree, 3);
                fec.div(s, fec.mul(fecThree, fec.mul(fecX1, fecX1)), divisor);
            }
            else
            {
                // Convert to RawFec::Element
                RawFec::Element fecX1, fecY1, fecX2, fecY2;
                fec.fromMpz(fecX1, x1.get_mpz_t());
                fec.fromMpz(fecY1, y1.get_mpz_t());
                fec.fromMpz(fecX2, x2.get_mpz_t());
                fec.fromMpz(fecY2, y2.get_mpz_t());

                // Division by zero must be managed by ROM before call ARITH
                RawFec::Element deltaX = fec.sub(fecX2, fecX1);
                if (fec.isZero(deltaX))
                {
                    zklog.info("Arith_verify() Invalid arithmetic op, DivisionByZero arithEquation=" + arith2string(arithEquation));
                    return ZKR_SM_MAIN_ARITH_ECRECOVER_DIVIDE_BY_ZERO;
                }
                fec.div(s, fec.sub(fecY2, fecY1), deltaX);
            }
        }

        mpz_class _x3,_y3;

        switch (arithEquation)
        {
            case ARITH_ECADD_DIFFERENT:
            {
                // Convert to RawFec::Element
                RawFec::Element fecX1, fecY1, fecX2, fecY2;
                fec.fromMpz(fecX1, x1.get_mpz_t());
                fec.fromMpz(fecY1, y1.get_mpz_t());
                fec.fromMpz(fecX2, x2.get_mpz_t());
                fec.fromMpz(fecY2, y2.get_mpz_t());

                RawFec::Element _fecX3;
                _fecX3 = fec.sub( fec.mul(s, s), fec.add(fecX1, fecX2) );
                fec.toMpz(_x3.get_mpz_t(), _fecX3);

                RawFec::Element _fecY3;
                _fecY3 = fec.sub( fec.mul(s, fec.sub(fecX1, _fecX3)), fecY1);
                fec.toMpz(_y3.get_mpz_t(), _fecY3);
                
                break;
            }
            case ARITH_ECADD_SAME:
            {
                // Convert to RawFec::Element
                RawFec::Element fecX1, fecY1;
                fec.fromMpz(fecX1, x1.get_mpz_t());
                fec.fromMpz(fecY1, y1.get_mpz_t());

                RawFec::Element _fecX3;
                _fecX3 = fec.sub( fec.mul(s, s), fec.add(fecX1, fecX1) );
                fec.toMpz(_x3.get_mpz_t(), _fecX3);

                RawFec::Element _fecY3;
                _fecY3 = fec.sub( fec.mul(s, fec.sub(fecX1, _fecX3)), fecY1);
                fec.toMpz(_y3.get_mpz_t(), _fecY3);

                break;
            }
            case ARITH_BN254_MULFP2:
            case ARITH_BLS12381_MULFP2:
            {
                if (is384) // BLS12_381_384
                {
                    // Convert to RawFec::Element
                    RawBLS12_381_384::Element fpX1, fpY1, fpX2, fpY2;
                    bls12_381_384.fromMpz(fpX1, x1.get_mpz_t());
                    bls12_381_384.fromMpz(fpY1, y1.get_mpz_t());
                    bls12_381_384.fromMpz(fpX2, x2.get_mpz_t());
                    bls12_381_384.fromMpz(fpY2, y2.get_mpz_t());

                    RawBLS12_381_384::Element _fpX3;
                    _fpX3 = bls12_381_384.sub( bls12_381_384.mul(fpX1, fpX2), bls12_381_384.mul(fpY1, fpY2));
                    bls12_381_384.toMpz(_x3.get_mpz_t(), _fpX3);

                    RawBLS12_381_384::Element _fpY3;
                    _fpY3 = bls12_381_384.add( bls12_381_384.mul(fpY1, fpX2), bls12_381_384.mul(fpX1, fpY2));
                    bls12_381_384.toMpz(_y3.get_mpz_t(), _fpY3);
                }
                else // BN256
                {
                    // Convert to RawFec::Element
                    RawFq::Element fpX1, fpY1, fpX2, fpY2;
                    bn254.fromMpz(fpX1, x1.get_mpz_t());
                    bn254.fromMpz(fpY1, y1.get_mpz_t());
                    bn254.fromMpz(fpX2, x2.get_mpz_t());
                    bn254.fromMpz(fpY2, y2.get_mpz_t());

                    RawFq::Element _fpX3;
                    _fpX3 = bn254.sub( bn254.mul(fpX1, fpX2), bn254.mul(fpY1, fpY2)) ;
                    bn254.toMpz(_x3.get_mpz_t(), _fpX3);

                    RawFq::Element _fpY3;
                    _fpY3 = bn254.add( bn254.mul(fpY1, fpX2), bn254.mul(fpX1, fpY2) );
                    bn254.toMpz(_y3.get_mpz_t(), _fpY3);                   
                }
                break;
            }
            case ARITH_BN254_ADDFP2:
            case ARITH_BLS12381_ADDFP2:
            {
                if (is384) // BLS12_381_384
                {
                    // Convert to RawFec::Element
                    RawBLS12_381_384::Element fpX1, fpY1, fpX2, fpY2;
                    bls12_381_384.fromMpz(fpX1, x1.get_mpz_t());
                    bls12_381_384.fromMpz(fpY1, y1.get_mpz_t());
                    bls12_381_384.fromMpz(fpX2, x2.get_mpz_t());
                    bls12_381_384.fromMpz(fpY2, y2.get_mpz_t());

                    RawBLS12_381_384::Element _fpX3;
                    _fpX3 = bls12_381_384.add(fpX1, fpX2);
                    bls12_381_384.toMpz(_x3.get_mpz_t(), _fpX3);

                    RawBLS12_381_384::Element _fpY3;
                    _fpY3 = bls12_381_384.add(fpY1, fpY2);
                    bls12_381_384.toMpz(_y3.get_mpz_t(), _fpY3);
                }
                else // BN256
                {
                    // Convert to RawFec::Element
                    RawFq::Element fpX1, fpY1, fpX2, fpY2;
                    bn254.fromMpz(fpX1, x1.get_mpz_t());
                    bn254.fromMpz(fpY1, y1.get_mpz_t());
                    bn254.fromMpz(fpX2, x2.get_mpz_t());
                    bn254.fromMpz(fpY2, y2.get_mpz_t());

                    RawFq::Element _fpX3;
                    _fpX3 = bn254.add(fpX1, fpX2);
                    bn254.toMpz(_x3.get_mpz_t(), _fpX3);

                    RawFq::Element _fpY3;
                    _fpY3 = bn254.add(fpY1, fpY2);
                    bn254.toMpz(_y3.get_mpz_t(), _fpY3);
                }
                break;
            }
            case ARITH_BN254_SUBFP2:
            case ARITH_BLS12381_SUBFP2:
            {
                if (is384) // BLS12_381_384
                {
                    // Convert to RawFec::Element
                    RawBLS12_381_384::Element fpX1, fpY1, fpX2, fpY2;
                    bls12_381_384.fromMpz(fpX1, x1.get_mpz_t());
                    bls12_381_384.fromMpz(fpY1, y1.get_mpz_t());
                    bls12_381_384.fromMpz(fpX2, x2.get_mpz_t());
                    bls12_381_384.fromMpz(fpY2, y2.get_mpz_t());

                    RawBLS12_381_384::Element _fpX3;
                    _fpX3 = bls12_381_384.sub(fpX1, fpX2);
                    bls12_381_384.toMpz(_x3.get_mpz_t(), _fpX3);

                    RawBLS12_381_384::Element _fpY3;
                    _fpY3 = bls12_381_384.sub(fpY1, fpY2);
                    bls12_381_384.toMpz(_y3.get_mpz_t(), _fpY3);
                }
                else // BN256
                {
                    // Convert to RawFec::Element
                    RawFq::Element fpX1, fpY1, fpX2, fpY2;
                    bn254.fromMpz(fpX1, x1.get_mpz_t());
                    bn254.fromMpz(fpY1, y1.get_mpz_t());
                    bn254.fromMpz(fpX2, x2.get_mpz_t());
                    bn254.fromMpz(fpY2, y2.get_mpz_t());

                    RawFq::Element _fpX3;
                    _fpX3 = bn254.sub(fpX1, fpX2);
                    bn254.toMpz(_x3.get_mpz_t(), _fpX3);

                    RawFq::Element _fpY3;
                    _fpY3 = bn254.sub(fpY1, fpY2);
                    bn254.toMpz(_y3.get_mpz_t(), _fpY3);
                }
                break;
            }
            default:
            {
                zklog.error("Arith_verify() invalid arithEquation=" + arith2string(arithEquation));
                exitProcess();
            }
        }

        bool x3eq = (x3 == _x3);
        bool y3eq = (y3 == _y3);

        if (!x3eq || !y3eq)
        {
            zklog.error("Arith_verify() Arithmetic point does not match arithEquation=" + arith2string(arithEquation) +
                " x3=" + x3.get_str(16) +
                " _x3=" + _x3.get_str(16) +
                " y3=" + y3.get_str(16) +
                " _y3=" + _y3.get_str(16));
            return ZKR_SM_MAIN_ARITH_MISMATCH;
        }

#ifdef USE_REQUIRED
        if (required != NULL)
        {
            ArithAction arithAction;

            arithAction.x1[0] = ctx.pols.A0[i];
            arithAction.x1[1] = ctx.pols.A1[i];
            arithAction.x1[2] = ctx.pols.A2[i];
            arithAction.x1[3] = ctx.pols.A3[i];
            arithAction.x1[4] = ctx.pols.A4[i];
            arithAction.x1[5] = ctx.pols.A5[i];
            arithAction.x1[6] = ctx.pols.A6[i];
            arithAction.x1[7] = ctx.pols.A7[i];

            arithAction.y1[0] = ctx.pols.B0[i];
            arithAction.y1[1] = ctx.pols.B1[i];
            arithAction.y1[2] = ctx.pols.B2[i];
            arithAction.y1[3] = ctx.pols.B3[i];
            arithAction.y1[4] = ctx.pols.B4[i];
            arithAction.y1[5] = ctx.pols.B5[i];
            arithAction.y1[6] = ctx.pols.B6[i];
            arithAction.y1[7] = ctx.pols.B7[i];

            arithAction.x2[0] = dbl ? ctx.pols.A0[i] : ctx.pols.C0[i];
            arithAction.x2[1] = dbl ? ctx.pols.A1[i] : ctx.pols.C1[i];
            arithAction.x2[2] = dbl ? ctx.pols.A2[i] : ctx.pols.C2[i];
            arithAction.x2[3] = dbl ? ctx.pols.A3[i] : ctx.pols.C3[i];
            arithAction.x2[4] = dbl ? ctx.pols.A4[i] : ctx.pols.C4[i];
            arithAction.x2[5] = dbl ? ctx.pols.A5[i] : ctx.pols.C5[i];
            arithAction.x2[6] = dbl ? ctx.pols.A6[i] : ctx.pols.C6[i];
            arithAction.x2[7] = dbl ? ctx.pols.A7[i] : ctx.pols.C7[i];

            arithAction.y2[0] = dbl ? ctx.pols.B0[i] : ctx.pols.D0[i];
            arithAction.y2[1] = dbl ? ctx.pols.B1[i] : ctx.pols.D1[i];
            arithAction.y2[2] = dbl ? ctx.pols.B2[i] : ctx.pols.D2[i];
            arithAction.y2[3] = dbl ? ctx.pols.B3[i] : ctx.pols.D3[i];
            arithAction.y2[4] = dbl ? ctx.pols.B4[i] : ctx.pols.D4[i];
            arithAction.y2[5] = dbl ? ctx.pols.B5[i] : ctx.pols.D5[i];
            arithAction.y2[6] = dbl ? ctx.pols.B6[i] : ctx.pols.D6[i];
            arithAction.y2[7] = dbl ? ctx.pols.B7[i] : ctx.pols.D7[i];

            arithAction.x3[0] = ctx.pols.E0[i];
            arithAction.x3[1] = ctx.pols.E1[i];
            arithAction.x3[2] = ctx.pols.E2[i];
            arithAction.x3[3] = ctx.pols.E3[i];
            arithAction.x3[4] = ctx.pols.E4[i];
            arithAction.x3[5] = ctx.pols.E5[i];
            arithAction.x3[6] = ctx.pols.E6[i];
            arithAction.x3[7] = ctx.pols.E7[i];

            arithAction.y3[0] = op0;
            arithAction.y3[1] = op1;
            arithAction.y3[2] = op2;
            arithAction.y3[3] = op3;
            arithAction.y3[4] = op4;
            arithAction.y3[5] = op5;
            arithAction.y3[6] = op6;
            arithAction.y3[7] = op7;

            arithAction.equation = arithEquation;

            required->Arith.push_back(arithAction);
        }
#endif

    } 
    else if (arithEquation == ARITH_256TO384)
    {
        mpz_class A, B, op;
        if (!fea2scalar(fr, A, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(A)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea2scalar(fr, B, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
        {
            zklog.error("Arith_verify() failed calling fea2scalar(A)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        if (!fea384ToScalar(fr, op, op0, op1, op2, op3, op4, op5, op6, op7))
        {
            zklog.error("Arith_verify() failed calling fea384ToScalar(op)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }

        if (B > ScalarMask128)
        {
            zklog.error("Arith_verify() ARITH_256TO384 B is too big B=" + B.get_str(16) + " > 128 bits ");
            return ZKR_SM_MAIN_ARITH_MISMATCH;
        }

        mpz_class expected = A + (B << 256);

        if (op != expected)
        {
            zklog.error("Arith_verify() Arithmetic ARITH_256TO384 point does not match op=" + op.get_str(16) + " expected=" + expected.get_str(16) + " A=" + A.get_str(16) + " B=" + B.get_str(16));
            return ZKR_SM_MAIN_ARITH_MISMATCH;
        }

        useCD = 0;
        useE = 0;

#ifdef USE_REQUIRED
        if (required != NULL)
        {
            ArithAction arithAction;

            arithAction.x1[0] = ctx.pols.A0[i];
            arithAction.x1[1] = ctx.pols.A1[i];
            arithAction.x1[2] = ctx.pols.A2[i];
            arithAction.x1[3] = ctx.pols.A3[i];
            arithAction.x1[4] = ctx.pols.A4[i];
            arithAction.x1[5] = ctx.pols.A5[i];
            arithAction.x1[6] = ctx.pols.A6[i];
            arithAction.x1[7] = ctx.pols.A7[i];

            arithAction.y1[0] = ctx.pols.B0[i];
            arithAction.y1[1] = ctx.pols.B1[i];
            arithAction.y1[2] = ctx.pols.B2[i];
            arithAction.y1[3] = ctx.pols.B3[i];
            arithAction.y1[4] = ctx.pols.B4[i];
            arithAction.y1[5] = ctx.pols.B5[i];
            arithAction.y1[6] = ctx.pols.B6[i];
            arithAction.y1[7] = ctx.pols.B7[i];

            arithAction.x2[0] = fr.zero();
            arithAction.x2[1] = fr.zero();
            arithAction.x2[2] = fr.zero();
            arithAction.x2[3] = fr.zero();
            arithAction.x2[4] = fr.zero();
            arithAction.x2[5] = fr.zero();
            arithAction.x2[6] = fr.zero();
            arithAction.x2[7] = fr.zero();

            arithAction.y2[0] = fr.zero();
            arithAction.y2[1] = fr.zero();
            arithAction.y2[2] = fr.zero();
            arithAction.y2[3] = fr.zero();
            arithAction.y2[4] = fr.zero();
            arithAction.y2[5] = fr.zero();
            arithAction.y2[6] = fr.zero();
            arithAction.y2[7] = fr.zero();

            arithAction.x3[0] = fr.zero();
            arithAction.x3[1] = fr.zero();
            arithAction.x3[2] = fr.zero();
            arithAction.x3[3] = fr.zero();
            arithAction.x3[4] = fr.zero();
            arithAction.x3[5] = fr.zero();
            arithAction.x3[6] = fr.zero();
            arithAction.x3[7] = fr.zero();

            arithAction.y3[0] = op0;
            arithAction.y3[1] = op1;
            arithAction.y3[2] = op2;
            arithAction.y3[3] = op3;
            arithAction.y3[4] = op4;
            arithAction.y3[5] = op5;
            arithAction.y3[6] = op6;
            arithAction.y3[7] = op7;

            arithAction.equation = arithEquation;
            
            required->Arith.push_back(arithAction);
        }
#endif
    }
    else
    {
        zklog.error("Arith_verify() invalid arithEquation=" + arith2string(arithEquation));
        exitProcess();
    }

    if (required != NULL)
    {
        ctx.pols.arith[i] = fr.one();
        ctx.pols.arithEquation[i] = fr.fromU64(arithEquation);
        ctx.pols.arithSame12[i] = fr.fromU64(same12);
        ctx.pols.arithUseE[i] = fr.fromU64(useE);
        ctx.pols.arithUseCD[i] = fr.fromU64(useCD);
    }

    return ZKR_SUCCESS;
}

} // namespace