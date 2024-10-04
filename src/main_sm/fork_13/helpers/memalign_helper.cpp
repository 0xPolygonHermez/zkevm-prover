#include "memalign_helper.hpp"
#include "zklog.hpp"
#include "definitions.hpp"

namespace fork_13
{

zkresult Memalign_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7)
{
    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].memAlignRD == 1);
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    mpz_class m0;
    if (!fea2scalar(fr, m0, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
    {
        zklog.error("Memalign_calculate() Failed calling fea2scalar(pols.A)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    mpz_class m1;
    if (!fea2scalar(fr, m1, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
    {
        zklog.error("Memalign_calculate() Failed calling fea2scalar(pols.B)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    mpz_class offsetScalar;
    if (!fea2scalar(fr, offsetScalar, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]))
    {
        zklog.error("Memalign_calculate() Failed calling fea2scalar(pols.C)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }

    if (offsetScalar<0 || offsetScalar>32)
    {
        zklog.error("Memalign_calculate() MemAlign out of range offset=" + offsetScalar.get_str());
        return ZKR_SM_MAIN_MEMALIGN_OFFSET_OUT_OF_RANGE;
    }

    uint64_t offset = offsetScalar.get_ui();
    mpz_class leftV;
    leftV = (m0 << (offset*8)) & ScalarMask256;
    mpz_class rightV;
    rightV = (m1 >> (256 - offset*8)) & (ScalarMask256 >> (256 - offset*8));
    mpz_class _V;
    _V = leftV | rightV;
    scalar2fea(fr, _V, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);

    return ZKR_SUCCESS;
}

zkresult Memalign_verify ( Context &ctx,
                           Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                           MainExecRequired *required)
{
    zkassert(ctx.pZKPC != NULL);
    uint64_t zkPC = *ctx.pZKPC;
    zkassert((ctx.rom.line[*ctx.pZKPC].memAlignRD == 1) || (ctx.rom.line[*ctx.pZKPC].memAlignWR == 1) || (ctx.rom.line[zkPC].memAlignWR8==1));
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;

    mpz_class m0;
    if (!fea2scalar(fr, m0, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
    {
        zklog.error("Memalign_verify() Failed calling fea2scalar(pols.A)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    mpz_class m1;
    if (!fea2scalar(fr, m1, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
    {
        zklog.error("Memalign_verify() Failed calling fea2scalar(pols.B)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    mpz_class v;
    if (!fea2scalar(fr, v, op0, op1, op2, op3, op4, op5, op6, op7))
    {
        zklog.error("Memalign_verify() Failed calling fea2scalar(pols.op)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    mpz_class offsetScalar;
    if (!fea2scalar(fr, offsetScalar, ctx.pols.C0[i], ctx.pols.C1[i], ctx.pols.C2[i], ctx.pols.C3[i], ctx.pols.C4[i], ctx.pols.C5[i], ctx.pols.C6[i], ctx.pols.C7[i]))
    {
        zklog.error("Memalign_verify() Failed calling fea2scalar(pols.C)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    if (offsetScalar<0 || offsetScalar>32)
    {
        zklog.error("Memalign_verify() MemAlign out of range offset=" + offsetScalar.get_str());
        return ZKR_SM_MAIN_MEMALIGN_OFFSET_OUT_OF_RANGE;
    }
    uint64_t offset = offsetScalar.get_ui();

    if (ctx.rom.line[zkPC].memAlignRD==0 && ctx.rom.line[zkPC].memAlignWR==1 && ctx.rom.line[zkPC].memAlignWR8==0)
    {
        ctx.pols.memAlignWR[i] = fr.one();

        mpz_class w0;
        if (!fea2scalar(fr, w0, ctx.pols.D0[i], ctx.pols.D1[i], ctx.pols.D2[i], ctx.pols.D3[i], ctx.pols.D4[i], ctx.pols.D5[i], ctx.pols.D6[i], ctx.pols.D7[i]))
        {
            zklog.error("Memalign_verify() Failed calling fea2scalar(pols.D)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        mpz_class w1;
        if (!fea2scalar(fr, w1, ctx.pols.E0[i], ctx.pols.E1[i], ctx.pols.E2[i], ctx.pols.E3[i], ctx.pols.E4[i], ctx.pols.E5[i], ctx.pols.E6[i], ctx.pols.E7[i]))
        {
            zklog.error("Memalign_verify() Failed calling fea2scalar(pols.E)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        mpz_class _W0;
        _W0 = (m0 & (ScalarTwoTo256 - (ScalarOne << (256-offset*8)))) | (v >> offset*8);
        mpz_class _W1;
        _W1 = (m1 & (ScalarMask256 >> offset*8)) | ((v << (256 - offset*8)) & ScalarMask256);
        if ( (w0 != _W0) || (w1 != _W1) )
        {
            zklog.error("Memalign_verify() MemAlign w0, w1 invalid: w0=" + w0.get_str(16) + " w1=" + w1.get_str(16) + " _W0=" + _W0.get_str(16) + " _W1=" + _W1.get_str(16) + " m0=" + m0.get_str(16) + " m1=" + m1.get_str(16) + " offset=" + to_string(offset) + " v=" + v.get_str(16));
            return ZKR_SM_MAIN_MEMALIGN_WRITE_MISMATCH;
        }

        if (required != NULL)
        {
            MemAlignAction memAlignAction;
            memAlignAction.m0 = m0;
            memAlignAction.m1 = m1;
            memAlignAction.w0 = w0;
            memAlignAction.w1 = w1;
            memAlignAction.v = v;
            memAlignAction.offset = offset;
            memAlignAction.wr256 = 1;
            memAlignAction.wr8 = 0;
            required->MemAlign.push_back(memAlignAction);
        }
    }
    else if (ctx.rom.line[zkPC].memAlignRD==0 && ctx.rom.line[zkPC].memAlignWR==0 && ctx.rom.line[zkPC].memAlignWR8==1)
    {
        ctx.pols.memAlignWR8[i] = fr.one();

        mpz_class w0;
        if (!fea2scalar(fr, w0, ctx.pols.D0[i], ctx.pols.D1[i], ctx.pols.D2[i], ctx.pols.D3[i], ctx.pols.D4[i], ctx.pols.D5[i], ctx.pols.D6[i], ctx.pols.D7[i]))
        {
            zklog.error("Memalign_verify() Failed calling fea2scalar(pols.D)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        mpz_class _W0;
        mpz_class byteMaskOn256("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
        _W0 = (m0 & (byteMaskOn256 >> (offset*8))) | ((v & 0xFF) << ((31-offset)*8));
        if (w0 != _W0)
        {
            zklog.error("Memalign_verify() MemAlign w0 invalid: w0=" + w0.get_str(16) + " _W0=" + _W0.get_str(16) + " m0=" + m0.get_str(16) + " offset=" + to_string(offset) + " v=" + v.get_str(16));
            return ZKR_SM_MAIN_MEMALIGN_WRITE8_MISMATCH;
        }

        if (required != NULL)
        {
            MemAlignAction memAlignAction;
            memAlignAction.m0 = m0;
            memAlignAction.m1 = 0;
            memAlignAction.w0 = w0;
            memAlignAction.w1 = 0;
            memAlignAction.v = v;
            memAlignAction.offset = offset;
            memAlignAction.wr256 = 0;
            memAlignAction.wr8 = 1;
            required->MemAlign.push_back(memAlignAction);
        }
    }
    else if (ctx.rom.line[zkPC].memAlignRD==1 && ctx.rom.line[zkPC].memAlignWR==0 && ctx.rom.line[zkPC].memAlignWR8==0)
    {
        ctx.pols.memAlignRD[i] = fr.one();

        mpz_class leftV;
        leftV = (m0 << offset*8) & ScalarMask256;
        mpz_class rightV;
        rightV = (m1 >> (256 - offset*8)) & (ScalarMask256 >> (256 - offset*8));
        mpz_class _V;
        _V = leftV | rightV;
        if (v != _V)
        {
            zklog.error("Memalign_verify() MemAlign v invalid: v=" + v.get_str(16) + " _V=" + _V.get_str(16) + " m0=" + m0.get_str(16) + " m1=" + m1.get_str(16) + " offset=" + to_string(offset));
            return ZKR_SM_MAIN_MEMALIGN_READ_MISMATCH;
        }

        if (required != NULL)
        {
            MemAlignAction memAlignAction;
            memAlignAction.m0 = m0;
            memAlignAction.m1 = m1;
            memAlignAction.w0 = 0;
            memAlignAction.w1 = 0;
            memAlignAction.v = v;
            memAlignAction.offset = offset;
            memAlignAction.wr256 = 0;
            memAlignAction.wr8 = 0;
            required->MemAlign.push_back(memAlignAction);
        }
    }
    else
    {
        zklog.error("Memalign_verify() Invalid memAlign operation");
        exitProcess();
    }

    return ZKR_SUCCESS;
}

} // namespace