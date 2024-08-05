#include "binary_helper.hpp"
#include "zklog.hpp"
#include "definitions.hpp"

namespace fork_12
{

zkresult Binary_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7)
{
    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].bin == 1);
    uint8_t binOpcode = ctx.rom.line[*ctx.pZKPC].binOpcode;
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;
    mpz_class a, b, c;

    // Read contents of registers A and B
    if (!fea2scalar(fr, a, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
    {
        zklog.error("Binary_calculate() Failed calling fea2scalar(pols.A)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    if (!fea2scalar(fr, b, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
    {
        zklog.error("Binary_calculate() Failed calling fea2scalar(pols.B)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }

    // Calculate result as c based on opcode
    switch (binOpcode)
    {
        case 0: // ADD
        {
            c = (a + b) & ScalarMask256;
            scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
            return ZKR_SUCCESS;
        }
        case 1: // SUB
        {
            c = (a - b + ScalarTwoTo256) & ScalarMask256;
            scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
            return ZKR_SUCCESS;
        }
        case 2: // LT
        {
            c = (a < b);
            scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
            return ZKR_SUCCESS;
        }
        case 3: // SLT
        {
            if (a >= ScalarTwoTo255) a = a - ScalarTwoTo256;
            if (b >= ScalarTwoTo255) b = b - ScalarTwoTo256;
            c = (a < b);
            scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
            return ZKR_SUCCESS;
        }
        case 4: // EQ
        {
            c = (a == b);
            scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
            return ZKR_SUCCESS;
        }
        case 5: // AND
        {
            c = (a & b);
            scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
            return ZKR_SUCCESS;
        }
        case 6: // OR
        {
            c = (a | b);
            scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
            return ZKR_SUCCESS;
        }
        case 7: // XOR
        {
            c = (a ^ b);
            scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
            return ZKR_SUCCESS;
        }
        case 8: // LT4
        {
            c = lt4(a, b);
            scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
            return ZKR_SUCCESS;
        }
        default:
        {
            zklog.error("Binary_calculate() Invalid binary operation: opcode=" + to_string(binOpcode));
            exitProcess();
        }
    }

    return ZKR_UNSPECIFIED;
}

zkresult Binary_verify ( Context &ctx,
                        Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7,
                        MainExecRequired *required)
{
    zkassert(ctx.pZKPC != NULL);
    zkassert(ctx.rom.line[*ctx.pZKPC].bin == 1);
    uint8_t binOpcode = ctx.rom.line[*ctx.pZKPC].binOpcode;
    zkassert(ctx.pStep != NULL);
    uint64_t i = *ctx.pStep;
    mpz_class a, b, c, expectedC;

    // Read contents of registers A and B
    if (!fea2scalar(fr, a, ctx.pols.A0[i], ctx.pols.A1[i], ctx.pols.A2[i], ctx.pols.A3[i], ctx.pols.A4[i], ctx.pols.A5[i], ctx.pols.A6[i], ctx.pols.A7[i]))
    {
        zklog.error("Binary_verify() Failed calling fea2scalar(pols.A)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    if (!fea2scalar(fr, b, ctx.pols.B0[i], ctx.pols.B1[i], ctx.pols.B2[i], ctx.pols.B3[i], ctx.pols.B4[i], ctx.pols.B5[i], ctx.pols.B6[i], ctx.pols.B7[i]))
    {
        zklog.error("Binary_verify() Failed calling fea2scalar(pols.B)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }

    // Convert op into c
    if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))
    {
        zklog.error("Binary_verify() Failed calling fea2scalar(op)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }

    // Calculate expected c and carry based on binary opcode
    switch (binOpcode)
    {
        case 0: // ADD
        {
            expectedC = (a + b) & ScalarMask256;
            if (c != expectedC)
            {
                zklog.error("Binary_verify() ADD operation does not match c=op=" + c.get_str(16) + " expectedC=(a + b) & ScalarMask256=" + expectedC.get_str(16));
                return ZKR_SM_MAIN_BINARY_ADD_MISMATCH;
            }

            ctx.pols.carry[i] = fr.fromU64(((a + b) >> 256) > 0);
            break;
        }
        case 1: // SUB
        {
            expectedC = (a - b + ScalarTwoTo256) & ScalarMask256;
            if (c != expectedC)
            {
                zklog.error("Binary_verify() SUB operation does not match c=op=" + c.get_str(16) + " expectedC=(a - b + ScalarTwoTo256) & ScalarMask256=" + expectedC.get_str(16));
                return ZKR_SM_MAIN_BINARY_ADD_MISMATCH;
            }

            ctx.pols.carry[i] = fr.fromU64((a - b) < 0);
            break;
        }
        case 2: // LT
        {
            expectedC = (a < b);
            if (c != expectedC)
            {
                zklog.error("Binary_verify() ADD operation does not match c=op=" + c.get_str(16) + " expectedC=a < b=" + expectedC.get_str(16));
                return ZKR_SM_MAIN_BINARY_ADD_MISMATCH;
            }

            ctx.pols.carry[i] = fr.fromU64(c.get_ui());
            break;
        }
        case 3: // SLT
        {
            mpz_class _a, _b;
            _a = a;
            _b = b;

            if (a >= ScalarTwoTo255) _a = a - ScalarTwoTo256;
            if (b >= ScalarTwoTo255) _b = b - ScalarTwoTo256;

            expectedC = (_a < _b);
            if (c != expectedC)
            {
                zklog.error("Binary_verify() SLT operation does not match a=" + a.get_str(16) + " b=" + b.get_str(16) + " c=" + c.get_str(16) + " _a=" + _a.get_str(16) + " _b=" + _b.get_str(16) + " expectedC=" + expectedC.get_str(16));
                return ZKR_SM_MAIN_BINARY_ADD_MISMATCH;
            }

            ctx.pols.carry[i] = fr.fromU64(c.get_ui());
            break;
        }
        case 4: // EQ
        {
            expectedC = (a == b);
            if (c != expectedC)
            {
                zklog.error("Binary_verify() EQ operation does not match c=op=" + c.get_str(16) + " expectedC=a == b=" + expectedC.get_str(16));
                return ZKR_SM_MAIN_BINARY_ADD_MISMATCH;
            }

            ctx.pols.carry[i] = fr.fromU64(c.get_ui());
            break;
        }
        case 5: // AND
        {
            expectedC = (a & b);
            if (c != expectedC)
            {
                zklog.error("Binary_verify() AND operation does not match c=op=" + c.get_str(16) + " expectedC=a & b=" + expectedC.get_str(16));
                return ZKR_SM_MAIN_BINARY_ADD_MISMATCH;
            }

            ctx.pols.carry[i] = (c != 0) ? fr.one() : fr.zero();
            break;
        }
        case 6: // OR
        {
            expectedC = (a | b);
            if (c != expectedC)
            {
                zklog.error("Binary_verify() OR operation does not match c=op=" + c.get_str(16) + " expectedC=a | b=" + expectedC.get_str(16));
                return ZKR_SM_MAIN_BINARY_ADD_MISMATCH;
            }

            ctx.pols.carry[i] = fr.zero();
            break;
        }
        case 7: // XOR
        {
            expectedC = (a ^ b);
            if (c != expectedC)
            {
                zklog.error("Binary_verify() OR operation does not match c=op=" + c.get_str(16) + " expectedC=a ^ b=" + expectedC.get_str(16));
                return ZKR_SM_MAIN_BINARY_ADD_MISMATCH;
            }

            ctx.pols.carry[i] = fr.zero();
            break;
        }
        case 8: // LT4
        {
            expectedC = lt4(a,b); 
            if (c != expectedC)
            {
                zklog.error("Binary_verify() OR operation does not match c=op=" + c.get_str(16) + " expectedC=a ^ b=" + expectedC.get_str(16));
                return ZKR_SM_MAIN_BINARY_ADD_MISMATCH;
            }

            ctx.pols.carry[i] = fr.fromScalar(c);
            break;
        }
        default:
        {
            zklog.error("Binary_verify() Invalid binary operation: opcode=" + to_string(binOpcode));
            exitProcess();
        }
    }

    if (!ctx.bProcessBatch)
    {
        ctx.pols.bin[i] = fr.one();
        ctx.pols.binOpcode[i] = fr.fromU64(binOpcode);
    }

    if(required != NULL)
    {
        // Store the binary action to execute it later with the binary SM
        BinaryAction binaryAction;
        binaryAction.a = a;
        binaryAction.b = b;
        binaryAction.c = c;
        binaryAction.opcode = binOpcode;
        binaryAction.type = 1;
        required->Binary.push_back(binaryAction);
    }

    return ZKR_SUCCESS;
}

} // namespace