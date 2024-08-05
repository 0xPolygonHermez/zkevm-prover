#include <iostream>
#include "definitions.hpp"
#include "config.hpp"
#include "main_sm/fork_12/main/eval_command.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "zkassert.hpp"
#include "zklog.hpp"
#include "zkglobals.hpp"

namespace fork_12
{

#ifdef DEBUG
#define CHECK_EVAL_COMMAND_PARAMETERS
#endif

void evalCommand (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    if (cmd.op == op_functionCall)
    {
        switch (cmd.function)
        {
            case f_getGlobalExitRoot:               return eval_getGlobalExitRoot(ctx, cmd, cr);
            case f_getSequencerAddr:                return eval_getSequencerAddr(ctx, cmd, cr);
            case f_getTimestamp:                    return eval_getTimestamp(ctx, cmd, cr);
            case f_getTxs:                          return eval_getTxs(ctx, cmd, cr);
            case f_getTxsLen:                       return eval_getTxsLen(ctx, cmd, cr);
            case f_eventLog:                        return eval_eventLog(ctx, cmd, cr);
            case f_cond:                            return eval_cond(ctx, cmd, cr);
            case f_inverseFpEc:                     return eval_inverseFpEc(ctx, cmd, cr);
            case f_inverseFnEc:                     return eval_inverseFnEc(ctx, cmd, cr);
            case f_sqrtFpEc:                        return eval_sqrtFpEc(ctx, cmd, cr);
            case f_sqrtFpEcParity:                  return eval_sqrtFpEcParity(ctx, cmd, cr);
            case f_xAddPointEc:                     return eval_xAddPointEc(ctx, cmd, cr);
            case f_yAddPointEc:                     return eval_yAddPointEc(ctx, cmd, cr);
            case f_xDblPointEc:                     return eval_xDblPointEc(ctx, cmd, cr);
            case f_yDblPointEc:                     return eval_yDblPointEc(ctx, cmd, cr);
            case f_bitwise_and:                     return eval_bitwise_and(ctx, cmd, cr);
            case f_bitwise_or:                      return eval_bitwise_or(ctx, cmd, cr);
            case f_bitwise_xor:                     return eval_bitwise_xor(ctx, cmd, cr);
            case f_bitwise_not:                     return eval_bitwise_not(ctx, cmd, cr);
            case f_comp_lt:                         return eval_comp_lt(ctx, cmd, cr);
            case f_comp_gt:                         return eval_comp_gt(ctx, cmd, cr);
            case f_comp_eq:                         return eval_comp_eq(ctx, cmd, cr);
            case f_loadScalar:                      return eval_loadScalar(ctx, cmd, cr);
            case f_log:                             return eval_log(ctx, cmd, cr);
            case f_exp:                             return eval_exp(ctx, cmd, cr);
            case f_storeLog:                        return eval_storeLog(ctx, cmd, cr);
            case f_memAlignWR_W0:                   return eval_memAlignWR_W0(ctx, cmd, cr);
            case f_memAlignWR_W1:                   return eval_memAlignWR_W1(ctx, cmd, cr);
            case f_memAlignWR8_W0:                  return eval_memAlignWR8_W0(ctx, cmd, cr);
            case f_beforeLast:                      return eval_beforeLast(ctx, cmd, cr);

            // Etrog (fork 7) new methods:
            case f_getL1InfoRoot:                   return eval_getL1InfoRoot(ctx, cmd, cr);
            case f_getL1InfoGER:                    return eval_getL1InfoGER(ctx, cmd, cr);
            case f_getL1InfoBlockHash:              return eval_getL1InfoBlockHash(ctx, cmd, cr);
            case f_getL1InfoTimestamp:              return eval_getL1InfoTimestamp(ctx, cmd, cr);
            case f_getTimestampLimit:               return eval_getTimestampLimit(ctx, cmd, cr);
            case f_getForcedBlockHashL1:            return eval_getForcedBlockHashL1(ctx, cmd, cr);
            case f_getSmtProof:                     return eval_getSmtProof(ctx, cmd, cr);
            case f_MPdiv:                           return eval_MPdiv(ctx, cmd, cr);
            case f_MPdiv_short:                     return eval_MPdiv_short(ctx, cmd, cr);
            case f_receiveLenQuotient_short:        return eval_receiveLenQuotient_short(ctx, cmd, cr);
            case f_receiveQuotientChunk_short:      return eval_receiveQuotientChunk_short(ctx, cmd, cr);
            case f_receiveRemainderChunk_short:     return eval_receiveRemainderChunk_short(ctx, cmd, cr);
            case f_receiveLenRemainder:             return eval_receiveLenRemainder(ctx, cmd, cr);
            case f_receiveRemainderChunk:           return eval_receiveRemainderChunk(ctx, cmd, cr);
            case f_receiveLenQuotient:              return eval_receiveLenQuotient(ctx, cmd, cr);
            case f_receiveQuotientChunk:            return eval_receiveQuotientChunk(ctx, cmd, cr);
            case f_receiveLen:                      return eval_receiveLen(ctx, cmd, cr);
            case f_ARITH_BN254_ADDFP2:              return eval_ARITH_BN254_ADDFP2(ctx, cmd, cr);
            case f_ARITH_BN254_SUBFP2:              return eval_ARITH_BN254_SUBFP2(ctx, cmd, cr);
            case f_ARITH_BN254_MULFP2_X:            return eval_ARITH_BN254_MULFP2_X(ctx, cmd, cr);
            case f_ARITH_BN254_MULFP2_Y:            return eval_ARITH_BN254_MULFP2_Y(ctx, cmd, cr);
            case f_fp2InvBN254_x:                   return eval_fp2InvBN254_x(ctx, cmd, cr);
            case f_fp2InvBN254_y:                   return eval_fp2InvBN254_y(ctx, cmd, cr);
            case f_fpBN254inv:                      return eval_fpBN254inv(ctx, cmd, cr);

            // Ignore diagnostic rom debug log functions
            case f_dumpRegs:                        cr.zkResult = ZKR_SUCCESS; cr.type = crt_u64; cr.u64 = 0; return;
            case f_dump:                            cr.zkResult = ZKR_SUCCESS; cr.type = crt_u64; cr.u64 = 0; return;
            case f_dumphex:                         cr.zkResult = ZKR_SUCCESS; cr.type = crt_u64; cr.u64 = 0; return;
            
            default:
                zklog.error("evalCommand() found invalid function=" + to_string(cmd.function) + "=" + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
                exitProcess();
        }
    }
    switch (cmd.op)
    {
        case op_number:         return eval_number(ctx, cmd, cr);
        case op_declareVar:     return eval_declareVar(ctx, cmd, cr);
        case op_setVar:         return eval_setVar(ctx, cmd, cr);
        case op_getVar:         return eval_getVar(ctx, cmd, cr);
        case op_getReg:         return eval_getReg(ctx, cmd, cr);
        case op_add:            return eval_add(ctx, cmd, cr);
        case op_sub:            return eval_sub(ctx, cmd, cr);
        case op_neg:            return eval_neg(ctx, cmd, cr);
        case op_mul:            return eval_mul(ctx, cmd, cr);
        case op_div:            return eval_div(ctx, cmd, cr);
        case op_mod:            return eval_mod(ctx, cmd, cr);
        case op_or:             return eval_logical_or(ctx, cmd, cr);
        case op_and:            return eval_logical_and(ctx, cmd, cr);
        case op_gt:             return eval_logical_gt(ctx, cmd, cr);
        case op_ge:             return eval_logical_ge(ctx, cmd, cr);
        case op_lt:             return eval_logical_lt(ctx, cmd, cr);
        case op_le:             return eval_logical_le(ctx, cmd, cr);
        case op_eq:             return eval_logical_eq(ctx, cmd, cr);
        case op_ne:             return eval_logical_ne(ctx, cmd, cr);
        case op_not:            return eval_logical_not(ctx, cmd, cr);
        case op_bitand:         return eval_bit_and(ctx, cmd, cr);
        case op_bitor:          return eval_bit_or(ctx, cmd, cr);
        case op_bitxor:         return eval_bit_xor(ctx, cmd, cr);
        case op_bitnot:         return eval_bit_not(ctx, cmd, cr);
        case op_shl:            return eval_bit_shl(ctx, cmd, cr);
        case op_shr:            return eval_bit_shr(ctx, cmd, cr);
        case op_if:             return eval_if(ctx, cmd, cr);
        case op_getMemValue:    return eval_getMemValue(ctx, cmd, cr);
        default:
            zklog.error("evalCommand() found invalid operation=" + op2String(cmd.op) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            exitProcess();
    }
}

void eval_number(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    cr.type = crt_scalar;
    cr.scalar = cmd.num;
}

/*************/
/* Variables */
/*************/

/* Declares a new variable, and fails if it already exists */
void eval_declareVar (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check the variable name
    if (cmd.varName == "")
    {
        zklog.error("eval_declareVar() Variable name not found step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }

    // Check that this variable does not exists
    if ( (cmd.varName[0] != '_') && (ctx.vars.find(cmd.varName) != ctx.vars.end()) )
    {
        zklog.error("eval_declareVar() Variable already declared: " + cmd.varName + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Create the new variable with a zero value
    ctx.vars[cmd.varName] = 0;

#ifdef LOG_VARIABLES
    zklog.info("Declare variable: " + cmd.varName);
#endif

    // Return the current value of this variable
    cr.type = crt_scalar;
    cr.scalar = 0;
}

/* Gets the value of the variable, and fails if it does not exist */
void eval_getVar (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check the variable name
    if (cmd.varName == "")
    {
        zklog.error("eval_getVar() Variable name not found step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Check that this variable exists
    std::unordered_map<std::string, mpz_class>::iterator it = ctx.vars.find(cmd.varName);
    if (it == ctx.vars.end())
    {
        zklog.error("eval_getVar() Undefined variable: " + cmd.varName + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }

#ifdef LOG_VARIABLES
    zklog.info("Get variable: " + cmd.varName + " scalar: " + ctx.vars[cmd.varName].get_str(16));
#endif

    // Return the current value of this variable
    cr.type = crt_scalar;
    cr.scalar = it->second;
}

// Forward declaration, used by eval_setVar
void eval_left (Context &ctx, const RomCommand &cmd, CommandResult &cr);

/* Sets variable to value, and fails if it does not exist */
void eval_setVar (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check that tag contains a values array
    if (cmd.values.size() == 0) {
        zklog.error("eval_setVar() could not find array values in setVar command step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get varName from the first element in values
    eval_left(ctx,*cmd.values[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_string)
    {
        zklog.error("eval_setVar() unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    string varName = cr.str;

    // Check that this variable exists
    std::unordered_map<std::string, mpz_class>::iterator it = ctx.vars.find(varName);
    if (it == ctx.vars.end())
    {
        zklog.error("eval_setVar() Undefined variable: " + varName + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }

    // Call evalCommand() to build the field element value for this variable
    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }

    // Get the field element value from the command result
    mpz_class auxScalar;
    cr2scalar(ctx, cr, auxScalar);

    // Store the value as the new variable value
    it->second = auxScalar;

    // Return the current value of the variable
    cr.type = crt_scalar;
    cr.scalar = auxScalar;

#ifdef LOG_VARIABLES
    zklog.info("Set variable: " + varName + " scalar: " + ctx.vars[varName].get_str(16));
#endif
}

void eval_left (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    switch (cmd.op)
    {
        case op_declareVar:
        {
            eval_declareVar(ctx, cmd, cr);
            cr.type = crt_string;
            cr.str = cmd.varName;
            return;
        }
        case op_getVar:
        {
            cr.type = crt_string;
            cr.str = cmd.varName;
            return;
        }
        default:
        {
            zklog.error("eval_left() invalid left expression, op: " + op2String(cmd.op) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            exitProcess();
        }
    }
}

/*************/
/* Registers */
/*************/

void eval_getReg (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Get registry value, with the proper registry type
    switch (cmd.reg)
    {
        case reg_A:
            cr.type = crt_scalar;
            if (!fea2scalar(ctx.fr, cr.scalar, ctx.pols.A0[*ctx.pStep], ctx.pols.A1[*ctx.pStep], ctx.pols.A2[*ctx.pStep], ctx.pols.A3[*ctx.pStep], ctx.pols.A4[*ctx.pStep], ctx.pols.A5[*ctx.pStep], ctx.pols.A6[*ctx.pStep], ctx.pols.A7[*ctx.pStep]))
            {
                cr.zkResult = ZKR_SM_MAIN_FEA2SCALAR;
                return;
            }
            break;
        case reg_B:
            cr.type = crt_scalar;
            if (!fea2scalar(ctx.fr, cr.scalar, ctx.pols.B0[*ctx.pStep], ctx.pols.B1[*ctx.pStep], ctx.pols.B2[*ctx.pStep], ctx.pols.B3[*ctx.pStep], ctx.pols.B4[*ctx.pStep], ctx.pols.B5[*ctx.pStep], ctx.pols.B6[*ctx.pStep], ctx.pols.B7[*ctx.pStep]))
            {
                cr.zkResult = ZKR_SM_MAIN_FEA2SCALAR;
                return;
            }
            break;
        case reg_C:
            cr.type = crt_scalar;
            if (!fea2scalar(ctx.fr, cr.scalar, ctx.pols.C0[*ctx.pStep], ctx.pols.C1[*ctx.pStep], ctx.pols.C2[*ctx.pStep], ctx.pols.C3[*ctx.pStep], ctx.pols.C4[*ctx.pStep], ctx.pols.C5[*ctx.pStep], ctx.pols.C6[*ctx.pStep], ctx.pols.C7[*ctx.pStep]))
            {
                cr.zkResult = ZKR_SM_MAIN_FEA2SCALAR;
                return;
            }
            break;
        case reg_D:
            cr.type = crt_scalar;
            if (!fea2scalar(ctx.fr, cr.scalar, ctx.pols.D0[*ctx.pStep], ctx.pols.D1[*ctx.pStep], ctx.pols.D2[*ctx.pStep], ctx.pols.D3[*ctx.pStep], ctx.pols.D4[*ctx.pStep], ctx.pols.D5[*ctx.pStep], ctx.pols.D6[*ctx.pStep], ctx.pols.D7[*ctx.pStep]))
            {
                cr.zkResult = ZKR_SM_MAIN_FEA2SCALAR;
                return;
            }
            break;
        case reg_E:
            cr.type = crt_scalar;
            if (!fea2scalar(ctx.fr, cr.scalar, ctx.pols.E0[*ctx.pStep], ctx.pols.E1[*ctx.pStep], ctx.pols.E2[*ctx.pStep], ctx.pols.E3[*ctx.pStep], ctx.pols.E4[*ctx.pStep], ctx.pols.E5[*ctx.pStep], ctx.pols.E6[*ctx.pStep], ctx.pols.E7[*ctx.pStep]))
            {
                cr.zkResult = ZKR_SM_MAIN_FEA2SCALAR;
                return;
            }
            break;
        case reg_SR:
            cr.type = crt_scalar;
            if (!fea2scalar(ctx.fr, cr.scalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep]))
            {
                cr.zkResult = ZKR_SM_MAIN_FEA2SCALAR;
                return;
            }
            break;
        case reg_CTX:
            cr.type = crt_u32;
            cr.u32 = ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep]);
            break;
        case reg_SP:
            cr.type = crt_u16;
            cr.u16 = ctx.fr.toU64(ctx.pols.SP[*ctx.pStep]);
            break;
        case reg_PC:
            cr.type = crt_u32;
            cr.u32 = ctx.fr.toU64(ctx.pols.PC[*ctx.pStep]);
            break;
        case reg_GAS:
            cr.type = crt_u64;
            cr.u64 = ctx.fr.toU64(ctx.pols.GAS[*ctx.pStep]);
            break;
        case reg_zkPC:
            cr.type = crt_u32;
            cr.u32 = ctx.fr.toU64(ctx.pols.zkPC[*ctx.pStep]);
            break;
        case reg_RR:
            cr.type = crt_u64;
            cr.u64 = ctx.fr.toU64(ctx.pols.RR[*ctx.pStep]);
            break;
        case reg_CNT_ARITH:
            cr.type = crt_u64;
            cr.u64 = ctx.fr.toU64(ctx.pols.cntArith[*ctx.pStep]);
            break;
        case reg_CNT_BINARY:
            cr.type = crt_u64;
            cr.u64 = ctx.fr.toU64(ctx.pols.cntBinary[*ctx.pStep]);
            break;
        case reg_CNT_KECCAK_F:
            cr.type = crt_u64;
            cr.u64 = ctx.fr.toU64(ctx.pols.cntKeccakF[*ctx.pStep]);
            break;
        case reg_CNT_MEM_ALIGN:
            cr.type = crt_u64;
            cr.u64 = ctx.fr.toU64(ctx.pols.cntMemAlign[*ctx.pStep]);
            break;
        case reg_CNT_PADDING_PG:
            cr.type = crt_u64;
            cr.u64 = ctx.fr.toU64(ctx.pols.cntPaddingPG[*ctx.pStep]);
            break;
        case reg_CNT_POSEIDON_G:
            cr.type = crt_u64;
            cr.u64 = ctx.fr.toU64(ctx.pols.cntPoseidonG[*ctx.pStep]);
            break;
        case reg_STEP:
            cr.type = crt_u64;
            cr.u64 = *ctx.pStep;
            break;
        case reg_HASHPOS:
            cr.type = crt_u64;
            cr.u64 = ctx.fr.toU64(ctx.pols.HASHPOS[*ctx.pStep]);
            break;
        default:
            zklog.error("eval_getReg() Invalid register=" + reg2string(cmd.reg) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            exitProcess();
    }
}

/*****************************/
/* Command result conversion */
/*****************************/

void cr2fe (Context &ctx, const CommandResult &cr, Goldilocks::Element &fe)
{
    switch (cr.type)
    {
        case crt_fe:
            fe = cr.fe;
            return;
        case crt_scalar:
            scalar2fe(ctx.fr, cr.scalar, fe);
            return;
        default:
            zklog.error("cr2fe() unexpected type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            exitProcess();
    }
}

void cr2scalar (Context &ctx, const CommandResult &cr, mpz_class &s)
{
    switch (cr.type)
    {
        case crt_scalar:
            s = cr.scalar;
            return;
        case crt_fe:
            fe2scalar(ctx.fr, s, cr.fe);
            return;
        case crt_u64:
            s = cr.u64;
            return;
        case crt_u32:
            s = cr.u32;
            return;
        case crt_u16:
            s = cr.u16;
            return;
        default:
            zklog.error("cr2scalar() unexpected type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            exitProcess();
    }
}

/*************************/
/* Arithmetic operations */
/*************************/

void eval_add(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_add() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a + b;
}

void eval_sub(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_sub() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a - b;
}

void eval_neg(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 1)
    {
        zklog.error("eval_neg() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    cr.type = crt_scalar;
    cr.scalar = -a;
}

void eval_mul(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_mul() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a * b;
}

void eval_div(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_div() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a / b;
}

void eval_mod(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_mod() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a % b;
}

/**********************/
/* Logical operations */
/**********************/

void eval_logical_or (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_logical_or() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = (a || b) ? 1 : 0;
}

void eval_logical_and (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_logical_and() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = (a && b) ? 1 : 0;
}

void eval_logical_gt (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_logical_gt() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = (a > b) ? 1 : 0;
}

void eval_logical_ge (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_logical_ge() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = (a >= b) ? 1 : 0;
}

void eval_logical_lt (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_logical_lt() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = (a < b) ? 1 : 0;
}

void eval_logical_le (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_logical_le() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = (a <= b) ? 1 : 0;
}

void eval_logical_eq (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_logical_eq() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = (a == b) ? 1 : 0;
}

void eval_logical_ne (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_logical_ne() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = (a != b) ? 1 : 0;
}

void eval_logical_not (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 1)
    {
        zklog.error("eval_logical_not() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    cr.type = crt_scalar;
    cr.scalar = (a) ? 0 : 1;
}

/*********************/
/* Binary operations */
/*********************/

void eval_bit_and (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_bit_and() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a & b;
}

void eval_bit_or (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_bit_or() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a | b;
}

void eval_bit_xor (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_bit_xor() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a ^ b;
}

void eval_bit_not (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 1)
    {
        zklog.error("eval_bit_not() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    cr.type = crt_scalar;
    cr.scalar = ~a;
}

void eval_bit_shl (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_bit_shl() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = (a << b.get_ui());
}

void eval_bit_shr (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        zklog.error("eval_bit_shr() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class b;
    cr2scalar(ctx, cr, b);

    cr.type = crt_scalar;
    cr.scalar = (a >> b.get_ui());
}

/*****************/
/* If: a ? b : c */
/*****************/

void eval_if (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 3)
    {
        zklog.error("eval_if() found invalid number of values=" + to_string(cmd.values.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
    mpz_class a;
    cr2scalar(ctx, cr, a);

    if (a)
    {
        evalCommand(ctx, *cmd.values[1], cr);
        if (cr.zkResult != ZKR_SUCCESS)
        {
            return;
        }
        mpz_class b;
        cr2scalar(ctx, cr, b);

        cr.type = crt_scalar;
        cr.scalar = b;
    }
    else
    {
        evalCommand(ctx, *cmd.values[2], cr);
        if (cr.zkResult != ZKR_SUCCESS)
        {
            return;
        }
        mpz_class c;
        cr2scalar(ctx, cr, c);

        cr.type = crt_scalar;
        cr.scalar = c;
    }
}

/***************/
/* Memory read */
/***************/

void eval_getMemValue (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    cr.type = crt_scalar;
    uint64_t addr = cmd.offset;
    if (cmd.useCTX == 1)
    {
        addr += ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep]) * 0x40000;
    }
    Fea fea = ctx.mem[addr];
    if (!fea2scalar(ctx.fr, cr.scalar, fea.fe0, fea.fe1, fea.fe2, fea.fe3, fea.fe4, fea.fe5, fea.fe6, fea.fe7))
    {
        cr.zkResult = ZKR_SM_MAIN_FEA2SCALAR;
        return;
    }
}

/**************/
/* Input data */
/**************/

void eval_getGlobalExitRoot(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 0)
    {
        zklog.error("eval_getGlobalExitRoot() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Return ctx.proverRequest.input.publicInputs.globalExitRoot as a field element array
    cr.type = crt_fea;
    scalar2fea(ctx.fr, ctx.proverRequest.input.publicInputsExtended.publicInputs.globalExitRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getSequencerAddr(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 0)
    {
        zklog.error("eval_getSequencerAddr() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Return ctx.proverRequest.input.publicInputs.sequencerAddr as a field element array
    cr.type = crt_fea;
    scalar2fea(ctx.fr, ctx.proverRequest.input.publicInputsExtended.publicInputs.sequencerAddr, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getTxsLen(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 0)
    {
        zklog.error("eval_getTxsLen() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Return ctx.proverRequest.input.txsLen/2 as a field element array
    cr.type = crt_fea;
    cr.fea0 = ctx.fr.fromU64(ctx.proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.size());
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

void eval_getTxs(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 2)
    {
        zklog.error("eval_getTxs() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get offset by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar) {
        zklog.error("eval_getTxs() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t offset = cr.scalar.get_ui();

    // Get offset by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_getTxs() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t len = cr.scalar.get_ui();

    // Return result as a field element array
    cr.type = crt_fea;
    ba2fea(ctx.fr, (uint8_t *)(ctx.proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.c_str()) + offset, len, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

/*********************/
/* Full tracer event */
/*********************/

void eval_eventLog(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() < 1)
    {
        zklog.error("eval_eventLog() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    zkassert(ctx.proverRequest.pFullTracer != NULL);
    zkassert(ctx.proverRequest.forkInfo.parentId == 12); // fork_12
    cr.zkResult = ((fork_12::FullTracer *)ctx.proverRequest.pFullTracer)->handleEvent(ctx, cmd);

    // Return an empty array of field elements
    cr.type = crt_fea;
    cr.fea0 = ctx.fr.zero();
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

void eval_getTimestamp(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 0)
    {
        zklog.error("eval_getTimestamp() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Return ctx.proverRequest.input.publicInputs.timestamp as a field element array
    cr.type = crt_fea;
    cr.fea0 = ctx.fr.fromU64(ctx.proverRequest.input.publicInputsExtended.publicInputs.timestamp);
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

/*************/
/* Condition */
/*************/

void eval_cond (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 1)
    {
        zklog.error("eval_cond() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get offset by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_cond() unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    cr.type = crt_fea;
    if (cr.scalar != 0)
    {
        cr.fea0 = ctx.fr.negone(); // -1
    }
    else
    {
        cr.fea0 = ctx.fr.zero();
    }
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

/***************/
/* Expotential */
/***************/

void eval_exp (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 2)
    {
        zklog.error("eval_exp() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_exp() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_exp() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class b = cr.scalar;

    // auxScalar = a^b
    mpz_class auxScalar;
    mpz_pow_ui(auxScalar.get_mpz_t(), a.get_mpz_t(), b.get_ui());

    // Return as a field element array
    cr.type = crt_fea;
    scalar2fea(ctx.fr, auxScalar, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

/*********************/
/* Binary operations */
/*********************/

void eval_bitwise_and (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 2)
    {
        zklog.error("eval_bitwise_and() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_bitwise_and() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_bitwise_and() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class b = cr.scalar;

    cr.type = crt_scalar;
    cr.scalar = a & b;
}

void eval_bitwise_or (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 2)
    {
        zklog.error("eval_bitwise_or() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_bitwise_or() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_bitwise_or() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class b = cr.scalar;

    cr.type = crt_scalar;
    cr.scalar = a | b;
}

void eval_bitwise_xor (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 2)
    {
        zklog.error("eval_bitwise_xor() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_bitwise_xor() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_bitwise_xor() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class b = cr.scalar;

    cr.type = crt_scalar;
    cr.scalar = a ^ b;
}

void eval_bitwise_not (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 1)
    {
        zklog.error("eval_bitwise_not() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_bitwise_not() unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class a = cr.scalar;

    cr.type = crt_scalar;
    cr.scalar = a ^ ScalarMask256;
}

/**************************/
/* Before last evaluation */
/**************************/

void eval_beforeLast (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 0)
    {
        zklog.error("eval_beforeLast() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Return a field element array
    cr.type = crt_fea;
    if (*ctx.pStep >= ctx.N-2)
    {
        cr.fea0 = ctx.fr.zero();
    }
    else
    {
        cr.fea0 = ctx.fr.negone();
    }
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

/*************************/
/* Comparison operations */
/*************************/

void eval_comp_lt (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 2)
    {
        zklog.error("eval_comp_lt() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_comp_lt() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_comp_lt() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class b = cr.scalar;

    cr.type = crt_scalar;
    cr.scalar = (a < b) ? 1 : 0;
}

void eval_comp_gt (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 2)
    {
        zklog.error("eval_comp_gt() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_comp_gt() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_comp_gt() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class b = cr.scalar;

    cr.type = crt_scalar;
    cr.scalar = (a > b) ? 1 : 0;
}

void eval_comp_eq (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 2)
    {
        zklog.error("eval_comp_eq() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_comp_eq() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_comp_eq() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class b = cr.scalar;

    cr.type = crt_scalar;
    cr.scalar = (a == b) ? 1 : 0;
}

void eval_loadScalar (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 1)
    {
        zklog.error("eval_loadScalar() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
}

/*************/
/* Store log */
/*************/

void eval_storeLog (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    zkassert(ctx.proverRequest.pFullTracer != NULL);
    zkassert(ctx.proverRequest.forkInfo.parentId == 12); // fork_12
    cr.zkResult = ((fork_12::FullTracer *)ctx.proverRequest.pFullTracer)->handleEvent(ctx, cmd);

    // Return an empty array of field elements
    cr.type = crt_fea;
    cr.fea0 = ctx.fr.zero();
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

/*******/
/* Log */
/*******/

void eval_log (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 1)
    {
        zklog.error("eval_log() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get indexLog by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }

    mpz_class scalarLog;
    switch (cr.type)
    {
        case crt_fea:
            if (!fea2scalar(ctx.fr, scalarLog, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7))
            {
                cr.zkResult = ZKR_SM_MAIN_FEA2SCALAR;
                return;
            }
            break;
        case crt_u64:
            scalarLog = cr.u64;
            break;
        default:
            zklog.error("eval_storeLog() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            exitProcess();
    }

    // Print the log
    string hexLog = Add0xIfMissing(scalarLog.get_str(16));
    zklog.info("Log regname=" + reg2string(cmd.params[0]->reg) + " hexLog=" + hexLog);

    // Return an empty array of field elements
    cr.type = crt_fea;
    cr.fea0 = ctx.fr.zero();
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

/***************************/
/* Memory align operations */
/***************************/

void eval_memAlignWR_W0 (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 3)
    {
        zklog.error("eval_memAlignWR_W0() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get m0 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_memAlignWR_W0() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class m0 = cr.scalar;

    // Get value by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_memAlignWR_W0() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class value = cr.scalar;

    // Get offset by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_memAlignWR_W0() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    int64_t offset = cr.scalar.get_si();

    int64_t shiftLeft = (32 - offset) * 8;
    int64_t shiftRight = offset * 8;
    mpz_class result = (m0 & (ScalarMask256 << shiftLeft)) | (ScalarMask256 & (value >> shiftRight));

    cr.type = crt_fea;
    scalar2fea(ctx.fr, result, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_memAlignWR_W1 (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 3)
    {
        zklog.error("eval_memAlignWR_W1() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get m1 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_memAlignWR_W1() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class m1 = cr.scalar;

    // Get value by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_memAlignWR_W1() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class value = cr.scalar;

    // Get offset by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_memAlignWR_W1() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    int64_t offset = cr.scalar.get_si();

    int64_t shiftRight = offset * 8;
    int64_t shiftLeft = (32 - offset) * 8;
    mpz_class result = (m1 & (ScalarMask256 >> shiftRight)) | (ScalarMask256 & (value << shiftLeft));

    cr.type = crt_fea;
    scalar2fea(ctx.fr, result, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_memAlignWR8_W0 (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 3)
    {
        zklog.error("eval_memAlignWR8_W0() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get m0 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_memAlignWR8_W0() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class m0 = cr.scalar;

    // Get value by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_memAlignWR8_W0() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class value = cr.scalar;

    // Get offset by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_memAlignWR8_W0() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    int64_t offset = cr.scalar.get_si();

    int64_t bits = (31 - offset) * 8;

    mpz_class result = (m0 & (ScalarMask256 - (ScalarMask8 << bits))) | ((ScalarMask8 & value ) << bits);

    cr.type = crt_fea;
    scalar2fea(ctx.fr, result, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

/*************************/
/* Inverse field element */
/*************************/

void eval_inverseFpEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 1)
    {
        zklog.error("eval_inverseFpEc() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_inverseFpEc() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFec::Element a;
    ctx.fec.fromString(a, cr.scalar.get_str(16), 16);
    if (ctx.fec.isZero(a))
    {
        zklog.error("eval_inverseFpEc() Division by zero step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }

    RawFec::Element r;
    ctx.fec.inv(r, a);

    cr.type = crt_scalar;
    ctx.fec.toMpz(cr.scalar.get_mpz_t(), r);
}

void eval_inverseFnEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 1)
    {
        zklog.error("eval_inverseFnEc() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_inverseFnEc() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFnec::Element a;
    ctx.fnec.fromMpz(a, cr.scalar.get_mpz_t());
    if (ctx.fnec.isZero(a))
    {
        zklog.error("eval_inverseFnEc() Division by zero step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }

    RawFnec::Element r;
    ctx.fnec.inv(r, a);

    cr.type = crt_scalar;
    ctx.fnec.toMpz(cr.scalar.get_mpz_t(), r);
}

/*****************************/
/* Square root field element */
/*****************************/

mpz_class pow ( const mpz_class &x, const mpz_class &n, const mpz_class &p )
{
    if (n == 0)
    {
        return 1;
    }
    if ((n & 1) == 1)
    {
        return (pow(x, n-1, p) * x) % p;
    }
    mpz_class x2 = pow(x, n/2, p);
    return (x2 * x2) % p;
}

mpz_class sqrtTonelliShanks ( const mpz_class &n, const mpz_class &p )
{
    mpz_class s = 1;
    mpz_class q = p - 1;
    while ((q & 1) == 0)
    {
        q = q / 2;
        ++s;
    }
    if (s == 1)
    {
        mpz_class r = pow(n, (p+1)/4, p);
        if ((r * r) % p == n)
        {
            return r;
        }
        return ScalarMask256;
    }

    mpz_class z = 1;
    while (pow(++z, (p - 1)/2, p) != (p - 1));
//    std::cout << "Z found: " << z << "\n";
    mpz_class c = pow(z, q, p);
    mpz_class r = pow(n, (q+1)/2, p);
    mpz_class t = pow(n, q, p);
    mpz_class m = s;
    while (t != 1)
    {
        mpz_class tt = t;
        mpz_class i = 0;
        while (tt != 1)
        {
            tt = (tt * tt) % p;
            ++i;
            if (i == m)
            {
                return ScalarMask256;
            }
        }
        mpz_class b = pow(c, pow(2, m-i-1, p-1), p);
        mpz_class b2 = (b * b) % p;
        r = (r * b) % p;
        t = (t * b2) % p;
        c = b2;
        m = i;
    }
    if (((r * r) % p) == n)
    {
        r = r % p;
        if (r > (p/2))
        {
            r = p - r; // return only the possitive solution of the square root
        }
        return r;
    }
    return ScalarMask256;
}

void eval_sqrtFpEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 1)
    {
        zklog.error("eval_sqrtFpEc() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_sqrtFpEc() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    
    mpz_class a = cr.scalar;
    cr.type = crt_scalar;
    sqrtF3mod4(cr.scalar, cr.scalar);
}

void eval_sqrtFpEcParity (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 2)
    {
        zklog.error("eval_sqrtFpEcParity() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_sqrtFpEc() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif    
    mpz_class a = cr.scalar;

    // Get parity by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_sqrtFpEc() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif    
    mpz_class parity = cr.scalar;

    // Call the sqrt function
    cr.type = crt_scalar;
    sqrtF3mod4(cr.scalar, a);

    // Post-process the result
    if (cr.scalar == ScalarMask256)
    {
        // This sqrt does not have a solution
    }
    else if ((cr.scalar & 1) == parity)
    {
        // Return r as it is, since it has the requested parity
    }
    else
    {
        // Negate the result
        RawFec::Element fe;
        fec.fromMpz(fe, cr.scalar.get_mpz_t());
        fe = fec.neg(fe);
        fec.toMpz(cr.scalar.get_mpz_t(), fe);
    }
}

/********************/
/* Point operations */
/********************/

void eval_AddPointEc (Context &ctx, const RomCommand &cmd, bool dbl, RawFec::Element &x3, RawFec::Element &y3);

void eval_xAddPointEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    RawFec::Element x3;
#ifdef ENABLE_EXPERIMENTAL_CODE
    if(ctx.ecRecoverPrecalcBuffer.filled == true){
        x3 = ctx.ecRecoverPrecalcBuffer.buffer[ctx.ecRecoverPrecalcBuffer.pos++];
    }else{
#endif
        RawFec::Element y3;
        eval_AddPointEc(ctx, cmd, false, x3, y3);    
#ifdef ENABLE_EXPERIMENTAL_CODE
    }
#endif
    cr.type = crt_scalar;
    ctx.fec.toMpz(cr.scalar.get_mpz_t(), x3);
}

void eval_yAddPointEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    RawFec::Element y3;
#ifdef ENABLE_EXPERIMENTAL_CODE
    if(ctx.ecRecoverPrecalcBuffer.filled == true){
        y3 = ctx.ecRecoverPrecalcBuffer.buffer[ctx.ecRecoverPrecalcBuffer.pos++];
    }else{
#endif
        RawFec::Element x3;
        eval_AddPointEc(ctx, cmd, false, x3, y3);  
#ifdef ENABLE_EXPERIMENTAL_CODE
    }
#endif
    cr.type = crt_scalar;
    ctx.fec.toMpz(cr.scalar.get_mpz_t(), y3);
}

void eval_xDblPointEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    RawFec::Element x3;
#ifdef ENABLE_EXPERIMENTAL_CODE
    if(ctx.ecRecoverPrecalcBuffer.filled == true){
        x3 = ctx.ecRecoverPrecalcBuffer.buffer[ctx.ecRecoverPrecalcBuffer.pos++];
    }else{
#endif
        RawFec::Element y3;
        eval_AddPointEc(ctx, cmd, true, x3, y3);    
#ifdef ENABLE_EXPERIMENTAL_CODE
    }
#endif
    cr.type = crt_scalar;
    ctx.fec.toMpz(cr.scalar.get_mpz_t(), x3);
}

void eval_yDblPointEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    RawFec::Element y3;
#ifdef ENABLE_EXPERIMENTAL_CODE
    if(ctx.ecRecoverPrecalcBuffer.filled == true){
        y3 = ctx.ecRecoverPrecalcBuffer.buffer[ctx.ecRecoverPrecalcBuffer.pos++];
    }else{
#endif
        RawFec::Element x3;
        eval_AddPointEc(ctx, cmd, true, x3, y3);    
#ifdef ENABLE_EXPERIMENTAL_CODE
    }
#endif
    cr.type = crt_scalar;
    ctx.fec.toMpz(cr.scalar.get_mpz_t(), y3);
}

void eval_AddPointEc (Context &ctx, const RomCommand &cmd, bool dbl, RawFec::Element &x3, RawFec::Element &y3)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != (dbl ? 2 : 4))
    {
        zklog.error("eval_AddPointEc() invalid number of parameters function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    CommandResult cr;

    // Get x1 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_AddPointEc() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFec::Element x1;
    ctx.fec.fromMpz(x1, cr.scalar.get_mpz_t());

    // Get y1 by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_AddPointEc() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFec::Element y1;
    ctx.fec.fromMpz(y1, cr.scalar.get_mpz_t());

    RawFec::Element x2, y2;
    if (dbl)
    {
        x2 = x1;
        y2 = y1;
    }
    else
    {
        // Get x2 by executing cmd.params[2]
        evalCommand(ctx, *cmd.params[2], cr);
        if (cr.zkResult != ZKR_SUCCESS)
        {
            return;
        }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
        if (cr.type != crt_scalar)
        {
            zklog.error("eval_AddPointEc() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            exitProcess();
        }
#endif
        ctx.fec.fromMpz(x2, cr.scalar.get_mpz_t());

        // Get y2 by executing cmd.params[3]
        evalCommand(ctx, *cmd.params[3], cr);
        if (cr.zkResult != ZKR_SUCCESS)
        {
            return;
        }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
        if (cr.type != crt_scalar)
        {
            zklog.error("eval_AddPointEc() 3 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            exitProcess();
        }
#endif
        ctx.fec.fromMpz(y2, cr.scalar.get_mpz_t());
    }

    cr.zkResult = AddPointEc(ctx, dbl, x1, y1, x2, y2, x3, y3);
}

zkresult AddPointEc (Context &ctx, bool dbl, const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &x2, const RawFec::Element &y2, RawFec::Element &x3, RawFec::Element &y3)
{
    
    // Check if results are buffered
#ifdef ENABLE_EXPERIMENTAL_CODE
    if(ctx.ecRecoverPrecalcBuffer.filled == true){
        if(ctx.ecRecoverPrecalcBuffer.pos < 2){
            zklog.error("ecRecoverPrecalcBuffer.buffer buffer is not filled, but pos < 2 (pos=" + to_string(ctx.ecRecoverPrecalcBuffer.pos) + ")");
            exitProcess();
        }
        x3 = ctx.ecRecoverPrecalcBuffer.buffer[ctx.ecRecoverPrecalcBuffer.pos-2];
        y3 = ctx.ecRecoverPrecalcBuffer.buffer[ctx.ecRecoverPrecalcBuffer.pos-1];
        return ZKR_SUCCESS;
    }
#endif

    // Check if we have just computed this operation
    if ( (ctx.lastECAdd.bDouble == dbl) &&
         ctx.fec.eq(ctx.lastECAdd.x1, x1) &&
         ctx.fec.eq(ctx.lastECAdd.y1, y1) &&
         ( dbl || (ctx.fec.eq(ctx.lastECAdd.x2, x2) && ctx.fec.eq(ctx.lastECAdd.y2, y2) ) ) )
    {
        //zklog.info("eval_AddPointEc() reading from cache");
        x3 = ctx.lastECAdd.x3;
        y3 = ctx.lastECAdd.y3;
        return ZKR_SUCCESS;
    }

    RawFec::Element aux1, aux2, s;

    if (dbl)
    {
        // s = 3*x1*x1/2*y1
        ctx.fec.mul(aux1, x1, x1);
        ctx.fec.fromUI(aux2, 3);
        ctx.fec.mul(aux1, aux1, aux2);
        ctx.fec.add(aux2, y1, y1);
        if (ctx.fec.isZero(aux2))
        {
            zklog.error("AddPointEc() got denominator=0 1");
            return ZKR_SM_MAIN_ARITH_ECRECOVER_DIVIDE_BY_ZERO;
        }
        ctx.fec.div(s, aux1, aux2);
    }
    else
    {
        // s = (y2-y1)/(x2-x1)
        ctx.fec.sub(aux1, y2, y1);
        ctx.fec.sub(aux2, x2, x1);
        if (ctx.fec.isZero(aux2))
        {
            zklog.error("AddPointEc() got denominator=0 2");
            return ZKR_SM_MAIN_ARITH_ECRECOVER_DIVIDE_BY_ZERO;
        }
        ctx.fec.div(s, aux1, aux2);
    }

    // x3 = s*s - (x1+x2)
    ctx.fec.mul(aux1, s, s);
    ctx.fec.add(aux2, x1, x2);
    ctx.fec.sub(x3, aux1, aux2);

    // y3 = s*(x1-x3) - y1
    ctx.fec.sub(aux1, x1, x3);;
    ctx.fec.mul(aux1, aux1, s);
    ctx.fec.sub(y3, aux1, y1);

    // Save parameters and result for later reuse
    ctx.lastECAdd.bDouble = dbl;
    ctx.lastECAdd.x1 = x1;
    ctx.lastECAdd.y1 = y1;
    ctx.lastECAdd.x2 = x2;
    ctx.lastECAdd.y2 = y2;
    ctx.lastECAdd.x3 = x3;
    ctx.lastECAdd.y3 = y3;

    return ZKR_SUCCESS;
}

zkresult eval_addReadWriteAddress (Context &ctx, const mpz_class value, const Goldilocks::Element (&key)[4])
{
    zkassert(ctx.proverRequest.pFullTracer != NULL);
    zkassert(ctx.proverRequest.forkInfo.parentId == 12); // fork_12
    return ((fork_12::FullTracer *)ctx.proverRequest.pFullTracer)->addReadWriteAddress(
        ctx.pols.A0[0], ctx.pols.A1[0], ctx.pols.A2[0], ctx.pols.A3[0], ctx.pols.A4[0], ctx.pols.A5[0], ctx.pols.A6[0], ctx.pols.A7[0],
        ctx.pols.B0[0], ctx.pols.B1[0], ctx.pols.B2[0], ctx.pols.B3[0], ctx.pols.B4[0], ctx.pols.B5[0], ctx.pols.B6[0], ctx.pols.B7[0],
        ctx.pols.C0[0], ctx.pols.C1[0], ctx.pols.C2[0], ctx.pols.C3[0], ctx.pols.C4[0], ctx.pols.C5[0], ctx.pols.C6[0], ctx.pols.C7[0],
        value,
        key);
}

void eval_getL1InfoRoot (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 0)
    {
        zklog.error("eval_getL1InfoRoot() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    cr.type = crt_fea;
    scalar2fea(fr, ctx.proverRequest.input.publicInputsExtended.publicInputs.l1InfoRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getL1InfoGER (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 1)
    {
        zklog.error("eval_getL1InfoGER() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get index by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_getL1InfoGER() unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t indexL1InfoTree = cr.scalar.get_ui();

    unordered_map<uint64_t, L1Data>::const_iterator it;
    it = ctx.proverRequest.input.l1InfoTreeData.find(indexL1InfoTree);
    if (ctx.proverRequest.input.l1InfoTreeData.find(indexL1InfoTree) == ctx.proverRequest.input.l1InfoTreeData.end())
    {
        zklog.error("eval_getL1InfoGER() could not find index=" + to_string(indexL1InfoTree) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        cr.zkResult = ZKR_SM_MAIN_INVALID_L1_INFO_TREE_INDEX;
        return;
    }

    cr.type = crt_fea;
    scalar2fea(fr, it->second.globalExitRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getL1InfoBlockHash (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{ 
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 1)
    {
        zklog.error("eval_getL1InfoBlockHash() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get index by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_getL1InfoBlockHash() unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t indexL1InfoTree = cr.scalar.get_ui();

    unordered_map<uint64_t, L1Data>::const_iterator it;
    it = ctx.proverRequest.input.l1InfoTreeData.find(indexL1InfoTree);
    if (ctx.proverRequest.input.l1InfoTreeData.find(indexL1InfoTree) == ctx.proverRequest.input.l1InfoTreeData.end())
    {
        zklog.error("eval_getL1InfoBlockHash() could not find index=" + to_string(indexL1InfoTree) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        cr.zkResult = ZKR_SM_MAIN_INVALID_L1_INFO_TREE_INDEX;
        return;
    }

    cr.type = crt_fea;
    scalar2fea(fr, it->second.blockHashL1, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getL1InfoTimestamp (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 1)
    {
        zklog.error("eval_getL1InfoTimestamp() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get index by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_getL1InfoTimestamp() unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t indexL1InfoTree = cr.scalar.get_ui();

    unordered_map<uint64_t, L1Data>::const_iterator it;
    it = ctx.proverRequest.input.l1InfoTreeData.find(indexL1InfoTree);
    if (ctx.proverRequest.input.l1InfoTreeData.find(indexL1InfoTree) == ctx.proverRequest.input.l1InfoTreeData.end())
    {
        zklog.error("eval_getL1InfoTimestamp() could not find index=" + to_string(indexL1InfoTree) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        cr.zkResult = ZKR_SM_MAIN_INVALID_L1_INFO_TREE_INDEX;
        return;
    }

    cr.type = crt_fea;
    scalar2fea(fr, it->second.minTimestamp, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getTimestampLimit (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 0)
    {
        zklog.error("eval_getTimestampLimit() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    cr.type = crt_fea;
    scalar2fea(fr, ctx.proverRequest.input.publicInputsExtended.publicInputs.timestampLimit, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getForcedBlockHashL1 (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 0)
    {
        zklog.error("eval_getForcedBlockHashL1() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    cr.type = crt_fea;
    scalar2fea(fr, ctx.proverRequest.input.publicInputsExtended.publicInputs.forcedBlockHashL1, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

mpz_class MOCK_VALUE_SMT_PROOF("0xd4e56740f876aef8c010b86a40d5f56745a118d0906a34e69aec8c0db1cb8fa3");

void eval_getSmtProof (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 2)
    {
        zklog.error("eval_getSmtProof() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get index by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_getSmtProof() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t index = cr.scalar.get_ui();

    // Get level by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_getSmtProof() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t level = cr.scalar.get_ui();

    mpz_class leafValue;
    if (ctx.proverRequest.input.bSkipVerifyL1InfoRoot)
    {
        leafValue = MOCK_VALUE_SMT_PROOF;
    }
    else
    {
        unordered_map<uint64_t, L1Data>::const_iterator it;
        it = ctx.proverRequest.input.l1InfoTreeData.find(index);
        if (ctx.proverRequest.input.l1InfoTreeData.find(index) == ctx.proverRequest.input.l1InfoTreeData.end())
        {
            zklog.error("eval_getSmtProof() could not find index=" + to_string(index) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            cr.zkResult = ZKR_SM_MAIN_INVALID_L1_INFO_TREE_INDEX;
            return;
        }
        if (level >= it->second.smtProof.size())
        {
            zklog.error("eval_getSmtProof() invalid level=" + to_string(level) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            cr.zkResult = ZKR_SM_MAIN_INVALID_L1_INFO_TREE_SMT_PROOF_VALUE;
            return;
        }
        leafValue = it->second.smtProof[level];
    }

    cr.type = crt_fea;
    scalar2fea(fr, leafValue, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

// Compares two unsigned integers represented as arrays of BigInts.
//    a - Unsigned integer represented as an array of BigInts.
//    b - Unsigned integer represented as an array of BigInts.
//    returns 1 if a > b, -1 if a < b, 0 if a == b.

int64_t compare (const vector<mpz_class> &a, const vector<mpz_class> &b)
{
    uint64_t aSize = a.size();
    uint64_t bSize = b.size();
    
    if (aSize != bSize)
    {
        return aSize >= bSize ? 1 : -1;
    }
    for (int64_t i = aSize - 1; i >= 0; i--)
    {
        if (a[i] != b[i])
        {
            return a[i] > b[i] ? 1 : -1;
        }
    }
    return 0;
}

// Removes leading zeros from a.
//  a - Unsigned integer represented as an array of BigInts.
//  returns a with leading zeros removed. It sets a.length = 0 if a = [0n]
void trim (vector<mpz_class> &a)
{
    while (a.size() > 0)
    {
        if (a[a.size()-1] == 0)
        {
            a.pop_back();
        }
        else
        {
            break;
        }
    }
}

// Computes the subtraction of two unsigned integers a,b represented as arrays of BigInts. Assumes a >= b.
//  param a - Unsigned integer represented as an array of BigInts.
//  param b - Unsigned integer represented as an array of BigInts.
//  returns a - b.
void _MP_sub (const vector<mpz_class> &a, const vector<mpz_class> &b, vector<mpz_class> &result)
{
    uint64_t aSize = a.size();
    uint64_t bSize = b.size();
    result.clear();
    mpz_class diff = 0;
    mpz_class carry = 0;

    uint64_t i = 0;
    for (i = 0; i < bSize; i++)
    {
        diff = a[i] - b[i] - carry;
        carry = diff < 0 ? 1 : 0;
        result.emplace_back(diff + carry * ScalarTwoTo256);
    }
    for (i = bSize; i < aSize; i++)
    {
        diff = a[i] - carry;
        if (diff < 0)
        {
            diff += ScalarTwoTo256;
        }
        else
        {
            result.emplace_back(diff);
            i++;
            break;
        }
        result.emplace_back(diff);
        i++;
    }
    for (; i < aSize; i++)
    {
        result.emplace_back(a[i]);
    }
    trim(result);
}

// Computes the subtraction of two unsigned integers represented as arrays of BigInts.
//  a - Unsigned integer represented as an array of BigInts.
//  b - Unsigned integer represented as an array of BigInts.
//  returns a - b.
void MP_sub (const vector<mpz_class> &a, const vector <mpz_class> &b, vector<mpz_class> &result)
{
    result.clear();
    if (compare(a, b) >= 0)
    {
        _MP_sub(a, b, result);
    }
    else
    {
        _MP_sub(b, a, result);
        result[result.size() - 1] = -result[result.size() - 1];
    }
    if (result.size() == 0)
    {
        result.emplace_back(0);
    }
}

// Computes the multiplication of an unsigned integer represented as an array of BigInts and an unsigned integer represented as a BigInt.
//  a - Unsigned integer represented as an array of BigInts.
//  b - Unsigned integer represented as a BigInt.
//  returns a * b
void MP_short_mul (const vector<mpz_class> &a, const mpz_class &b, vector<mpz_class> &result)
{
    uint64_t aSize = a.size();
    uint64_t size = aSize;
    result.clear();
    for (uint64_t i=0; i<size; i++)
    {
        result.emplace_back(0);
    }
    mpz_class product;
    mpz_class carry = 0;
    uint64_t i;
    for (i = 0; i < aSize; i++)
    {
        product = a[i] * b + carry;
        carry = product / ScalarTwoTo256;
        result[i] = product - carry * ScalarTwoTo256;
    }
    while (carry > 0)
    {
        result.emplace_back(carry % ScalarTwoTo256);
        carry /= ScalarTwoTo256;
    }
    trim(result);
}

// Computes the normalisation of two unsigned integers a,b as explained here https://www.codeproject.com/Articles/1276311/Multiple-Precision-Arithmetic-Division-Algorithm.
//  a - Unsigned integer represented as an array of BigInts.
//  b - Unsigned integer represented as an array of BigInts.
//  returns Normalised a and b to achieve better performance for MPdiv.
void normalize (vector<mpz_class> &a, vector<mpz_class> &b, mpz_class &shift)
{
    mpz_class bm = b[b.size() - 1];
    shift = 1; // shift cannot be larger than log2(base) - 1
    while (bm < (ScalarTwoTo256 / 2))
    {
        vector<mpz_class> aux;
        MP_short_mul(b, 2, aux); // left-shift b by 2
        b.swap(aux);
        bm = b[b.size() - 1];
        shift *= 2;
    }

    vector<mpz_class> result;
    MP_short_mul(a, shift, result); // left-shift a by 2^shift
    a.swap(result);
}

void _MPdiv_short (const vector<mpz_class> &a, const mpz_class &b, vector<mpz_class> &quotient, mpz_class &remainder);

// Computes the next digit of the quotient.
//  an - Unsigned integer represented as an array of BigInts. This is the current dividend.
//  b - Unsigned integer represented as an array of BigInts.
//  returns The next digit of the quotient.
void findQn (const vector<mpz_class> &an, const vector<mpz_class> &b, mpz_class &result)
{
    uint64_t b_l = b.size();
    mpz_class bm = b[b_l - 1];
    if (compare(an, b) == -1)
    {
        result = 0;
        return;
    }

    uint64_t n = an.size();
    vector<mpz_class> aguess;
    if (an[n-1] < bm)
    {
        aguess.emplace_back(an[n-2]);
        aguess.emplace_back(an[n-1]);
    }
    else
    {
        aguess.emplace_back(an[n-1]);
    }

    if (an[n-1] < bm)
    {
        vector<mpz_class> quotient;
        mpz_class remainder;
        _MPdiv_short(aguess, bm, quotient, remainder);
        if (quotient.size() == 0)
        {
            zklog.error("findQn() called _MPdiv_short() but got a quotient with size=0");
            exitProcess();
        }
        result = quotient[0]; // this is always a single digit
    }
    else if (an[n-1] == bm)
    {
        if (b_l < n)
        {
            result = ScalarTwoTo256 - 1;
            return;
        }
        else
        {
            result = 1;
            return;
        }
    }
    else
    {
        result = 1;
        return;
    }
}

// Computes the division of two unsigned integers represented as arrays of BigInts.
//  a - Unsigned integer represented as an array of BigInts.
//  b - Unsigned integer represented as an array of BigInts.
//  returns [quotient, remainder] of a / b.
void _MPdiv (vector<mpz_class> &a, vector<mpz_class> &b, vector<mpz_class> &quotient, vector<mpz_class> &remainder)
{
    mpz_class shift;
    normalize(a, b, shift);
    int64_t a_l = a.size();
    quotient.clear();
    remainder.clear();
    vector<mpz_class> an;
    while (compare(an, b) == -1)
    {
        an.emplace(an.begin(), a[--a_l]);
    }

    vector<mpz_class> test;
    mpz_class qn;
    while (a_l >= 0)
    {
        findQn(an, b, qn);
        MP_short_mul(b, qn, test);
        while (compare(test, an) == 1)
        {
            // maximum 2 iterations
            qn--;
            vector<mpz_class> aux;
            MP_sub(test, b, aux);
            test.swap(aux);
        }

        quotient.emplace(quotient.begin(), qn);
        MP_sub(an, test, remainder);
        an = remainder;
        if (a_l == 0)
        {
            break;
        }
        an.emplace(an.begin(), a[--a_l]);
    }
    vector<mpz_class> auxQuotient;
    mpz_class auxRemainder;
    _MPdiv_short(remainder, shift, auxQuotient, auxRemainder); // TODO: review with Carlos
    remainder.swap(auxQuotient);
    trim(quotient);
    trim(remainder);
}

// Computes the division of an unsigned integer represented as an array of BigInts and an unsigned integer represented as a BigInt.
//  a - Unsigned integer represented as an array of BigInts.
//  b - Unsigned integer represented as a BigInt.
//  returns [quotient, remainder] of a / b.
void _MPdiv_short (const vector<mpz_class> &a, const mpz_class &b, vector<mpz_class> &quotient, mpz_class &remainder)
{
    uint64_t a_l = a.size();
    quotient.clear();
    quotient.insert(quotient.begin(), a_l, ScalarZero);
    remainder = 0;

    mpz_class dividendi;
    mpz_class qi;
    for (int64_t i = a_l - 1; i >= 0; i--)
    {
        dividendi = remainder*ScalarTwoTo256 + a[i];
        qi = dividendi / b;
        remainder = dividendi - qi * b;
        quotient[i] = qi;
    }
    trim(quotient);
}

// Computes the division of two unsigned integers represented as arrays of BigInts.
//  sets ctx.quotient and ctx.remainder.
void eval_MPdiv (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 4)
    {
        zklog.error("eval_MPdiv() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get addr1 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_MPdiv() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t addr1 = cr.scalar.get_ui();

    // Get len1 by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_MPdiv() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t len1 = cr.scalar.get_ui();

    // Get addr2 by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_MPdiv() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t addr2 = cr.scalar.get_ui();

    // Get len2 by executing cmd.params[3]
    evalCommand(ctx, *cmd.params[3], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_MPdiv() 3 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t len2 = cr.scalar.get_ui();

    vector<mpz_class> input1;
    vector<mpz_class> input2;
    unordered_map<uint64_t, fork_12::Fea>::const_iterator it;
    mpz_class auxScalar;
    for (uint64_t i = 0; i < len1; i++)
    {
        it = ctx.mem.find(addr1 + i);
        if (it == ctx.mem.end())
        {
            zklog.error("eval_MPdiv() cound not find ctx.mem entry for address=" + to_string(addr1 + i) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            exitProcess();
        }
        if (!fea2scalar(fr, auxScalar, it->second.fe0, it->second.fe1, it->second.fe2, it->second.fe3, it->second.fe4, it->second.fe5, it->second.fe6, it->second.fe7))
        {
            zklog.error("eval_MPdiv() failed calling fea2scalar for address=" + to_string(addr1 + i) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            exitProcess();
        }
        input1.emplace_back(auxScalar);
    }
    for (uint64_t i = 0; i < len2; i++)
    {
        it = ctx.mem.find(addr2 + i);
        if (it == ctx.mem.end())
        {
            zklog.error("eval_MPdiv() cound not find ctx.mem entry for address=" + to_string(addr2 + i) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            exitProcess();
        }
        if (!fea2scalar(fr, auxScalar, it->second.fe0, it->second.fe1, it->second.fe2, it->second.fe3, it->second.fe4, it->second.fe5, it->second.fe6, it->second.fe7))
        {
            zklog.error("eval_MPdiv() failed calling fea2scalar for address=" + to_string(addr2 + i) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            exitProcess();
        }
        input2.emplace_back(auxScalar);
    }

    _MPdiv(input1, input2, ctx.quotient, ctx.remainder);
}

void eval_MPdiv_short (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 3)
    {
        zklog.error("eval_MPdiv_short() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get addr1 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_MPdiv_short() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t addr1 = cr.scalar.get_ui();

    // Get len1 by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_MPdiv_short() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t len1 = cr.scalar.get_ui();

    // Get addr2 by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_MPdiv_short() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    mpz_class &input2 = cr.scalar;

    vector<mpz_class> input1;
    unordered_map<uint64_t, fork_12::Fea>::const_iterator it;
    mpz_class auxScalar;
    for (uint64_t i = 0; i < len1; i++)
    {
        it = ctx.mem.find(addr1 + i);
        if (it == ctx.mem.end())
        {
            zklog.error("eval_MPdiv_short() cound not find ctx.mem entry for address=" + to_string(addr1 + i) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            exitProcess();
        }
        if (!fea2scalar(fr, auxScalar, it->second.fe0, it->second.fe1, it->second.fe2, it->second.fe3, it->second.fe4, it->second.fe5, it->second.fe6, it->second.fe7))
        {
            zklog.error("eval_MPdiv_short() failed calling fea2scalar for address=" + to_string(addr1 + i) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            exitProcess();
        }
        input1.emplace_back(auxScalar);
    }
    
    _MPdiv_short(input1, input2, ctx.quotientShort, ctx.remainderShort);
}

void eval_receiveLenQuotient_short (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    cr.type = crt_scalar;
    cr.scalar = ctx.quotientShort.size();
}

void eval_receiveQuotientChunk_short (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 1)
    {
        zklog.error("eval_receiveQuotientChunk_short() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get position by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_u64)
    {
        zklog.error("eval_receiveQuotientChunk_short() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t position = cr.u64;

    if (position >= ctx.quotientShort.size())
    {
        zklog.error("eval_receiveQuotientChunk_short() 0 unexpected position=" + to_string(position) + " >= ctx.quotientShort.size()=" + to_string(ctx.quotientShort.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }

    cr.type = crt_scalar;
    cr.scalar = ctx.quotientShort[position];
}

void eval_receiveRemainderChunk_short (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    cr.type = crt_scalar;
    cr.scalar = ctx.remainderShort;
}

void eval_receiveLenRemainder (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    cr.type = crt_scalar;
    cr.scalar = ctx.remainder.size();
}

void eval_receiveRemainderChunk (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 1)
    {
        zklog.error("eval_receiveRemainderChunk() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get position by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_receiveRemainderChunk() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t position = cr.scalar.get_ui();

    if (position >= ctx.remainder.size())
    {
        zklog.error("eval_receiveRemainderChunk() 0 unexpected position=" + to_string(position) + " >= ctx.remainder.size()=" + to_string(ctx.remainder.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }

    cr.type = crt_scalar;
    cr.scalar = ctx.remainder[position];
}

void eval_receiveLenQuotient (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    cr.type = crt_scalar;
    cr.scalar = ctx.quotient.size();
}

void eval_receiveQuotientChunk (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 1)
    {
        zklog.error("eval_receiveQuotientChunk() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get position by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_u64)
    {
        zklog.error("eval_receiveQuotientChunk() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    uint64_t position = cr.u64;

    if (position >= ctx.quotient.size())
    {
        zklog.error("eval_receiveQuotientChunk() 0 unexpected position=" + to_string(position) + " >= ctx.quotient.size()=" + to_string(ctx.quotient.size()) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }

    cr.type = crt_scalar;
    cr.scalar = ctx.quotient[position];
}

// Length of the binary representation of the input scalar. If there are multiple input scalars, it returns the maximum length
void eval_receiveLen (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() == 0)
    {
        zklog.error("eval_receiveLen() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    uint64_t len = 0;

    for (uint64_t i = 0; i < cmd.params.size(); i++)
    {
        evalCommand(ctx, *cmd.params[i], cr);
        if (cr.zkResult != ZKR_SUCCESS)
        {
            return;
        }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
        if (cr.type != crt_scalar)
        {
            zklog.error("eval_receiveLen() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
            exitProcess();
        }
#endif
        mpz_class ki = cr.scalar;
        if (ki == 0)
        {
            continue;
        }

        uint64_t leni = 0;
        while (ki != 1)
        {
            ki >>= 1;
            leni++;
        }
        len = zkmax(len, leni);
    }

    cr.type = crt_u64;
    cr.u64 = len;
}

void eval_ARITH_BN254_MULFP2_X (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 4)
    {
        zklog.error("eval_ARITH_BN254_MULFP2_X() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get x1 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_ARITH_BN254_MULFP2_X() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element x1;
    fq.fromMpz(x1, cr.scalar.get_mpz_t());

    // Get y1 by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_ARITH_BN254_MULFP2_X() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element y1;
    fq.fromMpz(y1, cr.scalar.get_mpz_t());

    // Get x2 by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_ARITH_BN254_MULFP2_X() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element x2;
    fq.fromMpz(x2, cr.scalar.get_mpz_t());

    // Get y2 by executing cmd.params[3]
    evalCommand(ctx, *cmd.params[3], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_ARITH_BN254_MULFP2_X() 3 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element y2;
    fq.fromMpz(y2, cr.scalar.get_mpz_t());

    // Calculate the point coordinate: x1*x2 - y1*y2
    RawFq::Element result;
    result = fq.sub(fq.mul(x1, x2), fq.mul(y1, y2));

    // Convert result to scalar
    cr.type = crt_scalar;
    fq.toMpz(cr.scalar.get_mpz_t(), result);
}

void eval_ARITH_BN254_MULFP2_Y (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 4)
    {
        zklog.error("eval_ARITH_BN254_MULFP2_Y() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get x1 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_ARITH_BN254_MULFP2_Y() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element x1;
    fq.fromMpz(x1, cr.scalar.get_mpz_t());

    // Get y1 by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_ARITH_BN254_MULFP2_Y() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element y1;
    fq.fromMpz(y1, cr.scalar.get_mpz_t());

    // Get x2 by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_ARITH_BN254_MULFP2_Y() 2 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element x2;
    fq.fromMpz(x2, cr.scalar.get_mpz_t());

    // Get y2 by executing cmd.params[3]
    evalCommand(ctx, *cmd.params[3], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_ARITH_BN254_MULFP2_Y() 3 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element y2;
    fq.fromMpz(y2, cr.scalar.get_mpz_t());

    // Calculate the point coordinate: x1*y2 + x2*y1
    RawFq::Element result;
    result = fq.add(fq.mul(x1, y2), fq.mul(x2, y1));

    // Convert result to scalar
    cr.type = crt_scalar;
    fq.toMpz(cr.scalar.get_mpz_t(), result);
}

void eval_ARITH_BN254_ADDFP2 (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 2)
    {
        zklog.error("eval_ARITH_BN254_ADDFP2() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get x1 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_ARITH_BN254_ADDFP2() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element x1;
    fq.fromMpz(x1, cr.scalar.get_mpz_t());

    // Get x2 by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_ARITH_BN254_ADDFP2() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element x2;
    fq.fromMpz(x2, cr.scalar.get_mpz_t());

    // Calculate the point coordinate
    RawFq::Element result;
    result = fq.add(x1, x2);

    // Convert result to scalar
    cr.type = crt_scalar;
    fq.toMpz(cr.scalar.get_mpz_t(), result);
}

void eval_ARITH_BN254_SUBFP2 (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 2)
    {
        zklog.error("eval_ARITH_BN254_SUBFP2() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get x1 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_ARITH_BN254_SUBFP2() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element x1;
    fq.fromMpz(x1, cr.scalar.get_mpz_t());

    // Get x2 by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_ARITH_BN254_SUBFP2() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element x2;
    fq.fromMpz(x2, cr.scalar.get_mpz_t());

    // Calculate the point coordinate
    RawFq::Element result;
    result = fq.sub(x1, x2);

    // Convert result to scalar
    cr.type = crt_scalar;
    fq.toMpz(cr.scalar.get_mpz_t(), result);
}

// Computes the "real" part of the inverse of the given Fp2 element.
void eval_fp2InvBN254_x (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 2)
    {
        zklog.error("eval_fp2InvBN254_x() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_fp2InvBN254_x() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element a;
    fq.fromMpz(a, cr.scalar.get_mpz_t());

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_fp2InvBN254_x() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element b;
    fq.fromMpz(b, cr.scalar.get_mpz_t());

    // Calculate the denominator
    RawFq::Element den;
    den = fq.add(fq.mul(a, a), fq.mul(b, b));

    // Calculate x
    RawFq::Element result;
    fq.div(result, a, den);

    // Convert back to scalar
    fq.toMpz(cr.scalar.get_mpz_t(), result);
}

// Computes the "imaginary" part of the inverse of the given Fp2 element.
void eval_fp2InvBN254_y (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 2)
    {
        zklog.error("eval_fp2InvBN254_y() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_fp2InvBN254_y() 0 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element a;
    fq.fromMpz(a, cr.scalar.get_mpz_t());

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_fp2InvBN254_y() 1 unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif
    RawFq::Element b;
    fq.fromMpz(b, cr.scalar.get_mpz_t());

    // Calculate the denominator
    RawFq::Element den;
    den = fq.add(fq.mul(a, a), fq.mul(b, b));

    // Calculate y
    RawFq::Element result;
    fq.div(result, fq.neg(b), den);

    // Convert back to scalar
    fq.toMpz(cr.scalar.get_mpz_t(), result);
}

// Computes the inverse of the given Fp element
void eval_fpBN254inv (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 1)
    {
        zklog.error("eval_fpBN254inv() invalid number of parameters=" + to_string(cmd.params.size()) + " function " + function2String(cmd.function) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.zkResult != ZKR_SUCCESS)
    {
        return;
    }
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        zklog.error("eval_fpBN254inv() unexpected command result type: " + to_string(cr.type) + " step=" + to_string(*ctx.pStep) + " zkPC=" + to_string(*ctx.pZKPC) + " line=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " uuid=" + ctx.proverRequest.uuid);
        exitProcess();
    }
#endif

    // Get the field element from the command result
    RawFq::Element a;
    fq.fromMpz(a, cr.scalar.get_mpz_t());

    // Calculate the inverse of this field element
    RawFq::Element aInv;
    fq.inv(aInv, a);

    // Convert back to scalar
    fq.toMpz(cr.scalar.get_mpz_t(), aInv);
}

void CommandResult::toFea (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7)
{
    // Copy fi=command result, depending on its type 
    switch (type)
    {
    case crt_fea:
        fi0 = fea0;
        fi1 = fea1;
        fi2 = fea2;
        fi3 = fea3;
        fi4 = fea4;
        fi5 = fea5;
        fi6 = fea6;
        fi7 = fea7;
        break;
    case crt_fe:
        fi0 = fe;
        fi1 = fr.zero();
        fi2 = fr.zero();
        fi3 = fr.zero();
        fi4 = fr.zero();
        fi5 = fr.zero();
        fi6 = fr.zero();
        fi7 = fr.zero();
        break;
    case crt_scalar:
        scalar2fea(fr, scalar, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
        break;
    case crt_u16:
        fi0 = fr.fromU64(u16);
        fi1 = fr.zero();
        fi2 = fr.zero();
        fi3 = fr.zero();
        fi4 = fr.zero();
        fi5 = fr.zero();
        fi6 = fr.zero();
        fi7 = fr.zero();
        break;
    case crt_u32:
        fi0 = fr.fromU64(u32);
        fi1 = fr.zero();
        fi2 = fr.zero();
        fi3 = fr.zero();
        fi4 = fr.zero();
        fi5 = fr.zero();
        fi6 = fr.zero();
        fi7 = fr.zero();
        break;
    case crt_u64:
        fi0 = fr.fromU64(u64);
        fi1 = fr.zero();
        fi2 = fr.zero();
        fi3 = fr.zero();
        fi4 = fr.zero();
        fi5 = fr.zero();
        fi6 = fr.zero();
        fi7 = fr.zero();
        break;
    default:
        zklog.error("CommandResult::toFea() Unexpected command result type: " + to_string(type));
        exitProcess();
    }
}

} // namespace