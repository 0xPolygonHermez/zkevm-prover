#include <iostream>
#include "definitions.hpp"
#include "config.hpp"
#include "main_sm/fork_1/main/eval_command.hpp"
#include "main_sm/fork_1/main/opcode_address.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "zkassert.hpp"

namespace fork_1
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
            default:
                cerr << "Error: evalCommand() found invalid function=" << cmd.function << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
            cerr << "Error: evalCommand() found invalid operation=" << op2String(cmd.op) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_declareVar() Variable name not found" << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }

    // Check that this variable does not exists
    if ( (cmd.varName[0] != '_') && (ctx.vars.find(cmd.varName) != ctx.vars.end()) )
    {
        cerr << "Error: eval_declareVar() Variable already declared: " << cmd.varName << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Create the new variable with a zero value
    ctx.vars[cmd.varName] = 0;

#ifdef LOG_VARIABLES
    cout << "Declare variable: " << cmd.varName << endl;
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
        cerr << "Error: eval_getVar() Variable name not found" << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Check that this variable exists
    std::unordered_map<std::string, mpz_class>::iterator it = ctx.vars.find(cmd.varName);
    if (it == ctx.vars.end())
    {
        cerr << "Error: eval_getVar() Undefined variable: " << cmd. varName << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }

#ifdef LOG_VARIABLES
    cout << "Get variable: " << cmd.varName << " scalar: " << ctx.vars[cmd.varName].get_str(16) << endl;
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
        cerr << "Error: eval_setVar() could not find array values in setVar command" << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get varName from the first element in values
    eval_left(ctx,*cmd.values[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_string)
    {
        cerr << "Error: eval_setVar() unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    string varName = cr.str;

    // Check that this variable exists
    std::unordered_map<std::string, mpz_class>::iterator it = ctx.vars.find(varName);
    if (it == ctx.vars.end())
    {
        cerr << "Error: eval_setVar() Undefined variable: " << varName << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }

    // Call evalCommand() to build the field element value for this variable
    evalCommand(ctx, *cmd.values[1], cr);

    // Get the field element value from the command result
    mpz_class auxScalar;
    cr2scalar(ctx, cr, auxScalar);

    // Store the value as the new variable value
    it->second = auxScalar;

    // Return the current value of the variable
    cr.type = crt_scalar;
    cr.scalar = auxScalar;

#ifdef LOG_VARIABLES
    cout << "Set variable: " << varName << " scalar: " << ctx.vars[varName].get_str(16) << endl;
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
            cerr << "Error: eval_left() invalid left expression, op: " << op2String(cmd.op) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
            fea2scalar(ctx.fr, cr.scalar, ctx.pols.A0[*ctx.pStep], ctx.pols.A1[*ctx.pStep], ctx.pols.A2[*ctx.pStep], ctx.pols.A3[*ctx.pStep], ctx.pols.A4[*ctx.pStep], ctx.pols.A5[*ctx.pStep], ctx.pols.A6[*ctx.pStep], ctx.pols.A7[*ctx.pStep]);
            break;
        case reg_B:
            cr.type = crt_scalar;
            fea2scalar(ctx.fr, cr.scalar, ctx.pols.B0[*ctx.pStep], ctx.pols.B1[*ctx.pStep], ctx.pols.B2[*ctx.pStep], ctx.pols.B3[*ctx.pStep], ctx.pols.B4[*ctx.pStep], ctx.pols.B5[*ctx.pStep], ctx.pols.B6[*ctx.pStep], ctx.pols.B7[*ctx.pStep]);
            break;
        case reg_C:
            cr.type = crt_scalar;
            fea2scalar(ctx.fr, cr.scalar, ctx.pols.C0[*ctx.pStep], ctx.pols.C1[*ctx.pStep], ctx.pols.C2[*ctx.pStep], ctx.pols.C3[*ctx.pStep], ctx.pols.C4[*ctx.pStep], ctx.pols.C5[*ctx.pStep], ctx.pols.C6[*ctx.pStep], ctx.pols.C7[*ctx.pStep]);
            break;
        case reg_D:
            cr.type = crt_scalar;
            fea2scalar(ctx.fr, cr.scalar, ctx.pols.D0[*ctx.pStep], ctx.pols.D1[*ctx.pStep], ctx.pols.D2[*ctx.pStep], ctx.pols.D3[*ctx.pStep], ctx.pols.D4[*ctx.pStep], ctx.pols.D5[*ctx.pStep], ctx.pols.D6[*ctx.pStep], ctx.pols.D7[*ctx.pStep]);
            break;
        case reg_E:
            cr.type = crt_scalar;
            fea2scalar(ctx.fr, cr.scalar, ctx.pols.E0[*ctx.pStep], ctx.pols.E1[*ctx.pStep], ctx.pols.E2[*ctx.pStep], ctx.pols.E3[*ctx.pStep], ctx.pols.E4[*ctx.pStep], ctx.pols.E5[*ctx.pStep], ctx.pols.E6[*ctx.pStep], ctx.pols.E7[*ctx.pStep]);
            break;
        case reg_SR:
            cr.type = crt_scalar;
            fea2scalar(ctx.fr, cr.scalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep]);
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
            cerr << "Error: eval_getReg() Invalid register: " << reg2string(cmd.reg) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
            cerr << "Error: cr2fe() unexpected type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
            cerr << "Error: cr2scalar() unexpected type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_add() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_sub() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_neg() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
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
        cerr << "Error: eval_mul() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx, cr, b);

    mpz_class mask256;
    mpz_class one(1);
    mask256 = (one << 256) - one;

    cr.type = crt_scalar;
    cr.scalar = a * b;
}

void eval_div(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check number of values
    if (cmd.values.size() != 2)
    {
        cerr << "Error: eval_div() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_mod() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_logical_or() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_logical_and() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_logical_gt() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_logical_ge() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_logical_lt() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_logical_le() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_logical_eq() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_logical_ne() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_logical_not() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
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
        cerr << "Error: eval_bit_and() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_bit_or() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_bit_xor() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_bit_not() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
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
        cerr << "Error: eval_bit_shl() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_bit_shr() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
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
        cerr << "Error: eval_if() found invalid number of values=" << cmd.values.size() << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx, cr, a);

    if (a)
    {
        evalCommand(ctx, *cmd.values[1], cr);
        mpz_class b;
        cr2scalar(ctx, cr, b);

        cr.type = crt_scalar;
        cr.scalar = b;
    }
    else
    {
        evalCommand(ctx, *cmd.values[2], cr);
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
    Fea fea = ctx.mem[cmd.offset];
    fea2scalar(ctx.fr, cr.scalar, fea.fe0, fea.fe1, fea.fe2, fea.fe3, fea.fe4, fea.fe5, fea.fe6, fea.fe7);
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
        cerr << "Error: eval_getGlobalExitRoot() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_getSequencerAddr() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_getTxsLen() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_getTxs() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get offset by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_getTxs() 1 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    uint64_t offset = cr.scalar.get_ui();

    // Get offset by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_getTxs() 2 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_eventLog() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    zkassert(ctx.proverRequest.input.publicInputsExtended.publicInputs.forkID == 1);
    cr.zkResult = ((fork_1::FullTracer *)ctx.proverRequest.pFullTracer)->handleEvent(ctx, cmd);

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
        cerr << "Error: eval_getTimestamp() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_cond() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get offset by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_cond() unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_exp() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_exp() 1 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_exp() 2 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_bitwise_and() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_bitwise_and() 1 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_bitwise_and() 2 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_bitwise_or() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_bitwise_or() 1 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_bitwise_or() 2 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_bitwise_xor() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_bitwise_xor() 1 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_bitwise_xor() 2 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_bitwise_not() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_bitwise_not() unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_beforeLast() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_comp_lt() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_comp_lt() 1 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_comp_lt() 2 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_comp_gt() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_comp_gt() 1 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_comp_gt() 2 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_comp_eq() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_comp_eq() 1 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_comp_eq() 2 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_loadScalar() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    evalCommand(ctx, *cmd.params[0], cr);
}

/*************/
/* Store log */
/*************/

void eval_storeLog (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 3)
    {
        cerr << "Error: eval_storeLog() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get indexLog by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_storeLog() param 0 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    uint64_t indexLog = cr.scalar.get_ui();;

    // Get isTopic by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_storeLog() param 1 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    uint32_t isTopic = cr.scalar.get_ui();

    // Get isTopic by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_storeLog() param 2 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    mpz_class data = cr.scalar;

    if (ctx.outLogs.find(indexLog) == ctx.outLogs.end())
    {
        OutLog outLog;
        ctx.outLogs[indexLog] = outLog;
    }

    if (isTopic)
    {
        ctx.outLogs[indexLog].topics.push_back(data.get_str(16));
    }
    else
    {
        ctx.outLogs[indexLog].data.push_back(data.get_str(16));
    }

    zkassert(ctx.proverRequest.input.publicInputsExtended.publicInputs.forkID == 1);
    ((fork_1::FullTracer *)ctx.proverRequest.pFullTracer)->handleEvent(ctx, cmd);

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
        cerr << "Error: eval_log() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get indexLog by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);

    mpz_class scalarLog;
    switch (cr.type)
    {
        case crt_fea:
            fea2scalar(ctx.fr, scalarLog, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
            break;
        case crt_u64:
            scalarLog = cr.u64;
            break;
        default:
            cerr << "Error: eval_storeLog() 1 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
            exitProcess();
    }

    // Print the log
    string hexLog = Add0xIfMissing(scalarLog.get_str(16));
    cout << "Log regname=" << reg2string(cmd.params[0]->reg) << " hexLog=" << hexLog << endl;

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
        cerr << "Error: eval_memAlignWR_W0() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get m0 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_memAlignWR_W0() 0 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    mpz_class m0 = cr.scalar;

    // Get value by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_memAlignWR_W0() 1 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    mpz_class value = cr.scalar;

    // Get offset by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_memAlignWR_W0() 2 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_memAlignWR_W1() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get m1 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_memAlignWR_W1() 0 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    mpz_class m1 = cr.scalar;

    // Get value by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_memAlignWR_W1() 1 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    mpz_class value = cr.scalar;

    // Get offset by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_memAlignWR_W1() 2 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_memAlignWR8_W0() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get m0 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_memAlignWR8_W0() 0 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    mpz_class m0 = cr.scalar;

    // Get value by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_memAlignWR8_W0() 1 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    mpz_class value = cr.scalar;

    // Get offset by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_memAlignWR8_W0() 2 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_inverseFpEc() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_inverseFpEc() 0 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    RawFec::Element a;
    ctx.fec.fromString(a, cr.scalar.get_str(16), 16);
    if (ctx.fec.isZero(a))
    {
        cerr << "Error: eval_inverseFpEc() Division by zero" << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        cerr << "Error: eval_inverseFnEc() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_inverseFnEc() 0 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    RawFnec::Element a;
    ctx.fnec.fromMpz(a, cr.scalar.get_mpz_t());
    if (ctx.fnec.isZero(a))
    {
        cerr << "Error: eval_inverseFnEc() Division by zero" << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
        return 0;
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
                return 0;
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
    return 0;
}

void eval_sqrtFpEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != 1)
    {
        cerr << "Error: eval_sqrtFpEc() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_sqrtFpEc() 0 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    RawFec::Element pfe = ctx.fec.negOne();
    mpz_class p;
    ctx.fec.toMpz(p.get_mpz_t(), pfe);
    p++;
    mpz_class a = cr.scalar;
    cr.type = crt_scalar;
    cr.scalar = sqrtTonelliShanks(a, p);
}

/********************/
/* Point operations */
/********************/

void eval_AddPointEc (Context &ctx, const RomCommand &cmd, bool dbl, RawFec::Element &x3, RawFec::Element &y3);

void eval_xAddPointEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    RawFec::Element x3;
    RawFec::Element y3;
    eval_AddPointEc(ctx, cmd, false, x3, y3);
    cr.type = crt_scalar;
    ctx.fec.toMpz(cr.scalar.get_mpz_t(), x3);
}

void eval_yAddPointEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    RawFec::Element x3;
    RawFec::Element y3;
    eval_AddPointEc(ctx, cmd, false, x3, y3);
    cr.type = crt_scalar;
    ctx.fec.toMpz(cr.scalar.get_mpz_t(), y3);
}

void eval_xDblPointEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    RawFec::Element x3;
    RawFec::Element y3;
    eval_AddPointEc(ctx, cmd, true, x3, y3);
    cr.type = crt_scalar;
    ctx.fec.toMpz(cr.scalar.get_mpz_t(), x3);
}

void eval_yDblPointEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    RawFec::Element x3;
    RawFec::Element y3;
    eval_AddPointEc(ctx, cmd, true, x3, y3);
    cr.type = crt_scalar;
    ctx.fec.toMpz(cr.scalar.get_mpz_t(), y3);
}

void eval_AddPointEc (Context &ctx, const RomCommand &cmd, bool dbl, RawFec::Element &x3, RawFec::Element &y3)
{
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    // Check parameters list size
    if (cmd.params.size() != (dbl ? 2 : 4))
    {
        cerr << "Error: eval_AddPointEc() invalid number of parameters function " << function2String(cmd.function) << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif

    CommandResult cr;

    // Get x1 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_AddPointEc() 0 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
        exitProcess();
    }
#endif
    RawFec::Element x1;
    ctx.fec.fromMpz(x1, cr.scalar.get_mpz_t());

    // Get y1 by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
    if (cr.type != crt_scalar)
    {
        cerr << "Error: eval_AddPointEc() 1 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
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
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
        if (cr.type != crt_scalar)
        {
            cerr << "Error: eval_AddPointEc() 2 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
            exitProcess();
        }
#endif
        ctx.fec.fromMpz(x2, cr.scalar.get_mpz_t());

        // Get y2 by executing cmd.params[3]
        evalCommand(ctx, *cmd.params[3], cr);
#ifdef CHECK_EVAL_COMMAND_PARAMETERS
        if (cr.type != crt_scalar)
        {
            cerr << "Error: eval_AddPointEc() 3 unexpected command result type: " << cr.type << " step=" << *ctx.pStep << " zkPC=" << *ctx.pZKPC << " line=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << " uuid=" << ctx.proverRequest.uuid << endl;
            exitProcess();
        }
#endif
        ctx.fec.fromMpz(y2, cr.scalar.get_mpz_t());
    }

    RawFec::Element aux1, aux2, s;

    if (dbl)
    {
        // s = 3*x1*x1/2*y1
        ctx.fec.mul(aux1, x1, x1);
        ctx.fec.fromUI(aux2, 3);
        ctx.fec.mul(aux1, aux1, aux2);
        ctx.fec.add(aux2, y1, y1);
        ctx.fec.div(s, aux1, aux2);
        // TODO: y1 == 0 => division by zero ==> how manage?
    }
    else
    {
        // s = (y2-y1)/(x2-x1)
        ctx.fec.sub(aux1, y2, y1);
        ctx.fec.sub(aux2, x2, x1);
        ctx.fec.div(s, aux1, aux2);
        // TODO: deltaX == 0 => division by zero ==> how manage?
    }

    // x3 = s*s - (x1+x2)
    ctx.fec.mul(aux1, s, s);
    ctx.fec.add(aux2, x1, x2);
    ctx.fec.sub(x3, aux1, aux2);

    // y3 = s*(x1-x3) - y1
    ctx.fec.sub(aux1, x1, x3);;
    ctx.fec.mul(aux1, aux1, s);
    ctx.fec.sub(y3, aux1, y1);
}

void eval_addReadWriteAddress (Context &ctx, const mpz_class value)
{
    zkassert(ctx.proverRequest.input.publicInputsExtended.publicInputs.forkID == 1);
    ((fork_1::FullTracer *)ctx.proverRequest.pFullTracer)->addReadWriteAddress(
        ctx.pols.A0[0], ctx.pols.A1[0], ctx.pols.A2[0], ctx.pols.A3[0], ctx.pols.A4[0], ctx.pols.A5[0], ctx.pols.A6[0], ctx.pols.A7[0],
        ctx.pols.B0[0], ctx.pols.B1[0], ctx.pols.B2[0], ctx.pols.B3[0], ctx.pols.B4[0], ctx.pols.B5[0], ctx.pols.B6[0], ctx.pols.B7[0],
        value);
}

} // namespace