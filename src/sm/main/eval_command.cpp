#include <iostream>

#include "config.hpp"
#include "eval_command.hpp"
#include "scalar.hpp"
#include "definitions.hpp"
#include "opcode_address.hpp"
#include "utils.hpp"

// Forwar declarations of internal functions
void eval_number            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getReg            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_declareVar        (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_setVar            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getVar            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_add               (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_sub               (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_neg               (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_mul               (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_div               (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_mod               (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_logical_operation (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_bit_operation     (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_if                (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getMemValue       (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_functionCall      (Context &ctx, const RomCommand &cmd, CommandResult &cr);

void evalCommand (Context &ctx, const RomCommand &cmd, CommandResult &cr) {
    switch (cmd.op)
    {
        case op_number:         return eval_number(ctx, cmd, cr);
        case op_declareVar:     return eval_declareVar(ctx, cmd, cr);
        case op_setVar:         return eval_setVar(ctx, cmd, cr);
        case op_getVar:         return eval_getVar(ctx, cmd, cr);
        case op_getReg:         return eval_getReg(ctx, cmd, cr);
        case op_functionCall:   return eval_functionCall(ctx, cmd, cr);
        case op_add:            return eval_add(ctx, cmd, cr);
        case op_sub:            return eval_sub(ctx, cmd, cr);
        case op_neg:            return eval_neg(ctx, cmd, cr);
        case op_mul:            return eval_mul(ctx, cmd, cr);
        case op_div:            return eval_div(ctx, cmd, cr);
        case op_mod:            return eval_mod(ctx, cmd, cr);
        case op_or:
        case op_and:
        case op_gt:
        case op_ge:
        case op_lt:
        case op_le:
        case op_eq:
        case op_ne:
        case op_not:            return eval_logical_operation(ctx, cmd, cr);
        case op_bitand:
        case op_bitor:
        case op_bitxor:
        case op_bitnot:
        case op_shl:
        case op_shr:            return eval_bit_operation(ctx, cmd, cr);
        case op_if:             return eval_if(ctx, cmd, cr);
        case op_getMemValue:    return eval_getMemValue(ctx, cmd, cr);
        default:
            cerr << "Error: evalCommand() found invalid operation: " << op2String(cmd.op) << " zkPC=" << *ctx.pZKPC << endl;
            exitProcess();
    }
}

void eval_number(Context &ctx, const RomCommand &cmd, CommandResult &cr) {
    cr.type = crt_scalar;
    cr.scalar = cmd.num;
}

/*************/
/* Variables */
/*************/

/* Declares a new variable, and fails if it already exists */
void eval_declareVar (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check the variable name
    if (cmd.varName == "") {
        cerr << "Error: eval_declareVar() Variable name not found" << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();  
    }

    // Check that this variable does not exists
    if ( (cmd.varName[0] != '_') && (ctx.vars.find(cmd.varName) != ctx.vars.end()) ) {
        cerr << "Error: eval_declareVar() Variable already declared: " << cmd.varName << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

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
    // Check the variable name
    if (cmd.varName == "") {
        cerr << "Error: eval_getVar() Variable name not found" << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();  
    }

    // Check that this variable exists
    if ( ctx.vars.find(cmd.varName) == ctx.vars.end() ) {
        cerr << "Error: eval_getVar() Undefined variable: " << cmd. varName << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

#ifdef LOG_VARIABLES
    cout << "Get variable: " << cmd.varName << " scalar: " << ctx.vars[cmd.varName].get_str(16) << endl;
#endif

    // Return the current value of this variable
    cr.type = crt_scalar;
    cr.scalar = ctx.vars[cmd.varName];
}

void eval_left (Context &ctx, const RomCommand &cmd, CommandResult &cr);

/* Sets variable to value, and fails if it does not exist */
void eval_setVar (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check that tag contains a values array
    if (cmd.values.size()==0) {
        cerr << "Error: eval_setVar() could not find array values in setVar command" << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get varName from the first element in values
    eval_left(ctx,*cmd.values[0], cr);
    if (cr.type != crt_string) {
        cerr << "Error: eval_setVar() unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    string varName = cr.str;

    // Check that this variable exists
    if ( ctx.vars.find(varName) == ctx.vars.end() ) {
        cerr << "Error: eval_setVar() Undefined variable: " << varName << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Call evalCommand() to build the field element value for this variable
    evalCommand(ctx, *cmd.values[1], cr);

    // Get the field element value from the command result
    mpz_class auxScalar;
    cr2scalar(ctx.fr, cr, auxScalar);

    // Store the value as the new variable value
    ctx.vars[varName] = auxScalar;

    // Return the current value of the variable
    cr.type = crt_scalar;
    cr.scalar = auxScalar;

#ifdef LOG_VARIABLES
    cout << "Set variable: " << varName << " scalar: " << ctx.vars[varName].get_str(16) << endl;
#endif
}

void eval_left (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    if (cmd.op == op_declareVar) {
        eval_declareVar(ctx, cmd, cr);
        cr.type = crt_string;
        cr.str = cmd.varName;
        return;
    } else if (cmd.op == op_getVar) {
        cr.type = crt_string;
        cr.str = cmd.varName;
        return;
    }
    cerr << "Error: eval_left() invalid left expression, op: " << op2String(cmd.op) << " zkPC=" << *ctx.pZKPC << endl;
    exitProcess();
}

void eval_getReg (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Get registry value, with the proper registry type
    if (cmd.regName=="A") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, ctx.pols.A0[*ctx.pStep], ctx.pols.A1[*ctx.pStep], ctx.pols.A2[*ctx.pStep], ctx.pols.A3[*ctx.pStep], ctx.pols.A4[*ctx.pStep], ctx.pols.A5[*ctx.pStep], ctx.pols.A6[*ctx.pStep], ctx.pols.A7[*ctx.pStep]);
    } else if (cmd.regName=="B") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, ctx.pols.B0[*ctx.pStep], ctx.pols.B1[*ctx.pStep], ctx.pols.B2[*ctx.pStep], ctx.pols.B3[*ctx.pStep], ctx.pols.B4[*ctx.pStep], ctx.pols.B5[*ctx.pStep], ctx.pols.B6[*ctx.pStep], ctx.pols.B7[*ctx.pStep]);
    } else if (cmd.regName=="C") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, ctx.pols.C0[*ctx.pStep], ctx.pols.C1[*ctx.pStep], ctx.pols.C2[*ctx.pStep], ctx.pols.C3[*ctx.pStep], ctx.pols.C4[*ctx.pStep], ctx.pols.C5[*ctx.pStep], ctx.pols.C6[*ctx.pStep], ctx.pols.C7[*ctx.pStep]);
    } else if (cmd.regName=="D") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, ctx.pols.D0[*ctx.pStep], ctx.pols.D1[*ctx.pStep], ctx.pols.D2[*ctx.pStep], ctx.pols.D3[*ctx.pStep], ctx.pols.D4[*ctx.pStep], ctx.pols.D5[*ctx.pStep], ctx.pols.D6[*ctx.pStep], ctx.pols.D7[*ctx.pStep]);
    } else if (cmd.regName=="E") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, ctx.pols.E0[*ctx.pStep], ctx.pols.E1[*ctx.pStep], ctx.pols.E2[*ctx.pStep], ctx.pols.E3[*ctx.pStep], ctx.pols.E4[*ctx.pStep], ctx.pols.E5[*ctx.pStep], ctx.pols.E6[*ctx.pStep], ctx.pols.E7[*ctx.pStep]);
    } else if (cmd.regName=="SR") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep]);
    } else if (cmd.regName=="CTX") {
        cr.type = crt_u32;
        cr.u32 = ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep]);
    } else if (cmd.regName=="SP") {
        cr.type = crt_u16;
        cr.u16 = ctx.fr.toU64(ctx.pols.SP[*ctx.pStep]);
    } else if (cmd.regName=="PC") {
        cr.type = crt_u32;
        cr.u32 = ctx.fr.toU64(ctx.pols.PC[*ctx.pStep]);
    } else if (cmd.regName=="MAXMEM") {
        cr.type = crt_u32;
        cr.u32 = ctx.fr.toU64(ctx.pols.MAXMEM[*ctx.pStep]);
    } else if (cmd.regName=="GAS") {
        cr.type = crt_u64;
        cr.u64 = ctx.fr.toU64(ctx.pols.GAS[*ctx.pStep]);
    } else if (cmd.regName=="zkPC") {
        cr.type = crt_u32;
        cr.u32 = ctx.fr.toU64(ctx.pols.zkPC[*ctx.pStep]);
    } else if (cmd.regName=="RR") {
        cr.type = crt_u64;
        cr.u64 = ctx.fr.toU64(ctx.pols.RR[*ctx.pStep]);
    } else if (cmd.regName=="CNT_ARITH") {
        cr.type = crt_u64;
        cr.u64 = ctx.fr.toU64(ctx.pols.cntArith[*ctx.pStep]);
    } else if (cmd.regName=="CNT_BINARY") {
        cr.type = crt_u64;
        cr.u64 = ctx.fr.toU64(ctx.pols.cntBinary[*ctx.pStep]);
    } else if (cmd.regName=="CNT_KECCAK_F") {
        cr.type = crt_u64;
        cr.u64 = ctx.fr.toU64(ctx.pols.cntKeccakF[*ctx.pStep]);
    } else if (cmd.regName=="CNT_MEM_ALIGN") {
        cr.type = crt_u64;
        cr.u64 = ctx.fr.toU64(ctx.pols.cntMemAlign[*ctx.pStep]);
    } else if (cmd.regName=="CNT_PADDING_PG") {
        cr.type = crt_u64;
        cr.u64 = ctx.fr.toU64(ctx.pols.cntPaddingPG[*ctx.pStep]);
    } else if (cmd.regName=="CNT_POSEIDON_G") {
        cr.type = crt_u64;
        cr.u64 = ctx.fr.toU64(ctx.pols.cntPoseidonG[*ctx.pStep]);
    } else if (cmd.regName=="STEP") {
        cr.type = crt_u64;
        cr.u64 = *ctx.pStep;
    } else if (cmd.regName=="HASHPOS") {
        cr.type = crt_u64;
        cr.u64 = ctx.fr.toU64(ctx.pols.HASHPOS[*ctx.pStep]);
    } else {
        cerr << "Error: eval_getReg() Invalid register: " << cmd.regName << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
}

void cr2fe (Goldilocks &fr, const CommandResult &cr, Goldilocks::Element &fe)
{
    if (cr.type == crt_fe)
    {
        fe = cr.fe;
    }
    else if (cr.type == crt_scalar)
    {
        scalar2fe(fr, cr.scalar, fe);
    }
    else
    {
        cerr << "Error: cr2fe() unexpected type: " << cr.type << endl;
        exitProcess();
    }
}

void cr2scalar (Goldilocks &fr, const CommandResult &cr, mpz_class &s)
{
    if (cr.type == crt_scalar)
    {
        s = cr.scalar;
    }
    else if (cr.type == crt_fe)
    {
        fe2scalar(fr, s, cr.fe);
    }
    else if (cr.type == crt_u64)
    {
        s = cr.u64;
    }
    else if (cr.type == crt_u32)
    {
        s = cr.u32;
    }
    else if (cr.type == crt_u16)
    {
        s = cr.u16;
    }
    else
    {
        cerr << "Error: cr2scalar() unexpected type: " << cr.type << endl;
        exitProcess();
    }
}

void eval_add(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a + b;
}

void eval_sub(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a - b;
}

void eval_neg(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    cr.type = crt_scalar;
    cr.scalar = -a;
}

void eval_mul(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    mpz_class mask256;
    mpz_class one(1);
    mask256 = (one << 256) - one;

    cr.type = crt_scalar;
    cr.scalar = a * b;
}

void eval_div(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a / b;
}

void eval_mod(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a % b;
}

void eval_logical_operation (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    if (cmd.op == op_not)
    {
        cr.type = crt_scalar;
        cr.scalar = (a) ? 0 : 1;
        return;
    }

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    cr.type = crt_scalar;
    
         if (cmd.op == op_or ) cr.scalar = (a || b) ? 1 : 0;
    else if (cmd.op == op_and) cr.scalar = (a && b) ? 1 : 0;
    else if (cmd.op == op_eq ) cr.scalar = (a == b) ? 1 : 0;
    else if (cmd.op == op_ne ) cr.scalar = (a != b) ? 1 : 0;
    else if (cmd.op == op_gt ) cr.scalar = (a >  b) ? 1 : 0;
    else if (cmd.op == op_ge ) cr.scalar = (a >= b) ? 1 : 0;
    else if (cmd.op == op_lt ) cr.scalar = (a <  b) ? 1 : 0;
    else if (cmd.op == op_le ) cr.scalar = (a <= b) ? 1 : 0;
    else
    {
        cerr << "Error: eval_logical_operation() operation not defined: " << op2String(cmd.op) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
}

void eval_bit_operation (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    if (cmd.op == op_bitnot)
    {
        cr.type = crt_scalar;
        cr.scalar = ~a;
        return;
    }

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    cr.type = crt_scalar;
    
         if (cmd.op == op_bitor ) cr.scalar = a | b;
    else if (cmd.op == op_bitand) cr.scalar = a & b;
    else if (cmd.op == op_bitxor) cr.scalar = a ^ b;
    else if (cmd.op == op_shl   ) cr.scalar = (a << b.get_ui());
    else if (cmd.op == op_shr   ) cr.scalar = (a >> b.get_ui());
    else if (cmd.op == op_ge ) cr.scalar = (a >= b) ? 1 : 0;
    else if (cmd.op == op_lt ) cr.scalar = (a <  b) ? 1 : 0;
    else if (cmd.op == op_le ) cr.scalar = (a <= b) ? 1 : 0;
    else
    {
        cerr << "Error: eval_bit_operation() operation not defined: " << op2String(cmd.op) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
}

void eval_if (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    if (a)
    {
        evalCommand(ctx, *cmd.values[1], cr);
        mpz_class b;
        cr2scalar(ctx.fr, cr, b);

        cr.type = crt_scalar;
        cr.scalar = b;
    }
    else
    {
        evalCommand(ctx, *cmd.values[2], cr);
        mpz_class c;
        cr2scalar(ctx.fr, cr, c);

        cr.type = crt_scalar;
        cr.scalar = c;
    }
}

void eval_getMemValue (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    cr.type = crt_scalar;
    Fea fea = ctx.mem[cmd.offset];
    fea2scalar(ctx.fr, cr.scalar, fea.fe0, fea.fe1, fea.fe2, fea.fe3, fea.fe4, fea.fe5, fea.fe6, fea.fe7);
}

// Forward declaration of internal callable functions
void eval_getGlobalHash       (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getGlobalExitRoot   (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getOldStateRoot     (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getNewStateRoot     (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getOldLocalExitRoot (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getNewLocalExitRoot (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getNTxs             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getRawTx            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getSequencerAddr    (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getBatchNum         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getBatchHashData    (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getTxs              (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getTxsLen           (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_addrOp              (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_eventLog            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getTimestamp        (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_cond                (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_inverseFpEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_inverseFnEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_sqrtFpEc            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_xAddPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_yAddPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_xDblPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_yDblPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getBytecode         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_beforeLast          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_bitwise             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_comp                (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_loadScalar          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getGlobalExitRootManagerAddr (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_log                 (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_exp                 (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_storeLog            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_memAlignWR_W0       (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_memAlignWR_W1       (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_memAlignWR8_W0      (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_saveContractBytecode(Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_isWarmedAddress     (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_checkpoint          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_revert              (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_commit              (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_clearWarmedStorage  (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_isWarmedStorage     (Context &ctx, const RomCommand &cmd, CommandResult &cr);

void eval_functionCall (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    switch (cmd.function)
    {
        case f_getGlobalHash:                   return eval_getGlobalHash(ctx, cmd, cr);
        case f_getGlobalExitRoot:               return eval_getGlobalExitRoot(ctx, cmd, cr);      
        case f_getOldStateRoot:                 return eval_getOldStateRoot(ctx, cmd, cr);        
        case f_getNewStateRoot:                 return eval_getNewStateRoot(ctx, cmd, cr);          
        case f_getSequencerAddr:                return eval_getSequencerAddr(ctx, cmd, cr);         
        case f_getOldLocalExitRoot:             return eval_getOldLocalExitRoot(ctx, cmd, cr);           
        case f_getNewLocalExitRoot:             return eval_getNewLocalExitRoot(ctx, cmd, cr);           
        case f_getNumBatch:                     return eval_getBatchNum(ctx, cmd, cr);             
        case f_getTimestamp:                    return eval_getTimestamp(ctx, cmd, cr);             
        case f_getBatchHashData:                return eval_getBatchHashData(ctx, cmd, cr);             
        case f_getTxs:                          return eval_getTxs(ctx, cmd, cr);              
        case f_getTxsLen:                       return eval_getTxsLen(ctx, cmd, cr);             
        case f_addrOp:                          return eval_addrOp(ctx, cmd, cr);           
        case f_eventLog:                        return eval_eventLog(ctx, cmd, cr);           
        case f_cond:                            return eval_cond(ctx, cmd, cr);             
        case f_inverseFpEc:                     return eval_inverseFpEc(ctx, cmd, cr);               
        case f_inverseFnEc:                     return eval_inverseFnEc(ctx, cmd, cr);                
        case f_sqrtFpEc:                        return eval_sqrtFpEc(ctx, cmd, cr);               
        case f_xAddPointEc:                     return eval_xAddPointEc(ctx, cmd, cr);                
        case f_yAddPointEc:                     return eval_yAddPointEc(ctx, cmd, cr);                
        case f_xDblPointEc:                     return eval_xDblPointEc(ctx, cmd, cr);               
        case f_yDblPointEc:                     return eval_yDblPointEc(ctx, cmd, cr);               
        case f_getBytecode:                     return eval_getBytecode(ctx, cmd, cr);  
        case f_bitwise_and:
        case f_bitwise_or:
        case f_bitwise_xor:
        case f_bitwise_not:                     return eval_bitwise(ctx, cmd, cr);           
        case f_comp_lt:
        case f_comp_gt:
        case f_comp_eq:                         return eval_comp(ctx, cmd, cr);            
        case f_loadScalar:                      return eval_loadScalar(ctx, cmd, cr);            
        case f_getGlobalExitRootManagerAddr:    return eval_getGlobalExitRootManagerAddr(ctx, cmd, cr);           
        case f_log:                             return eval_log(ctx, cmd, cr);         
        case f_exp:                             return eval_exp(ctx, cmd, cr);           
        case f_storeLog:                        return eval_storeLog(ctx, cmd, cr);             
        case f_memAlignWR_W0:                   return eval_memAlignWR_W0(ctx, cmd, cr);            
        case f_memAlignWR_W1:                   return eval_memAlignWR_W1(ctx, cmd, cr);           
        case f_memAlignWR8_W0:                  return eval_memAlignWR8_W0(ctx, cmd, cr);           
        case f_saveContractBytecode:            return eval_saveContractBytecode(ctx, cmd, cr); 
        case f_beforeLast:                      return eval_beforeLast(ctx, cmd, cr);
        case f_isWarmedAddress:                 return eval_isWarmedAddress(ctx, cmd, cr);
        case f_checkpoint:                      return eval_checkpoint(ctx, cmd, cr);
        case f_revert:                          return eval_revert(ctx, cmd, cr);
        case f_commit:                          return eval_commit(ctx, cmd, cr);
        case f_clearWarmedStorage:              return eval_clearWarmedStorage(ctx, cmd, cr);
        case f_isWarmedStorage:                 return eval_isWarmedStorage(ctx, cmd, cr);
        default:
            cerr << "Error: eval_functionCall() function not defined: " << cmd.function << " zkPC=" << *ctx.pZKPC << endl;
            exitProcess();
    }
}

void eval_getGlobalHash(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getGlobalHash() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Return ctx.proverRequest.input.globalHash as a field element
    cr.type = crt_fea;
    scalar2fea(ctx.fr, ctx.proverRequest.input.globalHash, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getGlobalExitRoot(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getGlobalExitRoot() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Return ctx.proverRequest.input.globalExitRoot as a field element array
    cr.type = crt_fea;
    mpz_class globalExitRoot(ctx.proverRequest.input.globalExitRoot);
    scalar2fea(ctx.fr, globalExitRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getSequencerAddr(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getSequencerAddr() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Return ctx.proverRequest.input.publicInputs.sequencerAddr as a field element array
    cr.type = crt_fea;
    mpz_class sequencerAddr(ctx.proverRequest.input.publicInputs.sequencerAddr);
    scalar2fea(ctx.fr, sequencerAddr, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getBatchNum(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getBatchNum() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Return ctx.proverRequest.input.publicInputs.batchNum as a field element array
    cr.type = crt_fea;
    cr.fea0 = ctx.fr.fromU64(ctx.proverRequest.input.publicInputs.batchNum);
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

void eval_getOldStateRoot(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getOldStateRoot() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Return ctx.proverRequest.input.publicInputs.oldStateRoot as a field element array
    cr.type = crt_fea;
    mpz_class oldStateRoot(ctx.proverRequest.input.publicInputs.oldStateRoot); // This field could be parsed out of the main loop, but it is only called once
    scalar2fea(ctx.fr, oldStateRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getNewStateRoot(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getNewStateRoot() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Return ctx.proverRequest.input.publicInputs.newStateRoot as a field element array
    cr.type = crt_fea;
    mpz_class newStateRoot(ctx.proverRequest.input.publicInputs.newStateRoot); // This field could be parsed out of the main loop, but it is only called once
    scalar2fea(ctx.fr, newStateRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getOldLocalExitRoot(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getOldLocalExitRoot() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Return ctx.proverRequest.input.publicInputs.oldLocalExitRoot as a field element array
    cr.type = crt_fea;
    mpz_class oldLocalExitRoot(ctx.proverRequest.input.publicInputs.oldLocalExitRoot);
    scalar2fea(ctx.fr, oldLocalExitRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getNewLocalExitRoot(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getNewLocalExitRoot() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Return ctx.proverRequest.input.publicInputs.newLocalExitRoot as a field element array
    cr.type = crt_fea;
    mpz_class newLocalExitRoot(ctx.proverRequest.input.publicInputs.newLocalExitRoot);
    scalar2fea(ctx.fr, newLocalExitRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getTxsLen(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getTxsLen() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Return ctx.proverRequest.input.txsLen/2 as a field element array
    cr.type = crt_fea;
    cr.fea0 = ctx.fr.fromU64((ctx.proverRequest.input.batchL2Data.size() - 2) / 2);
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
    // Check parameters list size
    if (cmd.params.size() != 2) {
        cerr << "Error: eval_getTxs() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get offset by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_getTxs() 1 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    uint64_t offset = cr.scalar.get_ui();

    // Get offset by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_getTxs() 2 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    uint64_t len = cr.scalar.get_ui();

    string resultString = ctx.proverRequest.input.batchL2Data.substr(2+offset*2, len*2);
    if (resultString.size() == 0) resultString += "0";

    // Return result as a field element array
    mpz_class resultScalar(resultString, 16);
    cr.type = crt_fea;
    scalar2fea(ctx.fr, resultScalar, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getBatchHashData(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getBatchHashData() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Return ctx.proverRequest.input.batchHashData as a field element array
    cr.type = crt_fea;
    scalar2fea(ctx.fr, ctx.proverRequest.input.batchHashData, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_addrOp(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_addrOp() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get offset by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_fe) {
        cerr << "Error: eval_addrOp() unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    uint64_t codeId = ctx.fr.toU64(cr.fe);
    cr.type = crt_fea;

    uint64_t addr = opcodeAddress[codeId];
    cr.fea0 = ctx.fr.fromU64(addr);
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}
void eval_eventLog(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() < 1) {
        cerr << "Error: eval_eventLog() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    ctx.proverRequest.fullTracer.handleEvent(ctx, cmd);

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
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getTimestamp() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Return ctx.proverRequest.input.publicInputs.timestamp as a field element array
    cr.type = crt_fea;
    cr.fea0 = ctx.fr.fromU64(ctx.proverRequest.input.publicInputs.timestamp);
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

void eval_cond (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_cond() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get offset by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_fe) {
        cerr << "Error: eval_cond() unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    uint64_t result = ctx.fr.toU64(cr.fe);
    
    cr.type = crt_fea;
    if (result)
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

void eval_getBytecode (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 2 && cmd.params.size() != 3) {
        cerr << "Error: eval_getBytecode() invalid number of parameters function " << function2String(cmd.function) <<" zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get hashcontract by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_getBytecode() unexpected command 0 result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    string aux = cr.scalar.get_str(16);
    string hashcontract = NormalizeTo0xNFormat(aux, 64);

    // Get offset by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_u64) {
        cerr << "Error: eval_getBytecode() unexpected command 1 result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    uint64_t offset = cr.u64;

    // Get length by executing cmd.params[2]
    uint64_t len = 1;
    if (cmd.params.size() == 3)
    {
        evalCommand(ctx, *cmd.params[2], cr);
        if (cr.type != crt_scalar) {
            cerr << "Error: eval_getBytecode() unexpected command 2 result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
            exitProcess();
        }
        len = cr.scalar.get_si();
    }


    if (ctx.proverRequest.input.contractsBytecode.find(hashcontract) == ctx.proverRequest.input.contractsBytecode.end())
    {
        // Get the contract hash key
        mpz_class scalar(hashcontract);
        Goldilocks::Element key[4];
        scalar2fea(ctx.fr, scalar, key);

        // Get the contract from the database
        vector<uint8_t> bytecode;
        zkresult zkResult = ctx.pStateDB->getProgram(key, bytecode);
        if (zkResult != ZKR_SUCCESS)
        {
            cerr << "Error: eval_getBytecode() failed calling ctx.pStateDB->getProgram() with key=" << hashcontract << " zkResult=" << zkResult << "=" << zkresult2string(zkResult) << endl;
            cr.type = crt_fea;
            cr.fea0 = ctx.fr.zero();
            cr.fea1 = ctx.fr.zero();
            cr.fea2 = ctx.fr.zero();
            cr.fea3 = ctx.fr.zero();
            cr.fea4 = ctx.fr.zero();
            cr.fea5 = ctx.fr.zero();
            cr.fea6 = ctx.fr.zero();
            cr.fea7 = ctx.fr.zero();
            cr.zkResult = zkResult;
            return;
        }

        // Store the bytecode locally
        ctx.proverRequest.input.contractsBytecode[hashcontract] = bytecode;
    }

    string d = "0x";
    for (uint64_t i=offset; i<offset+len; i++)
    {
        d += byte2string(ctx.proverRequest.input.contractsBytecode[hashcontract][i]);
    }
    if (len == 0) d += "0";
    mpz_class auxScalar(d);
    cr.type = crt_fea;
    scalar2fea(ctx.fr, auxScalar, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

/* Creates new storage checkpoint for warm slots and addresses */
void eval_checkpoint (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Push a new map into accessedStorage
    map< mpz_class, set<mpz_class> > auxMap;
    ctx.accessedStorage.push_back(auxMap);

    // Return zero
    cr.type = crt_fea;
    cr.fea0 = ctx.fr.zero();
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
    return;
}

/* Consolidates checkpoint, merge last access storage with beforeLast access storage
 * ctx: current rom context object */

void eval_commit (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    if (ctx.accessedStorage.size() > 1)
    {
        // Extract the last accessedStorage item
        map< mpz_class, set<mpz_class> > storageMap;
        storageMap = ctx.accessedStorage[ctx.accessedStorage.size() - 1];
        ctx.accessedStorage.pop_back();
        
        if (ctx.accessedStorage.size() > 1)
        {
            // Iterate all storageMap addresses
            map< mpz_class, set<mpz_class> >::const_iterator storageMapIterator;
            for (storageMapIterator = storageMap.begin(); storageMapIterator != storageMap.end(); storageMapIterator++)
            {
                mpz_class address = storageMapIterator->first;

                // If addresss is not present in destination map, then create a new set
                if ( ctx.accessedStorage[ctx.accessedStorage.size() - 1].find(address) == ctx.accessedStorage[ctx.accessedStorage.size() - 1].end() )
                {
                    set<mpz_class> auxSet;
                    ctx.accessedStorage[ctx.accessedStorage.size() - 1][address] = auxSet;
                }

                // Iterate for this address
                set<mpz_class>::const_iterator storageSetIterator;
                for (storageSetIterator = storageMap[address].begin(); storageSetIterator != storageMap[address].end(); storageSetIterator++)
                {
                    mpz_class key = *storageSetIterator;
                    ctx.accessedStorage[ctx.accessedStorage.size() - 1][address].insert(key);
                }
            }
        }
    }

    // Return zero
    cr.type = crt_fea;
    cr.fea0 = ctx.fr.zero();
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
    return;
}

/* Revert accessedStorage to last checkpoint
 * ctx: current rom context object */
void eval_revert (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Remove the last element of accessedStorage
    ctx.accessedStorage.pop_back();

    // Return zero
    cr.type = crt_fea;
    cr.fea0 = ctx.fr.zero();
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
    return;
}

/* Checks if the address is warm or cold. In case of cold, the address is added as warm
 * ctx: current rom context object
 * tag: tag inputs in rom function
 * returns 0 (fea) if address is warm, 1 (fea) if cold */
void eval_isWarmedAddress (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_isWarmedAddress() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get addr by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_isWarmedAddress() 1 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    mpz_class address = cr.scalar;

    // if address is precompiled smart contract considered warm access
    if ((address>0) && (address<10))
    {
        cr.type = crt_fea;
        cr.fea0 = ctx.fr.zero();
        cr.fea1 = ctx.fr.zero();
        cr.fea2 = ctx.fr.zero();
        cr.fea3 = ctx.fr.zero();
        cr.fea4 = ctx.fr.zero();
        cr.fea5 = ctx.fr.zero();
        cr.fea6 = ctx.fr.zero();
        cr.fea7 = ctx.fr.zero();
        return;
    }
    
    // If address is warm return 0
    for (int64_t i = ctx.accessedStorage.size() - 1; i >= 0; i--)
    {
        if (ctx.accessedStorage[i].find(address) != ctx.accessedStorage[i].end())
        {
            cr.type = crt_fea;
            cr.fea0 = ctx.fr.zero();
            cr.fea1 = ctx.fr.zero();
            cr.fea2 = ctx.fr.zero();
            cr.fea3 = ctx.fr.zero();
            cr.fea4 = ctx.fr.zero();
            cr.fea5 = ctx.fr.zero();
            cr.fea6 = ctx.fr.zero();
            cr.fea7 = ctx.fr.zero();
            return;
        }
    }

    // If address is not warm, return 1 and add it as warm. We add an emtpy set because is a warmed address (not warmed slot)
    if (ctx.accessedStorage[ctx.accessedStorage.size() - 1].find(address) == ctx.accessedStorage[ctx.accessedStorage.size()-1].end())
    {
        set<mpz_class> auxSet;
        ctx.accessedStorage[ctx.accessedStorage.size()-1][address] = auxSet;
    }

    // Return 1
    cr.type = crt_fea;
    cr.fea0 = ctx.fr.one();
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

/* Checks if the storage slot of the account is warm or cold. In case of cold, the slot is added as warm
 * ctx: current rom context object
 * tag: tag inputs in rom function
 * returns 0 (fea) if storage solt is warm, 1 (fea) if cold */
void eval_isWarmedStorage (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 2) {
        cerr << "Error: eval_isWarmedStorage() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get addr by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_isWarmedStorage() 1 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    mpz_class address = cr.scalar;

    // Get key by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_isWarmedStorage() 2 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    mpz_class key = cr.scalar;

    // If address in touchedStorageSlots return 0
    for (int64_t i = ctx.accessedStorage.size() - 1; i >= 0; i--)
    {
        if (ctx.accessedStorage[i].find(address) != ctx.accessedStorage[i].end())
        {
            if (ctx.accessedStorage[i][address].find(key) != ctx.accessedStorage[i][address].end())
            {
                cr.type = crt_fea;
                cr.fea0 = ctx.fr.zero();
                cr.fea1 = ctx.fr.zero();
                cr.fea2 = ctx.fr.zero();
                cr.fea3 = ctx.fr.zero();
                cr.fea4 = ctx.fr.zero();
                cr.fea5 = ctx.fr.zero();
                cr.fea6 = ctx.fr.zero();
                cr.fea7 = ctx.fr.zero();
                return;
            }
        }
    }

    // If address in touchedStorageSlots return 1 and add it as warm
    if (ctx.accessedStorage[ctx.accessedStorage.size() - 1].find(address) == ctx.accessedStorage[ctx.accessedStorage.size() - 1].end())
    {
        set<mpz_class> storageSet;
        ctx.accessedStorage[ctx.accessedStorage.size() - 1][address] = storageSet;
    }
    ctx.accessedStorage[ctx.accessedStorage.size() - 1][address].insert(key);

    // Return 1
    cr.type = crt_fea;
    cr.fea0 = ctx.fr.one();
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

/* Clears wamred storage array, ready to process a new tx */
void eval_clearWarmedStorage (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Clear accessedStorage
    ctx.accessedStorage.clear();

    // Add an empty map
    map<mpz_class, set<mpz_class>> auxMap;
    ctx.accessedStorage.push_back(auxMap);

    // Return 0
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

void eval_exp (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 2) {
        cerr << "Error: eval_exp() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_exp() 1 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_exp() 2 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    mpz_class b = cr.scalar;

    mpz_class auxScalar;
    mpz_pow_ui(auxScalar.get_mpz_t(), a.get_mpz_t(), b.get_ui());
    cr.type = crt_fea;
    scalar2fea(ctx.fr, auxScalar, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_bitwise (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1 && cmd.params.size() != 2) {
        cerr << "Error: eval_bitwise() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_bitwise() 1 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    mpz_class a = cr.scalar;

    if (cmd.function ==f_bitwise_and)
    {
        // Check parameters list size
        if (cmd.params.size() != 2) {
            cerr << "Error: eval_bitwise() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
            exitProcess();
        }

        // Get b by executing cmd.params[1]
        evalCommand(ctx, *cmd.params[1], cr);
        if (cr.type != crt_scalar) {
            cerr << "Error: eval_bitwise() 2 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
            exitProcess();
        }
        mpz_class b = cr.scalar;

        cr.type = crt_scalar;
        cr.scalar = a & b;
    }
    else if (cmd.function == f_bitwise_or)
    {
        // Check parameters list size
        if (cmd.params.size() != 2) {
            cerr << "Error: eval_bitwise() invalid number of parameters function " << function2String(cmd.function)<< " zkPC=" << *ctx.pZKPC << endl;
            exitProcess();
        }

        // Get b by executing cmd.params[1]
        evalCommand(ctx, *cmd.params[1], cr);
        if (cr.type != crt_scalar) {
            cerr << "Error: eval_bitwise() 3 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
            exitProcess();
        }
        mpz_class b = cr.scalar;

        cr.type = crt_scalar;
        cr.scalar = a | b;
    }
    else if (cmd.function == f_bitwise_xor)
    {
        // Check parameters list size
        if (cmd.params.size() != 2) {
            cerr << "Error: eval_bitwise() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
            exitProcess();
        }

        // Get b by executing cmd.params[1]
        evalCommand(ctx, *cmd.params[1], cr);
        if (cr.type != crt_scalar) {
            cerr << "Error: eval_bitwise() 4 unexpected command result type: " << cr.type << endl;
            exitProcess();
        }
        mpz_class b = cr.scalar;

        cr.type = crt_scalar;
        cr.scalar = a ^ b;
    }
    else if (cmd.function == f_bitwise_not)
    {
        // Check parameters list size
        if (cmd.params.size() != 1) {
            cerr << "Error: eval_bitwise() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
            exitProcess();
        }

        cr.type = crt_scalar;
        cr.scalar = a ^ Mask256;
    }
    else
    {
        cerr << "Error: eval_bitwise() invalid operation funcName=" << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
}

void eval_beforeLast (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_beforeLast() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // We record that this function was called in order to keep track of the last step
    cr.beforeLast = true;

    // Return a field element array
    cr.type = crt_fea;
    if (*ctx.pStep >= ctx.N-2) {
        cr.fea0 = ctx.fr.zero();
    } else {
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

void eval_comp (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 2) {
        cerr << "Error: eval_comp() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_comp() 1 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_comp() 2 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    mpz_class b = cr.scalar;

    cr.type = crt_scalar;
    if (cmd.function == f_comp_lt)
    {
        cr.scalar = (a < b) ? 1 : 0;
    }
    else if (cmd.function == f_comp_gt)
    {
        cr.scalar = (a > b) ? 1 : 0;
    }
    else if (cmd.function == f_comp_eq)
    {
        cr.scalar = (a = b) ? 1 : 0;
    }
    else
    {
        cerr << "Error: eval_comp() Invalid bitwise operation funcName=" << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
}

void eval_loadScalar (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_loadScalar() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    evalCommand(ctx, *cmd.params[0], cr);
}

// Will be replaced by hardcoding this address directly in the ROM once the CONST register can be 256 bits
void eval_getGlobalExitRootManagerAddr (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getGlobalExitRootManagerAddr() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Return ctx.proverRequest.input.publicInputs.oldLocalExitRoot as a field element array
    cr.type = crt_fea;
    mpz_class globalExitRootManagerAddr(ADDRESS_GLOBAL_EXIT_ROOT_MANAGER_L2);
    scalar2fea(ctx.fr, globalExitRootManagerAddr, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_storeLog (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 3) {
        cerr << "Error: eval_storeLog() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get indexLog by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_storeLog() param 0 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    uint64_t indexLog = cr.scalar.get_ui();;

    // Get isTopic by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_storeLog() param 1 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    uint32_t isTopic = cr.scalar.get_ui();

    // Get isTopic by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_storeLog() param 2 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    mpz_class data = cr.scalar;

    if (ctx.outLogs.find(indexLog) == ctx.outLogs.end())
    {
        OutLog outLog;
        ctx.outLogs[indexLog] = outLog;
    }

    if (isTopic) {
        ctx.outLogs[indexLog].topics.push_back(data.get_str(16));
    } else {
        ctx.outLogs[indexLog].data.push_back(data.get_str(16));
    }

    ctx.proverRequest.fullTracer.handleEvent(ctx, cmd);

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

void eval_log (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_log() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get indexLog by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_fea) {
        cerr << "Error: eval_storeLog() 1 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Print the log
    mpz_class scalarLog;
    fea2scalar(ctx.fr, scalarLog, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
    string hexLog = Add0xIfMissing(scalarLog.get_str(16));
    cout << "Log regname=" << cmd.params[0]->regName << " hexLog=" << hexLog << endl;

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

void eval_memAlignWR_W0 (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 3) {
        cerr << "Error: eval_memAlignWR_W0() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get m0 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_memAlignWR_W0() 0 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    mpz_class m0 = cr.scalar;

    // Get value by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_memAlignWR_W0() 1 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    mpz_class value = cr.scalar;

    // Get offset by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_memAlignWR_W0() 2 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    int64_t offset = cr.scalar.get_si();

    int64_t shiftLeft = (32 - offset) * 8;
    int64_t shiftRight = offset * 8;
    mpz_class result = (m0 & (Mask256 << shiftLeft)) | (Mask256 & (value >> shiftRight));
    
    cr.type = crt_fea;
    scalar2fea(ctx.fr, result, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_memAlignWR_W1 (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 3) {
        cerr << "Error: eval_memAlignWR_W1() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get m1 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_memAlignWR_W1() 0 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    mpz_class m1 = cr.scalar;

    // Get value by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_memAlignWR_W1() 1 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    mpz_class value = cr.scalar;

    // Get offset by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_memAlignWR_W1() 2 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    int64_t offset = cr.scalar.get_si();

    int64_t shiftRight = offset * 8;
    int64_t shiftLeft = (32 - offset) * 8;
    mpz_class result = (m1 & (Mask256 >> shiftRight)) | (Mask256 & (value << shiftLeft));
    
    cr.type = crt_fea;
    scalar2fea(ctx.fr, result, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_memAlignWR8_W0 (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 3) {
        cerr << "Error: eval_memAlignWR8_W0() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get m0 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_memAlignWR8_W0() 0 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    mpz_class m0 = cr.scalar;

    // Get value by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_memAlignWR8_W0() 1 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    mpz_class value = cr.scalar;

    // Get offset by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_memAlignWR8_W0() 2 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    int64_t offset = cr.scalar.get_si();

    int64_t bits = (31 - offset) * 8;
    
    mpz_class result = (m0 & (Mask256 - (Mask8 << bits))) | ((Mask8 & value ) << bits);
    
    cr.type = crt_fea;
    scalar2fea(ctx.fr, result, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_saveContractBytecode (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_saveContractBytecode() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get addr by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_saveContractBytecode() 0 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    uint64_t addr = cr.scalar.get_ui();

    string digestString = "0x" + ctx.hashP[addr].digest.get_str(16);
    ctx.proverRequest.input.contractsBytecode[digestString] = ctx.hashP[addr].data;

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

void eval_inverseFpEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_inverseFpEc() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_inverseFpEc() 0 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    RawFec::Element a;
    ctx.fec.fromString(a, cr.scalar.get_str(16), 16);
    if (ctx.fec.isZero(a))
    {
        cerr << "Error: eval_inverseFpEc() Division by zero" << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    RawFec::Element r;
    ctx.fec.inv(r, a);

    cr.type = crt_scalar;
    cr.scalar.set_str(ctx.fec.toString(r,16), 16);
}

void eval_inverseFnEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_inverseFnEc() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_inverseFnEc() 0 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    RawFnec::Element a;
    ctx.fnec.fromString(a, cr.scalar.get_str(16), 16);
    if (ctx.fnec.isZero(a))
    {
        cerr << "Error: eval_inverseFnEc() Division by zero" << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    RawFnec::Element r;
    ctx.fnec.inv(r, a);

    cr.type = crt_scalar;
    cr.scalar.set_str(ctx.fnec.toString(r,16), 16);
}

mpz_class pow ( const mpz_class &x, const mpz_class &n, const mpz_class &p ) 
{
    if (n == 0) return 1;
    if ((n & 1) == 1) {
        return (pow(x, n-1, p) * x) % p;
    }
    mpz_class x2 = pow(x, n/2, p);
    return (x2 * x2) % p;
}

mpz_class sqrtTonelliShanks ( const mpz_class &n, const mpz_class &p )
{
    mpz_class s = 1;
    mpz_class q = p - 1;
    while ((q & 1) == 0) {
        q = q / 2;
        ++s;
    }
    if (s == 1) {
        mpz_class r = pow(n, (p+1)/4, p);
        if ((r * r) % p == n) return r;
        return 0;
    }

    mpz_class z = 1;
    while (pow(++z, (p - 1)/2, p) != (p - 1));
//    std::cout << "Z found: " << z << "\n";
    mpz_class c = pow(z, q, p);
    mpz_class r = pow(n, (q+1)/2, p);
    mpz_class t = pow(n, q, p);
    mpz_class m = s;
    while (t != 1) {
        mpz_class tt = t;
        mpz_class i = 0;
        while (tt != 1) {
            tt = (tt * tt) % p;
            ++i;
            if (i == m) return 0;
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
        if (r > (p/2)) r = p - r; // return only the possitive solution of the square root
        return r;
    }
    return 0;
}

void eval_sqrtFpEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_sqrtFpEc() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_sqrtFpEc() 0 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    RawFec::Element pfe = ctx.fec.negOne();
    mpz_class p(ctx.fec.toString(pfe,16),16);
    p++;
    mpz_class a = cr.scalar;
    cr.type = crt_scalar;
    cr.scalar = sqrtTonelliShanks(a, p);
}

void eval_AddPointEc (Context &ctx, const RomCommand &cmd, bool dbl, RawFec::Element &x3, RawFec::Element &y3);

void eval_xAddPointEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    RawFec::Element x3;
    RawFec::Element y3;
    eval_AddPointEc(ctx, cmd, false, x3, y3);
    cr.type = crt_scalar;
    cr.scalar.set_str(ctx.fec.toString(x3, 16),16);
}

void eval_yAddPointEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    RawFec::Element x3;
    RawFec::Element y3;
    eval_AddPointEc(ctx, cmd, false, x3, y3);
    cr.type = crt_scalar;
    cr.scalar.set_str(ctx.fec.toString(y3, 16),16);
}

void eval_xDblPointEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    RawFec::Element x3;
    RawFec::Element y3;
    eval_AddPointEc(ctx, cmd, true, x3, y3);
    cr.type = crt_scalar;
    cr.scalar.set_str(ctx.fec.toString(x3, 16),16);
}

void eval_yDblPointEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    RawFec::Element x3;
    RawFec::Element y3;
    eval_AddPointEc(ctx, cmd, true, x3, y3);
    cr.type = crt_scalar;
    cr.scalar.set_str(ctx.fec.toString(y3, 16),16);
}

void eval_AddPointEc (Context &ctx, const RomCommand &cmd, bool dbl, RawFec::Element &x3, RawFec::Element &y3)
{
    // Check parameters list size
    if (cmd.params.size() != (dbl ? 2 : 4)) {
        cerr << "Error: eval_AddPointEc() invalid number of parameters function " << function2String(cmd.function) << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }

    CommandResult cr;

    // Get x1 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_AddPointEc() 0 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    RawFec::Element x1;
    ctx.fec.fromString(x1, cr.scalar.get_str(16), 16);

    // Get y1 by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_AddPointEc() 1 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
        exitProcess();
    }
    RawFec::Element y1;
    ctx.fec.fromString(y1, cr.scalar.get_str(16), 16);

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
        if (cr.type != crt_scalar) {
            cerr << "Error: eval_AddPointEc() 2 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
            exitProcess();
        }
        ctx.fec.fromString(x2, cr.scalar.get_str(16), 16);

        // Get y2 by executing cmd.params[3]
        evalCommand(ctx, *cmd.params[3], cr);
        if (cr.type != crt_scalar) {
            cerr << "Error: eval_AddPointEc() 3 unexpected command result type: " << cr.type << " zkPC=" << *ctx.pZKPC << endl;
            exitProcess();
        }
        ctx.fec.fromString(y2, cr.scalar.get_str(16), 16);
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