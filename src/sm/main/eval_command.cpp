#include <iostream>

#include "config.hpp"
#include "eval_command.hpp"
#include "scalar.hpp"
#include "pols.hpp"
#include "opcode_address.hpp"

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
    if (cmd.op=="number") {
        return eval_number(ctx, cmd, cr);
    } else if (cmd.op=="declareVar") {
        return eval_declareVar(ctx, cmd, cr);
    } else if (cmd.op=="setVar") {
        return eval_setVar(ctx, cmd, cr);
    } else if (cmd.op=="getVar") {
        return eval_getVar(ctx, cmd, cr);
    } else if (cmd.op=="getReg") {
        return eval_getReg(ctx, cmd, cr);
    } else if (cmd.op=="functionCall") {
        return eval_functionCall(ctx, cmd, cr);
    } else if (cmd.op=="add") {
        return eval_add(ctx, cmd, cr);
    } else if (cmd.op=="sub") {
        return eval_sub(ctx, cmd, cr);
    } else if (cmd.op=="neg") {
        return eval_neg(ctx, cmd, cr);
    } else if (cmd.op=="mul") {
        return eval_mul(ctx, cmd, cr);
    } else if (cmd.op=="div") {
        return eval_div(ctx, cmd, cr);
    } else if (cmd.op=="mod") {
        return eval_mod(ctx, cmd, cr);
    } else if (cmd.op == "or" || cmd.op == "and" || cmd.op == "gt" || cmd.op == "ge" || cmd.op == "lt" || cmd.op == "le" ||
               cmd.op == "eq" || cmd.op == "ne" || cmd.op == "not" ) {
        return eval_logical_operation(ctx, cmd, cr);
    } else if (cmd.op == "bitand" || cmd.op == "bitor" || cmd.op == "bitxor" || cmd.op == "bitnot"|| cmd.op == "shl" || cmd.op == "shr") {
        return eval_bit_operation(ctx, cmd, cr);
    } else if (cmd.op == "if") {
        return eval_if(ctx, cmd, cr);
    } else if (cmd.op == "getMemValue") {
        return eval_getMemValue(ctx, cmd, cr);
    } else {
        cerr << "Error: evalCommand() found invalid operation: " << cmd.op << endl;
        exit(-1);
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
        cerr << "Error: eval_declareVar() Variable name not found" << endl;
        exit(-1);  
    }

    // Check that this variable does not exists
    if ( ctx.vars.find(cmd.varName) != ctx.vars.end() ) {
        cerr << "Error: eval_declareVar() Variable already declared: " << cmd.varName << endl;
        exit(-1);
    }

    // Create the new variable with a zero value
    ctx.vars[cmd.varName] = ctx.fr.zero();

#ifdef LOG_VARIABLES
    cout << "Declare variable: " << cmd.varName << endl;
#endif

    // Return the current value of this variable
    cr.type = crt_fe;
    cr.fe = ctx.vars[cmd.varName];
}

/* Gets the value of the variable, and fails if it does not exist */
void eval_getVar (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check the variable name
    if (cmd.varName == "") {
        cerr << "Error: eval_getVar() Variable name not found" << endl;
        exit(-1);  
    }

    // Check that this variable exists
    if ( ctx.vars.find(cmd.varName) == ctx.vars.end() ) {
        cerr << "Error: eval_getVar() Undefined variable: " << cmd. varName << endl;
        exit(-1);
    }

#ifdef LOG_VARIABLES
    cout << "Get variable: " << cmd.varName << " fe: " << ctx.fr.toString(ctx.vars[cmd.varName], 16) << endl;
#endif

    // Return the current value of this variable
    cr.type = crt_fe;
    cr.fe = ctx.vars[cmd.varName];
}

void eval_left (Context &ctx, const RomCommand &cmd, CommandResult &cr);

/* Sets variable to value, and fails if it does not exist */
void eval_setVar (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check that tag contains a values array
    if (cmd.values.size()==0) {
        cerr << "Error: eval_setVar() could not find array values in setVar command" << endl;
        exit(-1);
    }

    // Get varName from the first element in values
    eval_left(ctx,*cmd.values[0], cr);
    if (cr.type != crt_string) {
        cerr << "Error: eval_setVar() unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    string varName = cr.str;

    // Check that this variable exists
    if ( ctx.vars.find(varName) == ctx.vars.end() ) {
        cerr << "Error: eval_setVar() Undefined variable: " << varName << endl;
        exit(-1);
    }

    // Call evalCommand() to build the field element value for this variable
    evalCommand(ctx, *cmd.values[1], cr);

    // Get the field element value from the command result
    FieldElement fe;
    cr2fe(ctx.fr, cr, fe);

    // Store the value as the new variable value
    ctx.vars[varName] = fe;

    // Return the current value of the variable
    cr.type = crt_fe;
    cr.fe = ctx.vars[cmd.varName];

#ifdef LOG_VARIABLES
    cout << "Set variable: " << varName << " fe: " << ctx.fr.toString(ctx.vars[varName], 16) << endl;
#endif
}

void eval_left (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    if (cmd.op == "declareVar") {
        eval_declareVar(ctx, cmd, cr);
        cr.type = crt_string;
        cr.str = cmd.varName;
        return;
    } else if (cmd.op == "getVar") {
        cr.type = crt_string;
        cr.str = cmd.varName;
        return;
    }
    cerr << "Error: eval_left() invalid left expression, op: " << cmd.op << "ln: " << *ctx.pZKPC << endl;
    exit(-1);
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
        cr.u32 = ctx.pols.CTX[*ctx.pStep];
    } else if (cmd.regName=="SP") {
        cr.type = crt_u16;
        cr.u16 = ctx.pols.SP[*ctx.pStep];
    } else if (cmd.regName=="PC") {
        cr.type = crt_u32;
        cr.u32 = ctx.pols.PC[*ctx.pStep];
    } else if (cmd.regName=="MAXMEM") {
        cr.type = crt_u32;
        cr.u32 = ctx.pols.MAXMEM[*ctx.pStep];
    } else if (cmd.regName=="GAS") {
        cr.type = crt_u64;
        cr.u64 = ctx.pols.CTX[*ctx.pStep];
    } else if (cmd.regName=="zkPC") {
        cr.type = crt_u32;
        cr.u32 = ctx.pols.zkPC[*ctx.pStep];
    } else {
        cerr << "Error: eval_getReg() Invalid register: " << cmd.regName << ": " << *ctx.pZKPC << endl;
        exit(-1);
    }
}

void cr2fe(FiniteField &fr, const CommandResult &cr, FieldElement &fe)
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
        exit(-1);
    }
}

void cr2scalar(FiniteField &fr, const CommandResult &cr, mpz_class &s)
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
        exit(-1);
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
    cr.scalar = (a * b) & mask256;
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

    if (cmd.op == "not")
    {
        cr.type = crt_scalar;
        cr.scalar = (a) ? 0 : 1;
        return;
    }

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    cr.type = crt_scalar;
    
         if (cmd.op == "or" ) cr.scalar = (a || b) ? 1 : 0;
    else if (cmd.op == "and") cr.scalar = (a && b) ? 1 : 0;
    else if (cmd.op == "eq" ) cr.scalar = (a == b) ? 1 : 0;
    else if (cmd.op == "ne" ) cr.scalar = (a != b) ? 1 : 0;
    else if (cmd.op == "gt" ) cr.scalar = (a >  b) ? 1 : 0;
    else if (cmd.op == "ge" ) cr.scalar = (a >= b) ? 1 : 0;
    else if (cmd.op == "lt" ) cr.scalar = (a <  b) ? 1 : 0;
    else if (cmd.op == "le" ) cr.scalar = (a <= b) ? 1 : 0;
    else
    {
        cerr << "Error: eval_logical_operation() operation not defined: " << cmd.op << endl;
        exit(-1);
    }
}

void eval_bit_operation (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    if (cmd.op == "bitnot")
    {
        cr.type = crt_scalar;
        cr.scalar = ~a;
        return;
    }

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    cr.type = crt_scalar;
    
         if (cmd.op == "bitor" ) cr.scalar = a | b;
    else if (cmd.op == "bitand") cr.scalar = a & b;
    else if (cmd.op == "bitxor") cr.scalar = a ^ b;
    else if (cmd.op == "shl"   ) cr.scalar = (a << b.get_ui());
    else if (cmd.op == "shr"   ) cr.scalar = (a >> b.get_ui());
    else if (cmd.op == "ge" ) cr.scalar = (a >= b) ? 1 : 0;
    else if (cmd.op == "lt" ) cr.scalar = (a <  b) ? 1 : 0;
    else if (cmd.op == "le" ) cr.scalar = (a <= b) ? 1 : 0;
    else
    {
        cerr << "Error: eval_bit_operation() operation not defined: " << cmd.op << endl;
        exit(-1);
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
void eval_getChainId          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getDefaultChainId   (Context &ctx, const RomCommand &cmd, CommandResult &cr);
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
void eval_dumpRegs            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_dump                (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_dumphex             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_xAddPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_yAddPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_xDblPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_yDblPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getBytecode         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getByte             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getBytecodeLength   (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getHashBytecode     (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_beforeLast          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_touchedAddress      (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_touchedStorageSlots (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_bitwise             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_comp                (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_loadScalar          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getGlobalExitRootManagerAddr (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_log                 (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_copyTouchedAddress  (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_exp                 (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_storeLog            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_precompiled         (Context &ctx, const RomCommand &cmd, CommandResult &cr);

void eval_functionCall (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Call the proper internal function
    if (cmd.funcName == "beforeLast") {
        return eval_beforeLast(ctx, cmd, cr);
    } else if (cmd.funcName == "getGlobalHash") {
        return eval_getGlobalHash(ctx, cmd, cr);
    } else if (cmd.funcName == "getGlobalExitRoot") {
        return eval_getGlobalExitRoot(ctx, cmd, cr);
    } else if (cmd.funcName == "getOldStateRoot") {
        return eval_getOldStateRoot(ctx, cmd, cr);
    } else if (cmd.funcName == "getNewStateRoot") {
        return eval_getNewStateRoot(ctx, cmd, cr);
    } else if (cmd.funcName == "getSequencerAddr") {
        return eval_getSequencerAddr(ctx, cmd, cr);
    } else if (cmd.funcName == "getChainId") {
        return eval_getChainId(ctx, cmd, cr);
    } else if (cmd.funcName == "getOldLocalExitRoot") {
        return eval_getOldLocalExitRoot(ctx, cmd, cr);
    } else if (cmd.funcName == "getNewLocalExitRoot") {
        return eval_getNewLocalExitRoot(ctx, cmd, cr);
    } else if (cmd.funcName == "getNumBatch") {
        return eval_getBatchNum(ctx, cmd, cr);
    } else if (cmd.funcName == "getTimestamp") {
        return eval_getTimestamp(ctx, cmd, cr);
    //} else if (cmd.funcName == "getDefaultChainId") {
    //    eval_getDefaultChainId(ctx, cmd, cr);
    } else if (cmd.funcName == "getBatchHashData") { // To be generated by preprocess_TX()
        return eval_getBatchHashData(ctx, cmd, cr);
    } else if (cmd.funcName == "getTxs") {
        return eval_getTxs(ctx, cmd, cr);
    } else if (cmd.funcName == "getTxsLen") {
        return eval_getTxsLen(ctx, cmd, cr);
    } else if (cmd.funcName == "addrOp") {
        return eval_addrOp(ctx, cmd, cr);
    } else if (cmd.funcName == "eventLog") {
        return eval_eventLog(ctx, cmd, cr);
    } else if (cmd.funcName == "cond") {
        return eval_cond(ctx, cmd, cr);
    } else if (cmd.funcName == "inverseFpEc") {
        return eval_inverseFpEc(ctx, cmd, cr);
    } else if (cmd.funcName == "inverseFnEc") {
        return eval_inverseFnEc(ctx, cmd, cr);
    } else if (cmd.funcName == "sqrtFpEc") {
        return eval_sqrtFpEc(ctx, cmd, cr);
    /*} else if (cmd.funcName == "dumpRegs") {
        return eval_dumpRegs(ctx, cmd, cr);
    } else if (cmd.funcName == "dump") {
        return eval_dump(ctx, cmd, cr);
    } else if (cmd.funcName == "dumphex") {
        return eval_dumphex(ctx, cmd, cr);*/
    } else if (cmd.funcName == "xAddPointEc") {
        return eval_xAddPointEc(ctx, cmd, cr);
    } else if (cmd.funcName == "yAddPointEc") {
        return eval_yAddPointEc(ctx, cmd, cr);
    } else if (cmd.funcName == "xDblPointEc") {
        return eval_xDblPointEc(ctx, cmd, cr);
    } else if (cmd.funcName == "yDblPointEc") {
        return eval_yDblPointEc(ctx, cmd, cr);
    /*} else if (cmd.funcName == "getBytecode") { // Added by opcodes
        return eval_getBytecode(ctx, cmd, cr);*/
    } else if (cmd.funcName == "getByte") {
        return eval_getByte(ctx, cmd, cr);
    /*} else if (cmd.funcName == "getBytecodeLength") {
        return eval_getBytecodeLength(ctx, cmd, cr);
    } else if (cmd.funcName == "getHashBytecode") {
        return eval_getHashBytecode(ctx, cmd, cr);*/
    } else if (cmd.funcName == "touchedAddress") {
        return eval_touchedAddress(ctx, cmd, cr);
    } else if (cmd.funcName == "touchedStorageSlots") {
        return eval_touchedStorageSlots(ctx, cmd, cr);
    } else if (cmd.funcName.find("bitwise_") == 0){
        return eval_bitwise(ctx, cmd, cr);
    } else if (cmd.funcName.find("comp_") == 0) {
        return eval_comp(ctx, cmd, cr);
    } else if (cmd.funcName == "loadScalar") {
        return eval_loadScalar(ctx, cmd, cr);
    } else if (cmd.funcName == "getGlobalExitRootManagerAddr") {
        return eval_getGlobalExitRootManagerAddr(ctx, cmd, cr);
    } else if (cmd.funcName == "log") {
        return eval_log(ctx, cmd, cr);
    } else if (cmd.funcName == "copyTouchedAddress"){
        return eval_copyTouchedAddress(ctx, cmd, cr);
    } else if (cmd.funcName == "exp") {
        return eval_exp(ctx, cmd, cr);
    } else if (cmd.funcName == "storeLog") {
        return eval_storeLog(ctx, cmd, cr);
    /*} else if (cmd.funcName.find("precompiled_") == 0) {
        return eval_precompiled(ctx, cmd, cr);*/
    } else {
        cerr << "Error: eval_functionCall() function not defined: " << cmd.funcName << " line: " << *ctx.pZKPC << endl;
        exit(-1);
    } 
}

void eval_getGlobalHash(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getGlobalHash() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Return ctx.input.globalHash as a field element
    cr.type = crt_fea;
    scalar2fea(ctx.fr, ctx.input.globalHash, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getGlobalExitRoot(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getGlobalExitRoot() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Return ctx.input.globalExitRoot as a field element array
    cr.type = crt_fea;
    mpz_class globalExitRoot(ctx.input.globalExitRoot);
    scalar2fea(ctx.fr, globalExitRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getSequencerAddr(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getSequencerAddr() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.sequencerAddr as a field element array
    cr.type = crt_fea;
    mpz_class sequencerAddr(ctx.input.publicInputs.sequencerAddr);
    scalar2fea(ctx.fr, sequencerAddr, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getChainId(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getChainId() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.chainId as a field element array
    cr.type = crt_fea;
    ctx.fr.fromUI(cr.fea0, ctx.input.publicInputs.chainId);
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

void eval_getDefaultChainId(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getDefaultChainId() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.defaultChainId as a field element array
    cr.type = crt_fea;
    ctx.fr.fromUI(cr.fea0, ctx.input.publicInputs.defaultChainId);
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

void eval_getBatchNum(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getBatchNum() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.batchNum as a field element array
    cr.type = crt_fea;
    ctx.fr.fromUI(cr.fea0, ctx.input.publicInputs.batchNum);
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
        cerr << "Error: eval_getOldStateRoot() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.oldStateRoot as a field element array
    cr.type = crt_fea;
    mpz_class oldStateRoot(ctx.input.publicInputs.oldStateRoot); // This field could be parsed out of the main loop, but it is only called once
    scalar2fea(ctx.fr, oldStateRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getNewStateRoot(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getNewStateRoot() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.newStateRoot as a field element array
    cr.type = crt_fea;
    mpz_class newStateRoot(ctx.input.publicInputs.newStateRoot); // This field could be parsed out of the main loop, but it is only called once
    scalar2fea(ctx.fr, newStateRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getOldLocalExitRoot(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getOldLocalExitRoot() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.oldLocalExitRoot as a field element array
    cr.type = crt_fea;
    mpz_class oldLocalExitRoot(ctx.input.publicInputs.oldLocalExitRoot);
    scalar2fea(ctx.fr, oldLocalExitRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getNewLocalExitRoot(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getNewLocalExitRoot() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.newLocalExitRoot as a field element array
    cr.type = crt_fea;
    mpz_class newLocalExitRoot(ctx.input.publicInputs.newLocalExitRoot);
    scalar2fea(ctx.fr, newLocalExitRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getTxsLen(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getTxsLen() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Return ctx.input.txsLen/2 as a field element array
    cr.type = crt_fea;
    u642fe(ctx.fr, cr.fea0, (ctx.input.batchL2Data.size() - 2) / 2);
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
        cerr << "Error: eval_getTxs() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Get offset by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_fe) {
        cerr << "Error: eval_getTxs() 1 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t offset = fe2n(ctx.fr, cr.fe);

    // Get offset by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_getTxs() 2 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t len = cr.scalar.get_ui();

    string resultString = ctx.input.batchL2Data.substr(2+offset*2, len*2);
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
        cerr << "Error: eval_getBatchHashData() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Return ctx.input.batchHashData as a field element array
    cr.type = crt_fea;
    scalar2fea(ctx.fr, ctx.input.batchHashData, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_addrOp(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_addrOp() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Get offset by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_fe) {
        cerr << "Error: eval_addrOp() unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t codeId = fe2n(ctx.fr, cr.fe);
    cr.type = crt_fea;

    uint64_t addr = opcodeAddress[codeId];
    ctx.fr.fromUI(cr.fea0, addr);
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
        cerr << "Error: eval_eventLog() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    ctx.fullTracer.handleEvent(ctx, cmd);

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
        cerr << "Error: eval_getTimestamp() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.chainId as a field element array
    cr.type = crt_fea;
    ctx.fr.fromUI(cr.fea0, ctx.input.publicInputs.timestamp);
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
        cerr << "Error: eval_cond() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Get offset by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_fe) {
        cerr << "Error: eval_cond() unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t result = fe2n(ctx.fr, cr.fe);
    
    cr.type = crt_fea;
    if (result)
    {
        cr.fea0 = ctx.fr.zero() - ctx.fr.one(); // -1
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

/*

function eval_getBytecode(ctx, tag) {
    if (tag.params.length != 2 && tag.params.length != 3) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    let hashcontract = evalCommand(ctx,tag.params[0]);
    hashcontract = "0x" + hashcontract.toString(16).padStart(64, '0');
    const bytecode = ctx.input.contractsBytecode[hashcontract];
    const offset = Number(evalCommand(ctx,tag.params[1]));
    let len;
    if(tag.params[2])
        len = Number(evalCommand(ctx,tag.params[2]));
    else
        len = 1;
    let d = "0x" + bytecode.slice(2+offset*2, 2+offset*2 + len*2);
    if (d.length == 2) d = d+'0';
    return scalar2fea(ctx.Fr, Scalar.e(d));
}*/


void eval_getByte (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 2 && cmd.params.size() != 3) {
        cerr << "Error: eval_getByte() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Get bytes by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_getByte() unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    string aux = cr.scalar.get_str(16);
    string bytes = NormalizeToNFormat(aux, 64);

    // Get offset by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_getByte() unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t offset = cr.scalar.get_ui();

    // Get length by executing cmd.params[2]
    uint64_t len = 1;
    if (cmd.params.size() == 3)
    {
        evalCommand(ctx, *cmd.params[2], cr);
        if (cr.type != crt_fe) {
            cerr << "Error: eval_getByte() unexpected command result type: " << cr.type << endl;
            exit(-1);
        }
        len = fe2n(ctx.fr, cr.fe);
    }

    // Check the total length
    if ((offset+2*len) > 64)
    {
        cerr << "Error: eval_getByte() invalid values offset=" << offset << " len=" << len << endl;
        exit(-1);
    }

    // Get the requested substring
    string auxString = bytes.substr(offset, 2*len);
    mpz_class auxScalar;
    auxScalar.set_str(auxString, 16);

    // Return as a field element array
    cr.type = crt_fea;
    scalar2fea(ctx.fr, auxScalar, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

/*function eval_getBytecodeLength(ctx, tag) {
    if (tag.params.length != 1) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    let hashcontract = evalCommand(ctx,tag.params[0]);
    hashcontract = "0x" + hashcontract.toString(16).padStart(64, '0');
    const bytecode = ctx.input.contractsBytecode[hashcontract];
    const d = (bytecode.length - 2)/2;
    return scalar2fea(ctx.Fr, Scalar.e(d));
}

function eval_getHashBytecode(ctx, tag) {
    if (tag.params.length != 1) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    let addr = evalCommand(ctx,tag.params[0]);
    addr = addr.toString(16).padStart(40, '0')
    return scalar2fea(ctx.Fr, Scalar.e(ctx.input.contractsBytecode["0x"+addr.toString(16).toLowerCase()]));
}*/

void eval_touchedAddress (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 2) {
        cerr << "Error: eval_touchedAddress() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Get addr by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_touchedAddress() 1 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    mpz_class addr = cr.scalar;

    // Get context by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_u32) {
        cerr << "Error: eval_touchedAddress() 2 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint32_t context = cr.u32;

    // if address is precompiled smart contract considered warm access
    if ((addr>0) && (addr<10))
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
    }
    else
    {
        // If address is not in touchedAddress, then return 1, and store it in the map's vector
        if(ctx.touchedAddress.find(context) != ctx.touchedAddress.end())
        {
            ctx.touchedAddress[context].push_back(addr);
        }
        else
        {
            vector<mpz_class> aux;
            aux.push_back(addr);
            ctx.touchedAddress[context] = aux;
        }
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
}

void eval_copyTouchedAddress (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 2) {
        cerr << "Error: eval_copyTouchedAddress() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Get ctx1 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_u32) {
        cerr << "Error: eval_copyTouchedAddress() 1 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint32_t ctx1 = cr.u32;

    // Get ctx2 by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_u32) {
        cerr << "Error: eval_copyTouchedAddress() 2 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint32_t ctx2 = cr.u32;

    if (ctx.touchedAddress.find(ctx1) != ctx.touchedAddress.end())
    {
        ctx.touchedAddress[ctx2] = ctx.touchedAddress[ctx1];
    }

    cr.type = crt_fea;
    cr.fea0 = 0;
    cr.fea1 = 0;
    cr.fea2 = 0;
    cr.fea3 = 0;
    cr.fea4 = 0;
    cr.fea5 = 0;
    cr.fea6 = 0;
    cr.fea7 = 0;
}

bool touchedStorageSlotsContains(Context &ctx, uint32_t context, uint32_t addr, uint32_t key)
{
    if (ctx.touchedStorageSlots.find(context) == ctx.touchedStorageSlots.end())
    {
        return false;
    }
    for (uint64_t i=0; i<ctx.touchedStorageSlots[context].size(); i++)
    {
        if ( (ctx.touchedStorageSlots[context][i].addr == addr) &&
             (ctx.touchedStorageSlots[context][i].key == key) )
        {
            return true;
        }
    }
    return false;
}

void eval_touchedStorageSlots (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 3) {
        cerr << "Error: eval_touchedStorageSlots() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Get addr by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_u32) {
        cerr << "Error: eval_touchedStorageSlots() 1 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint32_t addr = cr.u32;

    // Get key by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_u32) {
        cerr << "Error: eval_touchedStorageSlots() 2 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint32_t key = cr.u32;

    // Get context by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
    if (cr.type != crt_u32) {
        cerr << "Error: eval_touchedStorageSlots() 3 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint32_t context = cr.u32;

    // if address in touchedStorageSlots return 0
    if ( touchedStorageSlotsContains(ctx, context, addr, key) )
    {
        cr.type = crt_fea;
        cr.fea0 = 0;
        cr.fea1 = 0;
        cr.fea2 = 0;
        cr.fea3 = 0;
        cr.fea4 = 0;
        cr.fea5 = 0;
        cr.fea6 = 0;
        cr.fea7 = 0;
    }
    //if addres not in touchedStorageSlots, return 1
    else
    {
        if (ctx.touchedStorageSlots.find(context) != ctx.touchedStorageSlots.end())
        {
            TouchedStorageSlot slot;
            slot.addr = addr;
            slot.key = key;
            ctx.touchedStorageSlots[context].push_back(slot);
        }
        else
        {
            TouchedStorageSlot slot;
            slot.addr = addr;
            slot.key = key;
            vector<TouchedStorageSlot> slotVector;
            slotVector.push_back(slot);
            ctx.touchedStorageSlots[context] = slotVector;
        }
        cr.type = crt_fea;
        cr.fea0 = 1;
        cr.fea1 = 0;
        cr.fea2 = 0;
        cr.fea3 = 0;
        cr.fea4 = 0;
        cr.fea5 = 0;
        cr.fea6 = 0;
        cr.fea7 = 0;
    }
}

void eval_exp (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 2) {
        cerr << "Error: eval_exp() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_exp() 1 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_exp() 2 unexpected command result type: " << cr.type << endl;
        exit(-1);
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
        cerr << "Error: eval_bitwise() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_bitwise() 1 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    mpz_class a = cr.scalar;

    if (cmd.funcName == "bitwise_and")
    {
        // Check parameters list size
        if (cmd.params.size() != 2) {
            cerr << "Error: eval_bitwise() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
            exit(-1);
        }

        // Get b by executing cmd.params[1]
        evalCommand(ctx, *cmd.params[1], cr);
        if (cr.type != crt_scalar) {
            cerr << "Error: eval_bitwise() 2 unexpected command result type: " << cr.type << endl;
            exit(-1);
        }
        mpz_class b = cr.scalar;

        cr.type = crt_scalar;
        cr.scalar = a & b;
    }
    else if (cmd.funcName == "bitwise_or")
    {
        // Check parameters list size
        if (cmd.params.size() != 2) {
            cerr << "Error: eval_bitwise() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
            exit(-1);
        }

        // Get b by executing cmd.params[1]
        evalCommand(ctx, *cmd.params[1], cr);
        if (cr.type != crt_scalar) {
            cerr << "Error: eval_bitwise() 3 unexpected command result type: " << cr.type << endl;
            exit(-1);
        }
        mpz_class b = cr.scalar;

        cr.type = crt_scalar;
        cr.scalar = a | b;
    }
    else if (cmd.funcName == "bitwise_xor")
    {
        // Check parameters list size
        if (cmd.params.size() != 2) {
            cerr << "Error: eval_bitwise() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
            exit(-1);
        }

        // Get b by executing cmd.params[1]
        evalCommand(ctx, *cmd.params[1], cr);
        if (cr.type != crt_scalar) {
            cerr << "Error: eval_bitwise() 4 unexpected command result type: " << cr.type << endl;
            exit(-1);
        }
        mpz_class b = cr.scalar;

        cr.type = crt_scalar;
        cr.scalar = a ^ b;
    }
    else if (cmd.funcName == "bitwise_not")
    {
        // Check parameters list size
        if (cmd.params.size() != 1) {
            cerr << "Error: eval_bitwise() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
            exit(-1);
        }

        cr.type = crt_scalar;
        cr.scalar = a ^ Mask256;
    }
    else
    {
        cerr << "Error: eval_bitwise() invalid operation funcName=" << cmd.funcName << endl;
        exit(-1);
    }
}

void eval_beforeLast (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_beforeLast() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

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
        cerr << "Error: eval_comp() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_comp() 1 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    mpz_class a = cr.scalar;

    // Get b by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_comp() 2 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    mpz_class b = cr.scalar;

    cr.type = crt_scalar;
    if (cmd.funcName == "comp_lt")
    {
        cr.scalar = (a < b) ? 1 : 0;
    }
    else if (cmd.funcName == "comp_gt")
    {
        cr.scalar = (a > b) ? 1 : 0;
    }
    else if (cmd.funcName == "comp_gt")
    {
        cr.scalar = (a = b) ? 1 : 0;
    }
    else
    {
        cerr << "Error: eval_comp() Invalid bitwise operation funcName=" << cmd.funcName << endl;
        exit(-1);
    }
}

void eval_loadScalar (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_loadScalar() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    evalCommand(ctx, *cmd.params[0], cr);
}

// Will be replaced by hardcoding this address directly in the ROM once the CONST register can be 256 bits
void eval_getGlobalExitRootManagerAddr (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getGlobalExitRootManagerAddr() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.oldLocalExitRoot as a field element array
    cr.type = crt_fea;
    mpz_class globalExitRootManagerAddr(ADDRESS_GLOBAL_EXIT_ROOT_MANAGER_L2);
    scalar2fea(ctx.fr, globalExitRootManagerAddr, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_storeLog (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 3) {
        cerr << "Error: eval_storeLog() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Get indexLog by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_u32) {
        cerr << "Error: eval_storeLog() 1 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t indexLog = cr.u32;

    // Get isTopic by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_u32) {
        cerr << "Error: eval_storeLog() 2 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint32_t isTopic = cr.u32;

    // Get isTopic by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[2], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_storeLog() 3 unexpected command result type: " << cr.type << endl;
        exit(-1);
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
        cerr << "Error: eval_log() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Get indexLog by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_fea) {
        cerr << "Error: eval_storeLog() 1 unexpected command result type: " << cr.type << endl;
        exit(-1);
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

/*function eval_precompiled(ctx, tag){
    const primaryFunc = tag.funcName.split('_')[1];
    const secondaryFunc = tag.funcName.split('_')[2];

    switch (primaryFunc){
        case 'ecrecover':
            return preEcrecover(secondaryFunc, ctx, tag);
        case 'sha256':
            return preSha256(secondaryFunc, ctx, tag);
        case 'ripemd160':
            return preRipemd160(secondaryFunc, ctx, tag);
        case 'blake2f':
            return preBlake2f(secondaryFunc, ctx, tag);
        case 'ecAdd':
            return preEcAdd(secondaryFunc, ctx, tag);
        case 'ecMul':
            return preEcMul(secondaryFunc, ctx, tag);
        case 'ecPairing':
            return preEcPairing(secondaryFunc, ctx, tag);
        default:
            throw new Error(`Invalid precompiled operation ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function preSha256(func, ctx, tag){
    // initialize object
    if (typeof ctx.precompiled.sha256 === 'undefined'){
        ctx.precompiled.sha256 = {
            data: "0x",
        }
    }

    switch (func){
        case 'add':
            checkParams(ctx, tag, 2);
            const size = Number(evalCommand(ctx, tag.params[1]));
            const bytesToAdd = (evalCommand(ctx, tag.params[0])).toString(16).padStart(2*size, "0");
            ctx.precompiled.sha256.data += bytesToAdd;

            return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];

        case 'digest':
            ctx.precompiled.sha256.result = ethers.utils.soliditySha256(["bytes"], [ctx.precompiled.sha256.data]);
            return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];

        case 'read':
            return scalar2fea(ctx.Fr, ctx.precompiled.sha256.result);

        default:
            throw new Error(`Invalid precompiled Sha256 functionality ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function preRipemd160(func, ctx, tag){
    // initialize object
    if (typeof ctx.precompiled.ripemd160 === 'undefined'){
        ctx.precompiled.ripemd160 = {
            data: "0x",
        }
    }

    switch (func){
        case 'add':
            checkParams(ctx, tag, 2);
            const size = Number(evalCommand(ctx, tag.params[1]));
            const bytesToAdd = (evalCommand(ctx, tag.params[0])).toString(16).padStart(2*size, "0");
            ctx.precompiled.ripemd160.data += bytesToAdd;

            return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];

        case 'digest':
            const bytesToHash = ethers.utils.solidityPack(["bytes"], [ctx.precompiled.ripemd160.data]);
            ctx.precompiled.ripemd160.result = ethers.utils.ripemd160(bytesToHash);
            return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];

        case 'read':
            return scalar2fea(ctx.Fr, ctx.precompiled.ripemd160.result);

        default:
            throw new Error(`Invalid precompiled ripemd160 functionality ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function preBlake2f(func, ctx, tag){
    // initialize object
    if (typeof ctx.precompiled.blake2f === 'undefined'){
        ctx.precompiled.blake2f = {
            data: Buffer.alloc(0),
        }
    }

    switch (func){
        case 'add':
            checkParams(ctx, tag, 2);
            const size = Number(evalCommand(ctx, tag.params[1]));
            const bytesToAdd = (evalCommand(ctx, tag.params[0])).toString(16).padStart(2*size, "0");
            const buffToAdd = Buffer.from(bytesToAdd, "hex");
            ctx.precompiled.blake2f.data = Buffer.concat([ctx.precompiled.blake2f.data, buffToAdd]);

            return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];

        case 'digest':
            // prepare data
            const rounds = ctx.precompiled.blake2f.data.slice(0, 4).readUInt32BE(0)
            const hRaw = ctx.precompiled.blake2f.data.slice(4, 68)
            const mRaw = ctx.precompiled.blake2f.data.slice(68, 196)
            const tRaw = ctx.precompiled.blake2f.data.slice(196, 212)
            // final
            const f = (ctx.precompiled.blake2f.data.slice(212, 213)[0]) === 1

            const h = new Uint32Array(16)
            for (let i = 0; i < 16; i++) {
                h[i] = hRaw.readUInt32LE(i * 4)
            }

            const m = new Uint32Array(32)
            for (let i = 0; i < 32; i++) {
                m[i] = mRaw.readUInt32LE(i * 4)
            }

            const t = new Uint32Array(4)
            for (let i = 0; i < 4; i++) {
                t[i] = tRaw.readUInt32LE(i * 4)
            }

            const resBlake = blake2f(h, m, t, f, rounds);

            ctx.precompiled.blake2f.result = [];
            ctx.precompiled.blake2f.result.push("0x" + resBlake.slice(0, 32).toString("hex"));
            ctx.precompiled.blake2f.result.push("0x" + resBlake.slice(32, 64).toString("hex"));

            return scalar2fea(ctx.Fr, Number(rounds));

        case 'read':
            checkParams(ctx, tag, 1);
            const chunk = Number(evalCommand(ctx, tag.params[0]));
            return scalar2fea(ctx.Fr, ctx.precompiled.blake2f.result[chunk]);

        default:
            throw new Error(`Invalid precompiled blake2f functionality ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function preEcAdd(func, ctx, tag){
    // initialize object
    if (typeof ctx.precompiled.ecAdd === 'undefined'){
        ctx.precompiled.ecAdd = {
            data: "0x",
        }
    }

    switch (func){
        case 'add':
            checkParams(ctx, tag, 4);
            const input0 = "0x" + evalCommand(ctx, tag.params[0]).toString(16);
            const input1 = "0x" + evalCommand(ctx, tag.params[1]).toString(16);
            const input2 = "0x" + evalCommand(ctx, tag.params[2]).toString(16);
            const input3 = "0x" + evalCommand(ctx, tag.params[3]).toString(16);
            const buf0 = toBuffer(ethers.utils.hexZeroPad(input0, 32));
            const buf1 = toBuffer(ethers.utils.hexZeroPad(input1, 32));
            const buf2 = toBuffer(ethers.utils.hexZeroPad(input2, 32));
            const buf3 = toBuffer(ethers.utils.hexZeroPad(input3, 32));
            const buf = Buffer.concat([buf0, buf1, buf2, buf3]);
            ctx.precompiled.ecAdd.result = bn128.add(buf).toString("hex");
            ctx.precompiled.ecAdd.result0 = ctx.precompiled.ecAdd.result.substring(0,64);
            ctx.precompiled.ecAdd.result1 = ctx.precompiled.ecAdd.result.substring(64);
        case 'result0':
            if(!ctx.precompiled.ecAdd.result0)
                throw new Error(`First _add`)
            else
                return scalar2fea(ctx.Fr, Scalar.fromString(ctx.precompiled.ecAdd.result0, 16));
        case 'result1':
            if(!ctx.precompiled.ecAdd.result1)
                throw new Error(`First _add`)
            else
                return scalar2fea(ctx.Fr, Scalar.fromString(ctx.precompiled.ecAdd.result1, 16));
        default:
            throw new Error(`Invalid precompiled ecAdd functionality ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function preEcMul(func, ctx, tag){
    // initialize object
    if (typeof ctx.precompiled.ecMul === 'undefined'){
        ctx.precompiled.ecMul = {
            data: "0x",
        }
    }

    switch (func){
        case 'mul':
            checkParams(ctx, tag, 3);
            const input0 = "0x" + evalCommand(ctx, tag.params[0]).toString(16);
            const input1 = "0x" + evalCommand(ctx, tag.params[1]).toString(16);
            const input2 = "0x" + evalCommand(ctx, tag.params[2]).toString(16);
            const buf0 = toBuffer(ethers.utils.hexZeroPad(input0, 32));
            const buf1 = toBuffer(ethers.utils.hexZeroPad(input1, 32));
            const buf2 = toBuffer(ethers.utils.hexZeroPad(input2, 32));
            const buf = Buffer.concat([buf0, buf1, buf2]);
            ctx.precompiled.ecMul.result = bn128.mul(buf).toString("hex");
            ctx.precompiled.ecMul.result0 = ctx.precompiled.ecMul.result.substring(0,64);
            ctx.precompiled.ecMul.result1 = ctx.precompiled.ecMul.result.substring(64);
        case 'result0':
            if(!ctx.precompiled.ecMul.result0)
                throw new Error(`First _add`)
            else
                return scalar2fea(ctx.Fr, Scalar.fromString(ctx.precompiled.ecMul.result0, 16));
        case 'result1':
            if(!ctx.precompiled.ecMul.result1)
                throw new Error(`First _add`)
            else
                return scalar2fea(ctx.Fr, Scalar.fromString(ctx.precompiled.ecMul.result1, 16));
        default:
            throw new Error(`Invalid precompiled ecMul functionality ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function preEcPairing(func, ctx, tag){
    // initialize object
    if (typeof ctx.precompiled.ecPairing === 'undefined'){
        ctx.precompiled.ecPairing = {
            data: "0x",
        }
    }

    if (typeof ctx.precompiled.ecPairing.buf === 'undefined'){
        ctx.precompiled.ecPairing.buf = Buffer.from("");
    }

    switch (func){
        case 'pairing':
            ctx.precompiled.ecPairing.result = bn128.pairing(ctx.precompiled.ecPairing.buf).toString("hex");
            return scalar2fea(ctx.Fr, Scalar.fromString(ctx.precompiled.ecPairing.result, 16));
        case 'add':
            checkParams(ctx, tag, 1);
            const input0 = "0x" + evalCommand(ctx, tag.params[0]).toString(16);
            const buf0 = toBuffer(ethers.utils.hexZeroPad(input0, 32));
            ctx.precompiled.ecPairing.buf = Buffer.concat([ctx.precompiled.ecPairing.buf, buf0])
            return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
        default:
            throw new Error(`Invalid precompiled ecPairing functionality ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function eval_dumpRegs(ctx, tag) {

    console.log(`dumpRegs ${ctx.fileName}:${ctx.line}`);

    console.log(['A', fea2scalar(ctx.Fr, ctx.A)]);
    console.log(['B', fea2scalar(ctx.Fr, ctx.B)]);
    console.log(['C', fea2scalar(ctx.Fr, ctx.C)]);
    console.log(['D', fea2scalar(ctx.Fr, ctx.D)]);
    console.log(['E', fea2scalar(ctx.Fr, ctx.E)]);

    return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}

function eval_dump(ctx, tag) {
    console.log("\x1b[38;2;175;175;255mDUMP on " + ctx.fileName + ":" + ctx.line+"\x1b[0m");

    tag.params.forEach((value) => {
        let name = value.varName || value.paramName || value.regName || value.offsetLabel;
        if (typeof name == 'undefined' && value.path) {
            name = value.path.join('.');
        }
        console.log("\x1b[35m"+ name +"\x1b[0;35m: "+evalCommand(ctx, value)+"\x1b[0m");
    });

    return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}

function eval_dumphex(ctx, tag) {
    console.log("\x1b[38;2;175;175;255mDUMP on " + ctx.fileName + ":" + ctx.line+"\x1b[0m");

    tag.params.forEach((value) => {
        let name = value.varName || value.paramName || value.regName;
        if (typeof name == 'undefined' && value.path) {
            name = value.path.join('.');
        }
        console.log("\x1b[35m"+ name +"\x1b[0;35m: 0x"+evalCommand(ctx, value).toString(16)+"\x1b[0m");
    });

    return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}
}*/



void eval_inverseFpEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_inverseFpEc() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_inverseFpEc() 0 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    RawFec::Element a;
    ctx.fec.fromString(a, cr.scalar.get_str(16), 16);
    if (ctx.fec.isZero(a))
    {
        cerr << "Error: eval_inverseFpEc() Division by zero" << endl;
        exit(-1);
    }

    RawFec::Element r;
    ctx.fec.inv(r, a);

    cr.scalar.set_str(ctx.fec.toString(r,16), 16);
}

void eval_inverseFnEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_inverseFnEc() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_inverseFnEc() 0 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    RawFnec::Element a;
    ctx.fnec.fromString(a, cr.scalar.get_str(16), 16);
    if (ctx.fnec.isZero(a))
    {
        cerr << "Error: eval_inverseFnEc() Division by zero" << endl;
        exit(-1);
    }

    RawFnec::Element r;
    ctx.fnec.inv(r, a);

    cr.scalar.set_str(ctx.fnec.toString(r,16), 16);
}

void eval_sqrtFpEc (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_sqrtFpEc() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    // Get a by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_AddPointEc() 0 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    RawFec::Element a;
    ctx.fec.fromString(a, cr.scalar.get_str(16), 16);

    RawFec::Element r;
    ctx.fec.square(r, a);

    cr.scalar.set_str(ctx.fec.toString(r,16), 16);
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
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_log() invalid number of parameters function " << cmd.funcName << " : " << *ctx.pZKPC << endl;
        exit(-1);
    }

    CommandResult cr;

    // Get x1 by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_AddPointEc() 0 unexpected command result type: " << cr.type << endl; // TODO: Make sure all errors return *ctx.pZKPC
        exit(-1);
    }
    RawFec::Element x1;
    ctx.fec.fromString(x1, cr.scalar.get_str(16), 16);

    // Get y1 by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_AddPointEc() 1 unexpected command result type: " << cr.type << endl;
        exit(-1);
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
        evalCommand(ctx, *cmd.params[0], cr);
        if (cr.type != crt_scalar) {
            cerr << "Error: eval_AddPointEc() 2 unexpected command result type: " << cr.type << endl;
            exit(-1);
        }
        ctx.fec.fromString(x2, cr.scalar.get_str(16), 16);

        // Get y2 by executing cmd.params[3]
        evalCommand(ctx, *cmd.params[1], cr);
        if (cr.type != crt_scalar) {
            cerr << "Error: eval_AddPointEc() 3 unexpected command result type: " << cr.type << endl;
            exit(-1);
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