#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <gmpxx.h>
#include "../config/definitions.hpp" // This is the only project file allowed to be included

using namespace std;
using json = nlohmann::json;

// Fork namespace
const string forkNamespace = PROVER_FORK_NAMESPACE_STRING;

// Forward declaration
void file2json (json &rom, string &romFileName);
void string2file (const string & s, const string & fileName);
string generate(const json &rom, const string &functionName, const string &fileName, bool bFastMode, bool bHeader);
string selector8 (const string &regName, const string &regValue, bool opInitialized, bool bFastMode);
string selector1 (const string &regName, const string &regValue, bool opInitialized, bool bFastMode);
string selectorConst (int64_t CONST, bool opInitialized, bool bFastMode);
string selectorConstL (const string &CONSTL, bool opInitialized, bool bFastMode);
string setter8 (const string &reg, bool setReg, bool bFastMode, uint64_t zkPC, const json &rom);
string string2lower (const string &s);
string string2upper (const string &s);

int main(int argc, char **argv)
{
    cout << "Main generator" << endl;

    string codeGenerationName = "main_exec_generated";

    string functionName = codeGenerationName;
    string fileName = codeGenerationName;
    string directoryName = "src/main_sm/" + forkNamespace + "/" + codeGenerationName;

    // Load rom.json
    string romFileName = "src/main_sm/" + forkNamespace + "/scripts/rom.json";
    json rom;
    file2json(rom, romFileName);

    string code = generate(rom, functionName, fileName, false, false);
    string2file(code, directoryName + "/" + fileName + ".cpp");
    string header = generate(rom, functionName, fileName, false,  true);
    string2file(header, directoryName + "/" + fileName + ".hpp");
    functionName += "_fast";
    fileName += "_fast";
    string codeFast = generate(rom, functionName, fileName, true, false);
    string2file(codeFast, directoryName + "/" + fileName + ".cpp");
    string headerFast = generate(rom, functionName, fileName, true,  true);
    string2file(headerFast, directoryName + "/" + fileName + ".hpp");

    return 0;
}

void file2json (json &rom, string &romFileName)
{
    std::ifstream inputStream(romFileName);
    if (!inputStream.good())
    {
        cerr << "Error: Main generator failed loading input JSON file " << romFileName << endl;
        exit(-1);
    }
    try
    {
        inputStream >> rom;
    }
    catch (exception &e)
    {
        cerr << "Error: Main generator failed parsing input JSON file " << romFileName << " exception=" << e.what() << endl;
        exit(-1);
    }
    inputStream.close();
}

void string2file (const string & s, const string & fileName)
{
    ofstream outfile;
    outfile.open(fileName);
    outfile << s << endl;
    outfile.close();
}

void scalar2fea(const string &s, uint64_t (&fea)[8])
{
    mpz_class ScalarMask32  ("FFFFFFFF", 16);
    mpz_class scalar(s);
    mpz_class aux;
    aux = scalar & ScalarMask32;
    fea[0] = aux.get_ui();
    aux = scalar>>32 & ScalarMask32;
    fea[1] = aux.get_ui();
    aux = scalar>>64 & ScalarMask32;
    fea[2] = aux.get_ui();
    aux = scalar>>96 & ScalarMask32;
    fea[3] = aux.get_ui();
    aux = scalar>>128 & ScalarMask32;
    fea[4] = aux.get_ui();
    aux = scalar>>160 & ScalarMask32;
    fea[5] = aux.get_ui();
    aux = scalar>>192 & ScalarMask32;
    fea[6] = aux.get_ui();
    aux = scalar>>224 & ScalarMask32;
    fea[7] = aux.get_ui();
}

std::string removeDuplicateSpaces(std::string const &str)
{
    std::string s;
    std::string word;
    std::istringstream ss(str);

    while (ss >> word)
    {
        if (!s.empty())
        {
            s += ' ';
        }
        s += word;
    }

    return s;
}

string generate(const json &rom, const string &functionName, const string &fileName, bool bFastMode, bool bHeader)
{
    //const Fr = new F1Field(0xffffffff00000001n);

    string code = "";

    /*let usedLabels = [];
    for(var key in rom.labels)
    {
        let labelNumber = rom.labels[key];
        usedLabels.push(labelNumber);
        if (key=="mapping_opcodes")
        {
            for (let i=1; i<256; i++)
            {
                usedLabels.push(labelNumber + i);

            }
        }
    }*/

    // INCLUDES

    if (bHeader)
    {
        if (bFastMode)
        {
            code += "#ifndef MAIN_EXEC_GENERATED_FAST_HPP_" + forkNamespace + "\n";
            code += "#define MAIN_EXEC_GENERATED_FAST_HPP_" + forkNamespace + "\n";
        }
        else
        {
            code += "#ifndef MAIN_EXEC_GENERATED_HPP_" + forkNamespace + "\n";
            code += "#define MAIN_EXEC_GENERATED_HPP_" + forkNamespace + "\n";
        }
        code += "\n";
        code += "#include <string>\n";
        code += "#include \"main_sm/" + forkNamespace + "/main/main_executor.hpp\"\n";
        if (!bFastMode)
        {
            code += "#include \"main_sm/" + forkNamespace + "/main/main_exec_required.hpp\"\n";
        }
    }
    else
    {
        if (bFastMode)
        {
            code += "#define COMMIT_POL_FAST_MODE\n";
            code += "#include \"main_sm/" + forkNamespace + "/pols_generated/commit_pols.hpp\"\n";
        }
        code += "#include \"main_sm/" + forkNamespace + "/main_exec_generated/" + fileName + ".hpp\"\n";
        code += "#include \"scalar.hpp\"\n";
        code += "#include \"main_sm/"+ forkNamespace + "/main/eval_command.hpp\"\n";
        code += "#include <fstream>\n";
        code += "#include \"utils.hpp\"\n";
        code += "#include \"timer.hpp\"\n";
        code += "#include \"exit_process.hpp\"\n";
        code += "#include \"zkassert.hpp\"\n";
        code += "#include \"poseidon_g_permutation.hpp\"\n";
        code += "#include \"utils/time_metric.hpp\"\n";
        code += "#include \"zklog.hpp\"\n";
        if (!bFastMode)
            code += "#include \"goldilocks_precomputed.hpp\"\n";
        code += "#include \"ecrecover.hpp\"\n";

    }
    code += "\n";

    code += "namespace " + forkNamespace + "\n";
    code += "{\n";

    if (!bHeader)
    {
        code += "#define STACK_OFFSET 0x10000\n";
        code += "#define MEM_OFFSET   0x20000\n";
        code += "#define CTX_OFFSET   0x40000\n\n";

        code += "#define N_NO_COUNTERS_MULTIPLICATION_FACTOR 8\n\n";

        code += "#define FrFirst32Negative ( 0xFFFFFFFF00000001 - 0xFFFFFFFF )\n";
        code += "#define FrLast32Positive 0xFFFFFFFF\n\n";

        code += "#ifdef DEBUG\n";
        code += "#define CHECK_MAX_CNT_ASAP\n";
        code += "#endif\n";
        code += "#define CHECK_MAX_CNT_AT_THE_END\n\n";

        code += "vector<void *> " + functionName + "_labels;\n\n";

        code += "#pragma GCC push_options\n";
        code += "#pragma GCC optimize (\"O0\")\n\n";
    }

    if (bFastMode)
        code += "void " + functionName + " (" + forkNamespace + "::MainExecutor &mainExecutor, ProverRequest &proverRequest)";
    else
        code += "void "+ functionName + " (" + forkNamespace + "::MainExecutor &mainExecutor, ProverRequest &proverRequest, " + forkNamespace + "::MainCommitPols &pols, " + forkNamespace + "::MainExecRequired &required)";

    if (bHeader)
    {
        code += ";\n";
        code += "}\n";
        code += "\n";
        code += "#endif\n";
        return code;
    }
    else
        code += "\n{\n";

    if (bFastMode)
    {
        code += "    uint8_t polsBuffer[CommitPols::numPols()*sizeof(Goldilocks::Element)] = { 0 };\n";
        code += "    MainCommitPols pols((void *)polsBuffer, 1);\n";
    }
    code += "    int32_t addrRel = 0; // Relative and absolute address auxiliary variables\n";
    code += "    uint64_t addr = 0;\n";
    code += "    int32_t sp;\n";
    code += "    int64_t i64Aux;\n";
    //code += "    int64_t incHashPos = 0;\n"; // TODO: Remove initialization to check it is initialized before being used
    code += "    Rom &rom = mainExecutor.rom;\n";
    code += "    Goldilocks &fr = mainExecutor.fr;\n";
    code += "    uint64_t flushId;\n";
    code += "    uint64_t lastSentFlushId;\n";
    code += "\n";

    code += "    mainExecutor.labelsLock();\n";
    code += "    if (" + functionName + "_labels.size()==0)\n";
    code += "    {\n";
    for (uint64_t zkPC=0; zkPC<rom["program"].size(); zkPC++)
    {
        code += "        " + functionName + "_labels.push_back(&&" + functionName + "_rom_line_" + to_string(zkPC) + ");\n";
    }
    code += "    }\n";
    code += "    mainExecutor.labelsUnlock();\n\n";

    code += "    // Init execution flags\n";
    code += "    bool bProcessBatch = (proverRequest.type == prt_processBatch);\n";
    code += "    bool bUnsignedTransaction = (proverRequest.input.from != \"\") && (proverRequest.input.from != \"0x\");\n\n";
    
    code += "    // Unsigned transactions (from!=empty) are intended to be used to \"estimage gas\" (or \"call\")\n";
    code += "    // In prover mode, we cannot accept unsigned transactions, since the proof would not meet the PIL constrains\n";
    code += "    if (bUnsignedTransaction && !bProcessBatch)\n";
    code += "    {\n";
    code += "        proverRequest.result = ZKR_SM_MAIN_INVALID_UNSIGNED_TX;\n";
    code += "        zklog.error(\"MainExecutor::execute() failed called with bUnsignedTransaction=true but bProcessBatch=false\");\n";
    code += "        return;\n";
    code += "    }\n\n";

    code += "    Context ctx(mainExecutor.fr, mainExecutor.config, mainExecutor.fec, mainExecutor.fnec, pols, mainExecutor.rom, proverRequest, mainExecutor.pHashDB);\n\n";

    code += "    mainExecutor.initState(ctx);\n\n";

    code += "#ifdef LOG_COMPLETED_STEPS_TO_FILE\n";
    code += "    remove(\"c.txt\");\n";
    code += "#endif\n\n";

    code += "    // Copy input database content into context database\n";
    code += "    if (proverRequest.input.db.size() > 0)\n";
    code += "    {\n";
    code += "        mainExecutor.pHashDB->loadDB(proverRequest.input.db, true);\n";
    code += "        mainExecutor.pHashDB->flush(emptyString, emptyString, proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, flushId, lastSentFlushId);\n";
    code += "        if (mainExecutor.config.dbClearCache && (mainExecutor.config.databaseURL != \"local\"))\n";
    code += "        {\n";
    code += "            mainExecutor.pHashDB->clearCache();\n";
    code += "        }\n";
    code += "    }\n\n";

    code += "    // Copy input contracts database content into context database (dbProgram)\n";
    code += "    if (proverRequest.input.contractsBytecode.size() > 0)\n";
    code += "    {\n";
    code += "        mainExecutor.pHashDB->loadProgramDB(proverRequest.input.contractsBytecode, true);\n";
    code += "        mainExecutor.pHashDB->flush(emptyString, emptyString, proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, flushId, lastSentFlushId);\n";
    code += "        if (mainExecutor.config.dbClearCache && (mainExecutor.config.databaseURL != \"local\"))\n";
    code += "        {\n";
    code += "            mainExecutor.pHashDB->clearCache();\n";
    code += "        }\n";
    code += "    }\n\n";

    code += "    // opN are local, uncommitted polynomials\n";
    code += "    Goldilocks::Element op0, op1, op2, op3, op4, op5, op6, op7;\n";

    // Free in
    code += "    Goldilocks::Element fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7;\n";
    code += "    CommandResult cr;\n";

    // Storage free in
    code += "    Goldilocks::Element Kin0[12];\n";
    code += "    Goldilocks::Element Kin1[12];\n";
    code += "    mpz_class scalarD;\n";
    code += "    zkresult zkResult;\n";
    code += "    Goldilocks::Element Kin0Hash[4];\n";
    code += "    Goldilocks::Element Kin1Hash[4];\n";  // TODO: Reuse global variables
    code += "    Goldilocks::Element keyI[4];\n";
    code += "    Goldilocks::Element oldRoot[4];\n";
    code += "    Goldilocks::Element key[4];\n";
    code += "    SmtGetResult smtGetResult;\n";
    code += "    mpz_class opScalar;\n";
    code += "    mpz_class value;\n";
    if (!bFastMode)
        code += "    array<Goldilocks::Element,17> pg;\n";
    code += "    Goldilocks::Element fea[4];\n";
    code += "    SmtAction smtAction;\n";

    // Hash free in
    code += "    mpz_class s;\n";
    code += "    int64_t iPos;\n";
    code += "    uint64_t pos;\n";
    code += "    uint64_t size;\n";
    code += "    mpz_class result;\n";
    code += "    mpz_class dg;\n";
    code += "    uint64_t lm;\n";
    code += "    uint64_t lh;\n";
    code += "    mpz_class paddingA;\n";
    code += "    unordered_map< uint64_t, HashValue >::iterator hashIterator;\n";
    code += "    unordered_map<uint64_t, uint64_t>::iterator readsIterator;\n";
    code += "    HashValue emptyHashValue;\n";

    // Mem allign free in
    code += "    mpz_class m0;\n";
    code += "    mpz_class m1;\n";
    code += "    mpz_class offsetScalar;\n";
    code += "    uint64_t offset;\n";
    code += "    mpz_class leftV;\n";
    code += "    mpz_class rightV;\n";
    code += "    mpz_class v, _V;\n";
    code += "    mpz_class w0, w1, _W0, _W1;\n";
    code += "    MemAlignAction memAlignAction;\n";
    code += "    mpz_class byteMaskOn256(\"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF\", 16);\n";

    // Binary free in
    code += "    mpz_class a, b, c, _a, _b;\n";
    code += "    mpz_class expectedC;\n";
    code += "    BinaryAction binaryAction;\n";

    code += "    uint64_t b0;\n";
    code += "    bool bIsTouchedAddressTree;\n";

    code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
    code += "    struct timeval t;\n";
    code += "    TimeMetricStorage mainMetrics;\n";
    code += "    TimeMetricStorage evalCommandMetrics;\n";
    code += "#endif\n";

    // Arith
    code += "    mpz_class A, B, C, D, op;\n";
    code += "    mpz_class x1, y1, x2, y2, x3, y3;\n";
    code += "    ArithAction arithAction;\n";
    code += "    mpz_class _x3, _y3;\n";
    code += "    mpz_class left;\n";
    code += "    mpz_class right;\n";
    code += "    bool x3eq;\n";
    code += "    bool y3eq;\n";
    code += "    RawFec::Element fecX1, fecY1, fecX2, fecY2, fecX3, fecY3;\n";

    if (!bFastMode)
        code += "    MemoryAccess memoryAccess;\n";

    code += "    std::ofstream outfile;\n";
    code += "    std::unordered_map<uint64_t, Fea>::iterator memIterator;\n";
    code += "\n";

    code += "    uint64_t zkPC = 0; // Zero-knowledge program counter\n";
    code += "    uint64_t i=0; // Step, as it is used internally, set to 0 in fast mode to reuse the same evaluation all the time\n";
    if (!bFastMode)
        code += "    uint64_t nexti=1; // Next step, as it is used internally, set to 0 in fast mode to reuse the same evaluation all the time\n";
    code += "    ctx.N = mainExecutor.N; // Numer of evaluations\n";
    if (bFastMode)
    {
        code += "    uint64_t zero = 0;\n";
        code += "    ctx.pStep = &zero;\n";
    }
    else
    {
        code += "    ctx.pStep = &i; // ctx.pStep is used inside evaluateCommand() to find the current value of the registers, e.g. pols(A0)[ctx.step]\n";
    }
    code += "    ctx.pEvaluation = &i;\n";
    code += "    ctx.pZKPC = &zkPC; // Pointer to the zkPC\n\n";

    // Declare currentRCX only if repeat instruction is used
    for (uint64_t zkPC=0; zkPC<rom["program"].size(); zkPC++)
    {
        if (rom["program"][zkPC].contains("repeat") && (rom["program"][zkPC]["repeat"]==1))
        {
            code += "    Goldilocks::Element currentRCX = fr.zero();\n";
            break;
        }
    }

    code += "    uint64_t incHashPos = 0;\n";
    code += "    uint64_t incCounter = 0;\n\n";
    code += "    bool bJump = false;\n";
    code += "    uint64_t jmpnCondValue = 0;\n";
    code += "\n";

    code += "    uint64_t N_Max;\n";
    code += "    uint64_t N_Max_minus_one;\n";
    code += "    if (proverRequest.input.bNoCounters)\n";
    code += "    {\n";
    code += "        if (!bProcessBatch)\n";
    code += "        {\n";
    code += "            proverRequest.result = ZKR_SM_MAIN_INVALID_NO_COUNTERS;;\n";
    code += "            mainExecutor.logError(ctx, \"" + functionName + "()) found proverRequest.bNoCounters=true and bProcessBatch=false\");\n";
    code += "            return;\n";
    code += "        }\n";
    code += "        N_Max = mainExecutor.N_NoCounters;\n";
    code += "    }\n";
    code += "    else\n";
    code += "    {\n";
    code += "        N_Max = mainExecutor.N;\n";
    code += "    }\n\n";
    code += "    N_Max_minus_one = N_Max - 1;\n";

    for (uint64_t zkPC=0; zkPC<rom["program"].size(); zkPC++)
    {

        // When bConditionalJump=true, the code will go to the proper label after all the work has been done based on the content of bJump
        bool bConditionalJump = false;

        // When bForcedJump=true, the code will always jump
        bool bForcedJump = false;

        // When bIncHashPos=true, incHashPos will be added to HASHPOS
        bool bIncHashPos = false;

        // ROM instruction line, commented if not used to save compilation workload
        //if (!usedLabels.includes(zkPC))
        //    code += "// ";
        code += functionName + "_rom_line_" + to_string(zkPC) + ": //" + string(rom["program"][zkPC]["fileName"]) + ":" + to_string(rom["program"][zkPC]["line"]) + "=[" + removeDuplicateSpaces(string(rom["program"][zkPC]["lineStr"])) + "]\n\n";

        // START LOGS
        code += "#ifdef LOG_COMPLETED_STEPS_TO_FILE\n";
        code += "    fi0=fi1=fi2=fi3=fi4=fi5=fi6=fi7=fr.zero();\n";
        code += "#endif\n";
        code += "#ifdef LOG_START_STEPS\n";
        code += "    zklog.info(\"--> Starting step=\" + to_string(i) + \" zkPC=" + to_string(zkPC) + " zkasm=\" + rom.line[" + to_string(zkPC) + "].lineStr);\n";
        code += "#endif\n";
        code += "#ifdef LOG_PRINT_ROM_LINES\n";
        code += "    zklog.info(\"step=\" + to_string(i) + \" rom.line[" + to_string(zkPC) + "] =[\" + rom.line[" + to_string(zkPC) + "].toString(fr) + \"]\");\n";
        code += "#endif\n";
        code += "#ifdef LOG_START_STEPS_TO_FILE\n";
        code += "    outfile.open(\"c.txt\", std::ios_base::app); // append instead of overwrite\n";
        code += "    outfile << \"--> Starting step=\" << i << \" zkPC=" + to_string(zkPC) + " instruction= \" << rom.line[" + to_string(zkPC) + "].toString(fr) << endl;\n";
        code += "    outfile.close();\n";
        code += "#endif\n\n";

        // ECRECOVER PRE-CALCULATION 
        if(rom["labels"].contains("ecrecover_store_args") && zkPC == rom["labels"]["ecrecover_store_args"]){
            code += "    //ECRecover pre-calculation \n";
            code += "    if(mainExecutor.config.ECRecoverPrecalc){\n";
            code += "        zkassert(ctx.ecRecoverPrecalcBuffer.filled == false);\n";
            code += "        mpz_class signature_, r_, s_, v_;\n";
            if(bFastMode){   
                code += "       fea2scalar(fr, signature_, pols.A0[0], pols.A1[0], pols.A2[0], pols.A3[0], pols.A4[0], pols.A5[0], pols.A6[0], pols.A7[0]);\n";
                code += "       fea2scalar(fr, r_, pols.B0[0], pols.B1[0], pols.B2[0], pols.B3[0], pols.B4[0], pols.B5[0], pols.B6[0], pols.B7[0]);\n";
                code += "       fea2scalar(fr, s_, pols.C0[0], pols.C1[0], pols.C2[0], pols.C3[0], pols.C4[0], pols.C5[0], pols.C6[0], pols.C7[0]);\n";
                code += "       fea2scalar(fr, v_, pols.D0[0], pols.D1[0], pols.D2[0], pols.D3[0], pols.D4[0], pols.D5[0], pols.D6[0], pols.D7[0]);\n";
                
            }else{
                code += "       fea2scalar(fr, signature_, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);\n";
                code += "       fea2scalar(fr, r_, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);\n";
                code += "       fea2scalar(fr, s_, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]);\n";
                code += "       fea2scalar(fr, v_, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]);\n";
                
            }
            code += "       ctx.ecRecoverPrecalcBuffer.posUsed = ECRecoverPrecalc(signature_, r_, s_, v_, false, ctx.ecRecoverPrecalcBuffer.buffer, ctx.config.ECRecoverPrecalcNThreads);\n";
            code += "       ctx.ecRecoverPrecalcBuffer.pos=0;\n";
            code += "       if (ctx.ecRecoverPrecalcBuffer.posUsed > 0) ctx.ecRecoverPrecalcBuffer.filled = true;\n";
            code += "    }\n";

        }       
        if(rom["labels"].contains("ecrecover_end") && zkPC == rom["labels"]["ecrecover_end"]){

            code += "    //ECRecover destroy pre-calculaiton buffer\n";
            code += "    if( ctx.ecRecoverPrecalcBuffer.filled){\n";  
            code += "       zkassert(ctx.ecRecoverPrecalcBuffer.pos == ctx.ecRecoverPrecalcBuffer.posUsed);\n";
            code += "       ctx.ecRecoverPrecalcBuffer.filled = false;\n";
            code += "    }\n";
        }
        

        // INITIALIZATION

        bool opInitialized = false;

        // COMMAND BEFORE
        if (rom["program"][zkPC].contains("cmdBefore") &&
            (rom["program"][zkPC]["cmdBefore"].size()>0))
        {
            code += "    // Evaluate the list cmdBefore commands, and any children command, recursively\n";
            code += "    for (uint64_t j=0; j<rom.line[" + to_string(zkPC) + "].cmdBefore.size(); j++)\n";
            code += "    {\n";
            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "        gettimeofday(&t, NULL);\n";
            code += "#endif\n";
            code += "        cr.reset();\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        evalCommand(ctx, *rom.line[" + to_string(zkPC) + "].cmdBefore[j], cr);\n";
            code += "\n";
            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "        mainMetrics.add(\"Eval command\", TimeDiff(t));\n";
            code += "        evalCommandMetrics.add(rom.line[" + to_string(zkPC) + "].cmdBefore[j]->opAndFunction, TimeDiff(t));\n";
            code += "#endif\n";
            code += "        // In case of an external error, return it\n";
            code += "        if (cr.zkResult != ZKR_SUCCESS)\n";
            code += "        {\n";
            code += "            proverRequest.result = cr.zkResult;\n";
            code += "            mainExecutor.logError(ctx, string(\"Failed calling evalCommand() before result=\") + zkresult2string(proverRequest.result));\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            code += "        }\n";
            code += "    }\n";
            code += "\n";
        }

        /*************/
        /* SELECTORS */
        /*************/

        if (rom["program"][zkPC].contains("inA") && (rom["program"][zkPC]["inA"]!=0))
        {
            code += selector8("A", rom["program"][zkPC]["inA"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inB") && (rom["program"][zkPC]["inB"]!=0))
        {
            code += selector8("B", rom["program"][zkPC]["inB"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inC") && (rom["program"][zkPC]["inC"]!=0))
        {
            code += selector8("C", rom["program"][zkPC]["inC"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inD") && (rom["program"][zkPC]["inD"]!=0))
        {
            code += selector8("D", rom["program"][zkPC]["inD"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inE") && (rom["program"][zkPC]["inE"]!=0))
        {
            code += selector8("E", rom["program"][zkPC]["inE"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inSR") && (rom["program"][zkPC]["inSR"]!=0))
        {
            code += selector8("SR", rom["program"][zkPC]["inSR"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inCTX") && (rom["program"][zkPC]["inCTX"]!=0))
        {
            code += selector1("CTX", rom["program"][zkPC]["inCTX"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inSP") && (rom["program"][zkPC]["inSP"]!=0))
        {
            code += selector1("SP", rom["program"][zkPC]["inSP"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inPC") && (rom["program"][zkPC]["inPC"]!=0))
        {
            code += selector1("PC", rom["program"][zkPC]["inPC"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inGAS") && (rom["program"][zkPC]["inGAS"]!=0))
        {
            code += selector1("GAS", rom["program"][zkPC]["inGAS"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inSTEP") && (rom["program"][zkPC]["inSTEP"]!=0))
        {
            string inSTEPString = rom["program"][zkPC]["inSTEP"];
            int64_t inSTEP = atoi(inSTEPString.c_str());

            code += "    // op0 = op0 + inSTEP*step , where inSTEP=" + inSTEPString + "\n";

            string value = "";
            if (inSTEP == 1)
                value = "fr.fromU64(proverRequest.input.bNoCounters ? 0 : i)";
            else if (inSTEP == -1)
                value = "fr.neg(fr.fromU64(proverRequest.input.bNoCounters ? 0 : i))";
            else
                value = "fr.mul(fr.fromS32(" + inSTEPString + "), fr.fromU64(proverRequest.input.bNoCounters ? 0 : i))";
            if (opInitialized)
                value = "fr.add(op0, " + value + ")";
            code += "    op0 = " + value + ";\n";
            if (!opInitialized)
                for (uint64_t j=1; j<8; j++)
                {
                    code += "    op" + to_string(j) + " = fr.zero();\n";
                }
            if (!bFastMode)
                code += "    pols.inSTEP[i] = fr.fromS32(" + inSTEPString + ");\n";
            code += "\n";

            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inRR") && (rom["program"][zkPC]["inRR"]!=0))
        {
            code += selector1("RR", rom["program"][zkPC]["inRR"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inHASHPOS") && (rom["program"][zkPC]["inHASHPOS"]!=0))
        {
            code += selector1("HASHPOS", rom["program"][zkPC]["inHASHPOS"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inCntArith") && (rom["program"][zkPC]["inCntArith"]!=0))
        {
            code += selector1("cntArith", rom["program"][zkPC]["inCntArith"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inCntBinary") && (rom["program"][zkPC]["inCntBinary"]!=0))
        {
            code += selector1("cntBinary", rom["program"][zkPC]["inCntBinary"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inCntMemAlign") && (rom["program"][zkPC]["inCntMemAlign"]!=0))
        {
            code += selector1("cntMemAlign", rom["program"][zkPC]["inCntMemAlign"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inCntKeccakF") && (rom["program"][zkPC]["inCntKeccakF"]!=0))
        {
            code += selector1("cntKeccakF", rom["program"][zkPC]["inCntKeccakF"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inCntPoseidonG") && (rom["program"][zkPC]["inCntPoseidonG"]!=0))
        {
            code += selector1("cntPoseidonG", rom["program"][zkPC]["inCntPoseidonG"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inCntPaddingPG") && (rom["program"][zkPC]["inCntPaddingPG"]!=0))
        {
            code += selector1("cntPaddingPG", rom["program"][zkPC]["inCntPaddingPG"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("inROTL_C"))
        {
            string inROTL_CString = rom["program"][zkPC]["inROTL_C"];
            int64_t inROTL_C = atoi(inROTL_CString.c_str());
            if (inROTL_C != 0)
            {
                code += "    // If inROTL_C, op = C rotated left\n";
                if (opInitialized)
                {
                    code += "    op0 = fr.add(op0, fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C7[" + string(bFastMode?"0":"i") + "]));\n";
                    code += "    op1 = fr.add(op1, fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C0[" + string(bFastMode?"0":"i") + "]));\n";
                    code += "    op2 = fr.add(op2, fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C1[" + string(bFastMode?"0":"i") + "]));\n";
                    code += "    op3 = fr.add(op3, fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C2[" + string(bFastMode?"0":"i") + "]));\n";
                    code += "    op4 = fr.add(op4, fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C3[" + string(bFastMode?"0":"i") + "]));\n";
                    code += "    op5 = fr.add(op5, fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C4[" + string(bFastMode?"0":"i") + "]));\n";
                    code += "    op6 = fr.add(op6, fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C5[" + string(bFastMode?"0":"i") + "]));\n";
                    code += "    op7 = fr.add(op7, fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C6[" + string(bFastMode?"0":"i") + "]));\n";
                }
                else
                {
                    code += "    op0 = fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C7[" + string(bFastMode?"0":"i") + "]);\n";
                    code += "    op1 = fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C0[" + string(bFastMode?"0":"i") + "]);\n";
                    code += "    op2 = fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C1[" + string(bFastMode?"0":"i") + "]);\n";
                    code += "    op3 = fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C2[" + string(bFastMode?"0":"i") + "]);\n";
                    code += "    op4 = fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C3[" + string(bFastMode?"0":"i") + "]);\n";
                    code += "    op5 = fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C4[" + string(bFastMode?"0":"i") + "]);\n";
                    code += "    op6 = fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C5[" + string(bFastMode?"0":"i") + "]);\n";
                    code += "    op7 = fr.mul(rom.line[" + to_string(zkPC) + "].inROTL_C, pols.C6[" + string(bFastMode?"0":"i") + "]);\n";
                }
                if (!bFastMode)
                {
                    code += "    pols.inROTL_C[i] = rom.line[" + to_string(zkPC) + "].inROTL_C;\n";
                }
                code += "\n";
                opInitialized = true;
            }
        }

        if (rom["program"][zkPC].contains("inRCX") && (rom["program"][zkPC]["inRCX"]!=0))
        {
            code += selector1("RCX", rom["program"][zkPC]["inRCX"], opInitialized, bFastMode);
            opInitialized = true;
        }

        if (rom["program"][zkPC].contains("CONST") && (rom["program"][zkPC]["CONST"]!=0))
        {
            string aux = rom["program"][zkPC]["CONST"];
            int64_t iAux = atoi(aux.c_str());
            if (iAux!=0)
            {
                code += selectorConst(/*rom["program"][zkPC]["CONST"]*/iAux, opInitialized, bFastMode);
                opInitialized = true;
            }
        }

        if (rom["program"][zkPC].contains("CONSTL") && (rom["program"][zkPC]["CONSTL"]!="0"))
        {
            code += selectorConstL(rom["program"][zkPC]["CONSTL"], opInitialized, bFastMode);
            opInitialized = true;
        }

        bool bOnlyOffset = false;

        if ( (rom["program"][zkPC].contains("mOp") && (rom["program"][zkPC]["mOp"]==1)) ||
             (rom["program"][zkPC].contains("mWR") && (rom["program"][zkPC]["mWR"]==1)) ||
             (rom["program"][zkPC].contains("hashK") && (rom["program"][zkPC]["hashK"]==1)) ||
             (rom["program"][zkPC].contains("hashK1") && (rom["program"][zkPC]["hashK1"]==1)) ||
             (rom["program"][zkPC].contains("hashKLen") && (rom["program"][zkPC]["hashKLen"]==1)) ||
             (rom["program"][zkPC].contains("hashKDigest") && (rom["program"][zkPC]["hashKDigest"]==1)) ||
             (rom["program"][zkPC].contains("hashP") && (rom["program"][zkPC]["hashP"]==1)) ||
             (rom["program"][zkPC].contains("hashP1") && (rom["program"][zkPC]["hashP1"]==1)) ||
             (rom["program"][zkPC].contains("hashPLen") && (rom["program"][zkPC]["hashPLen"]==1)) ||
             (rom["program"][zkPC].contains("hashPDigest") && (rom["program"][zkPC]["hashPDigest"]==1)) ||
             (rom["program"][zkPC].contains("JMP") && (rom["program"][zkPC]["JMP"]==1)) ||
             (rom["program"][zkPC].contains("JMPN") && (rom["program"][zkPC]["JMPN"]==1)) ||
             (rom["program"][zkPC].contains("JMPC") && (rom["program"][zkPC]["JMPC"]==1)) ||
             (rom["program"][zkPC].contains("JMPZ") && (rom["program"][zkPC]["JMPZ"]==1)) ||
             (rom["program"][zkPC].contains("call") && (rom["program"][zkPC]["call"]==1)) )
        {
            bool bAddrRel = false;
            bool bOffset = false;
            code += "    // If address is involved, load offset into addr\n";
            if ( (rom["program"][zkPC].contains("ind") && (rom["program"][zkPC]["ind"]==1))  &&
                 (rom["program"][zkPC].contains("indRR") && (rom["program"][zkPC]["indRR"]==1)) )
            {
                cerr << "Error: Both ind and indRR are set to 1" << endl;
                exit(-1);
            }
            if (rom["program"][zkPC].contains("ind") && (rom["program"][zkPC]["ind"]==1))
            {
                code += "    if (!fr.toS32(addrRel, pols.E0[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_TOS32;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fr.toS32() with pols.E0[i]=\" + fr.toString(pols.E0[" + string(bFastMode?"0":"i") + "], 16));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                bAddrRel = true;
            }
            if (rom["program"][zkPC].contains("indRR") && (rom["program"][zkPC]["indRR"]==1))
            {
                code += "    if ( !fr.toS32(addrRel, pols.RR[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_TOS32;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fr.toS32() with pols.RR[i]=\" + fr.toString(pols.RR[" + string(bFastMode?"0":"i") + "], 16));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                bAddrRel = true;
            }
            if (rom["program"][zkPC].contains("offset") && (rom["program"][zkPC]["offset"] != 0))
            {
                if (bAddrRel)
                    code += "    addrRel += " + to_string(rom["program"][zkPC]["offset"]) + ";\n";
                else
                    code += "    addrRel = " + to_string(rom["program"][zkPC]["offset"]) + ";\n";
                bOffset = true;
            }
            if (rom["program"][zkPC].contains("isStack") && (rom["program"][zkPC]["isStack"]==1))
            {
                code += "    if (!fr.toS32(sp, pols.SP[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_TOS32;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fr.toS32() with pols.SP[i]=\" + fr.toString(pols.SP[" + string(bFastMode?"0":"i") + "], 16));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                if (bAddrRel || bOffset)
                    code += "    addrRel += sp;\n";
                else
                    code += "    addrRel = sp;\n";
                bAddrRel = true;
            }
            if (bAddrRel)
            {
                code += "    // If addrRel is possitive, and the sum is too big, fail\n";

                if (forkNamespace == "fork_4")
                {

                    code += "    if (proverRequest.input.publicInputsExtended.publicInputs.oldBatchNum > 382000)\n";
                    code += "    {\n";
                    code += "        if ( addrRel >= " + to_string( ( (rom["program"][zkPC].contains("isMem") && (rom["program"][zkPC]["isMem"]  == 1) ) ? 0x20000 : 0x10000 ) - 2048 ) + ")\n";
                    code += "        {\n";
                    code += "           proverRequest.result = ZKR_SM_MAIN_ADDRESS_OUT_OF_RANGE;\n";
                    code += "           zkPC=" + to_string(zkPC) +";\n";
                    code += "           mainExecutor.logError(ctx, \"addrRel too big addrRel=\" + to_string(addrRel));\n";
                    code += "           mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "           return;\n";
                    code += "       }\n";
                    code += "    }\n";
                    code += "    else\n";
                    code += "    {\n";
                    code += "        if (addrRel>=0x20000 || ((rom.line[" + to_string(zkPC) + "].isMem==1) && (addrRel >= 0x10000)))\n";
                    code += "        {\n";
                    code += "           proverRequest.result = ZKR_SM_MAIN_ADDRESS_OUT_OF_RANGE;\n";
                    code += "           zkPC=" + to_string(zkPC) +";\n";
                    code += "           mainExecutor.logError(ctx, \"addrRel too big addrRel=\" + to_string(addrRel));\n";
                    code += "           mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "           return;\n";
                    code += "       }\n";
                    code += "    }\n";
                }
                else
                {
                    code += "    if ( addrRel >= " + to_string( ( (rom["program"][zkPC].contains("isMem") && (rom["program"][zkPC]["isMem"]  == 1) ) ? 0x20000 : 0x10000 ) - 2048 ) + ")\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_ADDRESS_OUT_OF_RANGE;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, \"addrRel too big addrRel=\" + to_string(addrRel));\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                }
                
                code += "    // If addrRel is negative, fail\n";
                code += "    if (addrRel < 0)\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_ADDRESS_NEGATIVE;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"addrRel<0 addrRel=\" + to_string(addrRel));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    addr = addrRel;\n\n";
            }
            else if (!bAddrRel && bOffset)
            {
                if ((rom["program"][zkPC]["offset"] < 0) || (rom["program"][zkPC]["offset"] >= 0x10000))
                {
                    cerr << "Error: invalid offset=" + to_string(rom["program"][zkPC]["offset"]) << endl;
                    exit(-1);
                }
                if (!bFastMode)
                    code += "    addrRel = " + to_string(rom["program"][zkPC]["offset"]) + ";\n";
                code += "    addr = " + to_string(rom["program"][zkPC]["offset"]) + ";\n\n";
                bOnlyOffset = true;
            }
            else if (!bAddrRel && !bOffset)
            {
                if (!bFastMode)
                    code += "    addrRel = 0;\n";
                code += "    addr = 0;\n\n";
            }
        }
        else if ( (rom["program"][zkPC].contains("useCTX") && (rom["program"][zkPC]["useCTX"] == 1)) ||
                  (rom["program"][zkPC].contains("isStack") && (rom["program"][zkPC]["isStack"] == 1)) ||
                  (rom["program"][zkPC].contains("isMem") && (rom["program"][zkPC]["isMem"]  == 1)) )
        {
            code += "    addr = 0;\n\n";
        }
        else
        {
            code += "#if (defined LOG_COMPLETED_STEPS) || (defined LOG_COMPLETED_STEPS_TO_FILE)\n";
            code += "    addr = 0;\n";
            code += "#endif\n\n";
        }

        if (rom["program"][zkPC].contains("useCTX") && (rom["program"][zkPC]["useCTX"] == 1))
        {
            code += "    // If useCTX, addr = addr + CTX*CTX_OFFSET\n";
            code += "    addr += fr.toU64(pols.CTX[" + string(bFastMode?"0":"i") + "])*CTX_OFFSET;\n";
            if (!bFastMode)
                code += "    pols.useCTX[i] = fr.one();\n\n";
            else
                code += "\n";
            bOnlyOffset = false;
        }

        if (rom["program"][zkPC].contains("isStack") && (rom["program"][zkPC]["isStack"] == 1))
        {
            code += "    // If isStack, addr = addr + STACK_OFFSET\n";
            code += "    addr += STACK_OFFSET;\n";
            if (!bFastMode)
                code += "    pols.isStack[i] = fr.one();\n\n";
            else
                code += "\n";
            bOnlyOffset = false;
        }

        if (rom["program"][zkPC].contains("isMem") && (rom["program"][zkPC]["isMem"]  == 1))
        {
            code += "    // If isMem, addr = addr + MEM_OFFSET\n";
            code += "    addr += MEM_OFFSET;\n";
            if (!bFastMode)
                code += "    pols.isMem[i] = fr.one();\n\n";
            else
                code += "\n";
            bOnlyOffset = false;
        }

        if (rom["program"][zkPC].contains("incStack") && (rom["program"][zkPC]["incStack"] != 0) && !bFastMode)
        {
            code += "    pols.incStack[i] = fr.fromS32(" + to_string(rom["program"][zkPC]["incStack"]) + "); // Copy ROM flags into pols\n\n";
        }

        if (rom["program"][zkPC].contains("ind") && (rom["program"][zkPC]["ind"] != 0) && !bFastMode)
        {
            code += "    pols.ind[i] = fr.one();\n\n";
        }

        if (rom["program"][zkPC].contains("indRR") && (rom["program"][zkPC]["indRR"] != 0) && !bFastMode)
        {
            code += "    pols.indRR[i] = fr.one();\n\n";
        }

        // If offset, record it the committed polynomial
        if (rom["program"][zkPC].contains("offset") && (rom["program"][zkPC]["offset"] != 0) && !bFastMode)
        {
            code += "    pols.offset[i] = fr.fromS32(" + to_string(rom["program"][zkPC]["offset"]) + "); // Copy ROM flags into pols\n\n";
        }

        /**************/
        /* FREE INPUT */
        /**************/

        if (rom["program"][zkPC].contains("inFREE"))
        {

            if (!rom["program"][zkPC].contains("freeInTag"))
            {
                cerr << " Error: Instruction with freeIn without freeInTag" << endl;
                exit(-1);
            }

            if ( !rom["program"][zkPC]["freeInTag"].contains("op") ||
                 (rom["program"][zkPC]["freeInTag"]["op"] == "") )
            {
                uint64_t nHits = 0;

                // Memory read free in: get fi=mem[addr], if it exists
                if ( (rom["program"][zkPC].contains("mOp") && (rom["program"][zkPC]["mOp"]==1)) &&
                     (!rom["program"][zkPC].contains("mWR") || (rom["program"][zkPC]["mWR"]==0)) )
                {
                    code += "    // Memory read free in: get fi=mem[addr], if it exists\n";
                    code += "    memIterator = ctx.mem.find(addr);\n";
                    code += "    if (memIterator != ctx.mem.end()) {\n";
                    code += "        fi0 = memIterator->second.fe0;\n";
                    code += "        fi1 = memIterator->second.fe1;\n";
                    code += "        fi2 = memIterator->second.fe2;\n";
                    code += "        fi3 = memIterator->second.fe3;\n";
                    code += "        fi4 = memIterator->second.fe4;\n";
                    code += "        fi5 = memIterator->second.fe5;\n";
                    code += "        fi6 = memIterator->second.fe6;\n";
                    code += "        fi7 = memIterator->second.fe7;\n";
                    code += "    } else {\n";
                    code += "        fi0 = fr.zero();\n";
                    code += "        fi1 = fr.zero();\n";
                    code += "        fi2 = fr.zero();\n";
                    code += "        fi3 = fr.zero();\n";
                    code += "        fi4 = fr.zero();\n";
                    code += "        fi5 = fr.zero();\n";
                    code += "        fi6 = fr.zero();\n";
                    code += "        fi7 = fr.zero();\n";
                    code += "    }\n";
                    nHits++;
                }

                // Storage read free in: get a poseidon hash, and read fi=sto[hash]
                if (rom["program"][zkPC].contains("sRD") && (rom["program"][zkPC]["sRD"] == 1))
                {
                    code += "    // Storage read free in: get a poseidon hash, and read fi=sto[hash]\n";
                    code += "    Kin0[0] = pols.C0[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[1] = pols.C1[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[2] = pols.C2[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[3] = pols.C3[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[4] = pols.C4[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[5] = pols.C5[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[6] = pols.C6[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[7] = pols.C7[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[8] = fr.zero();\n";
                    code += "    Kin0[9] = fr.zero();\n";
                    code += "    Kin0[10] = fr.zero();\n";
                    code += "    Kin0[11] = fr.zero();\n";

                    code += "    Kin1[0] = pols.A0[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin1[1] = pols.A1[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin1[2] = pols.A2[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin1[3] = pols.A3[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin1[4] = pols.A4[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin1[5] = pols.A5[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin1[6] = pols.B0[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin1[7] = pols.B1[" + string(bFastMode?"0":"i") + "];\n";

                    code += "    b0 = fr.toU64(pols.B0[" + string(bFastMode?"0":"i") + "]);\n";
                    code += "    bIsTouchedAddressTree = (b0 == 5) || (b0 == 6);\n";
                    
                    code += "    if  ( !fr.isZero(pols.A5[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.A6[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.A7[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B2[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B3[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B4[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B5[" + string(bFastMode?"0":"i") + "])|| !fr.isZero(pols.B6[" + string(bFastMode?"0":"i") + "])|| !fr.isZero(pols.B7[" + string(bFastMode?"0":"i") + "]) )\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_STORAGE_INVALID_KEY;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, \"Storage read free in found non-zero A-B storage registers\");\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n\n";

                    code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
                    code += "    gettimeofday(&t, NULL);\n";
                    code += "#endif\n";

                    code += "    // Call poseidon and get the hash key\n";
                    code += "    mainExecutor.poseidon.hash(Kin0Hash, Kin0);\n";

                    code += "    // Reinject the first resulting hash as the capacity for the next poseidon hash\n";
                    code += "    Kin1[8] = Kin0Hash[0];\n";
                    code += "    Kin1[9] = Kin0Hash[1];\n";
                    code += "    Kin1[10] = Kin0Hash[2];\n";
                    code += "    Kin1[11] = Kin0Hash[3];\n";

                    code += "    // Call poseidon hash\n";
                    code += "    mainExecutor.poseidon.hash(Kin1Hash, Kin1);\n";

                    code += "    key[0] = Kin1Hash[0];\n";
                    code += "    key[1] = Kin1Hash[1];\n";
                    code += "    key[2] = Kin1Hash[2];\n";
                    code += "    key[3] = Kin1Hash[3];\n";
                    code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
                    code += "    mainMetrics.add(\"Poseidon\", TimeDiff(t), 3);\n";
                    code += "#endif\n";

                    code += "#ifdef LOG_STORAGE\n";
                    code += "    zklog.info(\"Storage read sRD got poseidon key: \" + ctx.fr.toString(ctx.lastSWrite.key, 16));\n";
                    code += "#endif\n";
                    code += "    sr8to4(fr, pols.SR0[" + string(bFastMode?"0":"i") + "], pols.SR1[" + string(bFastMode?"0":"i") + "], pols.SR2[" + string(bFastMode?"0":"i") + "], pols.SR3[" + string(bFastMode?"0":"i") + "], pols.SR4[" + string(bFastMode?"0":"i") + "], pols.SR5[" + string(bFastMode?"0":"i") + "], pols.SR6[" + string(bFastMode?"0":"i") + "], pols.SR7[" + string(bFastMode?"0":"i") + "], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);\n";

                    code += "    // Collect the keys used to read or write store data\n";
                    code += "    if (proverRequest.input.bGetKeys && !bIsTouchedAddressTree)\n";
                    code += "    {\n";
                    code += "        proverRequest.nodesKeys.insert(fea2string(fr, key));\n";
                    code += "    }\n";

                    code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
                    code += "    gettimeofday(&t, NULL);\n";
                    code += "#endif\n";
                    code += "    zkResult = mainExecutor.pHashDB->get(proverRequest.uuid, oldRoot, key, value, &smtGetResult, proverRequest.dbReadLog);\n";
                    code += "    if (zkResult != ZKR_SUCCESS)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = zkResult;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, string(\"Failed calling mainExecutor.pHashDB->get() result=\") + zkresult2string(zkResult));\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    code += "    incCounter = smtGetResult.proofHashCounter + 2;\n";

                    if (bFastMode)
                    {
                        code += "    zkResult = eval_addReadWriteAddress(ctx, smtGetResult.value);\n";
                        code += "    if (zkResult != ZKR_SUCCESS)\n";
                        code += "    {\n";
                        code += "        proverRequest.result = zkResult;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, string(\"Failed calling eval_addReadWriteAddress() 1 result=\") + zkresult2string(zkResult));\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                    }

                    code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
                    code += "    mainMetrics.add(\"SMT Get\", TimeDiff(t));\n";
                    code += "#endif\n";
                    code += "    scalar2fea(fr, smtGetResult.value, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";

                    code += "#ifdef LOG_STORAGE\n";
                    code += "    zklog.info(\"Storage read sRD read from key: \" + ctx.fr.toString(ctx.lastSWrite.key, 16) + \" value:\" + fr.toString(fi3, 16) + \":\" + fr.toString(fi2, 16) + \":\" + fr.toString(fi1, 16) + \":\" + fr.toString(fi0, 16));\n";
                    code += "#endif\n";

                    nHits++;
                }

                // Storage write free in: calculate the poseidon hash key, check its entry exists in storage, and update new root hash
                if (rom["program"][zkPC].contains("sWR") && (rom["program"][zkPC]["sWR"] == 1))
                {
                    code += "    // Storage write free in: calculate the poseidon hash key, check its entry exists in storage, and update new root hash\n";
                    code += "    // reset lastSWrite\n";
                    code += "    ctx.lastSWrite.reset();\n";
                    code += "    Kin0[0] = pols.C0[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[1] = pols.C1[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[2] = pols.C2[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[3] = pols.C3[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[4] = pols.C4[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[5] = pols.C5[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[6] = pols.C6[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[7] = pols.C7[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin0[8] = fr.zero();\n";
                    code += "    Kin0[9] = fr.zero();\n";
                    code += "    Kin0[10] = fr.zero();\n";
                    code += "    Kin0[11] = fr.zero();\n";

                    code += "    Kin1[0] = pols.A0[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin1[1] = pols.A1[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin1[2] = pols.A2[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin1[3] = pols.A3[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin1[4] = pols.A4[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin1[5] = pols.A5[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin1[6] = pols.B0[" + string(bFastMode?"0":"i") + "];\n";
                    code += "    Kin1[7] = pols.B1[" + string(bFastMode?"0":"i") + "];\n";

                    code += "    b0 = fr.toU64(pols.B0[" + string(bFastMode?"0":"i") + "]);\n";
                    code += "    bIsTouchedAddressTree = (b0 == 5) || (b0 == 6);\n";

                    code += "    if  ( !fr.isZero(pols.A5[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.A6[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.A7[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B2[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B3[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B4[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B5[" + string(bFastMode?"0":"i") + "])|| !fr.isZero(pols.B6[" + string(bFastMode?"0":"i") + "])|| !fr.isZero(pols.B7[" + string(bFastMode?"0":"i") + "]) )\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_STORAGE_INVALID_KEY;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, \"Storage write free in found non-zero A-B registers\");\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n\n";

                    code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
                    code += "    gettimeofday(&t, NULL);\n";
                    code += "#endif\n";

                    code += "    // Call poseidon and get the hash key\n";
                    code += "    mainExecutor.poseidon.hash(Kin0Hash, Kin0);\n";

                    code += "    Kin1[8] = Kin0Hash[0];\n";
                    code += "    Kin1[9] = Kin0Hash[1];\n";
                    code += "    Kin1[10] = Kin0Hash[2];\n";
                    code += "    Kin1[11] = Kin0Hash[3];\n";

                    code += "    ctx.lastSWrite.keyI[0] = Kin0Hash[0];\n";
                    code += "    ctx.lastSWrite.keyI[1] = Kin0Hash[1];\n";
                    code += "    ctx.lastSWrite.keyI[2] = Kin0Hash[2];\n";
                    code += "    ctx.lastSWrite.keyI[3] = Kin0Hash[3];\n";

                    code += "    // Call poseidon hash\n";
                    code += "    mainExecutor.poseidon.hash(Kin1Hash, Kin1);\n";

                    code += "    // Store a copy of the data in ctx.lastSWrite\n";
                    if (!bFastMode)
                    {
                        code += "    for (uint64_t j=0; j<12; j++)\n";
                        code += "        ctx.lastSWrite.Kin0[j] = Kin0[j];\n";
                        code += "    for (uint64_t j=0; j<12; j++)\n";
                        code += "        ctx.lastSWrite.Kin1[j] = Kin1[j];\n";
                    }
                    code += "    for (uint64_t j=0; j<4; j++)\n";
                    code += "        ctx.lastSWrite.keyI[j] = Kin0Hash[j];\n";
                    code += "    for (uint64_t j=0; j<4; j++)\n";
                    code += "        ctx.lastSWrite.key[j] = Kin1Hash[j];\n";

                    code += "    ctx.lastSWrite.key[0] = Kin1Hash[0];\n";
                    code += "    ctx.lastSWrite.key[1] = Kin1Hash[1];\n";
                    code += "    ctx.lastSWrite.key[2] = Kin1Hash[2];\n";
                    code += "    ctx.lastSWrite.key[3] = Kin1Hash[3];\n";
                    code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
                    code += "    mainMetrics.add(\"Poseidon\", TimeDiff(t));\n";
                    code += "#endif\n";

                    code += "#ifdef LOG_STORAGE\n";
                    code += "    zklog.info(\"Storage write sWR got poseidon key: \" + ctx.fr.toString(ctx.lastSWrite.key, 16));\n";
                    code += "#endif\n";
                    code += "    // Call SMT to get the new Merkel Tree root hash\n";
                    code += "    if (!fea2scalar(fr, scalarD, pols.D0[" + string(bFastMode?"0":"i") + "], pols.D1[" + string(bFastMode?"0":"i") + "], pols.D2[" + string(bFastMode?"0":"i") + "], pols.D3[" + string(bFastMode?"0":"i") + "], pols.D4[" + string(bFastMode?"0":"i") + "], pols.D5[" + string(bFastMode?"0":"i") + "], pols.D6[" + string(bFastMode?"0":"i") + "], pols.D7[" + string(bFastMode?"0":"i") + "]))\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar()\");\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
                    code += "    gettimeofday(&t, NULL);\n";
                    code += "#endif\n";
                    code += "    sr8to4(fr, pols.SR0[" + string(bFastMode?"0":"i") + "], pols.SR1[" + string(bFastMode?"0":"i") + "], pols.SR2[" + string(bFastMode?"0":"i") + "], pols.SR3[" + string(bFastMode?"0":"i") + "], pols.SR4[" + string(bFastMode?"0":"i") + "], pols.SR5[" + string(bFastMode?"0":"i") + "], pols.SR6[" + string(bFastMode?"0":"i") + "], pols.SR7[" + string(bFastMode?"0":"i") + "], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);\n";

                    code += "    // Collect the keys used to read or write store data\n";
                    code += "    if (proverRequest.input.bGetKeys && !bIsTouchedAddressTree)\n";
                    code += "    {\n";
                    code += "        proverRequest.nodesKeys.insert(fea2string(fr, ctx.lastSWrite.key));\n";
                    code += "    }\n";

                    code += "    zkResult = mainExecutor.pHashDB->set(proverRequest.uuid, proverRequest.pFullTracer->get_tx_number(), oldRoot, ctx.lastSWrite.key, scalarD, bIsTouchedAddressTree ? PERSISTENCE_TEMPORARY : ( proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE ), ctx.lastSWrite.newRoot, &ctx.lastSWrite.res, proverRequest.dbReadLog);\n";
                    code += "    if (zkResult != ZKR_SUCCESS)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = zkResult;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, string(\"Failed calling mainExecutor.pHashDB->set() result=\") + zkresult2string(zkResult));\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    code += "    incCounter = ctx.lastSWrite.res.proofHashCounter + 2;\n";

                    if (bFastMode)
                    {
                        code += "    zkResult = eval_addReadWriteAddress(ctx, scalarD);\n";
                        code += "    if (zkResult != ZKR_SUCCESS)\n";
                        code += "    {\n";
                        code += "        proverRequest.result = zkResult;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, string(\"Failed calling eval_addReadWriteAddress() 2 result=\") + zkresult2string(zkResult));\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                    }
                        
                    code += "    // If we just modified a balance\n";
                    code += "    if ( fr.isZero(pols.B0[" + string(bFastMode?"0":"i") + "]) && fr.isZero(pols.B1[" + string(bFastMode?"0":"i") + "]) )\n";
                    code += "        ctx.totalTransferredBalance += (ctx.lastSWrite.res.newValue - ctx.lastSWrite.res.oldValue);\n";

                    code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
                    code += "    mainMetrics.add(\"SMT Set\", TimeDiff(t));\n";
                    code += "#endif\n";
                    code += "    ctx.lastSWrite.step = i;\n";

                    code += "    sr4to8(fr, ctx.lastSWrite.newRoot[0], ctx.lastSWrite.newRoot[1], ctx.lastSWrite.newRoot[2], ctx.lastSWrite.newRoot[3], fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";

                    code += "#ifdef LOG_STORAGE\n";
                    code += "    zklog.info(\"Storage write sWR stored at key: \" + ctx.fr.toString(ctx.lastSWrite.key, 16) + \" newRoot: \" + fr.toString(ctx.lastSWrite.res.newRoot, 16));\n";
                    code += "#endif\n";

                    nHits++;
                }

                // HashK free in
                if ( (rom["program"][zkPC].contains("hashK") && (rom["program"][zkPC]["hashK"] == 1)) ||
                     (rom["program"][zkPC].contains("hashK1") && (rom["program"][zkPC]["hashK1"] == 1)))
                {
                    code += "    // HashK free in\n";
                    code += "    // If there is no entry in the hash database for this address, then create a new one\n";
                    code += "    hashIterator = ctx.hashK.find(addr);\n";
                    code += "    if (hashIterator == ctx.hashK.end())\n";
                    code += "    {\n";
                    code += "        ctx.hashK[addr] = emptyHashValue;\n";
                    code += "        hashIterator = ctx.hashK.find(addr);\n";
                    code += "        zkassert(hashIterator != ctx.hashK.end());\n";
                    code += "    }\n";

                    if (rom["program"][zkPC].contains("hashK") && (rom["program"][zkPC]["hashK"] == 1))
                    {
                        code += "    // Get the size of the hash from D0\n";
                        code += "    size = fr.toU64(pols.D0[" + string(bFastMode?"0":"i") + "]);\n";
                        code += "    if (size>32)\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_HASHK_SIZE_OUT_OF_RANGE;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Invalid size>32 for hashK 1: pols.D0[" + string(bFastMode?"0":"i") + "]=\" + fr.toString(pols.D0[" + string(bFastMode?"0":"i") + "], 16) + \" size=\" + to_string(size));\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n\n";
                    }
                    else
                    {
                        code += "    size = 1;\n";
                    }

                    code += "    // Get the positon of the hash from HASHPOS\n";
                    code += "    fr.toS64(iPos, pols.HASHPOS[" + string(bFastMode?"0":"i") + "]);\n";
                    code += "    if (iPos < 0)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_HASHK_POSITION_NEGATIVE;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, \"Invalid pos<0 for HashK 1: pols.HASHPOS[" + string(bFastMode?"0":"i") + "]=\" + fr.toString(pols.HASHPOS[" + string(bFastMode?"0":"i") + "], 16) + \" pos=\" + to_string(iPos));\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    code += "    pos = iPos;\n\n";

                    code += "    // Check that pos+size do not exceed data size\n";
                    code += "    if ( (pos+size) > hashIterator->second.data.size())\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_HASHK_POSITION_PLUS_SIZE_OUT_OF_RANGE;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, \"HashK 1 invalid size of hash: pos=\" + to_string(pos) + \" + size=\" + to_string(size) + \" > data.size=\" + to_string(hashIterator->second.data.size()));\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";

                    code += "    // Copy data into fi\n";
                    code += "    s = 0;\n";
                    code += "    for (uint64_t j=0; j<size; j++)\n";
                    code += "    {\n";
                    code += "        uint8_t data = hashIterator->second.data[pos+j];\n";
                    code += "        s = (s<<uint64_t(8)) + mpz_class(data);\n";
                    code += "    }\n";
                    code += "    scalar2fea(fr, s, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);\n";

                    code += "#ifdef LOG_HASHK\n";
                    code += "    zklog.info(\"hashK 1 i=\" + to_string(i) + \" zkPC=" + to_string(zkPC) + " addr=\" + to_string(addr) + \" pos=\" + to_string(pos) + \" size=\" + to_string(size) + \" data=\" + s.get_str(16));\n";
                    code += "#endif\n";

                    nHits++;
                }

                // HashKDigest free in
                if (rom["program"][zkPC].contains("hashKDigest") && (rom["program"][zkPC]["hashKDigest"] == 1))
                {
                    code += "    // HashKDigest free in\n";
                    code += "    // If there is no entry in the hash database for this address, this is an error\n";
                    code += "    hashIterator = ctx.hashK.find(addr);\n";
                    code += "    if (hashIterator == ctx.hashK.end())\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_HASHKDIGEST_ADDRESS_NOT_FOUND;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, \"HashKDigest 1: digest not defined for addr=\" + to_string(addr));\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";

                    code += "    // If digest was not calculated, this is an error\n";
                    code += "    if (!hashIterator->second.lenCalled)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_HASHKDIGEST_NOT_COMPLETED;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, \"HashKDigest 1: digest not calculated for addr=\" + to_string(addr) + \".  Call hashKLen to finish digest.\");\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";

                    code += "    // Copy digest into fi\n";
                    code += "    scalar2fea(fr, hashIterator->second.digest, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);\n";

                    code += "#ifdef LOG_HASHK\n";
                    code += "    zklog.info(\"hashKDigest 1 i=\" + to_string(i) + \" zkPC=" + to_string(zkPC) + " addr=\" + to_string(addr) + \" digest=\" + ctx.hashK[addr].digest.get_str(16));\n";
                    code += "#endif\n";

                    nHits++;
                }

                // HashP free in
                if ( (rom["program"][zkPC].contains("hashP") && (rom["program"][zkPC]["hashP"] == 1)) ||
                     (rom["program"][zkPC].contains("hashP1") && (rom["program"][zkPC]["hashP1"] == 1)) )
                {
                    code += "    // HashP free in\n";
                    code += "    // If there is no entry in the hash database for this address, then create a new one\n";
                    code += "    hashIterator = ctx.hashP.find(addr);\n";
                    code += "    if (hashIterator == ctx.hashP.end())\n";
                    code += "    {\n";
                    code += "        ctx.hashP[addr] = emptyHashValue;\n";
                    code += "        hashIterator = ctx.hashP.find(addr);\n";
                    code += "        zkassert(hashIterator != ctx.hashP.end());\n";
                    code += "    }\n";

                    if (rom["program"][zkPC].contains("hashP") && (rom["program"][zkPC]["hashP"] == 1))
                    {
                        code += "    // Get the size of the hash from D0\n";
                        code += "    size = fr.toU64(pols.D0[" + string(bFastMode?"0":"i") + "]);\n";
                        code += "    if (size>32)\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_HASHP_SIZE_OUT_OF_RANGE;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Invalid size>32 for hashP 1: pols.D0[" + string(bFastMode?"0":"i") + "]=\" + fr.toString(pols.D0[" + string(bFastMode?"0":"i") + "], 16) + \" size=\" + to_string(size));\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n\n";
                    }
                    else
                    {
                        code += "    size = 1;\n";
                    }

                    code += "    // Get the positon of the hash from HASHPOS\n";
                    code += "    fr.toS64(iPos, pols.HASHPOS[" + string(bFastMode?"0":"i") + "]);\n";
                    code += "    if (iPos < 0)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_HASHP_POSITION_NEGATIVE;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, \"Invalid pos<0 for HashP 1: pols.HASHPOS[" + string(bFastMode?"0":"i") + "]=\" + fr.toString(pols.HASHPOS[" + string(bFastMode?"0":"i") + "], 16) + \" pos=\" + to_string(iPos));\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    code += "    pos = iPos;\n\n";

                    code += "    // Check that pos+size do not exceed data size\n";
                    code += "    if ( (pos+size) > hashIterator->second.data.size())\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_HASHP_POSITION_PLUS_SIZE_OUT_OF_RANGE;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, \"HashP 1 invalid size of hash: pos=\" + to_string(pos) + \" size=\" + to_string(size) + \" data.size=\" + to_string(ctx.hashP[addr].data.size()));\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";

                    code += "    // Copy data into fi\n";
                    code += "    s = 0;\n";
                    code += "    for (uint64_t j=0; j<size; j++)\n";
                    code += "    {\n";
                    code += "        uint8_t data = hashIterator->second.data[pos+j];\n";
                    code += "        s = (s<<uint64_t(8)) + data;\n";
                    code += "    }\n";
                    code += "    scalar2fea(fr, s, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);\n";

                    nHits++;
                }

                // HashPDigest free in
                if (rom["program"][zkPC].contains("hashPDigest") && (rom["program"][zkPC]["hashPDigest"] == 1))
                {
                    code += "    // HashPDigest free in\n";
                    code += "    // If there is no entry in the hash database for this address, this is an error\n";
                    code += "    hashIterator = ctx.hashP.find(addr);\n";
                    code += "    if (hashIterator == ctx.hashP.end())\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_HASHPDIGEST_ADDRESS_NOT_FOUND;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, \"HashPDigest 1: digest not defined addr=\" + to_string(addr));\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    code += "    // If digest was not calculated, this is an error\n";
                    code += "    if (!hashIterator->second.lenCalled)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_HASHPDIGEST_NOT_COMPLETED;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, \"HashPDigest 1: digest not calculated.  Call hashPLen to finish digest.\");\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    code += "    // Copy digest into fi\n";
                    code += "    scalar2fea(fr, hashIterator->second.digest, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);\n";
                    nHits++;
                }

                // Binary free in
                if (rom["program"][zkPC].contains("bin") && (rom["program"][zkPC]["bin"] == 1))
                {
                    if (rom["program"][zkPC]["binOpcode"] == 0) // ADD
                    {
                        code += "    //Binary free in ADD\n";
                        code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    c = (a + b) & ScalarMask256;\n";
                        code += "    scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";
                        nHits++;
                    }
                    else if (rom["program"][zkPC]["binOpcode"] == 1) // SUB
                    {
                        code += "    //Binary free in SUB\n";
                        code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    c = (a - b + ScalarTwoTo256) & ScalarMask256;\n";
                        code += "    scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";
                        nHits++;
                    }
                    else if (rom["program"][zkPC]["binOpcode"] == 2) // LT
                    {
                        code += "    //Binary free in LT\n";
                        code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    c = (a < b);\n";
                        code += "    scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";
                        nHits++;
                    }
                    else if (rom["program"][zkPC]["binOpcode"] == 3) // SLT
                    {
                        code += "    //Binary free in SLT\n";
                        code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    if (a >= ScalarTwoTo255) a = a - ScalarTwoTo256;\n";
                        code += "    if (b >= ScalarTwoTo255) b = b - ScalarTwoTo256;\n";
                        code += "    c = (a < b);\n";
                        code += "    scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";
                        nHits++;
                    }
                    else if (rom["program"][zkPC]["binOpcode"] == 4) // EQ
                    {
                        code += "    //Binary free in EQ\n";
                        code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    c = (a == b);\n";
                        code += "    scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";
                        nHits++;
                    }
                    else if (rom["program"][zkPC]["binOpcode"] == 5) // AND
                    {
                        code += "    //Binary free in AND\n";
                        code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    c = (a & b);\n";
                        code += "    scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";
                        nHits++;
                    }
                    else if (rom["program"][zkPC]["binOpcode"] == 6) // OR
                    {
                        code += "    //Binary free in OR\n";
                        code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    c = (a | b);\n";
                        code += "    scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";
                        nHits++;
                    }
                    else if (rom["program"][zkPC]["binOpcode"] == 7) // XOR
                    {
                        code += "    //Binary free in XOR\n";
                        code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                        code += "    {\n";
                        code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                        code += "        zkPC=" + to_string(zkPC) +";\n";
                        code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                        code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                        code += "        return;\n";
                        code += "    }\n";
                        code += "    c = (a ^ b);\n";
                        code += "    scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";
                        nHits++;
                    }
                    else
                    {
                        cerr << "Error: Invalid binary operation: opcode=" << rom["program"][zkPC]["binOpcode"] << endl;
                        exit(-1);
                    }
                    code += "\n";
                }

                // Mem allign read free in
                if (rom["program"][zkPC].contains("memAlignRD") && (rom["program"][zkPC]["memAlignRD"]==1))
                {
                    code += "    // Mem allign read free in\n";
                    code += "    if (!fea2scalar(fr, m0, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    code += "    if (!fea2scalar(fr, m1, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    code += "    if (!fea2scalar(fr, offsetScalar, pols.C0[" + string(bFastMode?"0":"i") + "], pols.C1[" + string(bFastMode?"0":"i") + "], pols.C2[" + string(bFastMode?"0":"i") + "], pols.C3[" + string(bFastMode?"0":"i") + "], pols.C4[" + string(bFastMode?"0":"i") + "], pols.C5[" + string(bFastMode?"0":"i") + "], pols.C6[" + string(bFastMode?"0":"i") + "], pols.C7[" + string(bFastMode?"0":"i") + "]))\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.C)\");\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    code += "    if (offsetScalar<0 || offsetScalar>32)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = ZKR_SM_MAIN_MEMALIGN_OFFSET_OUT_OF_RANGE;\n";
                    code += "        mainExecutor.logError(ctx, \"MemAlign out of range offset=\" + offsetScalar.get_str());\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    code += "    offset = offsetScalar.get_ui();\n";
                    code += "    leftV = (m0 << (offset*8)) & ScalarMask256;\n";
                    code += "    rightV = (m1 >> (256 - offset*8)) & (ScalarMask256 >> (256 - offset*8));\n";
                    code += "    _V = leftV | rightV;\n";
                    code += "    scalar2fea(fr, _V, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";
                    nHits++;
                }

                // Check that one and only one instruction has been requested
                if (nHits != 1)
                {
                    cerr << "Error: Empty freeIn without any instruction: zkPC=" << zkPC << " nHits=" << nHits << endl;
                    exit(-1);
                }

            }
            // If freeInTag.op!="", then evaluate the requested command (recursively)
            else
            {
                code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
                code += "    gettimeofday(&t, NULL);\n";
                code += "#endif\n";

                if ( (rom["program"][zkPC]["freeInTag"]["op"]=="functionCall") && (rom["program"][zkPC]["freeInTag"]["funcName"]=="getBytecode") )
                {
                    code += "    cr.reset();\n";
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    eval_getBytecode(ctx, rom.line[" + to_string(zkPC) + "].freeInTag, cr);\n\n";
                    code += "    fi0 = cr.fea0;\n";
                    code += "    fi1 = cr.fea1;\n";
                    code += "    fi2 = cr.fea2;\n";
                    code += "    fi3 = cr.fea3;\n";
                    code += "    fi4 = cr.fea4;\n";
                    code += "    fi5 = cr.fea5;\n";
                    code += "    fi6 = cr.fea6;\n";
                    code += "    fi7 = cr.fea7;\n";
                }
                else if ( (rom["program"][zkPC]["freeInTag"]["op"]=="functionCall") && (rom["program"][zkPC]["freeInTag"]["funcName"]=="eventLog") )
                {
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    if (rom["program"][zkPC]["freeInTag"]["funcName"] == "storeLog")
                        code += "    ctx.proverRequest.fullTracer.onStoreLog(ctx, rom.line[" + to_string(zkPC) + "].freeInTag);\n";
                    else if (rom["program"][zkPC]["freeInTag"]["params"][0].contains("funcName") && (rom["program"][zkPC]["freeInTag"]["params"][0]["funcName"] == "onOpcode"))
                        code += "    ctx.proverRequest.fullTracer.onOpcode(ctx, rom.line[" + to_string(zkPC) + "].freeInTag);\n";
                    else if (rom["program"][zkPC]["freeInTag"]["params"][0].contains("varName") && (rom["program"][zkPC]["freeInTag"]["params"][0]["varName"] == "onError"))
                        code += "    ctx.proverRequest.fullTracer.onError(ctx, rom.line[" + to_string(zkPC) + "].freeInTag);\n";
                    else if (rom["program"][zkPC]["freeInTag"]["params"][0].contains("varName") && (rom["program"][zkPC]["freeInTag"]["params"][0]["varName"] == "onProcessTx"))
                        code += "    ctx.proverRequest.fullTracer.onProcessTx(ctx, rom.line[" + to_string(zkPC) + "].freeInTag);\n";
                    else if (rom["program"][zkPC]["freeInTag"]["params"][0].contains("varName") && (rom["program"][zkPC]["freeInTag"]["params"][0]["varName"] == "onUpdateStorage"))
                        code += "    ctx.proverRequest.fullTracer.onUpdateStorage(ctx, rom.line[" + to_string(zkPC) + "].freeInTag);\n";
                    else if (rom["program"][zkPC]["freeInTag"]["params"][0].contains("varName") && (rom["program"][zkPC]["freeInTag"]["params"][0]["varName"] == "onFinishTx"))
                        code += "    ctx.proverRequest.fullTracer.onFinishTx(ctx, rom.line[" + to_string(zkPC) + "].freeInTag);\n";
                    else if (rom["program"][zkPC]["freeInTag"]["params"][0].contains("varName") && (rom["program"][zkPC]["freeInTag"]["params"][0]["varName"] == "onStartBatch"))
                        code += "    ctx.proverRequest.fullTracer.onStartBatch(ctx, rom.line[" + to_string(zkPC) + "].freeInTag);\n";
                    else if (rom["program"][zkPC]["freeInTag"]["params"][0].contains("varName") && (rom["program"][zkPC]["freeInTag"]["params"][0]["varName"] == "onFinishBatch"))
                        code += "    ctx.proverRequest.fullTracer.onFinishBatch(ctx, rom.line[" + to_string(zkPC) + "].freeInTag);\n";
                    else
                    {
                        cerr << "Error: Invalid freeIn eventLog: zkPC=" << zkPC << endl;
                        exit(-1);
                    }
                    code += "    fi0 = fr.zero();\n";
                    code += "    fi1 = fr.zero();\n";
                    code += "    fi2 = fr.zero();\n";
                    code += "    fi3 = fr.zero();\n";
                    code += "    fi4 = fr.zero();\n";
                    code += "    fi5 = fr.zero();\n";
                    code += "    fi6 = fr.zero();\n";
                    code += "    fi7 = fr.zero();\n";
                }
                else if ( (rom["program"][zkPC]["freeInTag"]["op"]=="functionCall") && (rom["program"][zkPC]["freeInTag"]["funcName"]=="beforeLast") )
                {
                    code += "    if (*ctx.pStep >= ctx.N-2)\n";
                    code += "        fi0 = fr.zero();\n";
                    code += "    else\n";
                    code += "        fi0 = fr.negone();\n";
                    code += "    fi1 = fr.zero();\n";
                    code += "    fi2 = fr.zero();\n";
                    code += "    fi3 = fr.zero();\n";
                    code += "    fi4 = fr.zero();\n";
                    code += "    fi5 = fr.zero();\n";
                    code += "    fi6 = fr.zero();\n";
                    code += "    fi7 = fr.zero();\n";
                }
                else
                {
                    code += "    // Call evalCommand()\n";
                    code += "    cr.reset();\n";
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    evalCommand(ctx, rom.line[" + to_string(zkPC) + "].freeInTag, cr);\n\n";

                    code += "    // In case of an external error, return it\n";
                    code += "    if (cr.zkResult != ZKR_SUCCESS)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = cr.zkResult;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, string(\"Main exec failed calling evalCommand() result=\") + zkresult2string(proverRequest.result));\n";
                    code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n\n";

                    code += "    // Copy fi=command result, depending on its type \n";
                    code += "    switch (cr.type)\n";
                    code += "    {\n";
                    code += "    case crt_fea:\n";
                    code += "        fi0 = cr.fea0;\n";
                    code += "        fi1 = cr.fea1;\n";
                    code += "        fi2 = cr.fea2;\n";
                    code += "        fi3 = cr.fea3;\n";
                    code += "        fi4 = cr.fea4;\n";
                    code += "        fi5 = cr.fea5;\n";
                    code += "        fi6 = cr.fea6;\n";
                    code += "        fi7 = cr.fea7;\n";
                    code += "        break;\n";
                    code += "    case crt_fe:\n";
                    code += "        fi0 = cr.fe;\n";
                    code += "        fi1 = fr.zero();\n";
                    code += "        fi2 = fr.zero();\n";
                    code += "        fi3 = fr.zero();\n";
                    code += "        fi4 = fr.zero();\n";
                    code += "        fi5 = fr.zero();\n";
                    code += "        fi6 = fr.zero();\n";
                    code += "        fi7 = fr.zero();\n";
                    code += "        break;\n";
                    code += "    case crt_scalar:\n";
                    code += "        scalar2fea(fr, cr.scalar, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";
                    code += "        break;\n";
                    code += "    case crt_u16:\n";
                    code += "        fi0 = fr.fromU64(cr.u16);\n";
                    code += "        fi1 = fr.zero();\n";
                    code += "        fi2 = fr.zero();\n";
                    code += "        fi3 = fr.zero();\n";
                    code += "        fi4 = fr.zero();\n";
                    code += "        fi5 = fr.zero();\n";
                    code += "        fi6 = fr.zero();\n";
                    code += "        fi7 = fr.zero();\n";
                    code += "        break;\n";
                    code += "    case crt_u32:\n";
                    code += "        fi0 = fr.fromU64(cr.u32);\n";
                    code += "        fi1 = fr.zero();\n";
                    code += "        fi2 = fr.zero();\n";
                    code += "        fi3 = fr.zero();\n";
                    code += "        fi4 = fr.zero();\n";
                    code += "        fi5 = fr.zero();\n";
                    code += "        fi6 = fr.zero();\n";
                    code += "        fi7 = fr.zero();\n";
                    code += "        break;\n";
                    code += "    case crt_u64:\n";
                    code += "        fi0 = fr.fromU64(cr.u64);\n";
                    code += "        fi1 = fr.zero();\n";
                    code += "        fi2 = fr.zero();\n";
                    code += "        fi3 = fr.zero();\n";
                    code += "        fi4 = fr.zero();\n";
                    code += "        fi5 = fr.zero();\n";
                    code += "        fi6 = fr.zero();\n";
                    code += "        fi7 = fr.zero();\n";
                    code += "        break;\n";
                    code += "    default:\n";
                    code += "        mainExecutor.logError(ctx, \"Unexpected command result type: \" + to_string(cr.type));\n";
                    code += "        exitProcess();\n";
                    code += "    }\n";
                }

                code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
                code += "    mainMetrics.add(\"Eval command\", TimeDiff(t));\n";
                code += "    evalCommandMetrics.add(rom.line[" + to_string(zkPC) + "].freeInTag.opAndFunction, TimeDiff(t));\n";
                code += "#endif\n";

                /*
                code += "    // If we are in fast mode and we are consuming the last evaluations, exit the loop\n";
                code += "    if (cr.beforeLast)\n";
                code += "    {\n";
                code += "        if (ctx.lastStep == 0)\n";
                code += "        {\n";
                code += "            ctx.lastStep = step;\n";
                code += "        }\n";
                if (bFastMode)
                    code += "        goto " + functionName + "_end;\n";
                code += "    }\n\n";
                */

            }

            if (!bFastMode)
            {
                code += "    // Store polynomial FREE=fi\n";
                code += "    pols.FREE0[i] = fi0;\n";
                code += "    pols.FREE1[i] = fi1;\n";
                code += "    pols.FREE2[i] = fi2;\n";
                code += "    pols.FREE3[i] = fi3;\n";
                code += "    pols.FREE4[i] = fi4;\n";
                code += "    pols.FREE5[i] = fi5;\n";
                code += "    pols.FREE6[i] = fi6;\n";
                code += "    pols.FREE7[i] = fi7;\n\n";
            }

            code += "    // op = op + inFREE*fi\n";
            string inFREEString = rom["program"][zkPC]["inFREE"];
            int64_t inFREE = atoi(inFREEString.c_str());
            if (inFREE == 1)
            {
                if (opInitialized)
                {
                    code += "    op0 = fr.add(op0, fi0);\n";
                    code += "    op1 = fr.add(op1, fi1);\n";
                    code += "    op2 = fr.add(op2, fi2);\n";
                    code += "    op3 = fr.add(op3, fi3);\n";
                    code += "    op4 = fr.add(op4, fi4);\n";
                    code += "    op5 = fr.add(op5, fi5);\n";
                    code += "    op6 = fr.add(op6, fi6);\n";
                    code += "    op7 = fr.add(op7, fi7);\n\n";
                }
                else
                {
                    code += "    op0 = fi0;\n";
                    code += "    op1 = fi1;\n";
                    code += "    op2 = fi2;\n";
                    code += "    op3 = fi3;\n";
                    code += "    op4 = fi4;\n";
                    code += "    op5 = fi5;\n";
                    code += "    op6 = fi6;\n";
                    code += "    op7 = fi7;\n\n";
                    opInitialized = true;
                }
            }
            else
            {
                if (opInitialized)
                {
                    code += "    op0 = fr.add(op0, fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi0));\n";
                    code += "    op1 = fr.add(op1, fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi1));\n";
                    code += "    op2 = fr.add(op2, fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi2));\n";
                    code += "    op3 = fr.add(op3, fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi3));\n";
                    code += "    op4 = fr.add(op4, fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi4));\n";
                    code += "    op5 = fr.add(op5, fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi5));\n";
                    code += "    op6 = fr.add(op6, fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi6));\n";
                    code += "    op7 = fr.add(op7, fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi7));\n\n";
                }
                else
                {
                    code += "    op0 = fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi0);\n";
                    code += "    op1 = fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi1);\n";
                    code += "    op2 = fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi2);\n";
                    code += "    op3 = fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi3);\n";
                    code += "    op4 = fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi4);\n";
                    code += "    op5 = fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi5);\n";
                    code += "    op6 = fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi6);\n";
                    code += "    op7 = fr.mul(rom.line[" + to_string(zkPC) + "].inFREE, fi7);\n\n";
                    opInitialized = true;
                }
            }

            if (!bFastMode)
            {
                code += "    // Copy ROM flags into the polynomials\n";
                code += "    pols.inFREE[i] = rom.line[" + to_string(zkPC) + "].inFREE;\n\n";
            }
        }

        if (!opInitialized)
            code += "    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero\n\n";

        if (!bFastMode)
        {
            code += "    if (!fr.isZero(op0))\n";
            code += "    {\n";
            code += "        pols.op0Inv[i] = glp.inv(op0);\n";
            code += "    }\n";
        }

        /****************/
        /* INSTRUCTIONS */
        /****************/

        // If assert, check that A=op
        if (rom["program"][zkPC].contains("assert") && (rom["program"][zkPC]["assert"] == 1))
        {
            code += "    // If assert, check that A=op\n";
            code += "    if ( (!fr.equal(pols.A0[" + string(bFastMode?"0":"i") + "], op0)) ||\n";
            code += "         (!fr.equal(pols.A1[" + string(bFastMode?"0":"i") + "], op1)) ||\n";
            code += "         (!fr.equal(pols.A2[" + string(bFastMode?"0":"i") + "], op2)) ||\n";
            code += "         (!fr.equal(pols.A3[" + string(bFastMode?"0":"i") + "], op3)) ||\n";
            code += "         (!fr.equal(pols.A4[" + string(bFastMode?"0":"i") + "], op4)) ||\n";
            code += "         (!fr.equal(pols.A5[" + string(bFastMode?"0":"i") + "], op5)) ||\n";
            code += "         (!fr.equal(pols.A6[" + string(bFastMode?"0":"i") + "], op6)) ||\n";
            code += "         (!fr.equal(pols.A7[" + string(bFastMode?"0":"i") + "], op7)) )\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_ASSERT;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, string(\"ROM assert failed: AN!=opN\") + ";
            code += "\" A:\" + fr.toString(pols.A7[" + string(bFastMode?"0":"i") + "], 16) + \":\" + fr.toString(pols.A6[" + string(bFastMode?"0":"i") + "], 16) + \":\" + fr.toString(pols.A5[" + string(bFastMode?"0":"i") + "], 16) + \":\" + fr.toString(pols.A4[" + string(bFastMode?"0":"i") + "], 16) + \":\" + fr.toString(pols.A3[" + string(bFastMode?"0":"i") + "], 16) + \":\" + fr.toString(pols.A2[" + string(bFastMode?"0":"i") + "], 16) + \":\" + fr.toString(pols.A1[" + string(bFastMode?"0":"i") + "], 16) + \":\" + fr.toString(pols.A0[" + string(bFastMode?"0":"i") + "], 16) + ";
            code += "\" OP:\" + fr.toString(op7, 16) + \":\" + fr.toString(op6, 16) + \":\" + fr.toString(op5, 16) + \":\" + fr.toString(op4,16) + \":\" + fr.toString(op3, 16) + \":\" + fr.toString(op2, 16) + \":\" + fr.toString(op1, 16) + \":\" + fr.toString(op0, 16));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            if (!bFastMode)
                code += "    pols.assert_pol[i] = fr.one();\n";
            code += "\n";
        }


        // Memory operation instruction
        if (rom["program"][zkPC].contains("mOp") && (rom["program"][zkPC]["mOp"] == 1))
        {
            code += "    // Memory operation instruction\n";
            if (!bFastMode)
                code += "    pols.mOp[i] = fr.one();\n";

            // If mWR, mem[addr]=op
            if (rom["program"][zkPC].contains("mWR") && (rom["program"][zkPC]["mWR"] == 1))
            {
                if (!bFastMode)
                    code += "    pols.mWR[i] = fr.one();\n\n";

                code += "    memIterator = ctx.mem.find(addr);\n";
                code += "    if (memIterator == ctx.mem.end())\n";
                code += "    {\n";
                code += "        ctx.mem[addr].fe0 = op0;\n";
                code += "        memIterator = ctx.mem.find(addr);\n";
                code += "    }\n";
                code += "    else\n";
                code += "    {\n";
                code += "        memIterator->second.fe0 = op0;\n";
                code += "    }\n";
                code += "    memIterator->second.fe1 = op1;\n";
                code += "    memIterator->second.fe2 = op2;\n";
                code += "    memIterator->second.fe3 = op3;\n";
                code += "    memIterator->second.fe4 = op4;\n";
                code += "    memIterator->second.fe5 = op5;\n";
                code += "    memIterator->second.fe6 = op6;\n";
                code += "    memIterator->second.fe7 = op7;\n\n";

                if (!bFastMode)
                {
                    code += "    memoryAccess.bIsWrite = true;\n";
                    code += "    memoryAccess.address = addr;\n";
                    code += "    memoryAccess.pc = i;\n";
                    code += "    memoryAccess.fe0 = op0;\n";
                    code += "    memoryAccess.fe1 = op1;\n";
                    code += "    memoryAccess.fe2 = op2;\n";
                    code += "    memoryAccess.fe3 = op3;\n";
                    code += "    memoryAccess.fe4 = op4;\n";
                    code += "    memoryAccess.fe5 = op5;\n";
                    code += "    memoryAccess.fe6 = op6;\n";
                    code += "    memoryAccess.fe7 = op7;\n";
                    code += "    required.Memory.push_back(memoryAccess);\n\n";
                }
            }
            else
            {
                if (!bFastMode)
                {
                    code += "    memoryAccess.bIsWrite = false;\n";
                    code += "    memoryAccess.address = addr;\n";
                    code += "    memoryAccess.pc = i;\n";
                    code += "    memoryAccess.fe0 = op0;\n";
                    code += "    memoryAccess.fe1 = op1;\n";
                    code += "    memoryAccess.fe2 = op2;\n";
                    code += "    memoryAccess.fe3 = op3;\n";
                    code += "    memoryAccess.fe4 = op4;\n";
                    code += "    memoryAccess.fe5 = op5;\n";
                    code += "    memoryAccess.fe6 = op6;\n";
                    code += "    memoryAccess.fe7 = op7;\n";
                    code += "    required.Memory.push_back(memoryAccess);\n\n";
                }

                code += "    memIterator = ctx.mem.find(addr);\n";
                code += "    if (memIterator != ctx.mem.end()) \n";
                code += "    {\n";
                code += "        if ( (!fr.equal(memIterator->second.fe0, op0)) ||\n";
                code += "             (!fr.equal(memIterator->second.fe1, op1)) ||\n";
                code += "             (!fr.equal(memIterator->second.fe2, op2)) ||\n";
                code += "             (!fr.equal(memIterator->second.fe3, op3)) ||\n";
                code += "             (!fr.equal(memIterator->second.fe4, op4)) ||\n";
                code += "             (!fr.equal(memIterator->second.fe5, op5)) ||\n";
                code += "             (!fr.equal(memIterator->second.fe6, op6)) ||\n";
                code += "             (!fr.equal(memIterator->second.fe7, op7)) )\n";
                code += "        {\n";
                code += "            proverRequest.result = ZKR_SM_MAIN_MEMORY;\n";
                code += "            zkPC=" + to_string(zkPC) +";\n";
                code += "            mainExecutor.logError(ctx, \"Memory Read does not match op=\" + fea2string(fr, op0, op1, op2, op3, op4, op5, op6, op7) + \" mem=\" + fea2string(fr, memIterator->second.fe0, memIterator->second.fe1, memIterator->second.fe2, memIterator->second.fe3, memIterator->second.fe4, memIterator->second.fe5, memIterator->second.fe6, memIterator->second.fe7));\n";
                code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "            return;\n";
                code += "        }\n";
                code += "    }\n";
                code += "    else\n";
                code += "    {\n";
                code += "        if ( (!fr.isZero(op0)) ||\n";
                code += "             (!fr.isZero(op1)) ||\n";
                code += "             (!fr.isZero(op2)) ||\n";
                code += "             (!fr.isZero(op3)) ||\n";
                code += "             (!fr.isZero(op4)) ||\n";
                code += "             (!fr.isZero(op5)) ||\n";
                code += "             (!fr.isZero(op6)) ||\n";
                code += "             (!fr.isZero(op7)) )\n";
                code += "        {\n";
                code += "            proverRequest.result = ZKR_SM_MAIN_MEMORY;\n";
                code += "            zkPC=" + to_string(zkPC) +";\n";
                code += "            mainExecutor.logError(ctx, \"Memory Read does not match (op!=0)\");\n";
                code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "            return;\n";
                code += "        }\n";
                code += "    }\n\n";
            }
        }


        // Storage read instruction
        if (rom["program"][zkPC].contains("sRD") && (rom["program"][zkPC]["sRD"] == 1) )
        {
            code += "    // Storage read instruction\n";

            if (!bFastMode)
                code += "    pols.sRD[i] = fr.one();\n";

            code += "    Kin0[0] = pols.C0[" + string(bFastMode?"0":"i") + "];\n";
            code += "    Kin0[1] = pols.C1[" + string(bFastMode?"0":"i") + "];\n";
            code += "    Kin0[2] = pols.C2[" + string(bFastMode?"0":"i") + "];\n";
            code += "    Kin0[3] = pols.C3[" + string(bFastMode?"0":"i") + "];\n";
            code += "    Kin0[4] = pols.C4[" + string(bFastMode?"0":"i") + "];\n";
            code += "    Kin0[5] = pols.C5[" + string(bFastMode?"0":"i") + "];\n";
            code += "    Kin0[6] = pols.C6[" + string(bFastMode?"0":"i") + "];\n";
            code += "    Kin0[7] = pols.C7[" + string(bFastMode?"0":"i") + "];\n";
            code += "    Kin0[8] = fr.zero();\n";
            code += "    Kin0[9] = fr.zero();\n";
            code += "    Kin0[10] = fr.zero();\n";
            code += "    Kin0[11] = fr.zero();\n";

            code += "    Kin1[0] = pols.A0[" + string(bFastMode?"0":"i") + "];\n";
            code += "    Kin1[1] = pols.A1[" + string(bFastMode?"0":"i") + "];\n";
            code += "    Kin1[2] = pols.A2[" + string(bFastMode?"0":"i") + "];\n";
            code += "    Kin1[3] = pols.A3[" + string(bFastMode?"0":"i") + "];\n";
            code += "    Kin1[4] = pols.A4[" + string(bFastMode?"0":"i") + "];\n";
            code += "    Kin1[5] = pols.A5[" + string(bFastMode?"0":"i") + "];\n";
            code += "    Kin1[6] = pols.B0[" + string(bFastMode?"0":"i") + "];\n";
            code += "    Kin1[7] = pols.B1[" + string(bFastMode?"0":"i") + "];\n";

            code += "    b0 = fr.toU64(pols.B0[" + string(bFastMode?"0":"i") + "]);\n";
            code += "    bIsTouchedAddressTree = (b0 == 5) || (b0 == 6);\n";

            code += "    if  ( !fr.isZero(pols.A5[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.A6[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.A7[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B2[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B3[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B4[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B5[" + string(bFastMode?"0":"i") + "])|| !fr.isZero(pols.B6[" + string(bFastMode?"0":"i") + "])|| !fr.isZero(pols.B7[" + string(bFastMode?"0":"i") + "]) )\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_STORAGE_INVALID_KEY;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Storage read instruction found non-zero A-B registers\");\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";

            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "    gettimeofday(&t, NULL);\n";
            code += "#endif\n";

            code += "    // Call poseidon and get the hash key\n";
            code += "    mainExecutor.poseidon.hash(Kin0Hash, Kin0);\n";

            code += "    keyI[0] = Kin0Hash[0];\n";
            code += "    keyI[1] = Kin0Hash[1];\n";
            code += "    keyI[2] = Kin0Hash[2];\n";
            code += "    keyI[3] = Kin0Hash[3];\n";

            code += "    Kin1[8] = Kin0Hash[0];\n";
            code += "    Kin1[9] = Kin0Hash[1];\n";
            code += "    Kin1[10] = Kin0Hash[2];\n";
            code += "    Kin1[11] = Kin0Hash[3];\n";

            code += "    mainExecutor.poseidon.hash(Kin1Hash, Kin1);\n";

            // Store PoseidonG required data
            if (!bFastMode)
            {                
                code += "    // Store PoseidonG required data\n";
                code += "    for (uint64_t j=0; j<12; j++)\n";
                code += "        pg[j] = Kin0[j];\n";
                code += "    for (uint64_t j=0; j<4; j++)\n";
                code += "        pg[12+j] = Kin0Hash[j];\n";
                code += "    pg[16] = fr.fromU64(POSEIDONG_PERMUTATION1_ID);\n";
                code += "    required.PoseidonG.push_back(pg);\n";

                code += "    // Store PoseidonG required data\n";
                code += "    for (uint64_t j=0; j<12; j++)\n";
                code += "        pg[j] = Kin1[j];\n";
                code += "    for (uint64_t j=0; j<4; j++)\n";
                code += "        pg[12+j] = Kin1Hash[j];\n";
                code += "    pg[16] = fr.fromU64(POSEIDONG_PERMUTATION2_ID);\n";
                code += "    required.PoseidonG.push_back(pg);\n";
            }

            code += "    key[0] = Kin1Hash[0];\n";
            code += "    key[1] = Kin1Hash[1];\n";
            code += "    key[2] = Kin1Hash[2];\n";
            code += "    key[3] = Kin1Hash[3];\n";

            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "    mainMetrics.add(\"Poseidon\", TimeDiff(t), 3);\n";
            code += "#endif\n";

            code += "#ifdef LOG_STORAGE\n";
            code += "    zklog.info(\"Storage read sRD got poseidon key: \" + ctx.fr.toString(ctx.lastSWrite.key, 16));\n";
            code += "#endif\n";

            code += "    sr8to4(fr, pols.SR0[" + string(bFastMode?"0":"i") + "], pols.SR1[" + string(bFastMode?"0":"i") + "], pols.SR2[" + string(bFastMode?"0":"i") + "], pols.SR3[" + string(bFastMode?"0":"i") + "], pols.SR4[" + string(bFastMode?"0":"i") + "], pols.SR5[" + string(bFastMode?"0":"i") + "], pols.SR6[" + string(bFastMode?"0":"i") + "], pols.SR7[" + string(bFastMode?"0":"i") + "], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);\n";

            code += "    // Collect the keys used to read or write store data\n";
            code += "    if (proverRequest.input.bGetKeys && !bIsTouchedAddressTree)\n";
            code += "    {\n";
            code += "        proverRequest.nodesKeys.insert(fea2string(fr, key));\n";
            code += "    }\n";

            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "    gettimeofday(&t, NULL);\n";
            code += "#endif\n";
            code += "    zkResult = mainExecutor.pHashDB->get(proverRequest.uuid, oldRoot, key, value, &smtGetResult, proverRequest.dbReadLog);\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, string(\"Failed calling mainExecutor.pHashDB->get() result=\") + zkresult2string(zkResult));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    incCounter = smtGetResult.proofHashCounter + 2;\n";
                    
            if (bFastMode)
            {
                code += "    zkResult = eval_addReadWriteAddress(ctx, scalarD);\n";
                code += "    if (zkResult != ZKR_SUCCESS)\n";
                code += "    {\n";
                code += "        proverRequest.result = zkResult;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, string(\"Failed calling eval_addReadWriteAddress() 3 result=\") + zkresult2string(zkResult));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
            }

            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "    mainMetrics.add(\"SMT Get\", TimeDiff(t));\n";
            code += "#endif\n";
            if (!bFastMode)
            {
                code += "    smtAction.bIsSet = false;\n";
                code += "    smtAction.getResult = smtGetResult;\n";
                code += "    required.Storage.push_back(smtAction);\n";
            }

            code += "#ifdef LOG_STORAGE\n";
            code += "    zklog.info(\"Storage read sRD read from key: \" + ctx.fr.toString(ctx.lastSWrite.key, 16) + \" value:\" + fr.toString(fi3, 16) + \":\" + fr.toString(fi2, 16) + \":\" + fr.toString(fi1, 16) + \":\" + fr.toString(fi0, 16));\n";
            code += "#endif\n";

            code += "    if (!fea2scalar(fr, opScalar, op0, op1, op2, op3, op4, op5, op6, op7))\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    if (smtGetResult.value != opScalar)\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_STORAGE_READ_MISMATCH;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Storage read does not match: smtGetResult.value=\" + smtGetResult.value.get_str() + \" opScalar=\" + opScalar.get_str());\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";

            if (!bFastMode)
            {
                code += "    for (uint64_t k=0; k<4; k++)\n";
                code += "    {\n";
                code += "        pols.sKeyI[k][i] = keyI[k];\n";
                code += "        pols.sKey[k][i] = key[k];\n";
                code += "    }\n";
            }
        }

        // Storage write instruction
        if (rom["program"][zkPC].contains("sWR") && (rom["program"][zkPC]["sWR"] == 1))
        {
            code += "    // Storage write instruction\n";

            if (!bFastMode)
            {
                code += "    // Copy ROM flags into the polynomials\n";
                code += "    pols.sWR[i] = fr.one();\n";
            }

            code += "    if ( (ctx.lastSWrite.step == 0) || (ctx.lastSWrite.step != i) )\n";
            code += "    {\n";
            code += "        // Reset lastSWrite\n";
            code += "        ctx.lastSWrite.reset();\n";

            code += "        Kin0[0] = pols.C0[" + string(bFastMode?"0":"i") + "];\n";
            code += "        Kin0[1] = pols.C1[" + string(bFastMode?"0":"i") + "];\n";
            code += "        Kin0[2] = pols.C2[" + string(bFastMode?"0":"i") + "];\n";
            code += "        Kin0[3] = pols.C3[" + string(bFastMode?"0":"i") + "];\n";
            code += "        Kin0[4] = pols.C4[" + string(bFastMode?"0":"i") + "];\n";
            code += "        Kin0[5] = pols.C5[" + string(bFastMode?"0":"i") + "];\n";
            code += "        Kin0[6] = pols.C6[" + string(bFastMode?"0":"i") + "];\n";
            code += "        Kin0[7] = pols.C7[" + string(bFastMode?"0":"i") + "];\n";
            code += "        Kin0[8] = fr.zero();\n";
            code += "        Kin0[9] = fr.zero();\n";
            code += "        Kin0[10] = fr.zero();\n";
            code += "        Kin0[11] = fr.zero();\n";

            code += "        Kin1[0] = pols.A0[" + string(bFastMode?"0":"i") + "];\n";
            code += "        Kin1[1] = pols.A1[" + string(bFastMode?"0":"i") + "];\n";
            code += "        Kin1[2] = pols.A2[" + string(bFastMode?"0":"i") + "];\n";
            code += "        Kin1[3] = pols.A3[" + string(bFastMode?"0":"i") + "];\n";
            code += "        Kin1[4] = pols.A4[" + string(bFastMode?"0":"i") + "];\n";
            code += "        Kin1[5] = pols.A5[" + string(bFastMode?"0":"i") + "];\n";
            code += "        Kin1[6] = pols.B0[" + string(bFastMode?"0":"i") + "];\n";
            code += "        Kin1[7] = pols.B1[" + string(bFastMode?"0":"i") + "];\n";

            code += "        b0 = fr.toU64(pols.B0[" + string(bFastMode?"0":"i") + "]);;\n";
            code += "        bIsTouchedAddressTree = (b0 == 5) || (b0 == 6);;\n";

            code += "        if  ( !fr.isZero(pols.A5[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.A6[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.A7[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B2[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B3[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B4[" + string(bFastMode?"0":"i") + "]) || !fr.isZero(pols.B5[" + string(bFastMode?"0":"i") + "])|| !fr.isZero(pols.B6[" + string(bFastMode?"0":"i") + "])|| !fr.isZero(pols.B7[" + string(bFastMode?"0":"i") + "]) )\n";
            code += "        {\n";
            code += "            proverRequest.result = ZKR_SM_MAIN_STORAGE_INVALID_KEY;\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"Storage write instruction found non-zero A-B registers\");\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            code += "        }\n\n";

            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "        gettimeofday(&t, NULL);\n";
            code += "#endif\n";

            code += "        // Call poseidon and get the hash key\n";
            code += "        mainExecutor.poseidon.hash(Kin0Hash, Kin0);\n";

            code += "        ctx.lastSWrite.keyI[0] = Kin0Hash[0];\n";
            code += "        ctx.lastSWrite.keyI[1] = Kin0Hash[1];\n";
            code += "        ctx.lastSWrite.keyI[2] = Kin0Hash[2];\n";
            code += "        ctx.lastSWrite.keyI[3] = Kin0Hash[3];\n";

            code += "        Kin1[8] = Kin0Hash[0];\n";
            code += "        Kin1[9] = Kin0Hash[1];\n";
            code += "        Kin1[10] = Kin0Hash[2];\n";
            code += "        Kin1[11] = Kin0Hash[3];\n";

            code += "        mainExecutor.poseidon.hash(Kin1Hash, Kin1);\n";

            code += "        // Store a copy of the data in ctx.lastSWrite\n";
            if (!bFastMode)
            {
                code += "        for (uint64_t j=0; j<12; j++)\n";
                code += "            ctx.lastSWrite.Kin0[j] = Kin0[j];\n";
                code += "        for (uint64_t j=0; j<12; j++)\n";
                code += "            ctx.lastSWrite.Kin1[j] = Kin1[j];\n";
            }
            code += "        for (uint64_t j=0; j<4; j++)\n";
            code += "            ctx.lastSWrite.keyI[j] = Kin0Hash[j];\n";
            code += "        for (uint64_t j=0; j<4; j++)\n";
            code += "            ctx.lastSWrite.key[j] = Kin1Hash[j];\n";

            code += "        ctx.lastSWrite.key[0] = Kin1Hash[0];\n";
            code += "        ctx.lastSWrite.key[1] = Kin1Hash[1];\n";
            code += "        ctx.lastSWrite.key[2] = Kin1Hash[2];\n";
            code += "        ctx.lastSWrite.key[3] = Kin1Hash[3];\n";

            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "        mainMetrics.add(\"Poseidon\", TimeDiff(t));\n";
            code += "#endif\n";

            code += "        // Call SMT to get the new Merkel Tree root hash\n";
            code += "        if (!fea2scalar(fr, scalarD, pols.D0[" + string(bFastMode?"0":"i") + "], pols.D1[" + string(bFastMode?"0":"i") + "], pols.D2[" + string(bFastMode?"0":"i") + "], pols.D3[" + string(bFastMode?"0":"i") + "], pols.D4[" + string(bFastMode?"0":"i") + "], pols.D5[" + string(bFastMode?"0":"i") + "], pols.D6[" + string(bFastMode?"0":"i") + "], pols.D7[" + string(bFastMode?"0":"i") + "]))\n";
            code += "        {\n";
            code += "            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.D)\");\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            code += "        }\n";
            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "        gettimeofday(&t, NULL);\n";
            code += "#endif\n";

            code += "        sr8to4(fr, pols.SR0[" + string(bFastMode?"0":"i") + "], pols.SR1[" + string(bFastMode?"0":"i") + "], pols.SR2[" + string(bFastMode?"0":"i") + "], pols.SR3[" + string(bFastMode?"0":"i") + "], pols.SR4[" + string(bFastMode?"0":"i") + "], pols.SR5[" + string(bFastMode?"0":"i") + "], pols.SR6[" + string(bFastMode?"0":"i") + "], pols.SR7[" + string(bFastMode?"0":"i") + "], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);\n";

            code += "        // Collect the keys used to read or write store data\n";
            code += "        if (proverRequest.input.bGetKeys && !bIsTouchedAddressTree)\n";
            code += "        {\n";
            code += "            proverRequest.nodesKeys.insert(fea2string(fr, ctx.lastSWrite.key));\n";
            code += "        }\n";

            code += "        zkResult = mainExecutor.pHashDB->set(proverRequest.uuid, proverRequest.pFullTracer->get_tx_number(), oldRoot, ctx.lastSWrite.key, scalarD, bIsTouchedAddressTree ? PERSISTENCE_TEMPORARY : ( proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE ), ctx.lastSWrite.newRoot, &ctx.lastSWrite.res, proverRequest.dbReadLog);\n";
            code += "        if (zkResult != ZKR_SUCCESS)\n";
            code += "        {\n";
            code += "            proverRequest.result = zkResult;\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, string(\"Failed calling mainExecutor.pHashDB->set() result=\") + zkresult2string(zkResult));\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            code += "        }\n";
            code += "        incCounter = ctx.lastSWrite.res.proofHashCounter + 2;\n";
                    
            if (bFastMode)
            {
                code += "        zkResult = eval_addReadWriteAddress(ctx, scalarD);\n";
                code += "        if (zkResult != ZKR_SUCCESS)\n";
                code += "        {\n";
                code += "            proverRequest.result = zkResult;\n";
                code += "            zkPC=" + to_string(zkPC) +";\n";
                code += "            mainExecutor.logError(ctx, string(\"Failed calling eval_addReadWriteAddress() 4 result=\") + zkresult2string(zkResult));\n";
                code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "            return;\n";
                code += "        }\n";
            }
                        
            code += "        // If we just modified a balance\n";
            code += "        if ( fr.isZero(pols.B0[" + string(bFastMode?"0":"i") + "]) && fr.isZero(pols.B1[" + string(bFastMode?"0":"i") + "]) )\n";
            code += "            ctx.totalTransferredBalance += (ctx.lastSWrite.res.newValue - ctx.lastSWrite.res.oldValue);\n";

            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "        mainMetrics.add(\"SMT Set\", TimeDiff(t));\n";
            code += "#endif\n";

            code += "        ctx.lastSWrite.step = i;\n";
            code += "    }\n";

            // Store PoseidonG required data
            if (!bFastMode)
            {
                code += "    // Store PoseidonG required data\n";
                code += "    for (uint64_t j=0; j<12; j++)\n";
                code += "        pg[j] = ctx.lastSWrite.Kin0[j];\n";
                code += "    for (uint64_t j=0; j<4; j++)\n";
                code += "        pg[12+j] = ctx.lastSWrite.keyI[j];\n";
                code += "    pg[16] = fr.fromU64(POSEIDONG_PERMUTATION1_ID);\n";
                code += "    required.PoseidonG.push_back(pg);\n";
                code += "    // Store PoseidonG required data\n";
                code += "    for (uint64_t j=0; j<12; j++)\n";
                code += "        pg[j] = ctx.lastSWrite.Kin1[j];\n";
                code += "    for (uint64_t j=0; j<4; j++)\n";
                code += "       pg[12+j] = ctx.lastSWrite.key[j];\n";
                code += "    pg[16] = fr.fromU64(POSEIDONG_PERMUTATION2_ID);\n";
                code += "    required.PoseidonG.push_back(pg);\n";
            }

            if (!bFastMode)
            {
                code += "    smtAction.bIsSet = true;\n";
                code += "    smtAction.setResult = ctx.lastSWrite.res;\n";
                code += "    required.Storage.push_back(smtAction);\n";
            }

            code += "    // Check that the new root hash equals op0\n";
            code += "    sr8to4(fr, op0, op1, op2, op3, op4, op5, op6, op7, oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);\n";

            code += "    if ( !fr.equal(ctx.lastSWrite.newRoot[0], oldRoot[0]) ||\n";
            code += "         !fr.equal(ctx.lastSWrite.newRoot[1], oldRoot[1]) ||\n";
            code += "         !fr.equal(ctx.lastSWrite.newRoot[2], oldRoot[2]) ||\n";
            code += "         !fr.equal(ctx.lastSWrite.newRoot[3], oldRoot[3]) )\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_STORAGE_WRITE_MISMATCH;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Storage write does not match: ctx.lastSWrite.newRoot: \" + fr.toString(ctx.lastSWrite.newRoot[3], 16) + \":\" + fr.toString(ctx.lastSWrite.newRoot[2], 16) + \":\" + fr.toString(ctx.lastSWrite.newRoot[1], 16) + \":\" + fr.toString(ctx.lastSWrite.newRoot[0], 16) + \" oldRoot: \" + fr.toString(oldRoot[3], 16) + \":\" + fr.toString(oldRoot[2], 16) + \":\" + fr.toString(oldRoot[1], 16) + \":\" + fr.toString(oldRoot[0], 16));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";

            code += "    sr8to4(fr, op0, op1, op2, op3, op4, op5, op6, op7, fea[0], fea[1], fea[2], fea[3]);\n";
            code += "    if ( !fr.equal(ctx.lastSWrite.newRoot[0], fea[0]) ||\n";
            code += "         !fr.equal(ctx.lastSWrite.newRoot[1], fea[1]) ||\n";
            code += "         !fr.equal(ctx.lastSWrite.newRoot[2], fea[2]) ||\n";
            code += "         !fr.equal(ctx.lastSWrite.newRoot[3], fea[3]) )\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_STORAGE_WRITE_MISMATCH;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Storage write does not match: ctx.lastSWrite.newRoot=\" + fea2string(fr, ctx.lastSWrite.newRoot) + \" op=\" + fea2string(fr, fea));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";

            if (!bFastMode)
            {
                code += "    for (uint64_t k=0; k<4; k++)\n";
                code += "    {\n";
                code += "        pols.sKeyI[k][i] =  ctx.lastSWrite.keyI[k];\n";
                code += "        pols.sKey[k][i] = ctx.lastSWrite.key[k];\n";
                code += "    }\n";
            }
        }

        // HashK instruction
        if ( (rom["program"][zkPC].contains("hashK") && (rom["program"][zkPC]["hashK"] == 1)) ||
             (rom["program"][zkPC].contains("hashK1") && (rom["program"][zkPC]["hashK1"] == 1)) )
        {
            code += "    // HashK instruction\n";

            if (!bFastMode)
            {
                if (rom["program"][zkPC].contains("hashK") && (rom["program"][zkPC]["hashK"] == 1))
                {
                    code += "    pols.hashK[i] = fr.one();\n\n";
                }
                else
                {
                    code += "    pols.hashK1[i] = fr.one();\n\n";
                }
            }

            code += "    // If there is no entry in the hash database for this address, then create a new one\n";
            code += "    hashIterator = ctx.hashK.find(addr);\n";
            code += "    if (hashIterator == ctx.hashK.end())\n";
            code += "    {\n";
            code += "        ctx.hashK[addr] = emptyHashValue;\n";
            code += "        hashIterator = ctx.hashK.find(addr);\n";
            code += "        zkassert(hashIterator != ctx.hashK.end());\n";
            code += "    }\n\n";
            if (rom["program"][zkPC].contains("hashK") && (rom["program"][zkPC]["hashK"] == 1))
            {
                code += "    // Get the size of the hash from D0\n";
                code += "    size = fr.toU64(pols.D0[" + string(bFastMode?"0":"i") + "]);\n";
                code += "    if (size>32)\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_HASHK_SIZE_OUT_OF_RANGE;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Invalid size>32 for hashK 2: pols.D0[" + string(bFastMode?"0":"i") + "]=\" + fr.toString(pols.D0[" + string(bFastMode?"0":"i") + "], 16));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n\n";
            }
            else
            {
                code += "    size = 1;\n";
            }

            code += "    // Get the position of the hash from HASHPOS\n";
            code += "    fr.toS64(iPos, pols.HASHPOS[" + string(bFastMode?"0":"i") + "]);\n";
            code += "    if (iPos < 0)\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_HASHK_POSITION_NEGATIVE;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Invalid pos<0 for HashK 2: pols.HASHPOS[" + string(bFastMode?"0":"i") + "]=\" + fr.toString(pols.HASHPOS[" + string(bFastMode?"0":"i") + "], 16) + \" pos=\" + to_string(iPos));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    pos = iPos;\n\n";

            code += "    // Get contents of opN into a\n";
            code += "    if (!fea2scalar(fr, a, op0, op1, op2, op3, op4, op5, op6, op7))\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";

            code += "    // Fill the hash data vector with chunks of the scalar value\n";
            code += "    for (uint64_t j=0; j<size; j++)\n";
            code += "    {\n";
            code += "        result = (a >> ((size-j-1)*8)) & ScalarMask8;\n";
            code += "        uint8_t bm = result.get_ui();\n";
            code += "        if (hashIterator->second.data.size() == (pos+j))\n";
            code += "        {\n";
            code += "            hashIterator->second.data.push_back(bm);\n";
            code += "        }\n";
            code += "        else if (hashIterator->second.data.size() < (pos+j))\n";
            code += "        {\n";
            code += "            proverRequest.result = ZKR_SM_MAIN_HASHK_POSITION_PLUS_SIZE_OUT_OF_RANGE;\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"HashK 2: trying to insert data in a position:\" + to_string(pos+j) + \" higher than current data size:\" + to_string(ctx.hashK[addr].data.size()));\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            code += "        }\n";
            code += "        else\n";
            code += "        {\n";
            code += "            uint8_t bh;\n";
            code += "            bh = hashIterator->second.data[pos+j];\n";
            code += "            if (bm != bh)\n";
            code += "            {\n";
            code += "                proverRequest.result = ZKR_SM_MAIN_HASHK_VALUE_MISMATCH;\n";
            code += "                zkPC=" + to_string(zkPC) +";\n";
            code += "                mainExecutor.logError(ctx, \"HashK 2 bytes do not match: addr=\" + to_string(addr) + \" pos+j=\" + to_string(pos+j) + \" is bm=\" + to_string(bm) + \" and it should be bh=\" + to_string(bh));\n";
            code += "                mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "                return;\n";
            code += "            }\n";
            code += "        }\n";
            code += "    }\n\n";

            code += "    // Check that the remaining of a (op) is zero, i.e. no more data exists beyond size\n";
            code += "    paddingA = a >> (size*8);\n";
            code += "    if (paddingA != 0)\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_HASHK_PADDING_MISMATCH;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"HashK 2 incoherent size=\" + to_string(size) + \" a=\" + a.get_str(16) + \" paddingA=\" + paddingA.get_str(16));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";

            code += "    // Record the read operation\n";
            code += "    readsIterator = hashIterator->second.reads.find(pos);\n";
            code += "    if ( readsIterator != hashIterator->second.reads.end() )\n";
            code += "    {\n";
            code += "         if ( readsIterator->second != size )\n";
            code += "        {\n";
            code += "            proverRequest.result = ZKR_SM_MAIN_HASHK_SIZE_MISMATCH;\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"HashK 2 different read sizes in the same position addr=\" + to_string(addr) + \" pos=\" + to_string(pos) + \" ctx.hashK[addr].reads[pos]=\" + to_string(ctx.hashK[addr].reads[pos]) + \" size=\" + to_string(size));\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            code += "        }\n";
            code += "    }\n";
            code += "    else\n";
            code += "    {\n";
            code += "        ctx.hashK[addr].reads[pos] = size;\n";
            code += "    }\n\n";

            code += "    // Store the size\n";
            code += "    incHashPos = size;\n\n";
            bIncHashPos = true;

            code += "#ifdef LOG_HASHK\n";
            code += "    zklog.info(\"hashK 2 i=\" + to_string(i) + \" zkPC=" + to_string(zkPC) + " addr=\" + to_string(addr) + \" pos=\" + to_string(pos) + \" size=\" + to_string(size) + \" data=\" + a.get_str(16));\n";
            code += "#endif\n\n";
        }

        // HashKLen instruction
        if (rom["program"][zkPC].contains("hashKLen") && (rom["program"][zkPC]["hashKLen"] == 1))
        {
            code += "    // HashKLen instruction\n";

            if (!bFastMode)
                code += "    pols.hashKLen[i] = fr.one();\n";

            code += "    // Get the length\n";
            code += "    lm = fr.toU64(op0);\n\n";

            code += "    // Find the entry in the hash database for this address\n";
            code += "    hashIterator = ctx.hashK.find(addr);\n\n";

            code += "    // If it's undefined, compute a hash of 0 bytes\n";
            code += "    if (hashIterator == ctx.hashK.end())\n";
            code += "    {\n";
            code += "        // Check that length = 0\n";
            code += "        if (lm != 0)\n";
            code += "        {\n";
            code += "            proverRequest.result = ZKR_SM_MAIN_HASHKLEN_LENGTH_MISMATCH;\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"HashKLen 2 hashK[addr] is empty but lm is not 0 addr=\" + to_string(addr) + \" lm=\" + to_string(lm));\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            code += "        }\n\n";

            code += "        // Create an empty entry in this address slot\n";
            code += "        ctx.hashK[addr] = emptyHashValue;\n";
            code += "        hashIterator = ctx.hashK.find(addr);\n";
            code += "        zkassert(hashIterator != ctx.hashK.end());\n";
            code += "    }\n";

            code += "    if (ctx.hashK[addr].lenCalled)\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_HASHKLEN_CALLED_TWICE;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"HashKLen 2 called more than once addr=\" + to_string(addr));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    ctx.hashK[addr].lenCalled = true;\n";

            code += "    lh = hashIterator->second.data.size();\n";
            code += "    if (lm != lh)\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_HASHKLEN_LENGTH_MISMATCH;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"HashKLen 2 length does not match addr=\" + to_string(addr) + \" is lm=\" + to_string(lm) + \" and it should be lh=\" + to_string(lh));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    if (!hashIterator->second.digestCalled)\n";
            code += "    {\n";
            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "        gettimeofday(&t, NULL);\n";
            code += "#endif\n";
            code += "        keccak256(hashIterator->second.data.data(), hashIterator->second.data.size(), hashIterator->second.digest);\n";
            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "        mainMetrics.add(\"Keccak\", TimeDiff(t));\n";
            code += "#endif\n";

            code += "#ifdef LOG_HASHK\n";
            code += "        {\n";
            code += "           string s = \"hashKLen 2 calculate hashKLen: addr:\" + to_string(addr) + \" hash:\" + ctx.hashK[addr].digest.get_str(16) + \" size:\" + to_string(ctx.hashK[addr].data.size()) + \" data:\";\n";
            code += "           for (uint64_t k=0; k<ctx.hashK[addr].data.size(); k++) s += byte2string(ctx.hashK[addr].data[k]) + \":\";\n";
            code += "           zklog.info(s);\n";
            code += "        }\n";
            code += "#endif\n";
            code += "    }\n";

            code += "#ifdef LOG_HASHK\n";
            code += "    zklog.info(\"hashKLen 2 i=\" + to_string(i) + \" zkPC=" + to_string(zkPC) + " addr=\" + to_string(addr));\n";
            code += "#endif\n";
        }

        // HashKDigest instruction
        if (rom["program"][zkPC].contains("hashKDigest") && (rom["program"][zkPC]["hashKDigest"] == 1))
        {
            code += "    // HashKDigest instruction\n";

            if (!bFastMode)
                code += "    pols.hashKDigest[i] = fr.one();\n";

            code += "    // Find the entry in the hash database for this address\n";
            code += "    hashIterator = ctx.hashK.find(addr);\n";
            code += "    if (hashIterator == ctx.hashK.end())\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_HASHKDIGEST_NOT_FOUND;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"HashKDigest 2 could not find entry for addr=\" + to_string(addr));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";

            code += "    // Get contents of op into dg\n";
            code += "    if (!fea2scalar(fr, dg, op0, op1, op2, op3, op4, op5, op6, op7))\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";

            code += "    if (dg != hashIterator->second.digest)\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_HASHKDIGEST_DIGEST_MISMATCH;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"HashKDigest 2: Digest does not match op\");\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";

            code += "    if (ctx.hashK[addr].digestCalled)\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_HASHKDIGEST_CALLED_TWICE;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"HashKDigest 2 called more than once addr=\" + to_string(addr));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    ctx.hashK[addr].digestCalled = true;\n";

            code += "    incCounter = ceil((double(hashIterator->second.data.size()) + double(1)) / double(136));\n";

            code += "#ifdef LOG_HASHK\n";
            code += "    zklog.info(\"hashKDigest 2 i=\" + to_string(i) + \" zkPC=" + to_string(zkPC) + " addr=\" + to_string(addr) + \" digest=\" + ctx.hashK[addr].digest.get_str(16));\n";
            code += "#endif\n";
        }

        // HashP instruction
        if ( (rom["program"][zkPC].contains("hashP") && (rom["program"][zkPC]["hashP"] == 1)) ||
             (rom["program"][zkPC].contains("hashP1") && (rom["program"][zkPC]["hashP1"] == 1)) )
        {
            code += "    // HashP instruction\n";

            if (!bFastMode)
            {
                if (rom["program"][zkPC].contains("hashP") && (rom["program"][zkPC]["hashP"] == 1))
                {
                    code += "    pols.hashP[i] = fr.one();\n";
                }
                else
                {
                    code += "    pols.hashP1[i] = fr.one();\n";
                }
            }

            code += "    // If there is no entry in the hash database for this address, then create a new one\n";
            code += "    hashIterator = ctx.hashP.find(addr);\n";
            code += "    if (hashIterator == ctx.hashP.end())\n";
            code += "    {\n";
            code += "        ctx.hashP[addr] = emptyHashValue;\n";
            code += "        hashIterator = ctx.hashP.find(addr);\n";
            code += "        zkassert(hashIterator != ctx.hashP.end());\n";
            code += "    }\n";

            if (rom["program"][zkPC].contains("hashP") && (rom["program"][zkPC]["hashP"] == 1))
            {
                code += "    // Get the size of the hash from D0\n";
                code += "    size = fr.toU64(pols.D0[" + string(bFastMode?"0":"i") + "]);\n";
                code += "    if (size>32)\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_HASHP_SIZE_OUT_OF_RANGE;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Invalid size>32 for hashP 2: pols.D0[" + string(bFastMode?"0":"i") + "]=\" + fr.toString(pols.D0[" + string(bFastMode?"0":"i") + "], 16) + \" size=\" + to_string(size));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n\n";
            }
            else
            {
                code += "    size = 1;\n";
            }

            code += "    // Get the positon of the hash from HASHPOS\n";
            code += "    fr.toS64(iPos, pols.HASHPOS[" + string(bFastMode?"0":"i") + "]);\n";
            code += "    if (iPos < 0)\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_HASHP_POSITION_NEGATIVE;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Invalid pos<0 for HashP 2: pols.HASHPOS[" + string(bFastMode?"0":"i") + "]=\" + fr.toString(pols.HASHPOS[" + string(bFastMode?"0":"i") + "], 16) + \" pos=\" + to_string(iPos));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    pos = iPos;\n\n";

            code += "    // Get contents of opN into a\n";
            code += "    if (!fea2scalar(fr, a, op0, op1, op2, op3, op4, op5, op6, op7))\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";

            code += "    // Fill the hash data vector with chunks of the scalar value\n";
            code += "    for (uint64_t j=0; j<size; j++) {\n";
            code += "        result = (a >> (size-j-1)*8) & ScalarMask8;\n";
            code += "        uint8_t bm = result.get_ui();\n";
            code += "        if (hashIterator->second.data.size() == (pos+j))\n";
            code += "        {\n";
            code += "            hashIterator->second.data.push_back(bm);\n";
            code += "        }\n";
            code += "        else if (hashIterator->second.data.size() < (pos+j))\n";
            code += "        {\n";
            code += "            proverRequest.result = ZKR_SM_MAIN_HASHP_POSITION_PLUS_SIZE_OUT_OF_RANGE;\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"HashP 2: trying to insert data in a position:\" + to_string(pos+j) + \" higher than current data size:\" + to_string(ctx.hashP[addr].data.size()));\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            code += "        }\n";
            code += "        else\n";
            code += "        {\n";
            code += "            uint8_t bh;\n";
            code += "            bh = hashIterator->second.data[pos+j];\n";
            code += "            if (bm != bh)\n";
            code += "            {\n";
            code += "                proverRequest.result = ZKR_SM_MAIN_HASHP_VALUE_MISMATCH;\n";
            code += "                zkPC=" + to_string(zkPC) +";\n";
            code += "                mainExecutor.logError(ctx, \"HashP 2 bytes do not match: addr=\" + to_string(addr) + \" pos+j=\" + to_string(pos+j) + \" is bm=\" + to_string(bm) + \" and it should be bh=\" + to_string(bh));\n";
            code += "                mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "                return;\n";
            code += "            }\n";
            code += "        }\n";
            code += "    }\n";

            code += "    // Check that the remaining of a (op) is zero, i.e. no more data exists beyond size\n";
            code += "    paddingA = a >> (size*8);\n";
            code += "    if (paddingA != 0)\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_HASHP_PADDING_MISMATCH;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"HashP2 incoherent size=\" + to_string(size) + \" a=\" + a.get_str(16) + \" paddingA=\" + paddingA.get_str(16));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";

            code += "    // Record the read operation\n";
            code += "    readsIterator = hashIterator->second.reads.find(pos);\n";
            code += "    if ( readsIterator != hashIterator->second.reads.end() )\n";
            code += "    {\n";
            code += "        if ( readsIterator->second != size )\n";
            code += "        {\n";
            code += "            proverRequest.result = ZKR_SM_MAIN_HASHP_SIZE_MISMATCH;\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"HashP 2 diferent read sizes in the same position addr=\" + to_string(addr) + \" pos=\" + to_string(pos));\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            code += "        }\n";
            code += "    }\n";
            code += "    else\n";
            code += "    {\n";
            code += "        ctx.hashP[addr].reads[pos] = size;\n";
            code += "    }\n\n";

            code += "    // Store the size\n";
            code += "    incHashPos = size;\n";
            bIncHashPos = true;
        }

        // HashPLen instruction
        if (rom["program"][zkPC].contains("hashPLen") && (rom["program"][zkPC]["hashPLen"] == 1))
        {
            code += "    // HashPLen instruction\n";

            if (!bFastMode)
                code += "    pols.hashPLen[i] = fr.one();\n";

            code += "    // Get the length\n";
            code += "    lm = fr.toU64(op0);\n\n";

            code += "    // Find the entry in the hash database for this address\n";
            code += "    hashIterator = ctx.hashP.find(addr);\n\n";

            code += "    // If it's undefined, compute a hash of 0 bytes\n";
            code += "    if (hashIterator == ctx.hashP.end())\n";
            code += "    {\n";
            code += "        // Check that length = 0\n";
            code += "        if (lm != 0)\n";
            code += "        {\n";
            code += "            proverRequest.result = ZKR_SM_MAIN_HASHPLEN_LENGTH_MISMATCH;\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"HashPLen 2 hashP[addr] is empty but lm is not 0 addr=\" + to_string(addr) + \" lm=\" + to_string(lm));\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            code += "        }\n\n";

            code += "        // Create an empty entry in this address slot\n";
            code += "        ctx.hashP[addr] = emptyHashValue;\n";
            code += "        hashIterator = ctx.hashP.find(addr);\n";
            code += "        zkassert(hashIterator != ctx.hashP.end());\n";
            code += "    }\n";

            code += "    if (ctx.hashP[addr].lenCalled)\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_HASHPLEN_CALLED_TWICE;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"HashPLen 2 called more than once addr=\" + to_string(addr));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    ctx.hashP[addr].lenCalled = true;\n";

            code += "    lh = hashIterator->second.data.size();\n";
            code += "    if (lm != lh)\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_HASHPLEN_LENGTH_MISMATCH;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"HashPLen 2 does not match match addr=\" + to_string(addr) + \" is lm=\" + to_string(lm) + \" and it should be lh=\" + to_string(lh));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    if (!hashIterator->second.digestCalled)\n";
            code += "    {\n";
            
            code += "        // Get a local copy of the bytes vector\n";
            code += "        vector<uint8_t> data = hashIterator->second.data;\n";

            code += "        // Add padding = 0b1000...00001  up to a length of 56xN (7x8xN)\n";
            code += "        data.push_back(0x01);\n";
            code += "        while((data.size() % 56) != 0) data.push_back(0);\n";
            code += "        data[data.size()-1] |= 0x80;\n";

            code += "        // Create a FE buffer to store the transformed bytes into fe\n";
            code += "        uint64_t bufferSize = data.size()/7;\n";
            code += "        Goldilocks::Element * pBuffer = new Goldilocks::Element[bufferSize];\n";
            code += "        if (pBuffer == NULL)\n";
            code += "        {\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"hashPLen 2 failed allocating memory of \" + to_string(bufferSize) + \" field elements\");\n";
            code += "            exitProcess();\n";
            code += "        }\n";
            code += "        for (uint64_t j=0; j<bufferSize; j++) pBuffer[j] = fr.zero();\n";

            code += "        // Copy the bytes into the fe lower 7 sections\n";
            code += "        for (uint64_t j=0; j<data.size(); j++)\n";
            code += "        {\n";
            code += "            uint64_t fePos = j/7;\n";
            code += "            uint64_t shifted = uint64_t(data[j]) << ((j%7)*8);\n";
            code += "            pBuffer[fePos] = fr.add(pBuffer[fePos], fr.fromU64(shifted));\n";
            code += "        }\n";

            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "        gettimeofday(&t, NULL);\n";
            code += "#endif\n";
            code += "        Goldilocks::Element result[4];\n";
            code += "        mainExecutor.poseidon.linear_hash(result, pBuffer, bufferSize);\n";
            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "        mainMetrics.add(\"Poseidon\", TimeDiff(t));\n";
            code += "#endif\n";
            code += "        fea2scalar(fr, hashIterator->second.digest, result);\n";
            code += "        delete[] pBuffer;\n";

            code += "        // Collect the keys used to read or write store data\n";
            code += "        if (proverRequest.input.bGetKeys)\n";
            code += "        {\n";
            code += "            proverRequest.programKeys.insert(fea2string(fr, result));\n";
            code += "        }\n";

            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "        gettimeofday(&t, NULL);\n";
            code += "#endif\n";
            code += "        zkResult = mainExecutor.pHashDB->setProgram(result, hashIterator->second.data, proverRequest.input.bUpdateMerkleTree);\n";
            code += "        if (zkResult != ZKR_SUCCESS)\n";
            code += "        {\n";
            code += "            proverRequest.result = zkResult;\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, string(\"Failed calling mainExecutor.pHashDB->setProgram() result=\") + zkresult2string(zkResult));\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            code += "        }\n";
            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "        mainMetrics.add(\"Set program\", TimeDiff(t));\n";
            code += "#endif\n";
            code += "#ifdef LOG_HASH\n";
            code += "        {\n";
            code += "           string s = \"Hash calculate hashPLen 2: addr:\" + to_string(addr) + \" hash:\" + ctx.hashP[addr].digest.get_str(16) + \" size:\" + to_string(ctx.hashP[addr].data.size()) + \" data:\";\n";
            code += "           for (uint64_t k=0; k<ctx.hashP[addr].data.size(); k++) s += byte2string(ctx.hashP[addr].data[k]) + \":\";\n";
            code += "           zklog.info(s);\n";
            code += "        }\n";
            code += "#endif\n";
            code += "    }\n";
        }

        // HashPDigest instruction
        if (rom["program"][zkPC].contains("hashPDigest") && (rom["program"][zkPC]["hashPDigest"] == 1))
        {
            code += "    // HashPDigest instruction\n";

            if (!bFastMode)
                code += "    pols.hashPDigest[i] = fr.one();\n";

            code += "    // Get contents of op into dg\n";
            code += "    if (!fea2scalar(fr, dg, op0, op1, op2, op3, op4, op5, op6, op7))\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";

            code += "    hashIterator = ctx.hashP.find(addr);\n";
            code += "    if (hashIterator == ctx.hashP.end())\n";
            code += "    {\n";
            code += "        HashValue hashValue;\n";
            code += "        hashValue.digest = dg;\n";
            code += "        Goldilocks::Element aux[4];\n";
            code += "        scalar2fea(fr, dg, aux);\n";

            code += "        // Collect the keys used to read or write store data\n";
            code += "        if (proverRequest.input.bGetKeys)\n";
            code += "        {\n";
            code += "            proverRequest.programKeys.insert(fea2string(fr, aux));\n";
            code += "        }\n";

            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "        gettimeofday(&t, NULL);\n";
            code += "#endif\n";
            code += "        zkResult = mainExecutor.pHashDB->getProgram(aux, hashValue.data, proverRequest.dbReadLog);\n";
            code += "        if (zkResult != ZKR_SUCCESS)\n";
            code += "        {\n";
            code += "            proverRequest.result = zkResult;\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, string(\"Failed calling mainExecutor.pHashDB->getProgram() result=\") + zkresult2string(zkResult));\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            code += "        }\n";
            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "        mainMetrics.add(\"Get program\", TimeDiff(t));\n";
            code += "#endif\n";
            code += "        ctx.hashP[addr] = hashValue;\n";
            code += "        hashIterator = ctx.hashP.find(addr);\n";
            code += "        zkassert(hashIterator != ctx.hashP.end());\n";
            code += "    }\n";

            code += "    if (ctx.hashP[addr].digestCalled)\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_HASHPDIGEST_CALLED_TWICE;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"HashPDigest 2 called more than once addr=\" + to_string(addr));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    ctx.hashP[addr].digestCalled = true;\n";

            code += "    incCounter = ceil((double(hashIterator->second.data.size()) + double(1)) / double(56));\n";

            code += "    // Check that digest equals op\n";
            code += "    if (dg != hashIterator->second.digest)\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_HASHPDIGEST_DIGEST_MISMATCH;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"HashPDigest 2: ctx.hashP[addr].digest=\" + ctx.hashP[addr].digest.get_str(16) + \" does not match op=\" + dg.get_str(16));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
        }

        // HashP or Storage write instructions, required data
        if (!bFastMode && ( (rom["program"][zkPC].contains("hashPDigest") && (rom["program"][zkPC]["hashPDigest"]==1)) ||
                            (rom["program"][zkPC].contains("sWR") && (rom["program"][zkPC]["sWR"] ==1))) )
        {
            code += "    // HashP or Storage write instructions, required data\n";
            code += "    if (!fea2scalar(fr, op, op0, op1, op2, op3, op4, op5, op6, op7))\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    // Store the binary action to execute it later with the binary SM\n";
            code += "    binaryAction.a = op;\n";
            code += "    binaryAction.b = 0;\n";
            code += "    binaryAction.c = op;\n";
            code += "    binaryAction.opcode = 1;\n";
            code += "    binaryAction.type = 2;\n";
            code += "    required.Binary.push_back(binaryAction);\n";
        }


        if ( (rom["program"][zkPC].contains("arithEq0") && (rom["program"][zkPC]["arithEq0"]==1)) ||
             (rom["program"][zkPC].contains("arithEq1") && (rom["program"][zkPC]["arithEq1"]==1)) ||
             (rom["program"][zkPC].contains("arithEq2") && (rom["program"][zkPC]["arithEq2"]==1)) )
        {
            // Arith instruction: check that A*B + C = D<<256 + op, using scalars (result can be a big number)
            if ( (rom["program"][zkPC].contains("arithEq0") && rom["program"][zkPC]["arithEq0"]==1) &&
                 (!rom["program"][zkPC].contains("arithEq1") || rom["program"][zkPC]["arithEq1"]==0) &&
                 (!rom["program"][zkPC].contains("arithEq2") || rom["program"][zkPC]["arithEq2"]==0) )
            {
                code += "    // Arith instruction: check that A*B + C = D<<256 + op, using scalars (result can be a big number)\n";

                code += "    // Convert to scalar\n";
                code += "    if (!fea2scalar(fr, A, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, B, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, C, pols.C0[" + string(bFastMode?"0":"i") + "], pols.C1[" + string(bFastMode?"0":"i") + "], pols.C2[" + string(bFastMode?"0":"i") + "], pols.C3[" + string(bFastMode?"0":"i") + "], pols.C4[" + string(bFastMode?"0":"i") + "], pols.C5[" + string(bFastMode?"0":"i") + "], pols.C6[" + string(bFastMode?"0":"i") + "], pols.C7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.C)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, D, pols.D0[" + string(bFastMode?"0":"i") + "], pols.D1[" + string(bFastMode?"0":"i") + "], pols.D2[" + string(bFastMode?"0":"i") + "], pols.D3[" + string(bFastMode?"0":"i") + "], pols.D4[" + string(bFastMode?"0":"i") + "], pols.D5[" + string(bFastMode?"0":"i") + "], pols.D6[" + string(bFastMode?"0":"i") + "], pols.D7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.D)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, op, op0, op1, op2, op3, op4, op5, op6, op7))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    // Check the condition\n";
                code += "    if ( (A*B) + C != (D<<256) + op )\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_ARITH_MISMATCH;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        left = (A*B) + C;\n";
                code += "        right = (D<<256) + op;\n";
                code += "        mainExecutor.logError(ctx, \"Arithmetic does not match: (A*B) + C = \" + left.get_str(16) + \", (D<<256) + op = \" + right.get_str(16));;\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                if (!bFastMode)
                {
                    code += "    // Copy ROM flags into the polynomials\n";
                    code += "    pols.arithEq0[i] = fr.one();\n";
                    code += "    // Store the arith action to execute it later with the arith SM\n";
                    code += "    arithAction.x1 = A;\n";
                    code += "    arithAction.y1 = B;\n";
                    code += "    arithAction.x2 = C;\n";
                    code += "    arithAction.y2 = D;\n";
                    code += "    arithAction.x3 = 0;\n";
                    code += "    arithAction.y3 = op;\n";
                    code += "    arithAction.selEq0 = 1;\n";
                    code += "    arithAction.selEq1 = 0;\n";
                    code += "    arithAction.selEq2 = 0;\n";
                    code += "    arithAction.selEq3 = 0;\n";
                    code += "    required.Arith.push_back(arithAction);\n";
                }
            }
            // Arith instruction: check curve points
            else
            {
                code += "    // Arith instruction: check curve points\n";

                code += "    // Convert to scalar\n";
                code += "    if (!fea2scalar(fr, x1, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, y1, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, x2, pols.C0[" + string(bFastMode?"0":"i") + "], pols.C1[" + string(bFastMode?"0":"i") + "], pols.C2[" + string(bFastMode?"0":"i") + "], pols.C3[" + string(bFastMode?"0":"i") + "], pols.C4[" + string(bFastMode?"0":"i") + "], pols.C5[" + string(bFastMode?"0":"i") + "], pols.C6[" + string(bFastMode?"0":"i") + "], pols.C7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.C)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, y2, pols.D0[" + string(bFastMode?"0":"i") + "], pols.D1[" + string(bFastMode?"0":"i") + "], pols.D2[" + string(bFastMode?"0":"i") + "], pols.D3[" + string(bFastMode?"0":"i") + "], pols.D4[" + string(bFastMode?"0":"i") + "], pols.D5[" + string(bFastMode?"0":"i") + "], pols.D6[" + string(bFastMode?"0":"i") + "], pols.D7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.D)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, x3, pols.E0[" + string(bFastMode?"0":"i") + "], pols.E1[" + string(bFastMode?"0":"i") + "], pols.E2[" + string(bFastMode?"0":"i") + "], pols.E3[" + string(bFastMode?"0":"i") + "], pols.E4[" + string(bFastMode?"0":"i") + "], pols.E5[" + string(bFastMode?"0":"i") + "], pols.E6[" + string(bFastMode?"0":"i") + "], pols.E7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.E)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, y3, op0, op1, op2, op3, op4, op5, op6, op7))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    // Convert to RawFec::Element\n";
                code += "    mainExecutor.fec.fromMpz(fecX1, x1.get_mpz_t());\n";
                code += "    mainExecutor.fec.fromMpz(fecY1, y1.get_mpz_t());\n";
                code += "    mainExecutor.fec.fromMpz(fecX2, x2.get_mpz_t());\n";
                code += "    mainExecutor.fec.fromMpz(fecY2, y2.get_mpz_t());\n";

                bool dbl = false;
                if ( (!rom["program"][zkPC].contains("arithEq0") || (rom["program"][zkPC]["arithEq0"]==0)) &&
                     (rom["program"][zkPC].contains("arithEq1") && (rom["program"][zkPC]["arithEq1"]==1)) &&
                     (!rom["program"][zkPC].contains("arithEq2") || (rom["program"][zkPC]["arithEq2"]==0)) )
                {
                    dbl = false;
                }
                else if ( (!rom["program"][zkPC].contains("arithEq0") || (rom["program"][zkPC]["arithEq0"]==0)) &&
                          (!rom["program"][zkPC].contains("arithEq1") || (rom["program"][zkPC]["arithEq1"]==0)) &&
                          (rom["program"][zkPC].contains("arithEq2") && (rom["program"][zkPC]["arithEq2"]==1)) )
                {
                    dbl = true;
                }
                else
                {
                    cerr << "Error: Invalid arithmetic op zkPC=" << zkPC << endl;
                    exit(-1);
                }

                if (dbl)
                {
                    code += "    zkResult = AddPointEc(ctx, true, fecX1, fecY1, fecX1, fecY1, fecX3, fecY3);\n";
                }
                else
                {
                    code += "    zkResult = AddPointEc(ctx, false, fecX1, fecY1, fecX2, fecY2, fecX3, fecY3);\n";
                }
                
                code += "    if (zkResult != ZKR_SUCCESS)\n";
                code += "    {\n";
                code += "        proverRequest.result = zkResult;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling AddPointEc() in arith operation\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    mainExecutor.fec.toMpz(_x3.get_mpz_t(), fecX3);\n";
                code += "    mainExecutor.fec.toMpz(_y3.get_mpz_t(), fecY3);\n";

                code += "    // Compare\n";
                code += "    x3eq = (x3 == _x3);\n";
                code += "    y3eq = (y3 == _y3);\n";

                code += "    if (!x3eq || !y3eq)\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_ARITH_ECRECOVER_MISMATCH;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, string(\"Arithmetic curve " + string(dbl?"dbl":"add") + " point does not match x1=\") + x1.get_str() + \" y1=\" + y1.get_str() + \" x2=\" + x2.get_str() + \" y2=\" + y2.get_str() + \" x3=\" + x3.get_str() + \" y3=\" + y3.get_str() + \"_x3=\" + _x3.get_str() + \"_y3=\" + _y3.get_str());\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                if (!bFastMode)
                {
                    code += "    pols.arithEq0[i] = fr.fromU64(rom.line["+ to_string(zkPC) +"].arithEq0);\n";
                    code += "    pols.arithEq1[i] = fr.fromU64(rom.line["+ to_string(zkPC) +"].arithEq1);\n";
                    code += "    pols.arithEq2[i] = fr.fromU64(rom.line["+ to_string(zkPC) +"].arithEq2);\n";

                    code += "    // Store the arith action to execute it later with the arith SM\n";
                    code += "    arithAction.x1 = x1;\n";
                    code += "    arithAction.y1 = y1;\n";
                    code += "    arithAction.x2 = " + string(dbl?"x1":"x2") + ";\n";
                    code += "    arithAction.y2 = " + string(dbl?"y1":"y2") + ";\n";
                    code += "    arithAction.x3 = x3;\n";
                    code += "    arithAction.y3 = y3;\n";
                    code += "    arithAction.selEq0 = 0;\n";
                    code += "    arithAction.selEq1 = " + to_string(dbl?0:1) + ";\n";
                    code += "    arithAction.selEq2 = " + to_string(dbl?1:0) + ";\n";
                    code += "    arithAction.selEq3 = 1;\n";
                    code += "    required.Arith.push_back(arithAction);\n";
                }
            }
        }

        // Binary instruction
        if (rom["program"][zkPC].contains("bin") && (rom["program"][zkPC]["bin"] == 1))
        {
            if (rom["program"][zkPC]["binOpcode"] == 0) // ADD
            {
                code += "    // Binary instruction: ADD\n";

                code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    expectedC = (a + b) & ScalarMask256;\n";
                code += "    if (c != expectedC)\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_BINARY_ADD_MISMATCH;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Binary ADD operation does not match c=op=\" + c.get_str(16) + \" expectedC=(a + b) & ScalarMask256=\" + expectedC.get_str(16));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    pols.carry[" + string(bFastMode?"0":"i") + "] = fr.fromU64(((a + b) >> 256) > 0);\n";

                if (!bFastMode)
                {
                    code += "    pols.binOpcode[i] = fr.zero();\n";

                    code += "    // Store the binary action to execute it later with the binary SM\n";
                    code += "    binaryAction.a = a;\n";
                    code += "    binaryAction.b = b;\n";
                    code += "    binaryAction.c = c;\n";
                    code += "    binaryAction.opcode = 0;\n";
                    code += "    binaryAction.type = 1;\n";
                    code += "    required.Binary.push_back(binaryAction);\n";
                }
            }
            else if (rom["program"][zkPC]["binOpcode"] == 1) // SUB
            {
                code += "    // Binary instruction: SUB\n";

                code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    expectedC = (a - b + ScalarTwoTo256) & ScalarMask256;\n";
                code += "    if (c != expectedC)\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_BINARY_SUB_MISMATCH;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Binary SUB operation does not match c=op=\" + c.get_str(16) + \" expectedC=(a - b + ScalarTwoTo256) & ScalarMask256=\" + expectedC.get_str(16));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    pols.carry[" + string(bFastMode?"0":"i") + "] = fr.fromU64((a - b) < 0);\n";

                if (!bFastMode)
                {
                    code += "    pols.binOpcode[i] = fr.one();\n";

                    code += "    // Store the binary action to execute it later with the binary SM\n";
                    code += "    binaryAction.a = a;\n";
                    code += "    binaryAction.b = b;\n";
                    code += "    binaryAction.c = c;\n";
                    code += "    binaryAction.opcode = 1;\n";
                    code += "    binaryAction.type = 1;\n";
                    code += "    required.Binary.push_back(binaryAction);\n";
                }
            }
            else if (rom["program"][zkPC]["binOpcode"] == 2) // LT
            {
                code += "    // Binary instruction: LT\n";

                code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    expectedC = (a < b);\n";
                code += "    if (c != expectedC)\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_BINARY_LT_MISMATCH;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Binary LT operation does not match c=op=\" + c.get_str(16) + \" expectedC=(a < b)=\" + expectedC.get_str(16));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    pols.carry[" + string(bFastMode?"0":"i") + "] = fr.fromU64(a < b);\n";

                if (!bFastMode)
                {
                    code += "    pols.binOpcode[i] = fr.fromU64(2);\n";

                    code += "    // Store the binary action to execute it later with the binary SM\n";
                    code += "    binaryAction.a = a;\n";
                    code += "    binaryAction.b = b;\n";
                    code += "    binaryAction.c = c;\n";
                    code += "    binaryAction.opcode = 2;\n";
                    code += "    binaryAction.type = 1;\n";
                    code += "    required.Binary.push_back(binaryAction);\n";
                }
            }
            else if (rom["program"][zkPC]["binOpcode"] == 3) // SLT
            {
                code += "    // Binary instruction: SLT\n";

                code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    _a = a;\n";
                code += "    _b = b;\n";
                code += "    if (a >= ScalarTwoTo255) _a = a - ScalarTwoTo256;\n";
                code += "    if (b >= ScalarTwoTo255) _b = b - ScalarTwoTo256;\n";

                code += "    expectedC = (_a < _b);\n";
                code += "    if (c != expectedC)\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_BINARY_SLT_MISMATCH;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Binary SLT operation does not match a=\" + a.get_str(16) + \" b=\" + b.get_str(16) + \" c=\" + c.get_str(16) + \" _a=\" + _a.get_str(16) + \" _b=\" + _b.get_str(16) + \" expectedC=\" + expectedC.get_str(16));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    pols.carry[" + string(bFastMode?"0":"i") + "] = fr.fromU64(_a < _b);\n";

                if (!bFastMode)
                {
                    code += "    pols.binOpcode[i] = fr.fromU64(3);\n";

                    code += "    // Store the binary action to execute it later with the binary SM\n";
                    code += "    binaryAction.a = a;\n";
                    code += "    binaryAction.b = b;\n";
                    code += "    binaryAction.c = c;\n";
                    code += "    binaryAction.opcode = 3;\n";
                    code += "    binaryAction.type = 1;\n";
                    code += "    required.Binary.push_back(binaryAction);\n";
                }
            }
            else if (rom["program"][zkPC]["binOpcode"] == 4) // EQ
            {
                code += "    // Binary instruction: EQ\n";

                code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    expectedC = (a == b);\n";
                code += "    if (c != expectedC)\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_BINARY_EQ_MISMATCH;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError( ctx, \"Binary EQ operation does not match c=op=\" + c.get_str(16) + \" expectedC=(a==b)=\" + expectedC.get_str(16));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    pols.carry[" + string(bFastMode?"0":"i") + "] = fr.fromU64((a == b));\n";

                if (!bFastMode)
                {
                    code += "    pols.binOpcode[i] = fr.fromU64(4);\n";

                    code += "    // Store the binary action to execute it later with the binary SM\n";
                    code += "    binaryAction.a = a;\n";
                    code += "    binaryAction.b = b;\n";
                    code += "    binaryAction.c = c;\n";
                    code += "    binaryAction.opcode = 4;\n";
                    code += "    binaryAction.type = 1;\n";
                    code += "    required.Binary.push_back(binaryAction);\n";
                }
            }
            else if (rom["program"][zkPC]["binOpcode"] == 5) // AND
            {
                code += "    // Binary instruction: AND\n";

                code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    expectedC = (a & b);\n";
                code += "    if (c != expectedC)\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_BINARY_AND_MISMATCH;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Binary AND operation does not match c=op=\" + c.get_str(16) + \" expectedC=(a&b)=\" + expectedC.get_str(16));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    if (c != 0)\n";
                code += "        pols.carry[" + string(bFastMode?"0":"i") + "] = fr.one();\n";

                if (!bFastMode)
                {
                    code += "    pols.binOpcode[i] = fr.fromU64(5);\n";

                    code += "    // Store the binary action to execute it later with the binary SM\n";
                    code += "    binaryAction.a = a;\n";
                    code += "    binaryAction.b = b;\n";
                    code += "    binaryAction.c = c;\n";
                    code += "    binaryAction.opcode = 5;\n";
                    code += "    binaryAction.type = 1;\n";
                    code += "    required.Binary.push_back(binaryAction);\n";
                }
            }
            else if (rom["program"][zkPC]["binOpcode"] == 6) // OR
            {
                code += "    // Binary instruction: OR\n";

                code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    expectedC = (a | b);\n";
                code += "    if (c != expectedC)\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_BINARY_OR_MISMATCH;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Binary OR operation does not match c=op=\" + c.get_str(16) + \" expectedC=(a|b)=\" + expectedC.get_str(16));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                if (!bFastMode)
                {
                    code += "    pols.binOpcode[i] = fr.fromU64(6);\n";

                    code += "    // Store the binary action to execute it later with the binary SM\n";
                    code += "    binaryAction.a = a;\n";
                    code += "    binaryAction.b = b;\n";
                    code += "    binaryAction.c = c;\n";
                    code += "    binaryAction.opcode = 6;\n";
                    code += "    binaryAction.type = 1;\n";
                    code += "    required.Binary.push_back(binaryAction);\n";
                }
            }
            else if (rom["program"][zkPC]["binOpcode"] == 7) // XOR
            {
                code += "    // Binary instruction: XOR\n";

                code += "    if (!fea2scalar(fr, a, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, b, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                code += "    expectedC = (a ^ b);\n";
                code += "    if (c != expectedC)\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_BINARY_XOR_MISMATCH;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Binary XOR operation does not match c=op=\" + c.get_str(16) + \" expectedC=(a^b)=\" + expectedC.get_str(16));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                if (!bFastMode)
                {
                    code += "    pols.binOpcode[i] = fr.fromU64(7);\n";

                    code += "    // Store the binary action to execute it later with the binary SM\n";
                    code += "    binaryAction.a = a;\n";
                    code += "    binaryAction.b = b;\n";
                    code += "    binaryAction.c = c;\n";
                    code += "    binaryAction.opcode = 7;\n";
                    code += "    binaryAction.type = 1;\n";
                    code += "    required.Binary.push_back(binaryAction);\n";
                }
            }
            else
            {
                cerr << "Error: Invalid binary operation opcode=" << rom["program"][zkPC]["binOpcode"] << " zkPC=" << zkPC << endl;
                exit(-1);
            }

            if (!bFastMode)
                code += "    pols.bin[i] = fr.one();\n";

            code += "\n";
        }

        // MemAlign instruction
        if ( (rom["program"][zkPC].contains("memAlignRD") && (rom["program"][zkPC]["memAlignRD"]==1)) ||
             (rom["program"][zkPC].contains("memAlignWR") && (rom["program"][zkPC]["memAlignWR"]==1)) ||
             (rom["program"][zkPC].contains("memAlignWR8") && (rom["program"][zkPC]["memAlignWR8"]==1)) )
        {
            code += "    // MemAlign instruction\n";
            code += "    if (!fea2scalar(fr, m0, pols.A0[" + string(bFastMode?"0":"i") + "], pols.A1[" + string(bFastMode?"0":"i") + "], pols.A2[" + string(bFastMode?"0":"i") + "], pols.A3[" + string(bFastMode?"0":"i") + "], pols.A4[" + string(bFastMode?"0":"i") + "], pols.A5[" + string(bFastMode?"0":"i") + "], pols.A6[" + string(bFastMode?"0":"i") + "], pols.A7[" + string(bFastMode?"0":"i") + "]))\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.A)\");\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    if (!fea2scalar(fr, m1, pols.B0[" + string(bFastMode?"0":"i") + "], pols.B1[" + string(bFastMode?"0":"i") + "], pols.B2[" + string(bFastMode?"0":"i") + "], pols.B3[" + string(bFastMode?"0":"i") + "], pols.B4[" + string(bFastMode?"0":"i") + "], pols.B5[" + string(bFastMode?"0":"i") + "], pols.B6[" + string(bFastMode?"0":"i") + "], pols.B7[" + string(bFastMode?"0":"i") + "]))\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.B)\");\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    if (!fea2scalar(fr, v, op0, op1, op2, op3, op4, op5, op6, op7))\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    if (!fea2scalar(fr, offsetScalar, pols.C0[" + string(bFastMode?"0":"i") + "], pols.C1[" + string(bFastMode?"0":"i") + "], pols.C2[" + string(bFastMode?"0":"i") + "], pols.C3[" + string(bFastMode?"0":"i") + "], pols.C4[" + string(bFastMode?"0":"i") + "], pols.C5[" + string(bFastMode?"0":"i") + "], pols.C6[" + string(bFastMode?"0":"i") + "], pols.C7[" + string(bFastMode?"0":"i") + "]))\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.C)\");\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    if (offsetScalar<0 || offsetScalar>32)\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_MEMALIGN_OFFSET_OUT_OF_RANGE;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"MemAlign out of range offset=\" + offsetScalar.get_str());\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    offset = offsetScalar.get_ui();\n";

            if ( (!rom["program"][zkPC].contains("memAlignRD") || (rom["program"][zkPC]["memAlignRD"]==0)) &&
                 (rom["program"][zkPC].contains("memAlignWR") && (rom["program"][zkPC]["memAlignWR"]==1)) &&
                 (!rom["program"][zkPC].contains("memAlignWR8") || (rom["program"][zkPC]["memAlignWR8"]==0)) )
            {
                if (!bFastMode)
                    code += "    pols.memAlignWR[i] = fr.one();\n";

                code += "    if (!fea2scalar(fr, w0, pols.D0[" + string(bFastMode?"0":"i") + "], pols.D1[" + string(bFastMode?"0":"i") + "], pols.D2[" + string(bFastMode?"0":"i") + "], pols.D3[" + string(bFastMode?"0":"i") + "], pols.D4[" + string(bFastMode?"0":"i") + "], pols.D5[" + string(bFastMode?"0":"i") + "], pols.D6[" + string(bFastMode?"0":"i") + "], pols.D7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.D)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    if (!fea2scalar(fr, w1, pols.E0[" + string(bFastMode?"0":"i") + "], pols.E1[" + string(bFastMode?"0":"i") + "], pols.E2[" + string(bFastMode?"0":"i") + "], pols.E3[" + string(bFastMode?"0":"i") + "], pols.E4[" + string(bFastMode?"0":"i") + "], pols.E5[" + string(bFastMode?"0":"i") + "], pols.E6[" + string(bFastMode?"0":"i") + "], pols.E7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.E)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    _W0 = (m0 & (ScalarTwoTo256 - (ScalarOne << (256-offset*8)))) | (v >> offset*8);\n";
                code += "    _W1 = (m1 & (ScalarMask256 >> offset*8)) | ((v << (256 - offset*8)) & ScalarMask256);\n";
                code += "    if ( (w0 != _W0) || (w1 != _W1) )\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_MEMALIGN_WRITE_MISMATCH;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"MemAlign w0, w1 invalid: w0=\" + w0.get_str(16) + \" w1=\" + w1.get_str(16) + \" _W0=\" + _W0.get_str(16) + \" _W1=\" + _W1.get_str(16) + \" m0=\" + m0.get_str(16) + \" m1=\" + m1.get_str(16) + \" offset=\" + to_string(offset) + \" v=\" + v.get_str(16));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                if (!bFastMode)
                {
                    code += "    memAlignAction.m0 = m0;\n";
                    code += "    memAlignAction.m1 = m1;\n";
                    code += "    memAlignAction.w0 = w0;\n";
                    code += "    memAlignAction.w1 = w1;\n";
                    code += "    memAlignAction.v = v;\n";
                    code += "    memAlignAction.offset = offset;\n";
                    code += "    memAlignAction.wr256 = 1;\n";
                    code += "    memAlignAction.wr8 = 0;\n";
                    code += "    required.MemAlign.push_back(memAlignAction);\n";
                }
            }
            else if ( (!rom["program"][zkPC].contains("memAlignRD") || (rom["program"][zkPC]["memAlignRD"]==0)) &&
                      (!rom["program"][zkPC].contains("memAlignWR") || (rom["program"][zkPC]["memAlignWR"]==0)) &&
                      (rom["program"][zkPC].contains("memAlignWR8") && (rom["program"][zkPC]["memAlignWR8"]==1)) )
            {
                if (!bFastMode)
                    code += "    pols.memAlignWR8[i] = fr.one();\n";

                code += "    if (!fea2scalar(fr, w0, pols.D0[" + string(bFastMode?"0":"i") + "], pols.D1[" + string(bFastMode?"0":"i") + "], pols.D2[" + string(bFastMode?"0":"i") + "], pols.D3[" + string(bFastMode?"0":"i") + "], pols.D4[" + string(bFastMode?"0":"i") + "], pols.D5[" + string(bFastMode?"0":"i") + "], pols.D6[" + string(bFastMode?"0":"i") + "], pols.D7[" + string(bFastMode?"0":"i") + "]))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(pols.D)\");\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                code += "    _W0 = (m0 & (byteMaskOn256 >> (offset*8))) | ((v & 0xFF) << ((31-offset)*8));\n";
                code += "    if (w0 != _W0)\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_MEMALIGN_WRITE8_MISMATCH;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Error: MemAlign w0 invalid: w0=\" + w0.get_str(16) + \" _W0=\" + _W0.get_str(16) + \" m0=\" + m0.get_str(16) + \" offset=\" + to_string(offset) + \" v=\" + v.get_str(16));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                if (!bFastMode)
                {
                    code += "    memAlignAction.m0 = m0;\n";
                    code += "    memAlignAction.m1 = 0;\n";
                    code += "    memAlignAction.w0 = w0;\n";
                    code += "    memAlignAction.w1 = 0;\n";
                    code += "    memAlignAction.v = v;\n";
                    code += "    memAlignAction.offset = offset;\n";
                    code += "    memAlignAction.wr256 = 0;\n";
                    code += "    memAlignAction.wr8 = 1;\n";
                    code += "    required.MemAlign.push_back(memAlignAction);\n";
                }
            }
            else if ( (rom["program"][zkPC].contains("memAlignRD") && (rom["program"][zkPC]["memAlignRD"]==1)) &&
                      (!rom["program"][zkPC].contains("memAlignWR") || (rom["program"][zkPC]["memAlignWR"]==0)) &&
                      (!rom["program"][zkPC].contains("memAlignWR8") || (rom["program"][zkPC]["memAlignWR8"]==0)) )
            {
                if (!bFastMode)
                    code += "    pols.memAlignRD[i] = fr.one();\n";
                code += "    leftV = (m0 << offset*8) & ScalarMask256;\n";
                code += "    rightV = (m1 >> (256 - offset*8)) & (ScalarMask256 >> (256 - offset*8));\n";
                code += "    _V = leftV | rightV;\n";
                code += "    if (v != _V)\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_MEMALIGN_READ_MISMATCH;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"MemAlign v invalid: v=\" + v.get_str(16) + \" _V=\" + _V.get_str(16) + \" m0=\" + m0.get_str(16) + \" m1=\" + m1.get_str(16) + \" offset=\" + to_string(offset));\n";
                code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";

                if (!bFastMode)
                {
                    code += "    memAlignAction.m0 = m0;\n";
                    code += "    memAlignAction.m1 = m1;\n";
                    code += "    memAlignAction.w0 = 0;\n";
                    code += "    memAlignAction.w1 = 0;\n";
                    code += "    memAlignAction.v = v;\n";
                    code += "    memAlignAction.offset = offset;\n";
                    code += "    memAlignAction.wr256 = 0;\n";
                    code += "    memAlignAction.wr8 = 0;\n";
                    code += "    required.MemAlign.push_back(memAlignAction);\n";
                }
            }
            else
            {
                cerr << "Error: Invalid memAlign instruction zkPC=" << zkPC << endl;
                exit(-1);
            }

            code += "\n";
        }

        // Repeat instruction
        if ((rom["program"][zkPC].contains("repeat") && (rom["program"][zkPC]["repeat"] == 1)) && (!bFastMode))
        {
            code += "    pols.repeat[i] = fr.one();\n";
        }

        // Call instruction
        if ((rom["program"][zkPC].contains("call") && (rom["program"][zkPC]["call"] == 1)) && (!bFastMode))
        {
            code += "    pols.call[i] = fr.one();\n";
        }

        // Return instruction
        if ((rom["program"][zkPC].contains("return") && (rom["program"][zkPC]["return"] == 1)) && (!bFastMode))
        {
            code += "    pols.return_pol[i] = fr.one();\n";
        }

        /***********/
        /* SETTERS */
        /***********/

        code += setter8("A", rom["program"][zkPC].contains("setA") && (rom["program"][zkPC]["setA"]==1), bFastMode, zkPC, rom);
        code += setter8("B", rom["program"][zkPC].contains("setB") && (rom["program"][zkPC]["setB"]==1), bFastMode, zkPC, rom);
        code += setter8("C", rom["program"][zkPC].contains("setC") && (rom["program"][zkPC]["setC"]==1), bFastMode, zkPC, rom);
        code += setter8("D", rom["program"][zkPC].contains("setD") && (rom["program"][zkPC]["setD"]==1), bFastMode, zkPC, rom);
        code += setter8("E", rom["program"][zkPC].contains("setE") && (rom["program"][zkPC]["setE"]==1), bFastMode, zkPC, rom);
        code += setter8("SR", rom["program"][zkPC].contains("setSR") && (rom["program"][zkPC]["setSR"]==1), bFastMode, zkPC, rom);

        // If setCTX, CTX'=op
        if (rom["program"][zkPC].contains("setCTX") && (rom["program"][zkPC]["setCTX"]==1))
        {
            code += "    pols.CTX[" + string(bFastMode?"0":"nexti") + "] = op0; // If setCTX, CTX'=op\n";
            if (!bFastMode)
                code += "    pols.setCTX[i] = fr.one();\n";
        }
        else if (!bFastMode)
            code += "    pols.CTX[nexti] = pols.CTX[i];\n";

        // If setSP, SP'=op
        if (rom["program"][zkPC].contains("setSP") && (rom["program"][zkPC]["setSP"] == 1))
        {
            code += "    pols.SP[" + string(bFastMode?"0":"nexti") + "] = op0; // If setSP, SP'=op\n";
            if (!bFastMode)
                code += "    pols.setSP[i] = fr.one();\n";
        }
        else if (rom["program"][zkPC].contains("incStack") && (rom["program"][zkPC]["incStack"]!=0))
        {
            code += "    pols.SP[" + string(bFastMode?"0":"nexti") + "] = fr.add(pols.SP[" + string(bFastMode?"0":"i") + "], fr.fromS32(" + to_string(rom["program"][zkPC]["incStack"]) + ")); // SP' = SP + incStack\n";
        }
        else if (!bFastMode)
            code += "    pols.SP[nexti] = pols.SP[i];\n";

        // If setPC, PC'=op
        if ( rom["program"][zkPC].contains("setPC") && (rom["program"][zkPC]["setPC"] == 1) )
        {
            code += "    pols.PC[" + string(bFastMode?"0":"nexti") + "] = op0; // If setPC, PC'=op\n";
            if (!bFastMode)
                code += "    pols.setPC[i] = fr.one();\n";
        }
        else if (!bFastMode)
            code += "    pols.PC[nexti] = pols.PC[i];\n";

        // If setRR, RR'=op0
        if ( rom["program"][zkPC].contains("setRR") && (rom["program"][zkPC]["setRR"] == 1) )
        {
            code += "    pols.RR[" + string(bFastMode?"0":"nexti") + "] = op0; // If setRR, RR'=op0\n";
            if (!bFastMode)
                code += "    pols.setRR[i] = fr.one();\n";
        }
        else if ( rom["program"][zkPC].contains("call") && (rom["program"][zkPC]["call"] == 1) )
        {
            code += "    pols.RR[" + string(bFastMode?"0":"nexti") + "] = fr.fromU64(" + to_string(zkPC + 1) + ");\n";
        }
        else if (!bFastMode)
        {
            code += "    pols.RR[nexti] = pols.RR[i];\n";
        }

        // TODO: When regs are 0, do not copy to nexti.  Set bIsAZero to true at the beginning.

        // If arith, increment pols.cntArith
        if ( (rom["program"][zkPC].contains("arithEq0") && (rom["program"][zkPC]["arithEq0"]==1)) ||
             (rom["program"][zkPC].contains("arithEq1") && (rom["program"][zkPC]["arithEq1"]==1)) ||
             (rom["program"][zkPC].contains("arithEq2") && (rom["program"][zkPC]["arithEq2"]==1)) )
        {
            code += "    if (!proverRequest.input.bNoCounters)\n";
            code += "    {\n";
            code += "        pols.cntArith[" + string(bFastMode?"0":"nexti") + "] = fr.inc(pols.cntArith[" + string(bFastMode?"0":"i") + "]);\n";
            code += "#ifdef CHECK_MAX_CNT_ASAP\n";
            code += "        if (fr.toU64(pols.cntArith[" + string(bFastMode?"0":"nexti") + "]) > " + (string)rom["constants"]["MAX_CNT_ARITH_LIMIT"]["value"] + ")\n";
            code += "        {\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"Main Executor found pols.cntArith[nexti]=\" + fr.toString(pols.cntArith[" + string(bFastMode?"0":"nexti") + "], 10) + \" > MAX_CNT_ARITH_LIMIT=" + (string)rom["constants"]["MAX_CNT_ARITH_LIMIT"]["value"] + "\");\n";
            if (bFastMode)
            {
            code += "            proverRequest.result = ZKR_SM_MAIN_OOC_ARITH;\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            }
            else
            {
            code += "            exitProcess();\n";
            }
            code += "        }\n";
            code += "#endif\n";
            code += "    }\n\n";
        }
        else if (!bFastMode)
        {
            code += "    pols.cntArith[nexti] = pols.cntArith[i];\n";
        }

        // If bin, increment pols.cntBinary
        if ( (rom["program"][zkPC].contains("bin") && (rom["program"][zkPC]["bin"]==1)) ||
             (rom["program"][zkPC].contains("sWR") && (rom["program"][zkPC]["sWR"]==1)) ||
             (rom["program"][zkPC].contains("hashPDigest") && (rom["program"][zkPC]["hashPDigest"]==1)) )
        {
            code += "    if (!proverRequest.input.bNoCounters)\n";
            code += "    {\n";
            code += "        pols.cntBinary[" + string(bFastMode?"0":"nexti") + "] = fr.inc(pols.cntBinary[" + string(bFastMode?"0":"i") + "]);\n";
            code += "#ifdef CHECK_MAX_CNT_ASAP\n";
            code += "        if (fr.toU64(pols.cntBinary[" + string(bFastMode?"0":"nexti") + "]) > " + (string)rom["constants"]["MAX_CNT_BINARY_LIMIT"]["value"] + ")\n";
            code += "        {\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"Main Executor found pols.cntBinary[nexti]=\" + fr.toString(pols.cntBinary[" + string(bFastMode?"0":"nexti") + "], 10) + \" > MAX_CNT_BINARY_LIMIT=" + (string)rom["constants"]["MAX_CNT_BINARY_LIMIT"]["value"] + "\");\n";
            if (bFastMode)
            {
            code += "            proverRequest.result = ZKR_SM_MAIN_OOC_BINARY;\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            }
            else
            {
            code += "            exitProcess();\n";
            }
            code += "        }\n";
            code += "#endif\n";
            code += "    }\n\n";
        }
        else if (!bFastMode)
        {
            code += "    pols.cntBinary[nexti] = pols.cntBinary[i];\n";
        }

        // If memAlign, increment pols.cntMemAlign
        if ( (rom["program"][zkPC].contains("memAlignRD") && (rom["program"][zkPC]["memAlignRD"]==1)) ||
             (rom["program"][zkPC].contains("memAlignWR") && (rom["program"][zkPC]["memAlignWR"]==1)) ||
             (rom["program"][zkPC].contains("memAlignWR8") && (rom["program"][zkPC]["memAlignWR8"]==1)) )
        {
            code += "    if (!proverRequest.input.bNoCounters)\n";
            code += "    {\n";
            code += "        pols.cntMemAlign[" + string(bFastMode?"0":"nexti") + "] = fr.inc(pols.cntMemAlign[" + string(bFastMode?"0":"i") + "]);\n";
            code += "#ifdef CHECK_MAX_CNT_ASAP\n";
            code += "        if (fr.toU64(pols.cntMemAlign[" + string(bFastMode?"0":"nexti") + "]) > " + (string)rom["constants"]["MAX_CNT_MEM_ALIGN_LIMIT"]["value"] + ")\n";
            code += "        {\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"Main Executor found pols.cntMemAlign[nexti]=\" + fr.toString(pols.cntMemAlign[" + string(bFastMode?"0":"nexti") + "], 10) + \" > MAX_CNT_MEM_ALIGN_LIMIT=" + (string)rom["constants"]["MAX_CNT_MEM_ALIGN_LIMIT"]["value"] + "\");\n";
            if (bFastMode)
            {
            code += "            proverRequest.result = ZKR_SM_MAIN_OOC_MEM_ALIGN;\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            }
            else
            {
            code += "            exitProcess();\n";
            }
            code += "        }\n";
            code += "#endif\n";
            code += "    }\n\n";
        }
        else if (!bFastMode)
        {
            code += "    pols.cntMemAlign[nexti] = pols.cntMemAlign[i];\n";
        }

        // If setRCX, RCX=op, else if RCX>0, RCX--
        if (rom["program"][zkPC].contains("setRCX") && (rom["program"][zkPC]["setRCX"]==1))
        {
            code += "    pols.RCX[" + string(bFastMode?"0":"nexti") + "] = op0;\n";
            if (!bFastMode)
            code += "    pols.setRCX[i] = fr.one();\n";
        }
        else if (rom["program"][zkPC].contains("repeat") && (rom["program"][zkPC]["repeat"]==1))
        {
            code += "    currentRCX = pols.RCX[" + string(bFastMode?"0":"i") + "];\n";
            code += "    if (!fr.isZero(pols.RCX[" + string(bFastMode?"0":"i") + "]))\n";
            code += "    {\n";
            code += "        pols.RCX[" + string(bFastMode?"0":"nexti") + "] = fr.dec(pols.RCX[" + string(bFastMode?"0":"i") + "]);\n";
            code += "    }\n";
        }
        else
        {
            code += "    pols.RCX[" + string(bFastMode?"0":"nexti") + "] = pols.RCX[" + string(bFastMode?"0":"i") + "];\n";
        }

        // Calculate the inverse of RCX (if not zero)
        if (!bFastMode)
        {
        code += "    if (!fr.isZero(pols.RCX[nexti]))\n";
        code += "    {\n";
        code += "        pols.RCXInv[nexti] = glp.inv(pols.RCX[nexti]);\n";
        code += "    }\n";
        }

        // Record jump address data
        if ( rom["program"][zkPC].contains("jmpAddr") && rom["program"][zkPC]["jmpAddr"].is_number_unsigned() )
        {
            if (!bFastMode)
                code += "    pols.jmpAddr[i] = fr.fromU64(" + to_string(rom["program"][zkPC]["jmpAddr"]) + ");\n";
        }
        bool bUseJmpAddr = false;
        if ( rom["program"][zkPC].contains("useJmpAddr") && (rom["program"][zkPC]["useJmpAddr"] == 1) )
        {
            bUseJmpAddr = true;
            if (!bFastMode)
                code += "    pols.useJmpAddr[i] = fr.one();\n";
        }
        bool bUseElseAddr = false;
        if ( rom["program"][zkPC].contains("useElseAddr") && (rom["program"][zkPC]["useElseAddr"] == 1) )
        {
            bUseElseAddr = true;
            if (!bFastMode)
                code += "    pols.useElseAddr[i] = fr.one();\n";
        }

        if (!bFastMode)
        {
            if (bUseElseAddr)
            {
                if (!rom["program"][zkPC].contains("elseAddr"))
                {
                    cerr << "Error: useElseAddr=1 but elseAddr is not present" << endl;
                    exit(-1);
                }
                code += "    pols.elseAddr[i] = fr.fromU64(" + to_string(rom["program"][zkPC]["elseAddr"]) + ");\n";
            }
        }

        /*********/
        /* JUMPS */
        /*********/

        // If JMPN, jump conditionally if op0<0
        if (rom["program"][zkPC].contains("JMPN") && (rom["program"][zkPC]["JMPN"]==1))
        {
            if (!bFastMode)
                code += "    pols.JMPN[i] = fr.one();\n";

            code += "    jmpnCondValue = fr.toU64(op0);\n";
            // If op<0, jump to addr: zkPC'=addr
            code += "    if (jmpnCondValue >= FrFirst32Negative)\n";
            code += "    {\n";
            if (!bFastMode)
            {
                code += "        pols.isNeg[i] = fr.one();\n";
                if (bUseJmpAddr)
                    code += "        pols.zkPC[nexti] = fr.fromU64(" + to_string(rom["program"][zkPC]["jmpAddr"]) + "); // If op<0, jump to jmpAddr: zkPC'=jmpAddr\n";
                else
                    code += "        pols.zkPC[nexti] = fr.fromU64(addr); // If op<0, jump to addr: zkPC'=addr\n";
                code += "        jmpnCondValue = fr.toU64(fr.add(op0, fr.fromU64(0x100000000)));\n";
            }

            //code += "        goto *" + functionName + "_labels[addr]; // If op<0, jump to addr: zkPC'=addr\n";
            code += "        bJump = true;\n";
            bConditionalJump = true;
            code += "    }\n";
            // If op>=0, simply increase zkPC'=zkPC+1
            code += "    else if (jmpnCondValue <= FrLast32Positive)\n";
            code += "    {\n";
            if (!bFastMode)
            {
                if (bUseElseAddr)
                    code += "        pols.zkPC[nexti] = fr.fromU64(" + to_string(rom["program"][zkPC]["elseAddr"]) + "); // If op>=0, simply increase zkPC'=zkPC+1\n";
                else
                    code += "        pols.zkPC[nexti] = fr.inc(pols.zkPC[i]); // If op>=0, simply increase zkPC'=zkPC+1\n";
            }
            code += "    }\n";
            code += "    else\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_S33;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"JMPN invalid S33 value op0=\" + to_string(jmpnCondValue));\n";
            code += "        mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            if (!bFastMode)
            {
                code += "    pols.lJmpnCondValue[i] = fr.fromU64(jmpnCondValue & 0x7FFFFF);\n";
                code += "    jmpnCondValue = jmpnCondValue >> 23;\n";
                code += "    for (uint64_t index = 0; index < 9; ++index)\n";
                code += "    {\n";
                code += "        pols.hJmpnCondValueBit[index][i] = fr.fromU64(jmpnCondValue & 0x01);\n";
                code += "        jmpnCondValue = jmpnCondValue >> 1;\n";
                code += "    }\n";
            }
        }
        // If JMPC, jump conditionally if carry
        else if (rom["program"][zkPC].contains("JMPC") && (rom["program"][zkPC]["JMPC"] == 1))
        {
            if (!bFastMode)
                code += "    pols.JMPC[i] = fr.one();\n";
            code += "    if (!fr.isZero(pols.carry[" + string(bFastMode?"0":"i") + "]))\n";
            code += "    {\n";
            if (!bFastMode)
            {
                if (bUseJmpAddr)
                    code += "        pols.zkPC[nexti] = fr.fromU64(" + to_string(rom["program"][zkPC]["jmpAddr"]) + "); // If op<0, jump to jmpAddr: zkPC'=jmpAddr\n";
                else
                    code += "        pols.zkPC[nexti] = fr.fromU64(addr); // If carry, jump to addr: zkPC'=addr\n";
            }
            bConditionalJump = true;
            code += "        bJump = true;\n";
            if (bFastMode) // We reset the global variable to prevent jumping in next zkPC
                code += "        pols.carry[0] = fr.zero();\n";
            code += "    }\n";
            if (!bFastMode)
            {
                code += "    else\n";
                code += "{\n";
                if (bUseElseAddr)
                    code += "        pols.zkPC[nexti] = fr.fromU64(" + to_string(rom["program"][zkPC]["elseAddr"]) + "); // If op>=0, simply increase zkPC'=zkPC+1\n";
                else
                    code += "        pols.zkPC[nexti] = fr.inc(pols.zkPC[i]); // If not carry, simply increase zkPC'=zkPC+1\n";
                code += "}\n";
            }
        }
        // If JMPZ, jump
        else if (rom["program"][zkPC].contains("JMPZ") && (rom["program"][zkPC]["JMPZ"]==1))
        {
            code += "    if (fr.isZero(op0))\n";
            code += "    {\n";
            bConditionalJump = true;
            code += "        bJump = true;\n";
            if (!bFastMode)
            {
                if (bUseJmpAddr)
                    code += "        pols.zkPC[nexti] = fr.fromU64(" + to_string(rom["program"][zkPC]["jmpAddr"]) + "); // If op==0, jump to jmpAddr: zkPC'=jmpAddr\n";
                else
                    code += "        pols.zkPC[nexti] = fr.fromU64(addr);\n";
            }
            code += "    }\n";
            code += "    else\n";
            code += "    {\n";
            if (!bFastMode)
            {
                if (bUseElseAddr)
                    code += "        pols.zkPC[nexti] = fr.fromU64(" + to_string(rom["program"][zkPC]["elseAddr"]) + ");\n";
                else
                    code += "        pols.zkPC[nexti] = fr.inc(pols.zkPC[i]);\n";
            }
            code += "    }\n";
            if (!bFastMode)
            code += "    pols.JMPZ[i] = fr.one();\n";
        }
        // If JMP, directly jump zkPC'=addr
        else if (rom["program"][zkPC].contains("JMP") && (rom["program"][zkPC]["JMP"] == 1))
        {
            if (!bFastMode)
            {
                if (bUseJmpAddr)
                    code += "    pols.zkPC[nexti] = fr.fromU64(" + to_string(rom["program"][zkPC]["jmpAddr"]) + "); // If op==0, jump to jmpAddr: zkPC'=jmpAddr\n";
                else
                    code += "    pols.zkPC[nexti] = fr.fromU64(addr);\n";
                code += "    pols.JMP[i] = fr.one();\n";
            }
            //code += "    goto *" + functionName + "_labels[addr]; // If JMP, directly jump zkPC'=addr\n";
            bForcedJump = true;
            //code += "    bJump = true;\n";
        }
        // If call
        else if (rom["program"][zkPC].contains("call") && (rom["program"][zkPC]["call"] == 1))
        {
            if (!bFastMode)
            {
                if (bUseJmpAddr)
                {
                    code += "    pols.zkPC[nexti] = fr.fromU64(" + to_string(rom["program"][zkPC]["jmpAddr"]) + ");\n";
                }
                else if (bOnlyOffset)
                {
                    code += "    pols.zkPC[nexti] = fr.fromU64(" + to_string(rom["program"][zkPC]["offset"]) + ");\n";
                }
                else
                {
                    code += "    pols.zkPC[nexti] = fr.fromU64(addr);\n";
                }
            }
        }
        // If return
        else if (rom["program"][zkPC].contains("return") && (rom["program"][zkPC]["return"] == 1))
        {
            if (!bFastMode)
            {
                code += "    pols.zkPC[nexti] = pols.RR[i];\n";
            }
        }
        // If repeat
        else if (rom["program"][zkPC].contains("repeat") && (rom["program"][zkPC]["repeat"] == 1))
        {
            if (!bFastMode)
            {
                code += "    if (!fr.isZero(currentRCX))\n";
                code += "        pols.zkPC[nexti] = pols.zkPC[i];\n";
                code += "    else\n";
                code += "        pols.zkPC[nexti] = fr.inc(pols.zkPC[i]);\n";
            }
        }
        // Else, simply increase zkPC'=zkPC+1
        else if (!bFastMode)
        {
            code += "    pols.zkPC[nexti] = fr.inc(pols.zkPC[i]);\n";
        }

        // If setGAS, GAS'=op
        if ( rom["program"][zkPC].contains("setGAS") && (rom["program"][zkPC]["setGAS"] == 1) )
        {
            code += "    pols.GAS[" + string(bFastMode?"0":"nexti") + "] = op0; // If setGAS, GAS'=op\n";
            if (!bFastMode)
                code += "    pols.setGAS[i] = fr.one();\n";
        }
        else if (!bFastMode)
        {
            code += "    pols.GAS[nexti] = pols.GAS[i];\n";
        }

        // If setHASHPOS, HASHPOS' = op0 + incHashPos
        if ( rom["program"][zkPC].contains("setHASHPOS") && (rom["program"][zkPC]["setHASHPOS"] == 1) )
        {
            code += "    fr.toS64(i64Aux, op0);\n";
            if (bIncHashPos)
                code += "    pols.HASHPOS[" + string(bFastMode?"0":"nexti") + "] = fr.fromU64(i64Aux + incHashPos);\n";
            else
                code += "    pols.HASHPOS[" + string(bFastMode?"0":"nexti") + "] = fr.fromU64(i64Aux);\n";
            if (!bFastMode)
                code += "    pols.setHASHPOS[i] = fr.one();\n";
        }
        else //if (!bFastMode)
        {
            if (bIncHashPos)
                code += "    pols.HASHPOS[" + string(bFastMode?"0":"nexti") + "] = fr.add(pols.HASHPOS[" + string(bFastMode?"0":"i") + "], fr.fromU64(incHashPos));\n";
            else if (!bFastMode)
                code += "    pols.HASHPOS[nexti] = pols.HASHPOS[i];\n";
        }


        if (!bFastMode && ( (rom["program"][zkPC].contains("sRD") && (rom["program"][zkPC]["sRD"]==1)) ||
                            (rom["program"][zkPC].contains("sWR") && (rom["program"][zkPC]["sWR"]==1)) ||
                            (rom["program"][zkPC].contains("hashKDigest") && (rom["program"][zkPC]["hashKDigest"]==1)) ||
                            (rom["program"][zkPC].contains("hashPDigest") && (rom["program"][zkPC]["hashPDigest"]==1)) ))
        {
            code += "    pols.incCounter[i] = fr.fromU64(incCounter);\n";
        }

        if ( rom["program"][zkPC].contains("hashKDigest") && (rom["program"][zkPC]["hashKDigest"] == 1) )
        {
            code += "    if (!proverRequest.input.bNoCounters)\n";
            code += "    {\n";
            code += "        pols.cntKeccakF[" + string(bFastMode?"0":"nexti") + "] = fr.add(pols.cntKeccakF[" + string(bFastMode?"0":"i") + "], fr.fromU64(incCounter));\n";
            code += "#ifdef CHECK_MAX_CNT_ASAP\n";
            code += "        if (fr.toU64(pols.cntKeccakF[" + string(bFastMode?"0":"nexti") + "]) > " + (string)rom["constants"]["MAX_CNT_KECCAK_F_LIMIT"]["value"] + ")\n";
            code += "        {\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"Main Executor found pols.cntKeccakF[nexti]=\" + fr.toString(pols.cntKeccakF[" + string(bFastMode?"0":"nexti") + "], 10) + \" > MAX_CNT_KECCAK_F_LIMIT=" + (string)rom["constants"]["MAX_CNT_KECCAK_F_LIMIT"]["value"] + "\");\n";
            if (bFastMode)
            {
            code += "            proverRequest.result = ZKR_SM_MAIN_OOC_KECCAK_F;\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            }
            else
            {
            code += "            exitProcess();\n";
            }
            code += "        }\n";
            code += "#endif\n";
            code += "    }\n\n";
        }
        else if (!bFastMode)
        {
            code += "    pols.cntKeccakF[nexti] = pols.cntKeccakF[i];\n";
        }

        if ( rom["program"][zkPC].contains("hashPDigest") && (rom["program"][zkPC]["hashPDigest"] == 1) )
        {
            code += "    if (!proverRequest.input.bNoCounters)\n";
            code += "    {\n";
            code += "        pols.cntPaddingPG[" + string(bFastMode?"0":"nexti") + "] = fr.add(pols.cntPaddingPG[" + string(bFastMode?"0":"i") + "], fr.fromU64(incCounter));\n";
            code += "#ifdef CHECK_MAX_CNT_ASAP\n";
            code += "        if (fr.toU64(pols.cntPaddingPG[" + string(bFastMode?"0":"nexti") + "]) > " + (string)rom["constants"]["MAX_CNT_PADDING_PG_LIMIT"]["value"] + ")\n";
            code += "        {\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"Main Executor found pols.cntPaddingPG[nexti]=\" + fr.toString(pols.cntPaddingPG[" + string(bFastMode?"0":"nexti") + "], 10) + \" > MAX_CNT_PADDING_PG_LIMIT=" + (string)rom["constants"]["MAX_CNT_PADDING_PG_LIMIT"]["value"] + "\");\n";
            if (bFastMode)
            {
            code += "            proverRequest.result = ZKR_SM_MAIN_OOC_PADDING_PG;\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            }
            else
            {
            code += "            exitProcess();\n";
            }
            code += "        }\n";
            code += "#endif\n";
            code += "    }\n\n";
        }
        else if (!bFastMode)
        {
            code += "    pols.cntPaddingPG[nexti] = pols.cntPaddingPG[i];\n";
        }

        if ( (rom["program"][zkPC].contains("sRD") && (rom["program"][zkPC]["sRD"]==1)) ||
             (rom["program"][zkPC].contains("sWR") && (rom["program"][zkPC]["sWR"]==1)) ||
             (rom["program"][zkPC].contains("hashPDigest") && (rom["program"][zkPC]["hashPDigest"]==1) ) )
        {
            code += "    if (!proverRequest.input.bNoCounters)\n";
            code += "    {\n";
            code += "        pols.cntPoseidonG[" + string(bFastMode?"0":"nexti") + "] = fr.add(pols.cntPoseidonG[" + string(bFastMode?"0":"i") + "], fr.fromU64(incCounter));\n";
            code += "#ifdef CHECK_MAX_CNT_ASAP\n";
            code += "        if (fr.toU64(pols.cntPoseidonG[" + string(bFastMode?"0":"nexti") + "]) > " + (string)rom["constants"]["MAX_CNT_POSEIDON_G_LIMIT"]["value"] + ")\n";
            code += "        {\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"Main Executor found pols.cntPoseidonG[nexti]=\" + fr.toString(pols.cntPoseidonG[" + string(bFastMode?"0":"nexti") + "], 10) + \" > MAX_CNT_POSEIDON_G_LIMIT=" + (string)rom["constants"]["MAX_CNT_POSEIDON_G_LIMIT"]["value"] + "\");\n";
            if (bFastMode)
            {
            code += "            proverRequest.result = ZKR_SM_MAIN_OOC_POSEIDON_G;\n";
            code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            }
            else
            {
            code += "            exitProcess();\n";
            }
            code += "        }\n";
            code += "#endif\n";
            code += "    }\n\n";
        }
        else if (!bFastMode)
        {
            code += "    pols.cntPoseidonG[nexti] = pols.cntPoseidonG[i];\n";
        }

        // COMAND AFTER (of previous instruction)
        if ( rom["program"][zkPC].contains("cmdAfter") && (rom["program"][zkPC]["cmdAfter"].size()>0) )
        {
            code += "    // Evaluate the list cmdAfter commands of the previous ROM line,\n";
            code += "    // and any children command, recursively\n";
            code += "    if (i < N_Max_minus_one)\n";
            code += "    {\n";
            if (!bFastMode)
            code += "        i++;\n";
            code += "        for (uint64_t j=0; j<rom.line[" + to_string(zkPC) + "].cmdAfter.size(); j++)\n";
            code += "        {\n";
            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "            gettimeofday(&t, NULL);\n";
            code += "#endif\n";
            code += "            cr.reset();\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            evalCommand(ctx, *rom.line[" + to_string(zkPC) + "].cmdAfter[j], cr);\n";
            code += "    \n";
            code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
            code += "            mainMetrics.add(\"Eval command\", TimeDiff(t));\n";
            code += "            evalCommandMetrics.add(rom.line[" + to_string(zkPC) + "].cmdAfter[j]->opAndFunction, TimeDiff(t));\n";
            code += "#endif\n";
            code += "            // In case of an external error, return it\n";
            code += "            if (cr.zkResult != ZKR_SUCCESS)\n";
            code += "            {\n";
            code += "                proverRequest.result = cr.zkResult;\n";
            code += "                zkPC=" + to_string(zkPC) +";\n";
            code += "                mainExecutor.logError(ctx, string(\"Failed calling evalCommand() after result=\") + zkresult2string(proverRequest.result));\n";
            code += "                mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "                return;\n";
            code += "            }\n";
            code += "        }\n";
            if (!bFastMode)
            code += "        i--;\n";
            code += "    }\n\n";
        }

        code += "#ifdef LOG_COMPLETED_STEPS\n";
        code += "    zklog.info( \"<-- Completed step=\" + to_string(i) + \" zkPC=" + to_string(zkPC) + " op=\" + fr.toString(op7,16) + \":\" + fr.toString(op6,16) + \":\" + fr.toString(op5,16) + \":\" + fr.toString(op4,16) + \":\" + fr.toString(op3,16) + \":\" + fr.toString(op2,16) + \":\" + fr.toString(op1,16) + \":\" + fr.toString(op0,16) + \" ABCDE0=\" + fr.toString(pols.A0[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.B0[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.C0[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.D0[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.E0[" + string(bFastMode?"0":"i") + "],16) + \" FREE0:7=\" + fr.toString(fi0,16) + \":\" + fr.toString(fi7,16) + \" addr=\" + to_string(addr));\n";
        /*code += "    zklog.info(\"<-- Completed step=\" + to_string(i) + \" zkPC=" + to_string(zkPC) +
                " op=\" + fr.toString(op7,16) + \":\" + fr.toString(op6,16) + \":\" + fr.toString(op5,16) + \":\" + fr.toString(op4,16) + \":\" + fr.toString(op3,16) + \":\" + fr.toString(op2,16) + \":\" + fr.toString(op1,16) + \":\" + fr.toString(op0,16) + \"" +
                " A=\" + fr.toString(pols.A7[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.A6[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.A5[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.A4[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.A3[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.A2[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.A1[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.A0[" + string(bFastMode?"0":"i") + "],16) + \"" +
                " B=\" + fr.toString(pols.B7[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.B6[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.B5[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.B4[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.B3[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.B2[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.B1[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.B0[" + string(bFastMode?"0":"i") + "],16) + \"" +
                " C=\" + fr.toString(pols.C7[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.C6[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.C5[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.C4[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.C3[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.C2[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.C1[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.C0[" + string(bFastMode?"0":"i") + "],16) + \"" +
                " D=\" + fr.toString(pols.D7[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.D6[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.D5[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.D4[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.D3[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.D2[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.D1[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.D0[" + string(bFastMode?"0":"i") + "],16) + \"" +
                " E=\" + fr.toString(pols.E7[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.E6[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.E5[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.E4[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.E3[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.E2[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.E1[" + string(bFastMode?"0":"i") + "],16) + \":\" + fr.toString(pols.E0[" + string(bFastMode?"0":"i") + "],16) + \"" +
                " FREE0:7=\" + fr.toString(fi0,16) + \":\" + fr.toString(fi7],16) + \" addr=\" + to_string(addr));\n";*/
        code += "#endif\n";
        code += "#ifdef LOG_COMPLETED_STEPS_TO_FILE\n";
        code += "    outfile.open(\"c.txt\", std::ios_base::app); // append instead of overwrite\n";
        code += "    outfile << \"<-- Completed step=\" << i << \" zkPC=" + to_string(zkPC) + " op=\" << fr.toString(op7,16) << \":\" << fr.toString(op6,16) << \":\" << fr.toString(op5,16) << \":\" << fr.toString(op4,16) << \":\" << fr.toString(op3,16) << \":\" << fr.toString(op2,16) << \":\" << fr.toString(op1,16) << \":\" << fr.toString(op0,16) << \" ABCDE0=\" << fr.toString(pols.A0[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.B0[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.C0[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.D0[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.E0[" + string(bFastMode?"0":"i") + "],16) << \" FREE0:7=\" << fr.toString(fi0,16) << \":\" << fr.toString(fi7,16) << \" addr=\" << addr << endl;\n";
        /*code += "    outfile << \"<-- Completed step=\" << i << \" zkPC=" + to_string(zkPC) +
                " op=\" << fr.toString(op7,16) << \":\" << fr.toString(op6,16) << \":\" << fr.toString(op5,16) << \":\" << fr.toString(op4,16) << \":\" << fr.toString(op3,16) << \":\" << fr.toString(op2,16) << \":\" << fr.toString(op1,16) << \":\" << fr.toString(op0,16) << \"" +
                " A=\" << fr.toString(pols.A7[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.A6[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.A5[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.A4[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.A3[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.A2[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.A1[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.A0[" + string(bFastMode?"0":"i") + "],16) << \"" +
                " B=\" << fr.toString(pols.B7[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.B6[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.B5[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.B4[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.B3[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.B2[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.B1[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.B0[" + string(bFastMode?"0":"i") + "],16) << \"" +
                " C=\" << fr.toString(pols.C7[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.C6[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.C5[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.C4[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.C3[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.C2[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.C1[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.C0[" + string(bFastMode?"0":"i") + "],16) << \"" +
                " D=\" << fr.toString(pols.D7[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.D6[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.D5[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.D4[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.D3[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.D2[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.D1[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.D0[" + string(bFastMode?"0":"i") + "],16) << \"" +
                " E=\" << fr.toString(pols.E7[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.E6[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.E5[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.E4[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.E3[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.E2[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.E1[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.E0[" + string(bFastMode?"0":"i") + "],16) << \"" +
                " FREE0:7=\" << fr.toString(fi0,16) << \":\" << fr.toString(fi7,16) << \" addr=\" << addr << endl;\n";*/
        code += "    outfile.close();\n";
        code += "#endif\n\n";

        // Jump to the end label if we are done and we are in fast mode
        if (zkPC == rom["labels"]["finalizeExecution"])
        {
            code += "    if (ctx.lastStep != 0)\n";
            code += "    {\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Called finalizeExecutionLabel with a non-zero ctx.lastStep=\" + to_string(ctx.lastStep));\n";
            code += "        exitProcess();\n";
            code += "    }\n";
            code += "    ctx.lastStep = i;\n";
            if (bFastMode)
            code += "    goto " + functionName + "_end;\n\n";
        }

        // INCREASE EVALUATION INDEX

        code += "    if (i==N_Max_minus_one) goto " + functionName + "_end;\n";
        code += "    i++;\n";
        if (!bFastMode)
            code += "    nexti=(i+1)%N_Max;\n";
        code += "\n";

        // In case we had a pending jump, do it now, after the work has been done
        if (bForcedJump)
        {
            if (bUseJmpAddr)
                code += "    goto " + functionName + "_rom_line_" + to_string(rom["program"][zkPC]["jmpAddr"]) + ";\n";
            else if (bOnlyOffset)
                code += "    goto " + functionName + "_rom_line_" + to_string(rom["program"][zkPC]["offset"]) + ";\n";
            else
                code += "    goto *" + functionName + "_labels[addr];\n\n";
        }
        if (bConditionalJump)
        {
            code += "    if (bJump)\n";
            code += "    {\n";
            code += "        bJump = false;\n";

            if (bUseJmpAddr)
                code += "    goto " + functionName + "_rom_line_" + to_string(rom["program"][zkPC]["jmpAddr"]) + ";\n";
            else if (bOnlyOffset)
                code += "        goto " + functionName + "_rom_line_" + to_string(rom["program"][zkPC]["offset"]) + ";\n";
            else
                code += "        goto *" + functionName + "_labels[addr];\n";
            code += "    }\n";
            if (bUseElseAddr)
            {
                code += "    else\n";
                if (rom["program"][zkPC]["elseAddrLabel"] == "invalidIntrinsicTxSenderCode")
                {
                    code += "        if (bUnsignedTransaction)\n";
                    if (bUseJmpAddr)
                        code += "            goto " + functionName + "_rom_line_" + to_string(rom["program"][zkPC]["jmpAddr"]) + ";\n";
                    else
                        code += "            goto *" + functionName + "_labels[addr];\n";
                    code += "        else\n";
                }
                else
                {
                    code += "        goto " + functionName + "_rom_line_" + to_string(rom["program"][zkPC]["elseAddr"]) + ";\n";
                }
            }
        }
        if (rom["program"][zkPC].contains("repeat") && (rom["program"][zkPC]["repeat"]==1))
        {
            code += "    if (!fr.isZero(currentRCX))\n";
            code += "        goto " + functionName + "_rom_line_" + to_string(zkPC) + ";\n";
        }
        if (rom["program"][zkPC].contains("call") && (rom["program"][zkPC]["call"]==1))
        {
            if (bUseJmpAddr)
            {
                code += "    goto " + functionName + "_rom_line_" + to_string(rom["program"][zkPC]["jmpAddr"]) + ";\n";
            }
            else if (bOnlyOffset)
            {
                code += "    goto " + functionName + "_rom_line_" + to_string(rom["program"][zkPC]["offset"]) + ";\n";
            }
            else
            {
                code += "    goto *" + functionName + "_labels[addr];\n";
            }
        }
        if (rom["program"][zkPC].contains("return") && (rom["program"][zkPC]["return"]==1))
        {
            code += "    goto *" + functionName + "_labels[fr.toU64(pols.RR[" + string(bFastMode?"0":"i") + "])];\n";
        }
    }

    code += functionName + "_end:\n\n";


    code += "    // Copy the counters\n";
    code += "    proverRequest.counters.arith = fr.toU64(pols.cntArith[0]);\n";
    code += "    proverRequest.counters.binary = fr.toU64(pols.cntBinary[0]);\n";
    code += "    proverRequest.counters.keccakF = fr.toU64(pols.cntKeccakF[0]);\n";
    code += "    proverRequest.counters.memAlign = fr.toU64(pols.cntMemAlign[0]);\n";
    code += "    proverRequest.counters.paddingPG = fr.toU64(pols.cntPaddingPG[0]);\n";
    code += "    proverRequest.counters.poseidonG = fr.toU64(pols.cntPoseidonG[0]);\n";
    code += "    proverRequest.counters.steps = ctx.lastStep;\n\n";

    code += "    // Set the error (all previous errors generated a return)\n";
    code += "    proverRequest.result = ZKR_SUCCESS;\n";

    code += "    // Check that we did not run out of steps during the execution\n";
    code += "    if (ctx.lastStep == 0)\n";
    code += "    {\n";
    code += "        proverRequest.result = ZKR_SM_MAIN_OUT_OF_STEPS;\n";
    code += "        mainExecutor.logError(ctx, \"Found ctx.lastStep=0, so execution was not complete\");\n";
    if (!bFastMode)
    code += "        exitProcess();\n";
    code += "    }\n";
    code += "    if (!proverRequest.input.bNoCounters && (ctx.lastStep > " + (string)rom["constants"]["MAX_CNT_STEPS_LIMIT"]["value"] + "))\n";
    code += "    {\n";
    code += "        proverRequest.result = ZKR_SM_MAIN_OUT_OF_STEPS;\n";
    code += "        mainExecutor.logError(ctx, \"Found ctx.lastStep=\" + to_string(ctx.lastStep) + \" > MAX_CNT_STEPS_LIMIT=" + (string)rom["constants"]["MAX_CNT_STEPS_LIMIT"]["value"] + "\");\n";
    if (!bFastMode)
    {
    code += "        exitProcess();\n";
    }
    code += "    }\n\n";

    code += "#ifdef CHECK_MAX_CNT_AT_THE_END\n";
    code += "    if (!proverRequest.input.bNoCounters && (fr.toU64(pols.cntArith[0]) > " + (string)rom["constants"]["MAX_CNT_ARITH_LIMIT"]["value"] + "))\n";
    code += "    {\n";
    code += "        proverRequest.result = ZKR_SM_MAIN_OOC_ARITH;\n";
    code += "        mainExecutor.logError(ctx, \"Found pols.cntArith[0]=\" + to_string(fr.toU64(pols.cntArith[0])) + \" > MAX_CNT_ARITH_LIMIT=" + (string)rom["constants"]["MAX_CNT_ARITH_LIMIT"]["value"] + "\");\n";
    if (!bFastMode)
        code += "        exitProcess();\n";
    code += "    }\n";
    code += "    if (!proverRequest.input.bNoCounters && (fr.toU64(pols.cntBinary[0]) > " + (string)rom["constants"]["MAX_CNT_BINARY_LIMIT"]["value"] + "))\n";
    code += "    {\n";
    code += "        proverRequest.result = ZKR_SM_MAIN_OOC_BINARY;\n";
    code += "        mainExecutor.logError(ctx, \"Found pols.cntBinary[0]=\" + to_string(fr.toU64(pols.cntBinary[0])) + \" > MAX_CNT_BINARY_LIMIT=" + (string)rom["constants"]["MAX_CNT_BINARY_LIMIT"]["value"] + "\");\n";
    if (!bFastMode)
        code += "        exitProcess();\n";
    code += "    }\n";
    code += "    if (!proverRequest.input.bNoCounters && (fr.toU64(pols.cntMemAlign[0]) > " + (string)rom["constants"]["MAX_CNT_MEM_ALIGN_LIMIT"]["value"] + "))\n";
    code += "    {\n";
    code += "        proverRequest.result = ZKR_SM_MAIN_OOC_MEM_ALIGN;\n";
    code += "        mainExecutor.logError(ctx, \"Found pols.cntMemAlign[0]=\" + to_string(fr.toU64(pols.cntMemAlign[0])) + \" > MAX_CNT_MEM_ALIGN_LIMIT=" + (string)rom["constants"]["MAX_CNT_MEM_ALIGN_LIMIT"]["value"] + "\");\n";
    if (!bFastMode)
        code += "        exitProcess();\n";
    code += "    }\n";
    code += "    if (!proverRequest.input.bNoCounters && (fr.toU64(pols.cntKeccakF[0]) > " + (string)rom["constants"]["MAX_CNT_KECCAK_F_LIMIT"]["value"] + "))\n";
    code += "    {\n";
    code += "        proverRequest.result = ZKR_SM_MAIN_OOC_KECCAK_F;\n";
    code += "        mainExecutor.logError(ctx, \"Found pols.cntKeccakF[0]=\" + to_string(fr.toU64(pols.cntKeccakF[0])) + \" > MAX_CNT_KECCAK_F_LIMIT=" + (string)rom["constants"]["MAX_CNT_KECCAK_F_LIMIT"]["value"] + "\");\n";
    if (!bFastMode)
        code += "        exitProcess();\n";
    code += "    }\n";
    code += "    if (!proverRequest.input.bNoCounters && (fr.toU64(pols.cntPaddingPG[0]) > " + (string)rom["constants"]["MAX_CNT_PADDING_PG_LIMIT"]["value"] + "))\n";
    code += "    {\n";
    code += "        proverRequest.result = ZKR_SM_MAIN_OOC_PADDING_PG;\n";
    code += "        mainExecutor.logError(ctx, \"Found pols.cntPaddingPG[0]=\" + to_string(fr.toU64(pols.cntPaddingPG[0])) + \" > MAX_CNT_PADDING_PG_LIMIT=" + (string)rom["constants"]["MAX_CNT_PADDING_PG_LIMIT"]["value"] + "\");\n";
    if (!bFastMode)
        code += "        exitProcess();\n";
    code += "    }\n";
    code += "    if (!proverRequest.input.bNoCounters && (fr.toU64(pols.cntPoseidonG[0]) > " + (string)rom["constants"]["MAX_CNT_POSEIDON_G_LIMIT"]["value"] + "))\n";
    code += "    {\n";
    code += "        proverRequest.result = ZKR_SM_MAIN_OOC_POSEIDON_G;\n";
    code += "        mainExecutor.logError(ctx, \"Found pols.cntPoseidonG[0]=\" + to_string(fr.toU64(pols.cntPoseidonG[0])) + \" > MAX_CNT_POSEIDON_G_LIMIT=" + (string)rom["constants"]["MAX_CNT_POSEIDON_G_LIMIT"]["value"] + "\");\n";
    if (!bFastMode)
        code += "        exitProcess();\n";
    code += "    }\n";
    code += "#endif\n\n";

    if (!bFastMode) // In fast mode, last nexti was not 0 but 1, and pols have only 2 evaluations
    {
        code += "    // Check that all registers are set to 0\n";
        code += "    mainExecutor.checkFinalState(ctx);\n";
        code += "    mainExecutor.assertOutputs(ctx);\n\n";

        code += "    // Generate Padding KK required data\n";
        code += "    for (uint64_t i=0; i<ctx.hashK.size(); i++)\n";
        code += "    {\n";
        code += "        PaddingKKExecutorInput h;\n";
        code += "        h.dataBytes = ctx.hashK[i].data;\n";
        code += "        uint64_t p = 0;\n";
        code += "        while (p<ctx.hashK[i].data.size())\n";
        code += "        {\n";
        code += "            if (ctx.hashK[i].reads[p] != 0)\n";
        code += "            {\n";
        code += "                h.reads.push_back(ctx.hashK[i].reads[p]);\n";
        code += "                p += ctx.hashK[i].reads[p];\n";
        code += "            }\n";
        code += "            else\n";
        code += "            {\n";
        code += "                h.reads.push_back(1);\n";
        code += "                p++;\n";
        code += "            }\n";
        code += "        }\n";
        code += "        if (p != ctx.hashK[i].data.size())\n";
        code += "        {\n";
        code += "            proverRequest.result = ZKR_SM_MAIN_HASHK_READ_OUT_OF_RANGE;\n";
        code += "            mainExecutor.logError(ctx, \"Reading hashK out of limits: i=\" + to_string(i) + \" p=\" + to_string(p) + \" ctx.hashK[i].data.size()=\" + to_string(ctx.hashK[i].data.size()));\n";
        code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
        code += "            return;\n";
        code += "        }\n";
        code += "        h.digestCalled = ctx.hashK[i].digestCalled;\n";
        code += "        h.lenCalled = ctx.hashK[i].lenCalled;\n";
        code += "        required.PaddingKK.push_back(h);\n";
        code += "    }\n";

        code += "    // Generate Padding PG required data\n";
        code += "    for (uint64_t i=0; i<ctx.hashP.size(); i++)\n";
        code += "    {\n";
        code += "        PaddingPGExecutorInput h;\n";
        code += "        h.dataBytes = ctx.hashP[i].data;\n";
        code += "        uint64_t p = 0;\n";
        code += "        while (p<ctx.hashP[i].data.size())\n";
        code += "        {\n";
        code += "            if (ctx.hashP[i].reads[p] != 0)\n";
        code += "            {\n";
        code += "                h.reads.push_back(ctx.hashP[i].reads[p]);\n";
        code += "                p += ctx.hashP[i].reads[p];\n";
        code += "            }\n";
        code += "            else\n";
        code += "            {\n";
        code += "                h.reads.push_back(1);\n";
        code += "                p++;\n";
        code += "            }\n";
        code += "        }\n";
        code += "        if (p != ctx.hashP[i].data.size())\n";
        code += "        {\n";
        code += "            proverRequest.result = ZKR_SM_MAIN_HASHP_READ_OUT_OF_RANGE;\n";
        code += "            mainExecutor.logError(ctx, \"Reading hashP out of limits: i=\" + to_string(i) + \" p=\" + to_string(p) + \" ctx.hashP[i].data.size()=\" + to_string(ctx.hashP[i].data.size()));\n";
        code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
        code += "            return;\n";
        code += "        }\n";
        code += "        h.digestCalled = ctx.hashP[i].digestCalled;\n";
        code += "        h.lenCalled = ctx.hashP[i].lenCalled;\n";
        code += "        required.PaddingPG.push_back(h);\n";
        code += "    }\n";
    }

    code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
    code += "    gettimeofday(&t, NULL);\n";
    code += "#endif\n";

    code += "    if (ctx.config.hashDB64)\n";
    code += "    {\n";
    code += "        Goldilocks::Element newStateRoot[4];\n";
    code += "        string2fea(fr, proverRequest.pFullTracer->get_new_state_root(), newStateRoot);\n";
    code += "        zkresult zkr = mainExecutor.pHashDB->purge(proverRequest.uuid, newStateRoot, proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE);\n";
    code += "        if (zkr != ZKR_SUCCESS)\n";
    code += "        {\n";
    code += "            proverRequest.result = zkr;\n";
    code += "            mainExecutor.logError(ctx, string(\"Failed calling mainExecutor.pHashDB->purge() result=\") + zkresult2string(zkr));\n";
    code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
    code += "            return;\n";
    code += "        }\n";
    code += "        proverRequest.flushId = 0;\n";
    code += "        proverRequest.lastSentFlushId = 0;\n";
    code += "    }\n";
    code += "    else\n";
    code += "    {\n";
    code += "        zkresult zkr = mainExecutor.pHashDB->flush(proverRequest.uuid, proverRequest.pFullTracer->get_new_state_root(), proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, proverRequest.flushId, proverRequest.lastSentFlushId);\n";
    code += "        if (zkr != ZKR_SUCCESS)\n";
    code += "        {\n";
    code += "            proverRequest.result = zkr;\n";
    code += "            mainExecutor.logError(ctx, string(\"Failed calling mainExecutor.pHashDB->flush() result=\") + zkresult2string(zkr));\n";
    code += "            mainExecutor.pHashDB->cancelBatch(proverRequest.uuid);\n";
    code += "            return;\n";
    code += "        }\n";
    code += "    }\n\n";

    code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
    code += "    mainMetrics.add(\"Flush\", TimeDiff(t));\n";
    code += "#endif\n";

    code += "#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR\n";
    code += "    if (mainExecutor.config.executorTimeStatistics)\n";
    code += "    {\n";
    code += "        mainMetrics.print(\"Main Executor calls\");\n";
    code += "        evalCommandMetrics.print(\"Main Executor eval command calls\");\n";
    code += "    }\n";
    code += "#endif\n\n";
    
    code += "    if (mainExecutor.config.dbMetrics) proverRequest.dbReadLog->print();\n\n";

    code += "    zklog.info(\"" + functionName + "() done lastStep=\" + to_string(ctx.lastStep) + \" (\" + to_string((double(ctx.lastStep)*100)/mainExecutor.N) + \"%)\", &proverRequest.tags);\n\n";

    code += "    return;\n\n";

    code += "}\n\n";

    code += "#pragma GCC pop_options\n\n";

    code += "} // namespace\n\n";

    return code;
}

/*************/
/* SELECTORS */
/*************/

string selector8 (const string &regName, const string &regValue, bool opInitialized, bool bFastMode)
{
    int64_t inRegValue = atoi(regValue.c_str());
    string inRegName = "in" + string2upper(regName.substr(0, 1)) + regName.substr(1);
    string code = "";
    code += "    // op = op + " + inRegName + "*" + regName + ", where " + inRegName + "=" + to_string(inRegValue) + "\n";
    for (uint64_t j=0; j<8; j++)
    {
        string value = "";
        if (inRegValue == 1)
            value = "pols." + regName + to_string(j) + "[" + string(bFastMode?"0":"i") + "]";
        else if (inRegValue == -1)
            value = "fr.neg(pols." + regName + to_string(j) + "[" + string(bFastMode?"0":"i") + "])";
        else
            value = "fr.mul(fr.fromS32(" + to_string(inRegValue) + "), pols." + regName + to_string(j) + "[" + string(bFastMode?"0":"i") + "])";
        if (opInitialized)
            value = "fr.add(op" + to_string(j) + ", " + value + ")";
        code += "    op" + to_string(j) + " = " + value + ";\n";
    }
    if (!bFastMode)
        code += "    pols." + inRegName + "[i] = fr.fromS32(" + to_string(inRegValue) + ");\n";
    code += "\n";
    return code;
}

string selector1 (const string &regName, const string &regValue, bool opInitialized, bool bFastMode)
{
    int64_t inRegValue = atoi(regValue.c_str());
    string inRegName = "in" + string2upper(regName.substr(0, 1)) + regName.substr(1);
    string code = "";
    code += "    // op0 = op0 + " + inRegName + "*" + regName + ", where " + inRegName + "=" + to_string(inRegValue) + "\n";

    // Calculate value
    string value = "";
    if (inRegValue == 1)
        value = "pols." + regName + "[" + string(bFastMode?"0":"i") + "]";
    else if (inRegValue == -1)
        value = "fr.neg(pols." + regName + "[" + string(bFastMode?"0":"i") + "])";
    else
        value = "fr.mul(fr.fromS32(" + to_string(inRegValue) + "), pols." + regName + "[" + string(bFastMode?"0":"i") + "])";

    // Add to op0
    if (opInitialized)
        code += "    op0 = fr.add(op0, " + value + ");\n";
    else
    {
        code += "    op0 = " + value + ";\n";
        for (uint64_t j=1; j<8; j++)
        {
            code += "    op" + to_string(j) + " = fr.zero();\n";
        }
    }

    // Set selector
    if (!bFastMode)
        code += "    pols." + inRegName + "[i] = fr.fromS32(" + to_string(inRegValue) + ");\n";

    code += "\n";
    return code;
}

string selectorConst (int64_t CONST, bool opInitialized, bool bFastMode)
{
    string code = "";
    code += "    // op0 = op0 + CONST\n";

    string value = "";
    if (CONST > 0)
        value += "fr.fromU64(" + to_string(CONST) + ")";
    else
        value += "fr.neg(fr.fromU64(" + to_string(-CONST) + "))";
    if (opInitialized)
        value = "fr.add(op0, " + value + ")";
    code += "    op0 = " + value + ";\n";
    if (!opInitialized)
        for (uint64_t j=1; j<8; j++)
        {
            code += "    op" + to_string(j) + " = fr.zero();\n";
        }
    if (!bFastMode)
        code += "    pols.CONST0[i] = fr.fromS32(" + to_string(CONST) + ");\n\n";
    code += "\n";
    return code;
}

string selectorConstL (const string &CONSTL, bool opInitialized, bool bFastMode)
{
    string code = "";
    code += "    // op = op + CONSTL\n";
    uint64_t op[8];
    scalar2fea(CONSTL, op);

    for (uint64_t j=0; j<8; j++) // TODO: Should we ADD it, not just copy it?
    {
        code += "    op" + to_string(j) + " = fr.fromU64(" + to_string(op[j]) + ");\n";
    }

    if (!bFastMode)
        for (uint64_t j=0; j<8; j++)
        {
            code += "    pols.CONST" + to_string(j) + "[i] = fr.fromU64(" + to_string(op[j]) + ");\n";
        }

    code += "\n";
    return code;
}

/***********/
/* SETTERS */
/***********/

string setter8 (const string &reg, bool setReg, bool bFastMode, uint64_t zkPC, const json &rom)
{
    string code = "";

    if (setReg)
    {
        code += "    // " + reg + "' = op\n";
        for (uint64_t j=0; j<8; j++)
            code += "    pols." + reg + to_string(j) + "[" + (bFastMode?"0":"nexti") + "] = op" + to_string(j) + ";\n";
        if (!bFastMode)
            code += "    pols.set" + reg + "[i] = fr.one();\n";
        code += "\n";
    }
    else if ((rom["labels"].contains("checkAndSaveFrom") && (zkPC == rom["labels"]["checkAndSaveFrom"])) && (reg=="A"))
    {
        code += "    if (bUnsignedTransaction)\n";
        code += "    {\n";
        code += "        mpz_class from(proverRequest.input.from);\n";
        code += "        scalar2fea(fr, from, pols.A0[" + string(bFastMode?"0":"nexti") + "], pols.A1[" + (bFastMode?"0":"nexti") + "], pols.A2[" + (bFastMode?"0":"nexti") + "], pols.A3[" + (bFastMode?"0":"nexti") + "], pols.A4[" + (bFastMode?"0":"nexti") + "], pols.A5[" + (bFastMode?"0":"nexti") + "], pols.A6[" + (bFastMode?"0":"nexti") + "], pols.A7[" + (bFastMode?"0":"nexti") + "] );\n";
        code += "    }\n";
        if (!bFastMode)
        {
            code += "    else\n";
            code += "    {\n";
            code += "        // " + reg + "' = " + reg + "\n";
            for (uint64_t j=0; j<8; j++)
                code += "        pols." + reg + to_string(j) + "[nexti] = pols." + reg + to_string(j) + "[i];\n";
            code += "\n";
            code += "    }\n";
        }
        code += "\n";
    }
    else if (!bFastMode)
    {
        code += "    // " + reg + "' = " + reg + "\n";
        for (uint64_t j=0; j<8; j++)
            code += "    pols." + reg + to_string(j) + "[nexti] = pols." + reg + to_string(j) + "[i];\n";
        code += "\n";
    }

    return code;
}

string string2lower (const string &s)
{
    string result = s;
    transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

string string2upper (const string &s)
{
    string result = s;
    transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}