#include <iostream>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <gmpxx.h>
#include <sys/stat.h>

// The following are the only 2 project files allowed to be included, since they do not have internal dependencies
#include "../config/definitions.hpp"
#include "../config/fork_info.hpp"

/* This code generates forks 10 onwards */

//#define LOG_PRINT_ROM_LINES
//#define LOG_START_STEPS
//#define LOG_START_STEPS_TO_FILE
//#define LOG_COMPLETED_STEPS
//#define LOG_COMPLETED_STEPS_TO_FILE

//#define LOG_TIME_STATISTICS_MAIN_EXECUTOR

//#define LOG_HASHK
//#define LOG_HASHP
//#define LOG_HASHS
//#define LOG_SMT_KEY_DETAILS
//#define LOG_STORAGE
//#define LOG_SAVE_RESTORE

#ifdef DEBUG
//#define CHECK_MAX_CNT_ASAP
#endif

#define CHECK_MAX_CNT_AT_THE_END

using namespace std;
using json = nlohmann::json;

#define zkmin(a,b) (a < b ? a : b)
#define zkmax(a,b) (a > b ? a : b)

// Forward declaration
void file2json (json &rom, string &romFileName);
void string2file (const string & s, const string & fileName);
string generate(const json &rom, uint64_t forkID, string forkNamespace, const string &functionName, const string &fileName, bool bFastMode, bool bHeader, ForkInfo &forkInfo);
string selector8 (const string &regName, const string &regValue, bool opInitialized, bool bFastMode);
string selector1 (const string &regName, const string &regValue, bool opInitialized, bool bFastMode);
string selectorConst (int64_t CONST, bool opInitialized, bool bFastMode);
string selectorConstL (const string &CONSTL, bool opInitialized, bool bFastMode);
string setter8 (const string &reg, bool setReg, bool bFastMode, uint64_t zkPC, const json &rom, uint64_t forkID);
string string2lower (const string &s);
string string2upper (const string &s);
bool stringIsDec (const string &s);
void ensureDirectoryExists (const string &fileName);

int main (int argc, char **argv)
{
    cout << "Main generator 10" << endl;

    uint64_t firstForkID = PROVER_FORK_ID;
    uint64_t lastForkID = firstForkID;

    // Overwrite fork ID it one has been provided as a parameter
    if (argc == 2)
    {
        string argString = argv[1];
        if (argString == "all")
        {
            firstForkID = 10;
            lastForkID = PROVER_FORK_ID;
        }
        else if (!stringIsDec(argString))
        {
            cout << "Main generator 10 got invalid parameter=" + argString << endl;
            return -1;
        }
        else
        {
            firstForkID = atoi(argString.c_str());
            lastForkID = firstForkID;
        }
    }
    for (uint64_t forkID = firstForkID; forkID <= lastForkID; forkID ++)
    {
        cout << "Main generator 10 starting for fork ID=" << forkID << endl;

        // Get fork info for this fork ID
        ForkInfo forkInfo;
        if (!getForkInfo(forkID, forkInfo))
        {
            cerr << "Failed calling getForkInfo() with forkID=" << forkID << endl;
            exit(-1);
        }

        // Set fork namespace based on fork ID
        string forkNamespace = "fork_" + to_string(forkInfo.parentId);

        string codeGenerationName = "main_exec_generated";

        string functionName = codeGenerationName + "_" + to_string(forkID);
        string fileName = functionName;
        string directoryName = "src/main_sm/" + forkNamespace + "/" + codeGenerationName;

        // Load rom.json
        string romFileName;
        romFileName = "src/main_sm/" + forkNamespace + "/scripts/rom_" + to_string(forkID) + ".json";
        cout << "ROM file name=" << romFileName << endl;
        json rom;
        file2json(rom, romFileName);
        cout << "ROM file loaded" << endl;

        // Create directory
        ensureDirectoryExists(directoryName.c_str());

#ifdef MAIN_SM_PROVER_GENERATED_CODE
        if (forkID == PROVER_FORK_ID)
        {
            cout << "Generating code for prover with fork ID=" << forkID << endl;
            string code = generate(rom, forkID, forkNamespace, functionName, fileName, false, false, forkInfo);
            string2file(code, directoryName + "/" + fileName + ".cpp");
            string header = generate(rom, forkID, forkNamespace, functionName, fileName, false,  true, forkInfo);
            string2file(header, directoryName + "/" + fileName + ".hpp");
        }
#endif
        cout << "Generating code for executor with fork ID=" << forkID << endl;
        functionName += "_fast";
        fileName += "_fast";
        string codeFast = generate(rom, forkID, forkNamespace, functionName, fileName, true, false, forkInfo);
        string2file(codeFast, directoryName + "/" + fileName + ".cpp");
        string headerFast = generate(rom, forkID, forkNamespace, functionName, fileName, true,  true, forkInfo);
        string2file(headerFast, directoryName + "/" + fileName + ".hpp");
    }

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
    cout << "Wrote file " << fileName << endl;
}

void scalar2fea (const string &s, uint64_t (&fea)[8])
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

string generate(const json &rom, uint64_t forkID, string forkNamespace, const string &functionName, const string &fileName, bool bFastMode, bool bHeader, ForkInfo &forkInfo)
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
            code += "#ifndef MAIN_EXEC_GENERATED_FAST_HPP_" + forkNamespace + "_" + to_string(forkInfo.id) + "_" + to_string(forkInfo.parentId) + "\n";
            code += "#define MAIN_EXEC_GENERATED_FAST_HPP_" + forkNamespace + "_" + to_string(forkInfo.id) + "_" + to_string(forkInfo.parentId) + "\n";
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
        code += "#include \"main_sm/"+ forkNamespace + "/helpers/arith_helper.hpp\"\n";
        code += "#include \"main_sm/"+ forkNamespace + "/helpers/binary_helper.hpp\"\n";
        code += "#include \"main_sm/"+ forkNamespace + "/helpers/memory_helper.hpp\"\n";
        code += "#include \"main_sm/"+ forkNamespace + "/helpers/memalign_helper.hpp\"\n";
        code += "#include \"main_sm/"+ forkNamespace + "/helpers/storage_helper.hpp\"\n";
        code += "#include \"main_sm/"+ forkNamespace + "/helpers/keccak_helper.hpp\"\n";
        code += "#include \"main_sm/"+ forkNamespace + "/helpers/poseidon_helper.hpp\"\n";
        code += "#include \"main_sm/"+ forkNamespace + "/helpers/sha_helper.hpp\"\n";
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

    code += "#ifdef MAIN_SM_EXECUTOR_GENERATED_CODE\n";
    code += "\n";

    code += "namespace " + forkNamespace + "\n";
    code += "{\n";

    if (!bHeader)
    {
        code += "#define STACK_OFFSET 0x10000\n";
        code += "#define MEM_OFFSET   0x20000\n";
        code += "#define CTX_OFFSET   0x40000\n";
        code += "#define MAX_HASH_ADDRESS 0x100000000\n";
        code += "#define ZK_INT32_MAX 0x80000000\n";
        code += "#define CTX_MAX      ((ZK_INT32_MAX / CTX_OFFSET) - 1) // 8192 - 1\n\n";

        //code += "#define N_NO_COUNTERS_MULTIPLICATION_FACTOR 8\n\n";

        code += "#define FrFirst32Negative ( 0xFFFFFFFF00000001 - 0xFFFFFFFF )\n";
        code += "#define FrLast32Positive 0xFFFFFFFF\n\n";

        //code += "#ifdef DEBUG\n";
        //code += "#define CHECK_MAX_CNT_ASAP\n";
        //code += "#endif\n";
        //code += "#define CHECK_MAX_CNT_AT_THE_END\n\n";

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
        code += "#endif // MAIN_SM_EXECUTOR_GENERATED_CODE\n";
        code += "\n";
        code += "#endif // Header protection\n";
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
    code += "    uint64_t context;\n";
    code += "    int32_t sp;\n\n";
    
    //code += "    Rom &rom = config.loadDiagnosticRom ? mainExecutor.romDiagnostic : mainExecutor.romBatch;\n";

    if ((forkID == 10) || (forkID == 11))
    {
        code += "    if ( (proverRequest.input.publicInputsExtended.publicInputs.forkID != 10) && (proverRequest.input.publicInputsExtended.publicInputs.forkID != 11) )\n";
    }
    else
    {
        code += "    if ( proverRequest.input.publicInputsExtended.publicInputs.forkID != " + to_string(forkID) + ")\n";
    }
    code += "    {\n";
    code += "        proverRequest.result = ZKR_SM_MAIN_INVALID_FORK_ID;\n";
    code += "        zklog.error(\"MainExecutor::execute() called with invalid fork ID=\" + to_string(proverRequest.input.publicInputsExtended.publicInputs.forkID));\n";
    code += "        return;\n";
    code += "    }\n";

    if ((forkID == 10) || (forkID == 11))
    {
        code += "    Rom &rom = proverRequest.input.publicInputsExtended.publicInputs.forkID == 10 ? mainExecutor.romBatch_10 : mainExecutor.romBatch_11;\n";
    }
    else
    {
        code += "    Rom &rom = config.loadDiagnosticRom ? mainExecutor.romDiagnostic : mainExecutor.romBatch_" + to_string(forkID) + ";\n";
    }

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

    code += "    HashDBInterface *pHashDB;\n";
    code += "    if (config.hashDBSingleton)\n";
    code += "    {\n";
    code += "        pHashDB = mainExecutor.pHashDBSingleton;\n";
    code += "    }\n";
    code += "    else\n";
    code += "    {\n";
    code += "        pHashDB = HashDBClientFactory::createHashDBClient(fr, config);\n";
    code += "        if (pHashDB == NULL)\n";
    code += "        {\n";
    code += "            zklog.error(\"MainExecutor::execute() failed calling HashDBClientFactory::createHashDBClient()\");\n";
    code += "            exitProcess();\n";
    code += "        }\n";
    code += "    }\n\n";
    
    code += "    Context ctx(mainExecutor.fr, mainExecutor.config, mainExecutor.fec, mainExecutor.fnec, pols, rom, proverRequest, pHashDB);\n\n";

    code += "    mainExecutor.initState(ctx);\n\n";

#ifdef LOG_COMPLETED_STEPS_TO_FILE
    code += "    remove(\"c.txt\");\n";
#endif

    code += "   // Clear cache if configured and we are using a local database\n";
    code += "   if (mainExecutor.config.dbClearCache && (mainExecutor.config.databaseURL == \"local\"))\n";
    code += "   {\n";
    code += "       pHashDB->clearCache();\n";
    code += "   }\n";

    code += "    // Copy input database content into context database\n";
    code += "    if (proverRequest.input.db.size() > 0)\n";
    code += "    {\n";
    code += "        Goldilocks::Element stateRoot[4];\n";
    code += "        scalar2fea(fr, proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot, stateRoot);\n";
    code += "        pHashDB->loadDB(proverRequest.input.db, true, stateRoot);\n";
    code += "        pHashDB->flush(emptyString, emptyString, proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, flushId, lastSentFlushId);\n";
    code += "        if (mainExecutor.config.dbClearCache && (mainExecutor.config.databaseURL != \"local\"))\n";
    code += "        {\n";
    code += "            pHashDB->clearCache();\n";
    code += "        }\n";
    code += "    }\n\n";

    code += "    // Copy input contracts database content into context database (dbProgram)\n";
    code += "    if (proverRequest.input.contractsBytecode.size() > 0)\n";
    code += "    {\n";
    code += "        pHashDB->loadProgramDB(proverRequest.input.contractsBytecode, true);\n";
    code += "        pHashDB->flush(emptyString, emptyString, proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, flushId, lastSentFlushId);\n";
    code += "        if (mainExecutor.config.dbClearCache && (mainExecutor.config.databaseURL != \"local\"))\n";
    code += "        {\n";
    code += "            pHashDB->clearCache();\n";
    code += "        }\n";
    code += "    }\n\n";

    //code += "    // opN are local, uncommitted polynomials\n";
    code += "    Goldilocks::Element op0, op1, op2, op3, op4, op5, op6, op7;\n";

    // Free in
    code += "    Goldilocks::Element fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7;\n";
    code += "    CommandResult cr;\n";

    // Storage free in
    code += "    zkresult zkResult;\n";
    code += "    mpz_class opScalar;\n";
    code += "    mpz_class value;\n";

    // Hash free in
    code += "    mpz_class s;\n";
    code += "    mpz_class result;\n";
    code += "    mpz_class dg;\n";

    // Mem allign free in
    code += "    mpz_class m;\n";
    code += "    mpz_class v;\n";

    // Binary free in
    code += "    mpz_class a, b, c, _a, _b;\n";
    code += "    mpz_class expectedC;\n";
    code += "    BinaryAction binaryAction;\n";

    // Arith
    code += "    mpz_class op;\n";
    code += "    int64_t reserve;\n";

    code += "    std::ofstream outfile;\n";
    code += "    std::unordered_map<uint64_t, Fea>::iterator memIterator;\n";
    code += "\n";

    code += "    uint64_t zkPC = 0; // Zero-knowledge program counter\n";
    code += "    uint64_t i=0; // Step, as it is used internally, set to 0 in fast mode to reuse the same evaluation all the time\n";
    if (!bFastMode)
        code += "    uint64_t nexti=1; // Next step, as it is used internally, set to 0 in fast mode to reuse the same evaluation all the time\n";
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

    code += "    bool bJump = false;\n";
    code += "    uint64_t jmpnCondValue;\n";
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
    code += "        N_Max = " + to_string(forkInfo.N_NoCounters) + ";\n";
    code += "    }\n";
    code += "    else\n";
    code += "    {\n";
    code += "        N_Max = " + to_string(forkInfo.N) + ";\n";
    code += "    }\n\n";
    code += "    N_Max_minus_one = N_Max - 1;\n";
    code += "    ctx.N = N_Max; // Numer of evaluations\n";

    // This code is only used when 'skipFirstChangeL2Block = true'
    // This only is triggered when executong transaction by transaction across batches
    // This cannot be executed in prover mode
    // This code aims to set the timestamp of the batch to the one read from the state
    // Issue fixed: timestamp is set when processed a 'changeL2Block', stored on state and hold on memory.
    // Later on, 'opTIMESTAMP' loads the value hold on memory.
    // Hence, execution transaction by transaction lost track of the timestamp
    // This function aims to solve the abive issue by loading the timestamp from the state
    code += "    if (bProcessBatch && proverRequest.input.bSkipFirstChangeL2Block)\n";
    code += "    {\n";
        // this smt key is built with the following registers:
        // A: `0x000000000000000000000000000000005ca1ab1e` (%ADDRESS_SYSTEM)
        // B: `3` (%SMT_KEY_SC_STORAGE)
        // C: `2` (%TIMESTAMP_STORAGE_POS)
    code += "        Goldilocks::Element keyToRead[4];\n";
    code += "        keyToRead[0] = fr.fromU64(13748230500842749409ULL);\n";
    code += "        keyToRead[1] = fr.fromU64(4428676446262882967ULL);\n";
    code += "        keyToRead[2] = fr.fromU64(12167292013585018040ULL);\n";
    code += "        keyToRead[3] = fr.fromU64(12161933621946006603ULL);\n";

        // Get old state root (current state root)
    code += "        Goldilocks::Element oldStateRoot[4];\n";
    code += "        scalar2fea(fr, proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot, oldStateRoot);\n";

        // Get timestamp from storage
    code += "        mpz_class timestampFromSR;\n";
    code += "        zkResult = pHashDB->get(proverRequest.uuid, oldStateRoot, keyToRead, timestampFromSR, NULL, proverRequest.dbReadLog);\n";
    code += "        if (zkResult != ZKR_SUCCESS)\n";
    code += "        {\n";
    code += "            proverRequest.result = zkResult;\n";
    code += "            mainExecutor.logError(ctx, string(\"Copying timestamp from state to memory, failed calling pHashDB->get() result=\") + zkresult2string(zkResult));\n";
    code += "            pHashDB->cancelBatch(proverRequest.uuid);\n";
    code += "            return;\n";
    code += "        }\n";

        // Pre-load memory with this timestamp value
    code += "        Fea fea;\n";
    code += "        scalar2fea(fr, timestampFromSR, fea.fe0, fea.fe1, fea.fe2, fea.fe3, fea.fe4, fea.fe5, fea.fe6, fea.fe7);\n";
    code += "        ctx.mem[rom.timestampOffset] = fea;\n";
    code += "    }\n\n";

    ///////////////
    // MAIN LOOP //
    ///////////////

    for (uint64_t zkPC=0; zkPC < rom["program"].size(); zkPC++)
    {

        // When bConditionalJump=true, the code will go to the proper label after all the work has been done based on the content of bJump
        bool bConditionalJump = false;

        // When bForcedJump=true, the code will always jump
        bool bForcedJump = false;

        // When bIncHashPos=true, incHashPos will be added to HASHPOS
        bool bIncHashPos = false;

        // ROM instruction line label
        code += functionName + "_rom_line_" + to_string(zkPC) + ": //" + string(rom["program"][zkPC]["fileName"]) + ":" + to_string(rom["program"][zkPC]["line"]) + "=[" + removeDuplicateSpaces(string(rom["program"][zkPC]["lineStr"])) + "]\n\n";

        // START LOGS
#ifdef LOG_COMPLETED_STEPS_TO_FILE
        code += "    fi0=fi1=fi2=fi3=fi4=fi5=fi6=fi7=fr.zero();\n";
        code += "    pols.FREE0[" + string(bFastMode?"0":"i") + "] = fi0;\n";
        code += "    pols.FREE1[" + string(bFastMode?"0":"i") + "] = fi1;\n";
        code += "    pols.FREE2[" + string(bFastMode?"0":"i") + "] = fi2;\n";
        code += "    pols.FREE3[" + string(bFastMode?"0":"i") + "] = fi3;\n";
        code += "    pols.FREE4[" + string(bFastMode?"0":"i") + "] = fi4;\n";
        code += "    pols.FREE5[" + string(bFastMode?"0":"i") + "] = fi5;\n";
        code += "    pols.FREE6[" + string(bFastMode?"0":"i") + "] = fi6;\n";
        code += "    pols.FREE7[" + string(bFastMode?"0":"i") + "] = fi7;\n\n";
#endif
#ifdef LOG_START_STEPS
        code += "    zklog.info(\"--> Starting step=\" + to_string(i) + \" zkPC=" + to_string(zkPC) + " zkasm=\" + rom.line[" + to_string(zkPC) + "].lineStr);\n";
#endif
#ifdef LOG_PRINT_ROM_LINES
        code += "    zklog.info(\"step=\" + to_string(i) + \" rom.line[" + to_string(zkPC) + "] =[" + removeDuplicateSpaces(string(rom["program"][zkPC]["lineStr"])) + "]);\n";
#endif
#ifdef LOG_START_STEPS_TO_FILE
        code += "    outfile.open(\"c.txt\", std::ios_base::app); // append instead of overwrite\n";
        code += "    outfile << \"--> Starting step=\" << i << \" zkPC=" + to_string(zkPC) + " instruction= \" << rom.line[" + to_string(zkPC) + "].toString(fr) << endl;\n";
        code += "    outfile.close();\n";
#endif

#ifdef ENABLE_EXPERIMENTAL_CODE
        // ECRECOVER PRE-CALCULATION 
        if(rom["labels"].contains("ecrecover_store_args") && zkPC == rom["labels"]["ecrecover_store_args"]){
            //code += "    //ECRecover pre-calculation \n";
            code += "    if(mainExecutor.config.ECRecoverPrecalc_experimental){\n";
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

            //code += "    //ECRecover destroy pre-calculaiton buffer\n";
            code += "    if( ctx.ecRecoverPrecalcBuffer.filled){\n";  
            code += "       zkassert(ctx.ecRecoverPrecalcBuffer.pos == ctx.ecRecoverPrecalcBuffer.posUsed);\n";
            code += "       ctx.ecRecoverPrecalcBuffer.filled = false;\n";
            code += "    }\n";
        }
#endif

        // INITIALIZATION

        bool opInitialized = false;

        // COMMAND BEFORE
        if (rom["program"][zkPC].contains("cmdBefore") &&
            (rom["program"][zkPC]["cmdBefore"].size()>0))
        {
            //code += "    // Evaluate the list cmdBefore commands, and any children command, recursively\n";
            code += "    for (uint64_t j=0; j<rom.line[" + to_string(zkPC) + "].cmdBefore.size(); j++)\n";
            code += "    {\n";
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
            code += "        gettimeofday(&ctx.t, NULL);\n";
#endif
            code += "        cr.reset();\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        evalCommand(ctx, *rom.line[" + to_string(zkPC) + "].cmdBefore[j], cr);\n";
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
            code += "        ctx.mainMetrics.add(\"Eval command\", TimeDiff(ctx.t));\n";
            code += "        ctx.evalCommandMetrics.add(rom.line[" + to_string(zkPC) + "].cmdBefore[j]->opAndFunction, TimeDiff(ctx.t));\n";
#endif
            code += "        if (cr.zkResult != ZKR_SUCCESS)\n";
            code += "        {\n";
            code += "            proverRequest.result = cr.zkResult;\n";
            code += "            mainExecutor.logError(ctx, string(\"Failed calling evalCommand() before result=\") + zkresult2string(proverRequest.result));\n";
            code += "            pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            code += "        }\n";
            code += "    }\n\n";
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

            //code += "    // op0 = op0 + inSTEP*step , where inSTEP=" + inSTEPString + "\n";

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

        if (rom["program"][zkPC].contains("inCntSha256F") && (rom["program"][zkPC]["inCntSha256F"]!=0))
        {
            code += selector1("cntSha256F", rom["program"][zkPC]["inCntSha256F"], opInitialized, bFastMode);
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
                //code += "    // If inROTL_C, op = C rotated left\n";
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
            mpz_class auxScalar;
            auxScalar.set_str(aux, 10);
            int64_t iAux = auxScalar.get_si();
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
             (rom["program"][zkPC].contains("hashS") && (rom["program"][zkPC]["hashS"]==1)) ||
             (rom["program"][zkPC].contains("hashS1") && (rom["program"][zkPC]["hashS1"]==1)) ||
             (rom["program"][zkPC].contains("hashSLen") && (rom["program"][zkPC]["hashSLen"]==1)) ||
             (rom["program"][zkPC].contains("hashSDigest") && (rom["program"][zkPC]["hashSDigest"]==1)) ||
             (rom["program"][zkPC].contains("JMP") && (rom["program"][zkPC]["JMP"]==1)) ||
             (rom["program"][zkPC].contains("JMPN") && (rom["program"][zkPC]["JMPN"]==1)) ||
             (rom["program"][zkPC].contains("JMPC") && (rom["program"][zkPC]["JMPC"]==1)) ||
             (rom["program"][zkPC].contains("JMPZ") && (rom["program"][zkPC]["JMPZ"]==1)) ||
             (rom["program"][zkPC].contains("call") && (rom["program"][zkPC]["call"]==1)) )
        {
            bool bAddrRel = false;
            bool bOffset = false;
            //code += "    // If address is involved, load offset into addr\n";
            if ( (rom["program"][zkPC].contains("ind") && (rom["program"][zkPC]["ind"]==1))  &&
                 (rom["program"][zkPC].contains("indRR") && (rom["program"][zkPC]["indRR"]==1)) )
            {
                cerr << "Error: Both ind and indRR are set to 1" << endl;
                exit(-1);
            }
            if (rom["program"][zkPC].contains("ind") && (rom["program"][zkPC]["ind"]!=0))
            {
                if ((rom["program"][zkPC]["ind"]==1))
                    code += "    if (!fr.toS32(addrRel, pols.E0[" + string(bFastMode?"0":"i") + "]))\n";
                else
                    code += "    if (!fr.toS32(addrRel, fr.mul(rom.line[" + to_string(zkPC) + "].ind, pols.E0[" + string(bFastMode?"0":"i") + "])))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_TOS32;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fr.toS32() with pols.E0[i]=\" + fr.toString(pols.E0[" + string(bFastMode?"0":"i") + "], 16));\n";
                code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
                code += "        return;\n";
                code += "    }\n";
                bAddrRel = true;
            }
            if (rom["program"][zkPC].contains("indRR") && (rom["program"][zkPC]["indRR"]!=0))
            {
                int32_t indRR = rom["program"][zkPC]["indRR"];
                if ((indRR==1))
                    code += "    if ( !fr.toS32(addrRel, pols.RR[" + string(bFastMode?"0":"i") + "]))\n";
                else
                    code += "    if (!fr.toS32(addrRel, fr.mul(fr.fromS64(" + to_string(indRR) + "), pols.RR[" + string(bFastMode?"0":"i") + "])))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_TOS32;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"Failed calling fr.toS32() with pols.RR[i]=\" + fr.toString(pols.RR[" + string(bFastMode?"0":"i") + "], 16) + \" indRR=" + to_string(indRR) + "\");\n";
                code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
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
                code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
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
                //code += "    // If addrRel is possitive, and the sum is too big, fail\n";

                code += "    if ( (addrRel < 0) || (addrRel >= " + to_string( ( (rom["program"][zkPC].contains("isMem") && (rom["program"][zkPC]["isMem"]  == 1) ) ? 0x20000 : 0x10000 ) - 2048 ) + "))\n";
                code += "    {\n";
                code += "        proverRequest.result = ZKR_SM_MAIN_ADDRESS_OUT_OF_RANGE;\n";
                code += "        zkPC=" + to_string(zkPC) +";\n";
                code += "        mainExecutor.logError(ctx, \"addrRel out of range addrRel=\" + to_string(addrRel));\n";
                code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
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
#if (defined LOG_COMPLETED_STEPS) || (defined LOG_COMPLETED_STEPS_TO_FILE)
            code += "    addr = 0;\n";
#endif
        }

        if (rom["program"][zkPC].contains("useCTX") && (rom["program"][zkPC]["useCTX"] == 1))
        {
            //code += "    // If useCTX, addr = addr + CTX*CTX_OFFSET\n";            // Check context range
            code += "    context = fr.toU64(pols.CTX[" + string(bFastMode?"0":"i") + "]);\n";
            code += "    if (context > CTX_MAX)\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_INVALID_MEMORY_CTX;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"pols.CTX=\" + to_string(context) + \" > CTX_MAX=\" + to_string(CTX_MAX));\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            code += "    addr += fr.toU64(pols.CTX[" + string(bFastMode?"0":"i") + "])*CTX_OFFSET;\n";
            if (!bFastMode)
                code += "    pols.useCTX[i] = fr.one();\n\n";
            else
                code += "\n";
            bOnlyOffset = false;
        }

        if (rom["program"][zkPC].contains("isStack") && (rom["program"][zkPC]["isStack"] == 1))
        {
            code += "    addr += STACK_OFFSET; // If isStack\n";
            if (!bFastMode)
                code += "    pols.isStack[i] = fr.one();\n\n";
            else
                code += "\n";
            bOnlyOffset = false;
        }

        if (rom["program"][zkPC].contains("isMem") && (rom["program"][zkPC]["isMem"]  == 1))
        {
            code += "    addr += MEM_OFFSET; // If isMem\n";
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
            int64_t ind = rom["program"][zkPC]["ind"];
            code += "    pols.ind[i] = fr.fromS64(" + to_string(ind) + ");\n\n";
        }

        if (rom["program"][zkPC].contains("indRR") && (rom["program"][zkPC]["indRR"] != 0) && !bFastMode)
        {
            int64_t indRR = rom["program"][zkPC]["indRR"];
            code += "    pols.indRR[i] = fr.fromS64(" + to_string(indRR) + ");\n\n";
        }

        // If offset, record it the committed polynomial
        if (rom["program"][zkPC].contains("offset") && (rom["program"][zkPC]["offset"] != 0) && !bFastMode)
        {
            code += "    pols.offset[i] = fr.fromS32(" + to_string(rom["program"][zkPC]["offset"]) + "); // Copy ROM flags into pols\n\n";
        }

        /**************/
        /* FREE INPUT */
        /**************/

        if (rom["program"][zkPC].contains("inFREE")
            || rom["program"][zkPC].contains("inFREE0"))
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
                    //code += "    // Memory read free in: get fi=mem[addr], if it exists\n";
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    Memory_calculate(ctx, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7, addr);\n";
                    nHits++;
                }

                // Storage read free in: get a poseidon hash, and read fi=sto[hash]
                if (rom["program"][zkPC].contains("sRD") && (rom["program"][zkPC]["sRD"] == 1))
                {
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    zkResult = Storage_read_calculate(ctx, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";
                    code += "    if (zkResult != ZKR_SUCCESS)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = zkResult;\n";
                    code += "        mainExecutor.logError(ctx, \"Failed calling Storage_read_calculate() result=\" + zkresult2string(zkResult));\n";
                    code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    nHits++;
                }

                // Storage write free in: calculate the poseidon hash key, check its entry exists in storage, and update new root hash
                if (rom["program"][zkPC].contains("sWR") && (rom["program"][zkPC]["sWR"] == 1))
                {
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    zkResult = Storage_write_calculate(ctx, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";
                    code += "    if (zkResult != ZKR_SUCCESS)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = zkResult;\n";
                    code += "        mainExecutor.logError(ctx, \"Failed calling Storage_write_calculate() result=\" + zkresult2string(zkResult));\n";
                    code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    nHits++;
                }

                // HashK free in
                if ( (rom["program"][zkPC].contains("hashK") && (rom["program"][zkPC]["hashK"] == 1)) ||
                     (rom["program"][zkPC].contains("hashK1") && (rom["program"][zkPC]["hashK1"] == 1)))
                {
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    zkResult = HashK_calculate(ctx, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7, addr);\n";
                    code += "    if (zkResult != ZKR_SUCCESS)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = zkResult;\n";
                    code += "        mainExecutor.logError(ctx, \"Failed calling HashK_calculate() result=\" + zkresult2string(zkResult));\n";
                    code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    nHits++;
                }

                // HashKDigest free in
                if (rom["program"][zkPC].contains("hashKDigest") && (rom["program"][zkPC]["hashKDigest"] == 1))
                {
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    zkResult = HashKDigest_calculate(ctx, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7, addr);\n";
                    code += "    if (zkResult != ZKR_SUCCESS)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = zkResult;\n";
                    code += "        mainExecutor.logError(ctx, \"Failed calling HashKDigest_calculate() result=\" + zkresult2string(zkResult));\n";
                    code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    nHits++;
                }

                // HashKLen free in
                if (rom["program"][zkPC].contains("hashKLen") && (rom["program"][zkPC]["hashKLen"] == 1)) 
                {
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    HashKLen_calculate(ctx, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7, addr);\n";
                    nHits++;
                }

                // HashP free in
                if ( (rom["program"][zkPC].contains("hashP") && (rom["program"][zkPC]["hashP"] == 1)) ||
                     (rom["program"][zkPC].contains("hashP1") && (rom["program"][zkPC]["hashP1"] == 1)) )
                {
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    zkResult = HashP_calculate(ctx, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7, addr);\n";
                    code += "    if (zkResult != ZKR_SUCCESS)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = zkResult;\n";
                    code += "        mainExecutor.logError(ctx, \"Failed calling HashP_calculate() result=\" + zkresult2string(zkResult));\n";
                    code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    nHits++;
                }

                // HashPDigest free in
                if (rom["program"][zkPC].contains("hashPDigest") && (rom["program"][zkPC]["hashPDigest"] == 1))
                {
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    zkResult = HashPDigest_calculate(ctx, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7, addr);\n";
                    code += "    if (zkResult != ZKR_SUCCESS)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = zkResult;\n";
                    code += "        mainExecutor.logError(ctx, \"Failed calling HashPDigest_calculate() result=\" + zkresult2string(zkResult));\n";
                    code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    nHits++;
                }

                // HashPLen free in
                if (rom["program"][zkPC].contains("hashPLen") && (rom["program"][zkPC]["hashPLen"] == 1)) 
                {
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    HashPLen_calculate(ctx, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7, addr);\n";
                    nHits++;
                }

                // HashS free in
                if ( (rom["program"][zkPC].contains("hashS") && (rom["program"][zkPC]["hashS"] == 1)) ||
                     (rom["program"][zkPC].contains("hashS1") && (rom["program"][zkPC]["hashS1"] == 1)))
                {
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    zkResult = HashS_calculate(ctx, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7, addr);\n";
                    code += "    if (zkResult != ZKR_SUCCESS)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = zkResult;\n";
                    code += "        mainExecutor.logError(ctx, \"Failed calling HashS_calculate() result=\" + zkresult2string(zkResult));\n";
                    code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    nHits++;
                }

                // HashSDigest free in
                if (rom["program"][zkPC].contains("hashSDigest") && (rom["program"][zkPC]["hashSDigest"] == 1))
                {
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    zkResult = HashSDigest_calculate(ctx, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7, addr);\n";
                    code += "    if (zkResult != ZKR_SUCCESS)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = zkResult;\n";
                    code += "        mainExecutor.logError(ctx, \"Failed calling HashSDigest_calculate() result=\" + zkresult2string(zkResult));\n";
                    code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";
                    nHits++;
                }

                // HashSLen free in
                if (rom["program"][zkPC].contains("hashSLen") && (rom["program"][zkPC]["hashSLen"] == 1)) 
                {
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    HashSLen_calculate(ctx, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7, addr);\n";
                    nHits++;
                }

                // Binary free in
                if (rom["program"][zkPC].contains("bin") && (rom["program"][zkPC]["bin"] == 1))
                {
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    zkResult = Binary_calculate(ctx, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";
                    code += "    if (zkResult != ZKR_SUCCESS)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = zkResult;\n";
                    code += "        mainExecutor.logError(ctx, \"Failed calling Binary_calculate() result=\" + zkresult2string(zkResult));\n";
                    code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n\n";
                    nHits++;
                }

                // Mem align read free in
                if (rom["program"][zkPC].contains("memAlignRD") && (rom["program"][zkPC]["memAlignRD"]==1))
                {
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    zkResult = Memalign_calculate(ctx, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n";
                    code += "    if (zkResult != ZKR_SUCCESS)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = zkResult;\n";
                    code += "        mainExecutor.logError(ctx, \"Failed calling Memalign_calculate() result=\" + zkresult2string(zkResult));\n";
                    code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n\n";
                    nHits++;
                }

                // Check that one and only one instruction has been requested
                if (nHits != 1)
                {
                    cerr << "Error: Empty freeIn without any instruction: zkPC=" << zkPC << " nHits=" << nHits << " instruction=" << rom["program"][zkPC]["lineStr"] << endl;
                    exit(-1);
                }

            }
            // If freeInTag.op!="", then evaluate the requested command (recursively)
            else
            {
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                code += "    gettimeofday(&ctx.t, NULL);\n";
#endif

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
                    //code += "    // Call evalCommand()\n";
                    code += "    cr.reset();\n";
                    code += "    zkPC=" + to_string(zkPC) +";\n";
                    code += "    evalCommand(ctx, rom.line[" + to_string(zkPC) + "].freeInTag, cr);\n";
                    code += "    if (cr.zkResult != ZKR_SUCCESS)\n";
                    code += "    {\n";
                    code += "        proverRequest.result = cr.zkResult;\n";
                    code += "        zkPC=" + to_string(zkPC) +";\n";
                    code += "        mainExecutor.logError(ctx, string(\"Main exec failed calling evalCommand() result=\") + zkresult2string(proverRequest.result));\n";
                    code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
                    code += "        return;\n";
                    code += "    }\n";

                    code += "    cr.toFea(ctx, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);\n\n";
                }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                code += "    ctx.mainMetrics.add(\"Eval command\", TimeDiff(ctx.t));\n";
                code += "    ctx.evalCommandMetrics.add(rom.line[" + to_string(zkPC) + "].freeInTag.opAndFunction, TimeDiff(ctx.t));\n";
#endif

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

            if (!bFastMode || (rom["program"][zkPC].contains("assumeFree") && (rom["program"][zkPC]["assumeFree"] == 1)))
            {
                //code += "    // Store polynomial FREE=fi\n";
                code += "    pols.FREE0[" + string(bFastMode?"0":"i") + "] = fi0;\n";
                code += "    pols.FREE1[" + string(bFastMode?"0":"i") + "] = fi1;\n";
                code += "    pols.FREE2[" + string(bFastMode?"0":"i") + "] = fi2;\n";
                code += "    pols.FREE3[" + string(bFastMode?"0":"i") + "] = fi3;\n";
                code += "    pols.FREE4[" + string(bFastMode?"0":"i") + "] = fi4;\n";
                code += "    pols.FREE5[" + string(bFastMode?"0":"i") + "] = fi5;\n";
                code += "    pols.FREE6[" + string(bFastMode?"0":"i") + "] = fi6;\n";
                code += "    pols.FREE7[" + string(bFastMode?"0":"i") + "] = fi7;\n\n";
            }

            //code += "    // op = op + inFREE*fi\n";
            string inFREEString = rom["program"][zkPC]["inFREE"];
            int64_t inFREE;
            string inFREE0String = rom["program"][zkPC]["inFREE0"];
            inFREE = atoi(inFREEString.c_str()) + atoi(inFREE0String.c_str());

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
                    code += "    op0 = fr.add(op0, fr.mul(fr.add(rom.line[" + to_string(zkPC) + "].inFREE, rom.line[" + to_string(zkPC) + "].inFREE0), fi0));\n";
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
                    code += "    op0 = fr.mul(fr.add(rom.line[" + to_string(zkPC) + "].inFREE, rom.line[" + to_string(zkPC) + "].inFREE0), fi0);\n";
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
                //code += "    // Copy ROM flags into the polynomials\n";
                code += "    pols.inFREE[i] = rom.line[" + to_string(zkPC) + "].inFREE;\n\n";
                code += "    pols.inFREE0[i] = rom.line[" + to_string(zkPC) + "].inFREE0;\n\n";
            }
        }

        if (!opInitialized)
            code += "    op7 = op6 = op5 = op4 = op3 = op2 = op1 = op0 = fr.zero(); // Initialize op to zero\n\n";

        // Set pols.op0Inv
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
            //code += "    // If assert, check that A=op\n";
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
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            if (!bFastMode)
                code += "    pols.assert_pol[i] = fr.one();\n";
            code += "\n";
        }


        // Assume free
        if (!bFastMode && rom["program"][zkPC].contains("assumeFree") && (rom["program"][zkPC]["assumeFree"] == 1))
        {
            code += "    pols.assumeFree[i] = fr.one();\n";
        }

        // Memory operation instruction
        if (rom["program"][zkPC].contains("mOp") && (rom["program"][zkPC]["mOp"] == 1))
        {
            //code += "    // Memory operation instruction\n";
            code += "    zkPC=" + to_string(zkPC) +";\n";
            if (rom["program"][zkPC].contains("assumeFree") && (rom["program"][zkPC]["assumeFree"] == 1))
            {
                code += "    zkResult = Memory_verify(ctx, pols.FREE0[" + string(bFastMode?"0":"i") + "], pols.FREE1[" + string(bFastMode?"0":"i") + "], pols.FREE2[" + string(bFastMode?"0":"i") + "], pols.FREE3[" + string(bFastMode?"0":"i") + "], pols.FREE4[" + string(bFastMode?"0":"i") + "], pols.FREE5[" + string(bFastMode?"0":"i") + "], pols.FREE6[" + string(bFastMode?"0":"i") + "], pols.FREE7[" + string(bFastMode?"0":"i") + "], " + (bFastMode ? string("NULL") : string("&required")) + ", addr);\n";
            }
            else
                code += "    zkResult = Memory_verify(ctx, op0, op1, op2, op3, op4, op5, op6, op7, " + (bFastMode ? string("NULL") : string("&required")) + ", addr);\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling Memory_verify() result=\" + zkresult2string(zkResult));\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";
        }

        // overwrite 'op' when hiting 'checkFirstTxType' label
        if ( rom["labels"].contains("checkFirstTxType") &&
            (zkPC == rom["labels"]["checkFirstTxType"]) )
        {
            code += "    if (proverRequest.input.bSkipFirstChangeL2Block)\n";
            code += "    {\n";
            code += "        op0 = fr.one();\n";
            code += "        op1 = fr.one();\n";
            code += "        op2 = fr.one();\n";
            code += "        op3 = fr.one();\n";
            code += "        op4 = fr.one();\n";
            code += "        op5 = fr.one();\n";
            code += "        op6 = fr.one();\n";
            code += "        op7 = fr.one();\n";
            code += "    }\n";
        }
        // overwrite 'op' when hiting 'writeBlockInfoRoot' label
        if ( rom["labels"].contains("writeBlockInfoRoot") &&
            (zkPC == rom["labels"]["writeBlockInfoRoot"]) )
        {
            code += "    if (proverRequest.input.bSkipWriteBlockInfoRoot)\n";
            code += "    {\n";
            code += "        op0 = fr.zero();\n";
            code += "        op1 = fr.zero();\n";
            code += "        op2 = fr.zero();\n";
            code += "        op3 = fr.zero();\n";
            code += "        op4 = fr.zero();\n";
            code += "        op5 = fr.zero();\n";
            code += "        op6 = fr.zero();\n";
            code += "        op7 = fr.zero();\n";
            code += "    }\n";
        }
        // Storage read instruction
        if (rom["program"][zkPC].contains("sRD") && (rom["program"][zkPC]["sRD"] == 1) )
        {
            code += "    zkPC=" + to_string(zkPC) +";\n";
            code += "    zkResult = Storage_read_verify(ctx, op0, op1, op2, op3, op4, op5, op6, op7, " + (bFastMode ? string("NULL") : string("&required")) + ");\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling Storage_read_verify() result=\" + zkresult2string(zkResult));\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";
        }

        // Storage write instruction
        if (rom["program"][zkPC].contains("sWR") && (rom["program"][zkPC]["sWR"] == 1))
        {
            code += "    zkPC=" + to_string(zkPC) +";\n";
            code += "    zkResult = Storage_write_verify(ctx, op0, op1, op2, op3, op4, op5, op6, op7, " + (bFastMode ? string("NULL") : string("&required")) + ");\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling Storage_write_verify() result=\" + zkresult2string(zkResult));\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";
        }

        

        // HashK instruction
        if ( (rom["program"][zkPC].contains("hashK") && (rom["program"][zkPC]["hashK"] == 1)) ||
             (rom["program"][zkPC].contains("hashK1") && (rom["program"][zkPC]["hashK1"] == 1)) )
        {
            code += "    zkPC=" + to_string(zkPC) +";\n";
            code += "    zkResult = HashK_verify(ctx, op0, op1, op2, op3, op4, op5, op6, op7, " + (bFastMode ? string("NULL") : string("&required")) + ", addr);\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling HashK_verify() result=\" + zkresult2string(zkResult));\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";
            bIncHashPos = true;
        }

        // HashKLen instruction
        if (rom["program"][zkPC].contains("hashKLen") && (rom["program"][zkPC]["hashKLen"] == 1))
        {
            code += "    zkPC=" + to_string(zkPC) +";\n";
            code += "    zkResult = HashKLen_verify(ctx, op0, op1, op2, op3, op4, op5, op6, op7, " + (bFastMode ? string("NULL") : string("&required")) + ", addr);\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling HashKLen_verify() result=\" + zkresult2string(zkResult));\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";
        }

        // HashKDigest instruction
        if (rom["program"][zkPC].contains("hashKDigest") && (rom["program"][zkPC]["hashKDigest"] == 1))
        {
            code += "    zkPC=" + to_string(zkPC) +";\n";
            code += "    zkResult = HashKDigest_verify(ctx, op0, op1, op2, op3, op4, op5, op6, op7, " + (bFastMode ? string("NULL") : string("&required")) + ", addr);\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling HashKDigest_verify() result=\" + zkresult2string(zkResult));\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";
        }

        // HashP instruction
        if ( (rom["program"][zkPC].contains("hashP") && (rom["program"][zkPC]["hashP"] == 1)) ||
             (rom["program"][zkPC].contains("hashP1") && (rom["program"][zkPC]["hashP1"] == 1)) )
        {
            code += "    zkPC=" + to_string(zkPC) +";\n";
            code += "    zkResult = HashP_verify(ctx, op0, op1, op2, op3, op4, op5, op6, op7, " + (bFastMode ? string("NULL") : string("&required")) + ", addr);\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling HashP_verify() result=\" + zkresult2string(zkResult));\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";
            bIncHashPos = true;
        }

        // HashPLen instruction
        if (rom["program"][zkPC].contains("hashPLen") && (rom["program"][zkPC]["hashPLen"] == 1))
        {
            code += "    zkPC=" + to_string(zkPC) +";\n";
            code += "    zkResult = HashPLen_verify(ctx, op0, op1, op2, op3, op4, op5, op6, op7, " + (bFastMode ? string("NULL") : string("&required")) + ", addr);\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling HashPLen_verify() result=\" + zkresult2string(zkResult));\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";
        }

        // HashPDigest instruction
        if (rom["program"][zkPC].contains("hashPDigest") && (rom["program"][zkPC]["hashPDigest"] == 1))
        {
            code += "    zkPC=" + to_string(zkPC) +";\n";
            code += "    zkResult = HashPDigest_verify(ctx, op0, op1, op2, op3, op4, op5, op6, op7, " + (bFastMode ? string("NULL") : string("&required")) + ", addr);\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling HashPDigest_verify() result=\" + zkresult2string(zkResult));\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";
        }

        // HashS instruction
        if ( (rom["program"][zkPC].contains("hashS") && (rom["program"][zkPC]["hashS"] == 1)) ||
             (rom["program"][zkPC].contains("hashS1") && (rom["program"][zkPC]["hashS1"] == 1)) )
        {
            code += "    zkPC=" + to_string(zkPC) +";\n";
            code += "    zkResult = HashS_verify(ctx, op0, op1, op2, op3, op4, op5, op6, op7, " + (bFastMode ? string("NULL") : string("&required")) + ", addr);\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling HashS_verify() result=\" + zkresult2string(zkResult));\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";
            bIncHashPos = true;
        }

        // HashSLen instruction
        if (rom["program"][zkPC].contains("hashSLen") && (rom["program"][zkPC]["hashSLen"] == 1))
        {
            code += "    zkPC=" + to_string(zkPC) +";\n";
            code += "    zkResult = HashSLen_verify(ctx, op0, op1, op2, op3, op4, op5, op6, op7, " + (bFastMode ? string("NULL") : string("&required")) + ", addr);\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling HashSLen_verify() result=\" + zkresult2string(zkResult));\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";
        }

        // HashSDigest instruction
        if (rom["program"][zkPC].contains("hashSDigest") && (rom["program"][zkPC]["hashSDigest"] == 1))
        {
            code += "    zkPC=" + to_string(zkPC) +";\n";
            code += "    zkResult = HashSDigest_verify(ctx, op0, op1, op2, op3, op4, op5, op6, op7, " + (bFastMode ? string("NULL") : string("&required")) + ", addr);\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling HashDigest_verify() result=\" + zkresult2string(zkResult));\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";
        }

        // HashP or Storage write instructions, required data
        if (!bFastMode && ( (rom["program"][zkPC].contains("hashPDigest") && (rom["program"][zkPC]["hashPDigest"]==1)) ||
                            (rom["program"][zkPC].contains("sWR") && (rom["program"][zkPC]["sWR"] ==1))) )
        {
            //code += "    // HashP or Storage write instructions, required data\n";
            code += "    if (!fea2scalar(fr, op, op0, op1, op2, op3, op4, op5, op6, op7))\n";
            code += "    {\n";
            code += "        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;\n";
            code += "        zkPC=" + to_string(zkPC) +";\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling fea2scalar(op)\");\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            //code += "    // Store the binary action to execute it later with the binary SM\n";
            code += "    binaryAction.a = op;\n";
            code += "    binaryAction.b = Scalar4xGoldilocksPrime;\n";
            code += "    binaryAction.c = 1;\n";
            code += "    binaryAction.opcode = 8;\n";
            code += "    binaryAction.type = 2;\n";
            code += "    required.Binary.push_back(binaryAction);\n";
        }

        // Arith instruction
        if ( ((forkID >= 13) && (rom["program"][zkPC].contains("arith") && (rom["program"][zkPC]["arith"]==1)) ) ||
             ((forkID < 13) &&
                ( (rom["program"][zkPC].contains("arithEq0") && (rom["program"][zkPC]["arithEq0"]==1)) ||
                  (rom["program"][zkPC].contains("arithEq1") && (rom["program"][zkPC]["arithEq1"]==1)) ||
                  (rom["program"][zkPC].contains("arithEq2") && (rom["program"][zkPC]["arithEq2"]==1)) ||
                  (rom["program"][zkPC].contains("arithEq3") && (rom["program"][zkPC]["arithEq3"]==1)) ||
                  (rom["program"][zkPC].contains("arithEq4") && (rom["program"][zkPC]["arithEq4"]==1)) ||
                  (rom["program"][zkPC].contains("arithEq5") && (rom["program"][zkPC]["arithEq5"]==1)) )
             )
           )
        {
            code += "    zkPC=" + to_string(zkPC) +";\n";
            code += "    zkResult = Arith_verify(ctx, op0, op1, op2, op3, op4, op5, op6, op7, " + (bFastMode ? string("NULL") : string("&required")) + ");\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling Arith_verify()\");\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
        }

        // Binary instruction
        if (rom["program"][zkPC].contains("bin") && (rom["program"][zkPC]["bin"] == 1))
        {
            code += "    zkPC=" + to_string(zkPC) +";\n";
            code += "    zkResult = Binary_verify(ctx, op0, op1, op2, op3, op4, op5, op6, op7, " + (bFastMode ? string("NULL") : string("&required")) + ");\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling Binary_verify() result=\" + zkresult2string(zkResult));\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";
        }

        // MemAlign instruction
        if ( (rom["program"][zkPC].contains("memAlignRD") && (rom["program"][zkPC]["memAlignRD"]==1)) ||
             (rom["program"][zkPC].contains("memAlignWR") && (rom["program"][zkPC]["memAlignWR"]==1)) ||
             (rom["program"][zkPC].contains("memAlignWR8") && (rom["program"][zkPC]["memAlignWR8"]==1)) )
        {
            code += "    zkPC=" + to_string(zkPC) +";\n";
            code += "    zkResult = Memalign_verify(ctx, op0, op1, op2, op3, op4, op5, op6, op7, " + (bFastMode ? string("NULL") : string("&required")) + ");\n";
            code += "    if (zkResult != ZKR_SUCCESS)\n";
            code += "    {\n";
            code += "        proverRequest.result = zkResult;\n";
            code += "        mainExecutor.logError(ctx, \"Failed calling Memalign_verify() result=\" + zkresult2string(zkResult));\n";
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n\n";
        }

        // Repeat instruction
        bool repeat = (rom["program"][zkPC].contains("repeat") && (rom["program"][zkPC]["repeat"] == 1));
        if (repeat && (!bFastMode))
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

        // Set op
        if (!bFastMode)
        {
            code += "    pols.op0[i] = op0;\n";
            code += "    pols.op1[i] = op1;\n";
            code += "    pols.op2[i] = op2;\n";
            code += "    pols.op3[i] = op3;\n";
            code += "    pols.op4[i] = op4;\n";
            code += "    pols.op5[i] = op5;\n";
            code += "    pols.op6[i] = op6;\n";
            code += "    pols.op7[i] = op7;\n";
        }

        code += setter8("A",  rom["program"][zkPC].contains("setA") && (rom["program"][zkPC]["setA"]==1),   bFastMode, zkPC, rom, forkID);
        code += setter8("B",  rom["program"][zkPC].contains("setB") && (rom["program"][zkPC]["setB"]==1),   bFastMode, zkPC, rom, forkID);
        code += setter8("C",  rom["program"][zkPC].contains("setC") && (rom["program"][zkPC]["setC"]==1),   bFastMode, zkPC, rom, forkID);
        code += setter8("D",  rom["program"][zkPC].contains("setD") && (rom["program"][zkPC]["setD"]==1),   bFastMode, zkPC, rom, forkID);
        code += setter8("E",  rom["program"][zkPC].contains("setE") && (rom["program"][zkPC]["setE"]==1),   bFastMode, zkPC, rom, forkID);
        code += setter8("SR", rom["program"][zkPC].contains("setSR") && (rom["program"][zkPC]["setSR"]==1), bFastMode, zkPC, rom, forkID);

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
        if ( ((forkID >= 13) && (rom["program"][zkPC].contains("arith") && (rom["program"][zkPC]["arith"]==1)) ) ||
             ((forkID < 13) && 
             ( (rom["program"][zkPC].contains("arithEq0") && (rom["program"][zkPC]["arithEq0"]==1)) ||
               (rom["program"][zkPC].contains("arithEq1") && (rom["program"][zkPC]["arithEq1"]==1)) ||
               (rom["program"][zkPC].contains("arithEq2") && (rom["program"][zkPC]["arithEq2"]==1)) ||
               (rom["program"][zkPC].contains("arithEq3") && (rom["program"][zkPC]["arithEq3"]==1)) ||
               (rom["program"][zkPC].contains("arithEq4") && (rom["program"][zkPC]["arithEq4"]==1)) ||
               (rom["program"][zkPC].contains("arithEq5") && (rom["program"][zkPC]["arithEq5"]==1)) ))
           )
        {
            code += "    if (!proverRequest.input.bNoCounters)\n";
            code += "    {\n";
            code += "        pols.cntArith[" + string(bFastMode?"0":"nexti") + "] = fr.inc(pols.cntArith[" + string(bFastMode?"0":"i") + "]);\n";
#ifdef CHECK_MAX_CNT_ASAP
            code += "        if (fr.toU64(pols.cntArith[" + string(bFastMode?"0":"nexti") + "]) > " + (string)rom["constants"]["MAX_CNT_ARITH_LIMIT"]["value"] + ")\n";
            code += "        {\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"Main Executor found pols.cntArith[nexti]=\" + fr.toString(pols.cntArith[" + string(bFastMode?"0":"nexti") + "], 10) + \" > MAX_CNT_ARITH_LIMIT=" + (string)rom["constants"]["MAX_CNT_ARITH_LIMIT"]["value"] + "\");\n";
            if (bFastMode)
            {
            code += "            proverRequest.result = ZKR_SM_MAIN_OOC_ARITH;\n";
            code += "            pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            }
            else
            {
            code += "            exitProcess();\n";
            }
            code += "        }\n";
#endif
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
#ifdef CHECK_MAX_CNT_ASAP
            code += "        if (fr.toU64(pols.cntBinary[" + string(bFastMode?"0":"nexti") + "]) > " + (string)rom["constants"]["MAX_CNT_BINARY_LIMIT"]["value"] + ")\n";
            code += "        {\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"Main Executor found pols.cntBinary[nexti]=\" + fr.toString(pols.cntBinary[" + string(bFastMode?"0":"nexti") + "], 10) + \" > MAX_CNT_BINARY_LIMIT=" + (string)rom["constants"]["MAX_CNT_BINARY_LIMIT"]["value"] + "\");\n";
            if (bFastMode)
            {
            code += "            proverRequest.result = ZKR_SM_MAIN_OOC_BINARY;\n";
            code += "            pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            }
            else
            {
            code += "            exitProcess();\n";
            }
            code += "        }\n";
#endif
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
#ifdef CHECK_MAX_CNT_ASAP
            code += "        if (fr.toU64(pols.cntMemAlign[" + string(bFastMode?"0":"nexti") + "]) > " + (string)rom["constants"]["MAX_CNT_MEM_ALIGN_LIMIT"]["value"] + ")\n";
            code += "        {\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"Main Executor found pols.cntMemAlign[nexti]=\" + fr.toString(pols.cntMemAlign[" + string(bFastMode?"0":"nexti") + "], 10) + \" > MAX_CNT_MEM_ALIGN_LIMIT=" + (string)rom["constants"]["MAX_CNT_MEM_ALIGN_LIMIT"]["value"] + "\");\n";
            if (bFastMode)
            {
            code += "            proverRequest.result = ZKR_SM_MAIN_OOC_MEM_ALIGN;\n";
            code += "            pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            }
            else
            {
            code += "            exitProcess();\n";
            }
            code += "        }\n";
#endif
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
        else if (!bFastMode)
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
            if (rom["program"][zkPC].contains("jmpAddrLabel"))
            {
                if (rom["program"][zkPC]["jmpAddrLabel"] == "outOfCountersStep")
                {
                    code += "    reserve = int64_t(" + (string)rom["constants"]["MAX_CNT_STEPS"]["value"] + ") - fr.toS64(op0);\n";
                    code += "    if ((reserve > 0) && (uint64_t(reserve) > proverRequest.countersReserve.steps))\n";
                    code += "    {\n";
                    code += "        proverRequest.countersReserve.steps = uint64_t(reserve);\n";
                    code += "        proverRequest.countersReserveZkpc.steps = " + to_string(zkPC) + ";\n";
                    code += "    }\n";
                }
                else if (rom["program"][zkPC]["jmpAddrLabel"] == "outOfCountersArith")
                {
                    code += "    reserve = int64_t(" + (string)rom["constants"]["MAX_CNT_ARITH"]["value"] + ") - fr.toS64(op0);\n";
                    code += "    if ((reserve > 0) && (uint64_t(reserve) > proverRequest.countersReserve.arith))\n";
                    code += "    {\n";
                    code += "        proverRequest.countersReserve.arith = uint64_t(reserve);\n";
                    code += "        proverRequest.countersReserveZkpc.arith = " + to_string(zkPC) + ";\n";
                    code += "    }\n";
                }
                else if (rom["program"][zkPC]["jmpAddrLabel"] == "outOfCountersBinary")
                {
                    code += "    reserve = int64_t(" + (string)rom["constants"]["MAX_CNT_BINARY"]["value"] + ") - fr.toS64(op0);\n";
                    code += "    if ((reserve > 0) && (uint64_t(reserve) > proverRequest.countersReserve.binary))\n";
                    code += "    {\n";
                    code += "        proverRequest.countersReserve.binary = uint64_t(reserve);\n";
                    code += "        proverRequest.countersReserveZkpc.binary = " + to_string(zkPC) + ";\n";
                    code += "    }\n";
                }
                else if (rom["program"][zkPC]["jmpAddrLabel"] == "outOfCountersKeccak")
                {
                    code += "    reserve = int64_t(" + (string)rom["constants"]["MAX_CNT_KECCAK_F"]["value"] + ") - fr.toS64(op0);\n";
                    code += "    if ((reserve > 0) && (uint64_t(reserve) > proverRequest.countersReserve.keccakF))\n";
                    code += "    {\n";
                    code += "        proverRequest.countersReserve.keccakF = uint64_t(reserve);\n";
                    code += "        proverRequest.countersReserveZkpc.keccakF = " + to_string(zkPC) + ";\n";
                    code += "    }\n";
                }
                else if (rom["program"][zkPC]["jmpAddrLabel"] == "outOfCountersSha256")
                {
                    code += "    reserve = int64_t(" + (string)rom["constants"]["MAX_CNT_SHA256_F"]["value"] + ") - fr.toS64(op0);\n";
                    code += "    if ((reserve > 0) && (uint64_t(reserve) > proverRequest.countersReserve.sha256F))\n";
                    code += "    {\n";
                    code += "        proverRequest.countersReserve.sha256F = uint64_t(reserve);\n";
                    code += "        proverRequest.countersReserveZkpc.sha256F = " + to_string(zkPC) + ";\n";
                    code += "    }\n";
                }
                else if (rom["program"][zkPC]["jmpAddrLabel"] == "outOfCountersMemalign")
                {
                    code += "    reserve = int64_t(" + (string)rom["constants"]["MAX_CNT_MEM_ALIGN"]["value"] + ") - fr.toS64(op0);\n";
                    code += "    if ((reserve > 0) && (uint64_t(reserve) > proverRequest.countersReserve.memAlign))\n";
                    code += "    {\n";
                    code += "        proverRequest.countersReserve.memAlign = uint64_t(reserve);\n";
                    code += "        proverRequest.countersReserveZkpc.memAlign = " + to_string(zkPC) + ";\n";
                    code += "    }\n";
                }
                else if (rom["program"][zkPC]["jmpAddrLabel"] == "outOfCountersPoseidon")
                {
                    code += "    reserve = int64_t(" + (string)rom["constants"]["MAX_CNT_POSEIDON_G"]["value"] + ") - fr.toS64(op0);\n";
                    code += "    if ((reserve > 0) && (uint64_t(reserve) > proverRequest.countersReserve.poseidonG))\n";
                    code += "    {\n";
                    code += "        proverRequest.countersReserve.poseidonG = uint64_t(reserve);\n";
                    code += "        proverRequest.countersReserveZkpc.poseidonG = " + to_string(zkPC) + ";\n";
                    code += "    }\n";
                }
                else if (rom["program"][zkPC]["jmpAddrLabel"] == "outOfCountersPadding")
                {
                    code += "    reserve = int64_t(" + (string)rom["constants"]["MAX_CNT_PADDING_PG"]["value"] + ") - fr.toS64(op0);\n";
                    code += "    if ((reserve > 0) && (uint64_t(reserve) > proverRequest.countersReserve.paddingPG))\n";
                    code += "    {\n";
                    code += "        proverRequest.countersReserve.paddingPG = uint64_t(reserve);\n";
                    code += "        proverRequest.countersReserveZkpc.paddingPG = " + to_string(zkPC) + ";\n";
                    code += "    }\n";
                }
            }

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
            code += "        pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "        return;\n";
            code += "    }\n";
            if (!bFastMode)
            {
                code += "    pols.lJmpnCondValue[i] = fr.fromU64(jmpnCondValue &" + to_string(forkInfo.N - 1) + ");\n";
                code += "    jmpnCondValue = jmpnCondValue >>" + to_string(forkInfo.Nbits) + ";\n";
                if (forkID < 12)
                {
                    code += "    for (uint64_t index = 0; index < 8; ++index)\n";
                }
                else
                {
                    code += "    for (uint64_t index = 0; index < 7; ++index)\n";
                }
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

        /***********************/
        /* Set GAS and HASHPOS */
        /***********************/

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
            if (bIncHashPos)
                code += "    pols.HASHPOS[" + string(bFastMode?"0":"nexti") + "] = fr.add(op0, fr.fromU64(ctx.incHashPos));\n";
            else
                code += "    pols.HASHPOS[" + string(bFastMode?"0":"nexti") + "] = op0;\n";
            if (!bFastMode)
                code += "    pols.setHASHPOS[i] = fr.one();\n";
        }
        else //if (!bFastMode)
        {
            if (bIncHashPos)
                code += "    pols.HASHPOS[" + string(bFastMode?"0":"nexti") + "] = fr.add(pols.HASHPOS[" + string(bFastMode?"0":"i") + "], fr.fromU64(ctx.incHashPos));\n";
            else if (!bFastMode)
                code += "    pols.HASHPOS[nexti] = pols.HASHPOS[i];\n";
        }

        /************/
        /* COUNTERS */
        /************/

        if (!bFastMode && ( (rom["program"][zkPC].contains("sRD") && (rom["program"][zkPC]["sRD"]==1)) ||
                            (rom["program"][zkPC].contains("sWR") && (rom["program"][zkPC]["sWR"]==1)) ||
                            (rom["program"][zkPC].contains("hashKDigest") && (rom["program"][zkPC]["hashKDigest"]==1)) ||
                            (rom["program"][zkPC].contains("hashPDigest") && (rom["program"][zkPC]["hashPDigest"]==1)) ||
                            (rom["program"][zkPC].contains("hashSDigest") && (rom["program"][zkPC]["hashSDigest"]==1)) ) )
        {
            code += "    pols.incCounter[i] = fr.fromU64(ctx.incCounter);\n";
        }

        if ( rom["program"][zkPC].contains("hashKDigest") && (rom["program"][zkPC]["hashKDigest"] == 1) )
        {
            code += "    if (!proverRequest.input.bNoCounters)\n";
            code += "    {\n";
            code += "        pols.cntKeccakF[" + string(bFastMode?"0":"nexti") + "] = fr.add(pols.cntKeccakF[" + string(bFastMode?"0":"i") + "], fr.fromU64(ctx.incCounter));\n";
#ifdef CHECK_MAX_CNT_ASAP
            code += "        if (fr.toU64(pols.cntKeccakF[" + string(bFastMode?"0":"nexti") + "]) > " + (string)rom["constants"]["MAX_CNT_KECCAK_F_LIMIT"]["value"] + ")\n";
            code += "        {\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"Main Executor found pols.cntKeccakF[nexti]=\" + fr.toString(pols.cntKeccakF[" + string(bFastMode?"0":"nexti") + "], 10) + \" > MAX_CNT_KECCAK_F_LIMIT=" + (string)rom["constants"]["MAX_CNT_KECCAK_F_LIMIT"]["value"] + "\");\n";
            if (bFastMode)
            {
            code += "            proverRequest.result = ZKR_SM_MAIN_OOC_KECCAK_F;\n";
            code += "            pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            }
            else
            {
            code += "            exitProcess();\n";
            }
            code += "        }\n";
#endif
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
            code += "        pols.cntPaddingPG[" + string(bFastMode?"0":"nexti") + "] = fr.add(pols.cntPaddingPG[" + string(bFastMode?"0":"i") + "], fr.fromU64(ctx.incCounter));\n";
#ifdef CHECK_MAX_CNT_ASAP
            code += "        if (fr.toU64(pols.cntPaddingPG[" + string(bFastMode?"0":"nexti") + "]) > " + (string)rom["constants"]["MAX_CNT_PADDING_PG_LIMIT"]["value"] + ")\n";
            code += "        {\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"Main Executor found pols.cntPaddingPG[nexti]=\" + fr.toString(pols.cntPaddingPG[" + string(bFastMode?"0":"nexti") + "], 10) + \" > MAX_CNT_PADDING_PG_LIMIT=" + (string)rom["constants"]["MAX_CNT_PADDING_PG_LIMIT"]["value"] + "\");\n";
            if (bFastMode)
            {
            code += "            proverRequest.result = ZKR_SM_MAIN_OOC_PADDING_PG;\n";
            code += "            pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            }
            else
            {
            code += "            exitProcess();\n";
            }
            code += "        }\n";
#endif
            code += "    }\n\n";
        }
        else if (!bFastMode)
        {
            code += "    pols.cntPaddingPG[nexti] = pols.cntPaddingPG[i];\n";
        }

        if ( rom["program"][zkPC].contains("hashSDigest") && (rom["program"][zkPC]["hashSDigest"] == 1) )
        {
            code += "    if (!proverRequest.input.bNoCounters)\n";
            code += "    {\n";
            code += "        pols.cntSha256F[" + string(bFastMode?"0":"nexti") + "] = fr.add(pols.cntSha256F[" + string(bFastMode?"0":"i") + "], fr.fromU64(ctx.incCounter));\n";
#ifdef CHECK_MAX_CNT_ASAP
            code += "        if (fr.toU64(pols.cntSha256F[" + string(bFastMode?"0":"nexti") + "]) > " + (string)rom["constants"]["MAX_CNT_SHA256_F_LIMIT"]["value"] + ")\n";
            code += "        {\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"Main Executor found pols.cntSha256F[nexti]=\" + fr.toString(pols.cntSha256F[" + string(bFastMode?"0":"nexti") + "], 10) + \" > MAX_CNT_SHA256_F_LIMIT=" + (string)rom["constants"]["MAX_CNT_SHA256_F_LIMIT"]["value"] + "\");\n";
            if (bFastMode)
            {
            code += "            proverRequest.result = ZKR_SM_MAIN_OOC_SHA256_F;\n";
            code += "            pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            }
            else
            {
            code += "            exitProcess();\n";
            }
            code += "        }\n";
#endif
            code += "    }\n\n";
        }
        else if (!bFastMode)
        {
            code += "    pols.cntSha256F[nexti] = pols.cntSha256F[i];\n";
        }

        if ( (rom["program"][zkPC].contains("sRD") && (rom["program"][zkPC]["sRD"]==1)) ||
             (rom["program"][zkPC].contains("sWR") && (rom["program"][zkPC]["sWR"]==1)) ||
             (rom["program"][zkPC].contains("hashPDigest") && (rom["program"][zkPC]["hashPDigest"]==1) ) )
        {
            code += "    if (!proverRequest.input.bNoCounters)\n";
            code += "    {\n";
            code += "        pols.cntPoseidonG[" + string(bFastMode?"0":"nexti") + "] = fr.add(pols.cntPoseidonG[" + string(bFastMode?"0":"i") + "], fr.fromU64(ctx.incCounter));\n";
#ifdef CHECK_MAX_CNT_ASAP
            code += "        if (fr.toU64(pols.cntPoseidonG[" + string(bFastMode?"0":"nexti") + "]) > " + (string)rom["constants"]["MAX_CNT_POSEIDON_G_LIMIT"]["value"] + ")\n";
            code += "        {\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            mainExecutor.logError(ctx, \"Main Executor found pols.cntPoseidonG[nexti]=\" + fr.toString(pols.cntPoseidonG[" + string(bFastMode?"0":"nexti") + "], 10) + \" > MAX_CNT_POSEIDON_G_LIMIT=" + (string)rom["constants"]["MAX_CNT_POSEIDON_G_LIMIT"]["value"] + "\");\n";
            if (bFastMode)
            {
            code += "            proverRequest.result = ZKR_SM_MAIN_OOC_POSEIDON_G;\n";
            code += "            pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "            return;\n";
            }
            else
            {
            code += "            exitProcess();\n";
            }
            code += "        }\n";
#endif
            code += "    }\n\n";
        }
        else if (!bFastMode)
        {
            code += "    pols.cntPoseidonG[nexti] = pols.cntPoseidonG[i];\n";
        }

        // COMAND AFTER (of previous instruction)
        if ( rom["program"][zkPC].contains("cmdAfter") && (rom["program"][zkPC]["cmdAfter"].size()>0) )
        {
            //code += "    // Evaluate the list cmdAfter commands of the previous ROM line,\n";
            //code += "    // and any children command, recursively\n";
            code += "    if (i < N_Max_minus_one)\n";
            code += "    {\n";
            if (!bFastMode)
            code += "        i++;\n";
            code += "        for (uint64_t j=0; j<rom.line[" + to_string(zkPC) + "].cmdAfter.size(); j++)\n";
            code += "        {\n";
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
            code += "            gettimeofday(&ctx.t, NULL);\n";
#endif
            code += "            cr.reset();\n";
            code += "            zkPC=" + to_string(zkPC) +";\n";
            code += "            evalCommand(ctx, *rom.line[" + to_string(zkPC) + "].cmdAfter[j], cr);\n";
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
            code += "            ctx.mainMetrics.add(\"Eval command\", TimeDiff(ctx.t));\n";
            code += "            ctx.evalCommandMetrics.add(rom.line[" + to_string(zkPC) + "].cmdAfter[j]->opAndFunction, TimeDiff(ctx.t));\n";
#endif
            code += "            if (cr.zkResult != ZKR_SUCCESS)\n";
            code += "            {\n";
            code += "                proverRequest.result = cr.zkResult;\n";
            code += "                zkPC=" + to_string(zkPC) +";\n";
            code += "                mainExecutor.logError(ctx, string(\"Failed calling evalCommand() after result=\") + zkresult2string(proverRequest.result));\n";
            code += "                pHashDB->cancelBatch(proverRequest.uuid);\n";
            code += "                return;\n";
            code += "            }\n";
            code += "        }\n\n";
            if (!bFastMode)
            code += "        i--;\n";
            code += "    }\n\n";
        }
#ifdef LOG_COMPLETED_STEPS
        code += "    zklog.info( \"<-- Completed step=\" + to_string(i) + \" zkPC=" + to_string(zkPC) + " op=\" + fr.toString(op7,16) + \":\" + fr.toString(op6,16) + \":\" + fr.toString(op5,16) + \":\" + fr.toString(op4,16) + \":\" + fr.toString(op3,16) + \":\" + fr.toString(op2,16) + \":\" + fr.toString(op1,16) + \":\" + fr.toString(op0,16) + \" ABCDE0=\" + fr.toString(pols.A0[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.B0[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.C0[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.D0[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.E0[" + string(bFastMode?"0":"nexti") + "],16) + \" FREE0:7=\" + fr.toString(fi0,16) + \":\" + fr.toString(fi7,16) + \" addr=\" + to_string(addr));\n";
        /*code += "    zklog.info(\"<-- Completed step=\" + to_string(i) + \" zkPC=" + to_string(zkPC) +
                " op=\" + fr.toString(op7,16) + \":\" + fr.toString(op6,16) + \":\" + fr.toString(op5,16) + \":\" + fr.toString(op4,16) + \":\" + fr.toString(op3,16) + \":\" + fr.toString(op2,16) + \":\" + fr.toString(op1,16) + \":\" + fr.toString(op0,16) + \"" +
                " A=\" + fr.toString(pols.A7[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.A6[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.A5[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.A4[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.A3[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.A2[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.A1[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.A0[" + string(bFastMode?"0":"nexti") + "],16) + \"" +
                " B=\" + fr.toString(pols.B7[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.B6[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.B5[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.B4[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.B3[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.B2[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.B1[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.B0[" + string(bFastMode?"0":"nexti") + "],16) + \"" +
                " C=\" + fr.toString(pols.C7[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.C6[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.C5[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.C4[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.C3[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.C2[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.C1[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.C0[" + string(bFastMode?"0":"nexti") + "],16) + \"" +
                " D=\" + fr.toString(pols.D7[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.D6[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.D5[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.D4[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.D3[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.D2[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.D1[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.D0[" + string(bFastMode?"0":"nexti") + "],16) + \"" +
                " E=\" + fr.toString(pols.E7[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.E6[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.E5[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.E4[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.E3[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.E2[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.E1[" + string(bFastMode?"0":"nexti") + "],16) + \":\" + fr.toString(pols.E0[" + string(bFastMode?"0":"nexti") + "],16) + \"" +
                " FREE0:7=\" + fr.toString(fi0,16) + \":\" + fr.toString(fi7],16) + \" addr=\" + to_string(addr));\n";*/
#endif
#ifdef LOG_COMPLETED_STEPS_TO_FILE
        code += "    outfile.open(\"c.txt\", std::ios_base::app); // append instead of overwrite\n";
        //code += "    outfile << \"<-- Completed step=\" << i << \" zkPC=" + to_string(zkPC) + " op=\" << fr.toString(op7,16) << \":\" << fr.toString(op6,16) << \":\" << fr.toString(op5,16) << \":\" << fr.toString(op4,16) << \":\" << fr.toString(op3,16) << \":\" << fr.toString(op2,16) << \":\" << fr.toString(op1,16) << \":\" << fr.toString(op0,16) << \" ABCDE0=\" << fr.toString(pols.A0[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.B0[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.C0[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.D0[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.E0[" + string(bFastMode?"0":"nexti") + "],16) << \" FREE0:7=\" << fr.toString(fi0,16) << \":\" << fr.toString(fi7,16) << \" addr=\" << addr << endl;\n";
        /*code += "    outfile << \"<-- Completed step=\" << i << \" zkPC=" + to_string(zkPC) +
                " op=\" << fr.toString(op7,16) << \":\" << fr.toString(op6,16) << \":\" << fr.toString(op5,16) << \":\" << fr.toString(op4,16) << \":\" << fr.toString(op3,16) << \":\" << fr.toString(op2,16) << \":\" << fr.toString(op1,16) << \":\" << fr.toString(op0,16) << \"" +
                " A=\" << fr.toString(pols.A7[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.A6[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.A5[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.A4[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.A3[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.A2[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.A1[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.A0[" + string(bFastMode?"0":"nexti") + "],16) << \"" +
                " B=\" << fr.toString(pols.B7[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.B6[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.B5[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.B4[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.B3[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.B2[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.B1[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.B0[" + string(bFastMode?"0":"nexti") + "],16) << \"" +
                " C=\" << fr.toString(pols.C7[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.C6[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.C5[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.C4[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.C3[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.C2[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.C1[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.C0[" + string(bFastMode?"0":"nexti") + "],16) << \"" +
                " D=\" << fr.toString(pols.D7[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.D6[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.D5[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.D4[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.D3[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.D2[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.D1[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.D0[" + string(bFastMode?"0":"nexti") + "],16) << \"" +
                " E=\" << fr.toString(pols.E7[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.E6[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.E5[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.E4[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.E3[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.E2[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.E1[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.E0[" + string(bFastMode?"0":"nexti") + "],16) << \"" +
                " FREE0:7=\" << fr.toString(fi0,16) << \":\" << fr.toString(fi7,16) << \" addr=\" << addr << endl;\n";*/
        code += "    outfile << \"<-- Completed step=\" << i << \" zkPC=\" << " + to_string(zkPC) + " << \"" +
                " op=\" << fr.toString(op7,16) << \":\" << fr.toString(op6,16) << \":\" << fr.toString(op5,16) << \":\" << fr.toString(op4,16) << \":\" << fr.toString(op3,16) << \":\" << fr.toString(op2,16) << \":\" << fr.toString(op1,16) << \":\" << fr.toString(op0,16) << \"" +
                " A=\" << fr.toString(pols.A7[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.A6[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.A5[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.A4[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.A3[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.A2[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.A1[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.A0[" + string(bFastMode?"0":"nexti") + "],16) << \"" +
                " B=\" << fr.toString(pols.B7[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.B6[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.B5[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.B4[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.B3[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.B2[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.B1[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.B0[" + string(bFastMode?"0":"nexti") + "],16) << \"" +
                " C=\" << fr.toString(pols.C7[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.C6[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.C5[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.C4[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.C3[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.C2[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.C1[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.C0[" + string(bFastMode?"0":"nexti") + "],16) << \"" +
                " D=\" << fr.toString(pols.D7[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.D6[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.D5[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.D4[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.D3[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.D2[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.D1[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.D0[" + string(bFastMode?"0":"nexti") + "],16) << \"" +
                " E=\" << fr.toString(pols.E7[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.E6[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.E5[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.E4[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.E3[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.E2[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.E1[" + string(bFastMode?"0":"nexti") + "],16) << \":\" << fr.toString(pols.E0[" + string(bFastMode?"0":"nexti") + "],16) << \"" +
                " FREE=\" << fr.toString(pols.FREE7[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.FREE6[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.FREE5[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.FREE4[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.FREE3[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.FREE2[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.FREE1[" + string(bFastMode?"0":"i") + "],16) << \":\" << fr.toString(pols.FREE0[" + string(bFastMode?"0":"i") + "],16) << \"" +
                " addr=\" << addr << \"" +
                " RR=\" << fr.toString(pols.RR[" + string(bFastMode?"0":"nexti") + "],16) << \"" +
                " RCX=\" << fr.toString(pols.RCX[" + string(bFastMode?"0":"nexti") + "],16) << \"" +
                " HASHPOS=\" << fr.toString(pols.HASHPOS[" + string(bFastMode?"0":"nexti") + "],16) << " +
                " endl;\n";
        code += "    outfile.close();\n";
#endif

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
            code += "    nexti = (i==N_Max_minus_one) ? 0 : i+1;\n";
        code += "\n";

        /********/
        /* GOTO */
        /********/

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

    } // End of main executor loop, for all rom instructions

    code += functionName + "_end:\n\n";

    //code += "    // Copy the counters\n";
    code += "    proverRequest.counters.arith = fr.toU64(pols.cntArith[0]);\n";
    code += "    proverRequest.counters.binary = fr.toU64(pols.cntBinary[0]);\n";
    code += "    proverRequest.counters.keccakF = fr.toU64(pols.cntKeccakF[0]);\n";
    code += "    proverRequest.counters.memAlign = fr.toU64(pols.cntMemAlign[0]);\n";
    code += "    proverRequest.counters.paddingPG = fr.toU64(pols.cntPaddingPG[0]);\n";
    code += "    proverRequest.counters.poseidonG = fr.toU64(pols.cntPoseidonG[0]);\n";
    code += "    proverRequest.counters.sha256F = fr.toU64(pols.cntSha256F[0]);\n";
    code += "    proverRequest.counters.steps = ctx.lastStep;\n\n";

    code += "    proverRequest.countersReserve.arith = zkmax(proverRequest.countersReserve.arith, proverRequest.counters.arith);\n";
    code += "    proverRequest.countersReserve.binary = zkmax(proverRequest.countersReserve.binary, proverRequest.counters.binary);\n";
    code += "    proverRequest.countersReserve.keccakF = zkmax(proverRequest.countersReserve.keccakF, proverRequest.counters.keccakF);\n";
    code += "    proverRequest.countersReserve.memAlign = zkmax(proverRequest.countersReserve.memAlign, proverRequest.counters.memAlign);\n";
    code += "    proverRequest.countersReserve.paddingPG = zkmax(proverRequest.countersReserve.paddingPG, proverRequest.counters.paddingPG);\n";
    code += "    proverRequest.countersReserve.poseidonG = zkmax(proverRequest.countersReserve.poseidonG, proverRequest.counters.poseidonG);\n";
    code += "    proverRequest.countersReserve.sha256F = zkmax(proverRequest.countersReserve.sha256F, proverRequest.counters.sha256F);\n";
    code += "    proverRequest.countersReserve.steps = zkmax(proverRequest.countersReserve.steps, proverRequest.counters.steps);\n";

    //code += "    // Set the error (all previous errors generated a return)\n";
    code += "    proverRequest.result = ZKR_SUCCESS;\n";

    //code += "    // Check that we did not run out of steps during the execution\n";
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

#ifdef CHECK_MAX_CNT_AT_THE_END
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
    code += "    if (!proverRequest.input.bNoCounters && (fr.toU64(pols.cntSha256F[0]) > " + (string)rom["constants"]["MAX_CNT_SHA256_F_LIMIT"]["value"] + "))\n";
    code += "    {\n";
    code += "        proverRequest.result = ZKR_SM_MAIN_OOC_SHA256_F;\n";
    code += "        mainExecutor.logError(ctx, \"Found pols.cntSha256F[0]=\" + to_string(fr.toU64(pols.cntSha256F[0])) + \" > MAX_CNT_SHA256_F_LIMIT=" + (string)rom["constants"]["MAX_CNT_SHA256_F_LIMIT"]["value"] + "\");\n";
    if (!bFastMode)
        code += "        exitProcess();\n";
    code += "    }\n";
#endif

    if (!bFastMode) // In fast mode, last nexti was not 0 but 1, and pols have only 2 evaluations
    {
        //code += "    // Check that all registers are set to 0\n";
        code += "    mainExecutor.checkFinalState(ctx);\n";
        code += "    mainExecutor.assertOutputs(ctx);\n\n";

        //code += "    // Generate Padding KK required data\n";
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
        code += "            pHashDB->cancelBatch(proverRequest.uuid);\n";
        code += "            return;\n";
        code += "        }\n";
        code += "        h.digestCalled = ctx.hashK[i].digestCalled;\n";
        code += "        h.lenCalled = ctx.hashK[i].lenCalled;\n";
        code += "        required.PaddingKK.push_back(h);\n";
        code += "    }\n";

        //code += "    // Generate Padding PG required data\n";
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
        code += "            pHashDB->cancelBatch(proverRequest.uuid);\n";
        code += "            return;\n";
        code += "        }\n";
        code += "        h.digestCalled = ctx.hashP[i].digestCalled;\n";
        code += "        h.lenCalled = ctx.hashP[i].lenCalled;\n";
        code += "        required.PaddingPG.push_back(h);\n";
        code += "    }\n";

        //code += "    // Generate Padding SHA required data\n";
        code += "    for (uint64_t i=0; i<ctx.hashS.size(); i++)\n";
        code += "    {\n";
        code += "        PaddingSha256ExecutorInput h;\n";
        code += "        h.dataBytes = ctx.hashS[i].data;\n";
        code += "        uint64_t p = 0;\n";
        code += "        while (p<ctx.hashS[i].data.size())\n";
        code += "        {\n";
        code += "            if (ctx.hashS[i].reads[p] != 0)\n";
        code += "            {\n";
        code += "                h.reads.push_back(ctx.hashS[i].reads[p]);\n";
        code += "                p += ctx.hashS[i].reads[p];\n";
        code += "            }\n";
        code += "            else\n";
        code += "            {\n";
        code += "                h.reads.push_back(1);\n";
        code += "                p++;\n";
        code += "            }\n";
        code += "        }\n";
        code += "        if (p != ctx.hashS[i].data.size())\n";
        code += "        {\n";
        code += "            proverRequest.result = ZKR_SM_MAIN_HASHS_READ_OUT_OF_RANGE;\n";
        code += "            mainExecutor.logError(ctx, \"Reading hashS out of limits: i=\" + to_string(i) + \" p=\" + to_string(p) + \" ctx.hashS[i].data.size()=\" + to_string(ctx.hashS[i].data.size()));\n";
        code += "            pHashDB->cancelBatch(proverRequest.uuid);\n";
        code += "            return;\n";
        code += "        }\n";
        code += "        h.digestCalled = ctx.hashS[i].digestCalled;\n";
        code += "        h.lenCalled = ctx.hashS[i].lenCalled;\n";
        code += "        required.PaddingSha256.push_back(h);\n";
        code += "    }\n";
    }

    code += "    if (ctx.config.hashDBSingleton)\n";
    code += "    {\n";
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    code += "        gettimeofday(&ctx.t, NULL);\n";
#endif
    code += "        zkResult = pHashDB->flush(proverRequest.uuid, proverRequest.pFullTracer->get_new_state_root(), proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, proverRequest.flushId, proverRequest.lastSentFlushId);\n";
    code += "        if (zkResult != ZKR_SUCCESS)\n";
    code += "        {\n";
    code += "            proverRequest.result = zkResult;\n";
    code += "            mainExecutor.logError(ctx, string(\"Failed calling pHashDB->flush() result=\") + zkresult2string(zkResult));\n";
    code += "            pHashDB->cancelBatch(proverRequest.uuid);\n";
    code += "            return;\n";
    code += "        }\n";
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    code += "        ctx.mainMetrics.add(\"Flush\", TimeDiff(ctx.t));\n";
#endif
    code += "    }\n";
    code += "    else\n";
    code += "    {\n";
    code += "        proverRequest.flushId = 0;\n";
    code += "        proverRequest.lastSentFlushId = 0;\n";
    code += "    }\n";
    code += "\n";

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    code += "    if (mainExecutor.config.executorTimeStatistics)\n";
    code += "    {\n";
    code += "        ctx.mainMetrics.print(\"Main Executor calls\");\n";
    code += "        ctx.evalCommandMetrics.print(\"Main Executor eval command calls\");\n";
    code += "    }\n";
#endif
    
    code += "    if (mainExecutor.config.dbMetrics) proverRequest.dbReadLog->print();\n\n";

    code += "    zklog.info(\"" + functionName + "() done lastStep=\" + to_string(ctx.lastStep) + \" (\" + to_string((double(ctx.lastStep)*100)/N_Max) + \"%)\", &proverRequest.tags);\n\n";

    code += "    return;\n\n";

    code += "}\n\n";

    code += "#pragma GCC pop_options\n\n";

    code += "} // namespace\n\n";

    code += "\n";
    code += "#endif // MAIN_SM_EXECUTOR_GENERATED_CODE\n";

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
    //code += "    // op = op + " + inRegName + "*" + regName + ", where " + inRegName + "=" + to_string(inRegValue) + "\n";
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
    //code += "    // op0 = op0 + " + inRegName + "*" + regName + ", where " + inRegName + "=" + to_string(inRegValue) + "\n";

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
    //code += "    // op0 = op0 + CONST\n";

    string value = "";
    string valueCopy;
    if (CONST >= 0)
        value += "fr.fromU64(" + to_string(CONST) + ")";
    else
        value += "fr.neg(fr.fromU64(" + to_string(-CONST) + "))";
    valueCopy = value;
    if (opInitialized)
        value = "fr.add(op0, " + value + ")";
    code += "    op0 = " + value + ";\n";
    if (!opInitialized)
        for (uint64_t j=1; j<8; j++)
        {
            code += "    op" + to_string(j) + " = fr.zero();\n";
        }
    if (!bFastMode)
    {
        code += "    pols.CONST0[i] = " + valueCopy + ";\n\n";
    }
    code += "\n";
    return code;
}

string selectorConstL (const string &CONSTL, bool opInitialized, bool bFastMode)
{
    string code = "";
    //code += "    // op = op + CONSTL\n";
    uint64_t op[8];
    scalar2fea(CONSTL, op);

    for (uint64_t j=0; j<8; j++) // TODO: Should we ADD it, not just copy it?
    {
        if (opInitialized)
        {
            code += "    op" + to_string(j) + " = fr.add(op" + to_string(j) + ", fr.fromU64(" + to_string(op[j]) + "));\n";
        }
        else
        {
            code += "    op" + to_string(j) + " = fr.fromU64(" + to_string(op[j]) + ");\n";
        }
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

string setter8 (const string &reg, bool setReg, bool bFastMode, uint64_t zkPC, const json &rom, uint64_t forkID)
{
    string code = "";

    if (setReg)
    {
        //code += "    // " + reg + "' = op\n";
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
    else if ((forkID >= 7) && rom["labels"].contains("verifyMerkleProofEnd") && (zkPC == rom["labels"]["verifyMerkleProofEnd"]) && (reg=="C"))
    {
        code += "    if (proverRequest.input.bSkipVerifyL1InfoRoot)\n";
        code += "    {\n";
        code += "        scalar2fea(fr, proverRequest.input.publicInputsExtended.publicInputs.l1InfoRoot, pols.C0[" + string(bFastMode?"0":"nexti") + "], pols.C1[" + (bFastMode?"0":"nexti") + "], pols.C2[" + (bFastMode?"0":"nexti") + "], pols.C3[" + (bFastMode?"0":"nexti") + "], pols.C4[" + (bFastMode?"0":"nexti") + "], pols.C5[" + (bFastMode?"0":"nexti") + "], pols.C6[" + (bFastMode?"0":"nexti") + "], pols.C7[" + (bFastMode?"0":"nexti") + "] );\n";
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
        //code += "    // " + reg + "' = " + reg + "\n";
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

inline bool charIsDec (char c)
{
    if ( (c >= '0') && (c <= '9') ) return true;
    return false;
}

bool stringIsDec (const string &s)
{
    for (uint64_t i=0; i<s.size(); i++)
    {
        if (!charIsDec(s.at(i))) return false;
    }
    return true;
}

void ensureDirectoryExists (const string &fileName)
{
    string command = "[ -d " + fileName + " ] || mkdir -p " + fileName;
    int iResult = system(command.c_str());
    if (iResult != 0)
    {
        cout <<"ensureDirectoryExists() system() returned: " << iResult << endl;
        exit(-1);
    }
}