#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <gmpxx.h>
#include <unistd.h>
#include "config.hpp"
#include "main_sm/fork_8/main/main_executor.hpp"
#include "main_sm/fork_8/main/rom_line.hpp"
#include "main_sm/fork_8/main/rom_command.hpp"
#include "main_sm/fork_8/main/rom.hpp"
#include "main_sm/fork_8/main/context.hpp"
#include "main_sm/fork_8/main/eval_command.hpp"
#include "utils/time_metric.hpp"
#include "input.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "hashdb_factory.hpp"
#include "goldilocks_base_field.hpp"
#include "ffiasm/fec.hpp"
#include "ffiasm/fnec.hpp"
#include "timer.hpp"
#include "zkresult.hpp"
#include "database_map.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "poseidon_g_permutation.hpp"
#include "goldilocks_precomputed.hpp"
#include "zklog.hpp"
#include "ecrecover.hpp"
#include "sha256.hpp"


using namespace std;
using json = nlohmann::json;

namespace fork_8
{

#define STACK_OFFSET 0x10000
#define MEM_OFFSET   0x20000
#define CTX_OFFSET   0x40000

#define N_NO_COUNTERS_MULTIPLICATION_FACTOR 8

#define FrFirst32Negative ( 0xFFFFFFFF00000001 - 0xFFFFFFFF )
#define FrLast32Positive 0xFFFFFFFF

#ifdef DEBUG
#define CHECK_MAX_CNT_ASAP
#endif
#define CHECK_MAX_CNT_AT_THE_END

//#define LOG_COMPLETED_STEPS_TO_FILE
//#define LOG_COMPLETED_STEPS

MainExecutor::MainExecutor (Goldilocks &fr, PoseidonGoldilocks &poseidon, const Config &config) :
    fr(fr),
    N(MainCommitPols::pilDegree()),
    N_NoCounters(N_NO_COUNTERS_MULTIPLICATION_FACTOR*MainCommitPols::pilDegree()),
    poseidon(poseidon),
    rom(config),
#ifdef MULTI_ROM_TEST
    rom_gas_limit_100000000(config),
    rom_gas_limit_2147483647(config),
    rom_gas_limit_89128960(config),
#endif
    config(config)
{
    /* Load and parse ROM JSON file */

    TimerStart(ROM_LOAD);

    // Load file contents into a json instance
    json romJson;
    file2json("src/main_sm/fork_8/scripts/rom.json", romJson);

    // Load ROM data from JSON data
    rom.load(fr, romJson);

#ifdef MULTI_ROM_TEST
    romJson.clear();
    file2json("src/main_sm/fork_8/scripts/rom_gas_limit_100000000.json", romJson);
    rom_gas_limit_100000000.load(fr, romJson);
    romJson.clear();
    file2json("src/main_sm/fork_8/scripts/rom_gas_limit_2147483647.json", romJson);
    rom_gas_limit_2147483647.load(fr, romJson);
    romJson.clear();
    file2json("src/main_sm/fork_8/scripts/rom_gas_limit_89128960.json", romJson);
    rom_gas_limit_89128960.load(fr, romJson);
#endif

    // Get labels
    finalizeExecutionLabel     = rom.getLabel(string("finalizeExecution"));
    checkAndSaveFromLabel      = rom.getLabel(string("checkAndSaveFrom"));
    ecrecoverStoreArgsLabel    = rom.getLabel(string("ecrecover_store_args"));
    ecrecoverEndLabel          = rom.getLabel(string("ecrecover_end"));
    checkFirstTxTypeLabel      = rom.getLabel(string("checkFirstTxType"));
    writeBlockInfoRootLabel    = rom.getLabel(string("writeBlockInfoRoot"));
    verifyMerkleProofEndLabel  = rom.getLabel(string("verifyMerkleProofEnd"));
    outOfCountersStepLabel     = rom.getLabel(string("outOfCountersStep"));
    outOfCountersArithLabel    = rom.getLabel(string("outOfCountersArith"));
    outOfCountersBinaryLabel   = rom.getLabel(string("outOfCountersBinary"));
    outOfCountersKeccakLabel   = rom.getLabel(string("outOfCountersKeccak"));
    outOfCountersSha256Label   = rom.getLabel(string("outOfCountersSha256"));
    outOfCountersMemalignLabel = rom.getLabel(string("outOfCountersMemalign"));
    outOfCountersPoseidonLabel = rom.getLabel(string("outOfCountersPoseidon"));
    outOfCountersPaddingLabel  = rom.getLabel(string("outOfCountersPadding"));

    // Init labels mutex
    pthread_mutex_init(&labelsMutex, NULL);

    /* Get a HashDBInterface interface, according to the configuration */
    pHashDB = HashDBClientFactory::createHashDBClient(fr, config);
    if (pHashDB == NULL)
    {
        zklog.error("MainExecutor::MainExecutor() failed calling HashDBClientFactory::createHashDBClient()");
        exitProcess();
    }

    TimerStopAndLog(ROM_LOAD);
};

MainExecutor::~MainExecutor ()
{
    TimerStart(MAIN_EXECUTOR_DESTRUCTOR_fork_8);

    HashDBClientFactory::freeHashDBClient(pHashDB);

    TimerStopAndLog(MAIN_EXECUTOR_DESTRUCTOR_fork_8);
}

void MainExecutor::execute (ProverRequest &proverRequest, MainCommitPols &pols, MainExecRequired &required)
{
    TimerStart(MAIN_EXECUTOR_EXECUTE);

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    struct timeval t;
    TimeMetricStorage mainMetrics;
    TimeMetricStorage evalCommandMetrics;
#endif

#ifdef MULTI_ROM_TEST

    // Get the right rom based on gas limit
    Rom * pRom = &rom;
    if (proverRequest.input.debug.gasLimit == 100000000)
    {
        pRom = &rom_gas_limit_100000000;
    }
    else if (proverRequest.input.debug.gasLimit == 2147483647)
    {
        pRom = &rom_gas_limit_2147483647;
    }
    else if (proverRequest.input.debug.gasLimit == 89128960)
    {
        pRom = &rom_gas_limit_89128960;
    }
    Rom &rom = *pRom;

    // Get labels
    finalizeExecutionLabel    = rom.getLabel(string("finalizeExecution"));
    checkAndSaveFromLabel     = rom.getLabel(string("checkAndSaveFrom"));
    ecrecoverStoreArgsLabel   = rom.getLabel(string("ecrecover_store_args"));
    ecrecoverEndLabel         = rom.getLabel(string("ecrecover_end"));
    checkFirstTxTypeLabel     = rom.getLabel(string("checkFirstTxType"));
    writeBlockInfoRootLabel   = rom.getLabel(string("writeBlockInfoRoot"));
    verifyMerkleProofEndLabel = rom.getLabel(string("verifyMerkleProofEnd"));
    
#endif

    // Init execution flags
    bool bProcessBatch = (proverRequest.type == prt_processBatch);
    bool bUnsignedTransaction = (proverRequest.input.from != "") && (proverRequest.input.from != "0x");

    // Unsigned transactions (from!=empty) are intended to be used to "estimage gas" (or "call")
    // In prover mode, we cannot accept unsigned transactions, since the proof would not meet the PIL constrains
    if (bUnsignedTransaction && !bProcessBatch)
    {
        proverRequest.result = ZKR_SM_MAIN_INVALID_UNSIGNED_TX;
        zklog.error("MainExecutor::execute() failed called with bUnsignedTransaction=true but bProcessBatch=false");
        return;
    }

    // Create context and store a finite field reference in it
    Context ctx(fr, config, fec, fnec, pols, rom, proverRequest, pHashDB);

    // Init the state of the polynomials first evaluation
    initState(ctx);

#ifdef LOG_COMPLETED_STEPS_TO_FILE
    remove("c.txt");
#endif

    // Copy input database content into context database
    if (proverRequest.input.db.size() > 0)
    {
        Goldilocks::Element stateRoot[4];
        scalar2fea(fr, proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot, stateRoot);
        pHashDB->loadDB(proverRequest.input.db, true, stateRoot);
        uint64_t flushId, lastSentFlushId;
        pHashDB->flush(emptyString, emptyString, proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, flushId, lastSentFlushId);
        if (config.dbClearCache && (config.databaseURL != "local"))
        {
            pHashDB->clearCache();
        }
    }

    // Copy input contracts database content into context database (dbProgram)
    if (proverRequest.input.contractsBytecode.size() > 0)
    {
        pHashDB->loadProgramDB(proverRequest.input.contractsBytecode, true);
        uint64_t flushId, lastSentFlushId;
        pHashDB->flush(emptyString, emptyString, proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, flushId, lastSentFlushId);
        if (config.dbClearCache && (config.databaseURL != "local"))
        {
            pHashDB->clearCache();
        }
    }

    // opN are local, uncommitted polynomials
    Goldilocks::Element op0, op1, op2, op3, op4, op5, op6, op7;

    uint64_t zkPC = 0; // Zero-knowledge program counter
    uint64_t step = 0; // Step, number of polynomial evaluation
    uint64_t i; // Step, as it is used internally, set to 0 in fast mode to reuse the same evaluation all the time
    uint64_t nexti; // Next step, as it is used internally, set to 0 in fast mode to reuse the same evaluation all the time
    ctx.N = N; // Numer of evaluations
    ctx.pStep = &i; // ctx.pStep is used inside evaluateCommand() to find the current value of the registers, e.g. pols(A0)[ctx.step]
    ctx.pEvaluation = &step;
    ctx.pZKPC = &zkPC; // Pointer to the zkPC
    Goldilocks::Element currentRCX = fr.zero();

    uint64_t N_Max;
    if (proverRequest.input.bNoCounters)
    {
        if (!bProcessBatch)
        {
            proverRequest.result = ZKR_SM_MAIN_INVALID_NO_COUNTERS;
            logError(ctx, "MainExecutor::execute() found proverRequest.bNoCounters=true and bProcessBatch=false");
            return;
        }
        N_Max = N_NoCounters;
    }
    else
    {
        N_Max = N;
    }

    // This code is only used when 'skipFirstChangeL2Block = true'
    // This only is triggered when executong transaction by transaction across batches
    // This cannot be executed in prover mode
    // This code aims to set the timestamp of the batch to the one read from the state
    // Issue fixed: timestamp is set when processed a 'changeL2Block', stored on state and hold on memory.
    // Later on, 'opTIMESTAMP' loads the value hold on memory.
    // Hence, execution transaction by transaction lost track of the timestamp
    // This function aims to solve the abive issue by loading the timestamp from the state
    if (bProcessBatch && proverRequest.input.bSkipFirstChangeL2Block)
    {
        // this smt key is built with the following registers:
        // A: `0x000000000000000000000000000000005ca1ab1e` (%ADDRESS_SYSTEM)
        // B: `3` (%SMT_KEY_SC_STORAGE)
        // C: `2` (%TIMESTAMP_STORAGE_POS)
        Goldilocks::Element keyToRead[4];
        keyToRead[0] = fr.fromU64(13748230500842749409ULL);
        keyToRead[1] = fr.fromU64(4428676446262882967ULL);
        keyToRead[2] = fr.fromU64(12167292013585018040ULL);
        keyToRead[3] = fr.fromU64(12161933621946006603ULL);

        // Get old state root (current state root)
        Goldilocks::Element oldStateRoot[4];
        scalar2fea(fr, proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot, oldStateRoot);

        // Get timestamp from storage
        mpz_class timestampFromSR;
        zkresult zkr = pHashDB->get(proverRequest.uuid, oldStateRoot, keyToRead, timestampFromSR, NULL, proverRequest.dbReadLog);
        if (zkr != ZKR_SUCCESS)
        {
            proverRequest.result = zkr;
            logError(ctx, string("Copying timestamp from state to memory, failed calling pHashDB->get() result=") + zkresult2string(zkr) + " key=" + fea2string(fr, keyToRead));
            pHashDB->cancelBatch(proverRequest.uuid);
            return;
        }

        // Pre-load memory with this timestamp value
        Fea fea;
        scalar2fea(fr, timestampFromSR, fea.fe0, fea.fe1, fea.fe2, fea.fe3, fea.fe4, fea.fe5, fea.fe6, fea.fe7);
        ctx.mem[rom.timestampOffset] = fea;
    }

    for (step=0; step<N_Max; step++)
    {
        if (bProcessBatch)
        {
            i = 0;
            nexti = 0;
            pols.FREE0[i] = fr.zero();
            pols.FREE1[i] = fr.zero();
            pols.FREE2[i] = fr.zero();
            pols.FREE3[i] = fr.zero();
            pols.FREE4[i] = fr.zero();
            pols.FREE5[i] = fr.zero();
            pols.FREE6[i] = fr.zero();
            pols.FREE7[i] = fr.zero();
        }
        else
        {
            i = step;
            // Calculate nexti to write the next evaluation register values according to setX
            // The registers of the evaluation 0 will be overwritten with the values from the last evaluation, closing the evaluation circle
            nexti = (i+1)%N;
        }

        zkPC = fr.toU64(pols.zkPC[i]); // This is the read line of ZK code

        uint64_t incHashPos = 0;
        uint64_t incCounter = 0;

#ifdef LOG_START_STEPS
        zklog.info("--> Starting step=" + to_string(step) + " zkPC=" + to_string(zkPC) + " zkasm=" + rom.line[zkPC].lineStr);
#endif
        if (config.executorROMLineTraces)
        {
            zklog.info("step=" + to_string(step) + " rom.line[" + to_string(zkPC) + "] =[" + rom.line[zkPC].toString(fr) + "]");
        }
#ifdef LOG_START_STEPS_TO_FILE
        {
        std::ofstream outfile;
        outfile.open("c.txt", std::ios_base::app); // append instead of overwrite
        outfile << "--> Starting step=" << step << " zkPC=" << zkPC << " instruction= " << rom.line[zkPC].toString(fr) << endl;
        outfile.close();
        }
#endif

        if (zkPC == ecrecoverStoreArgsLabel && config.ECRecoverPrecalc)
        {
            zkassert(ctx.ecRecoverPrecalcBuffer.filled == false);
            mpz_class signature_, r_, s_, v_;
            fea2scalar(fr, signature_, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
            fea2scalar(fr, r_, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
            fea2scalar(fr, s_, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]);
            fea2scalar(fr, v_, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]);
            ctx.ecRecoverPrecalcBuffer.posUsed = ECRecoverPrecalc(signature_, r_, s_, v_, false, ctx.ecRecoverPrecalcBuffer.buffer, ctx.config.ECRecoverPrecalcNThreads);
            ctx.ecRecoverPrecalcBuffer.pos = 0;
            if (ctx.ecRecoverPrecalcBuffer.posUsed > 0)
            {
                ctx.ecRecoverPrecalcBuffer.filled = true;
            }
        }
        if (zkPC == ecrecoverEndLabel)
        {
            if ( ctx.ecRecoverPrecalcBuffer.filled)
            {
                zkassert(ctx.ecRecoverPrecalcBuffer.pos == ctx.ecRecoverPrecalcBuffer.posUsed);
                ctx.ecRecoverPrecalcBuffer.filled = false;
            }
        }

        // Consolidate the state and store it in SR, just before we save SR into SMT
        if (config.hashDB64 && bProcessBatch && (zkPC == consolidateStateRootZKPC))
        {
            // Convert pols.SR to virtualStateRoot fea
            Goldilocks::Element virtualStateRoot[4];
            if (!fea2fea(virtualStateRoot, pols.SR0[i], pols.SR1[i], pols.SR2[i], pols.SR3[i], pols.SR4[i], pols.SR5[i], pols.SR6[i], pols.SR7[i]))
            {
                proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                logError(ctx, string("Failed calling fea2fea()"));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            // Call purge()
            zkresult zkr = pHashDB->purge(proverRequest.uuid, virtualStateRoot, proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE);
            if (zkr != ZKR_SUCCESS)
            {
                proverRequest.result = zkr;
                logError(ctx, string("Failed calling pHashDB->purge() result=") + zkresult2string(zkr));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            // Call consolidateState()
            Goldilocks::Element consolidatedStateRoot[4];
            uint64_t flushId, storedFlushId;
            zkr = pHashDB->consolidateState(virtualStateRoot, proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE , consolidatedStateRoot, flushId, storedFlushId);
            if (zkr != ZKR_SUCCESS)
            {
                proverRequest.result = zkr;
                logError(ctx, string("Failed calling pHashDB->consolidateState() result=") + zkresult2string(proverRequest.result));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            // Convert consolidatedState fea to pols.SR
            fea2fea(pols.SR0[i], pols.SR1[i], pols.SR2[i], pols.SR3[i], pols.SR4[i], pols.SR5[i], pols.SR6[i], pols.SR7[i], consolidatedStateRoot);
            //zklog.info("SR=" + fea2string(fr, pols.SR0[i], pols.SR1[i], pols.SR2[i], pols.SR3[i], pols.SR4[i], pols.SR5[i], pols.SR6[i], pols.SR7[i]));
        }

#ifdef LOG_FILENAME
        // Store fileName and line
        ctx.fileName = rom.line[zkPC].fileName;
        ctx.line = rom.line[zkPC].line;
#endif

        // Evaluate the list cmdBefore commands, and any children command, recursively
        for (uint64_t j=0; j<rom.line[zkPC].cmdBefore.size(); j++)
        {
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
            gettimeofday(&t, NULL);
#endif
            CommandResult cr;
            evalCommand(ctx, *rom.line[zkPC].cmdBefore[j], cr);

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
            mainMetrics.add("Eval command", TimeDiff(t));
            evalCommandMetrics.add(rom.line[zkPC].cmdBefore[j]->opAndFunction, TimeDiff(t));
#endif
            // In case of an external error, return it
            if (cr.zkResult != ZKR_SUCCESS)
            {
                proverRequest.result = cr.zkResult;
                logError(ctx, string("Failed calling evalCommand() before, result=") + zkresult2string(proverRequest.result));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
        }

        // Initialize the local registers to zero
        op0 = fr.zero();
        op1 = fr.zero();
        op2 = fr.zero();
        op3 = fr.zero();
        op4 = fr.zero();
        op5 = fr.zero();
        op6 = fr.zero();
        op7 = fr.zero();

        /*************/
        /* SELECTORS */
        /*************/

        // inX adds the corresponding register values to the op local register set, multiplied by inX
        // In case several inXs are set to !=0, those values will be added together to opN
        // e.g. op0 = inX*X0 + inY*Y0 + inZ*Z0 +...

        // If inA, op = op + inA*A
        if (!fr.isZero(rom.line[zkPC].inA))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inA, pols.A0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inA, pols.A1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inA, pols.A2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inA, pols.A3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inA, pols.A4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inA, pols.A5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inA, pols.A6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inA, pols.A7[i]));

            pols.inA[i] = rom.line[zkPC].inA;

#ifdef LOG_INX
            zklog.info("inA op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inB, op = op + inB*B
        if (!fr.isZero(rom.line[zkPC].inB))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inB, pols.B0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inB, pols.B1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inB, pols.B2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inB, pols.B3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inB, pols.B4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inB, pols.B5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inB, pols.B6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inB, pols.B7[i]));

            pols.inB[i] = rom.line[zkPC].inB;

#ifdef LOG_INX
            zklog.info("inB op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inA, op = op + inA*A
        if (!fr.isZero(rom.line[zkPC].inC))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inC, pols.C0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inC, pols.C1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inC, pols.C2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inC, pols.C3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inC, pols.C4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inC, pols.C5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inC, pols.C6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inC, pols.C7[i]));

            pols.inC[i] = rom.line[zkPC].inC;

#ifdef LOG_INX
            zklog.info("inC op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inD, op = op + inD*D
        if (!fr.isZero(rom.line[zkPC].inD))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inD, pols.D0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inD, pols.D1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inD, pols.D2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inD, pols.D3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inD, pols.D4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inD, pols.D5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inD, pols.D6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inD, pols.D7[i]));

            pols.inD[i] = rom.line[zkPC].inD;

#ifdef LOG_INX
            zklog.info("inD op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inE, op = op + inE*E
        if (!fr.isZero(rom.line[zkPC].inE))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inE, pols.E0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inE, pols.E1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inE, pols.E2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inE, pols.E3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inE, pols.E4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inE, pols.E5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inE, pols.E6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inE, pols.E7[i]));

            pols.inE[i] = rom.line[zkPC].inE;

#ifdef LOG_INX
            zklog.info("inE op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inSR, op = op + inSR*SR
        if (!fr.isZero(rom.line[zkPC].inSR))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inSR, pols.SR0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inSR, pols.SR1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inSR, pols.SR2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inSR, pols.SR3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inSR, pols.SR4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inSR, pols.SR5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inSR, pols.SR6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inSR, pols.SR7[i]));

            pols.inSR[i] = rom.line[zkPC].inSR;

#ifdef LOG_INX
            zklog.info("inSR op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inCTX, op = op + inCTX*CTX
        if (!fr.isZero(rom.line[zkPC].inCTX))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCTX, pols.CTX[i]));
            pols.inCTX[i] = rom.line[zkPC].inCTX;
#ifdef LOG_INX
            zklog.info("inCTX op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inSP, op = op + inSP*SP
        if (!fr.isZero(rom.line[zkPC].inSP))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inSP, pols.SP[i]));
            pols.inSP[i] = rom.line[zkPC].inSP;
#ifdef LOG_INX
            zklog.info("inSP op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inPC, op = op + inPC*PC
        if (!fr.isZero(rom.line[zkPC].inPC))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inPC, pols.PC[i]));
            pols.inPC[i] = rom.line[zkPC].inPC;
#ifdef LOG_INX
            zklog.info("inPC op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inGAS, op = op + inGAS*GAS
        if (!fr.isZero(rom.line[zkPC].inGAS))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inGAS, pols.GAS[i]));
            pols.inGAS[i] = rom.line[zkPC].inGAS;
#ifdef LOG_INX
            zklog.info("inGAS op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inSTEP, op = op + inSTEP*STEP
        if (!fr.isZero(rom.line[zkPC].inSTEP))
        {
            op0 = fr.add(op0, fr.mul( rom.line[zkPC].inSTEP, fr.fromU64(proverRequest.input.bNoCounters ? 0 : step) ));
            pols.inSTEP[i] = rom.line[zkPC].inSTEP;
#ifdef LOG_INX
            zklog.info("inSTEP op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inRR, op = op + inRR*RR
        if (!fr.isZero(rom.line[zkPC].inRR))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inRR, pols.RR[i]));
            pols.inRR[i] = rom.line[zkPC].inRR;
#ifdef LOG_INX
            zklog.info("inRR op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inHASHPOS, op = op + inHASHPOS*HASHPOS
        if (!fr.isZero(rom.line[zkPC].inHASHPOS))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inHASHPOS, pols.HASHPOS[i]));
            pols.inHASHPOS[i] = rom.line[zkPC].inHASHPOS;
#ifdef LOG_INX
            zklog.info("inHASHPOS op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inCntArith, op = op + inCntArith*cntArith
        if (!fr.isZero(rom.line[zkPC].inCntArith))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCntArith, pols.cntArith[i]));
            pols.inCntArith[i] = rom.line[zkPC].inCntArith;
#ifdef LOG_INX
            zklog.info("inCntArith op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inCntBinary, op = op + inCntBinary*cntBinary
        if (!fr.isZero(rom.line[zkPC].inCntBinary))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCntBinary, pols.cntBinary[i]));
            pols.inCntBinary[i] = rom.line[zkPC].inCntBinary;
#ifdef LOG_INX
            zklog.info("inCntBinary op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inCntMemAlign, op = op + inCntMemAlign*cntMemAlign
        if (!fr.isZero(rom.line[zkPC].inCntMemAlign))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCntMemAlign, pols.cntMemAlign[i]));
            pols.inCntMemAlign[i] = rom.line[zkPC].inCntMemAlign;
#ifdef LOG_INX
            zklog.info("inCntMemAlign op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inCntKeccakF, op = op + inCntKeccakF*cntKeccakF
        if (!fr.isZero(rom.line[zkPC].inCntKeccakF))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCntKeccakF, pols.cntKeccakF[i]));
            pols.inCntKeccakF[i] = rom.line[zkPC].inCntKeccakF;
#ifdef LOG_INX
            zklog.info("inCntKeccakF op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inCntSha256F, op = op + inCntSha256F*cntSha256F
        if (!fr.isZero(rom.line[zkPC].inCntSha256F))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCntSha256F, pols.cntSha256F[i]));
            pols.inCntSha256F[i] = rom.line[zkPC].inCntSha256F;
#ifdef LOG_INX
            zklog.info("inCntSha256F op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inCntPoseidonG, op = op + inCntPoseidonG*cntPoseidonG
        if (!fr.isZero(rom.line[zkPC].inCntPoseidonG))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCntPoseidonG, pols.cntPoseidonG[i]));
            pols.inCntPoseidonG[i] = rom.line[zkPC].inCntPoseidonG;
#ifdef LOG_INX
            zklog.info("inCntPoseidonG op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inCntPaddingPG, op = op + inCntPaddingPG*cntPaddingPG
        if (!fr.isZero(rom.line[zkPC].inCntPaddingPG))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCntPaddingPG, pols.cntPaddingPG[i]));
            pols.inCntPaddingPG[i] = rom.line[zkPC].inCntPaddingPG;
#ifdef LOG_INX
            zklog.info("inCntPaddingPG op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inROTL_C, op = C rotated left
        if (!fr.isZero(rom.line[zkPC].inROTL_C))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inROTL_C, pols.C7[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inROTL_C, pols.C0[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inROTL_C, pols.C1[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inROTL_C, pols.C2[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inROTL_C, pols.C3[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inROTL_C, pols.C4[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inROTL_C, pols.C5[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inROTL_C, pols.C6[i]));

            pols.inROTL_C[i] = rom.line[zkPC].inROTL_C;
        }

        // If inRCX, op = op + inRCX*RCS
        if (!fr.isZero(rom.line[zkPC].inRCX))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inRCX, pols.RCX[i]));
            pols.inRCX[i] = rom.line[zkPC].inRCX;
#ifdef LOG_INX
            zklog.info("inCntPaddingPG op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // If inCONST, op = op + CONST
        if (rom.line[zkPC].bConstLPresent)
        {
            scalar2fea(fr, rom.line[zkPC].CONSTL, pols.CONST0[i], pols.CONST1[i], pols.CONST2[i], pols.CONST3[i], pols.CONST4[i], pols.CONST5[i], pols.CONST6[i], pols.CONST7[i]);
            op0 = fr.add(op0, pols.CONST0[i]);
            op1 = fr.add(op1, pols.CONST1[i]);
            op2 = fr.add(op2, pols.CONST2[i]);
            op3 = fr.add(op3, pols.CONST3[i]);
            op4 = fr.add(op4, pols.CONST4[i]);
            op5 = fr.add(op5, pols.CONST5[i]);
            op6 = fr.add(op6, pols.CONST6[i]);
            op7 = fr.add(op7, pols.CONST7[i]);
#ifdef LOG_INX
            zklog.info("CONSTL op=" + rom.line[zkPC].CONSTL.get_str(16));
#endif
        }
        else if (rom.line[zkPC].bConstPresent)
        {
            op0 = fr.add(op0, rom.line[zkPC].CONST);
            pols.CONST0[i] = rom.line[zkPC].CONST;
#ifdef LOG_INX
            zklog.info("CONST op=" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0, 16));
#endif
        }

        // Relative and absolute address auxiliary variables
        int32_t addrRel = 0;
        uint64_t addr = 0;

        // If address is involved, load offset into addr
        if (rom.line[zkPC].mOp==1 ||
            rom.line[zkPC].mWR==1 ||
            rom.line[zkPC].hashK==1 ||
            rom.line[zkPC].hashK1==1 ||
            rom.line[zkPC].hashKLen==1 ||
            rom.line[zkPC].hashKDigest==1 ||
            rom.line[zkPC].hashP==1 ||
            rom.line[zkPC].hashP1==1 ||
            rom.line[zkPC].hashPLen==1 ||
            rom.line[zkPC].hashPDigest==1 ||
            rom.line[zkPC].hashS==1 ||
            rom.line[zkPC].hashS1==1 ||
            rom.line[zkPC].hashSLen==1 ||
            rom.line[zkPC].hashSDigest==1 ||
            rom.line[zkPC].JMP==1 ||
            rom.line[zkPC].JMPN==1 ||
            rom.line[zkPC].JMPC==1 ||
            rom.line[zkPC].JMPZ==1 ||
            rom.line[zkPC].call==1)
        {
            if (rom.line[zkPC].ind == 1)
            {
                if (!fr.toS32(addrRel, pols.E0[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_TOS32;
                    logError(ctx, "Failed calling fr.toS32() with pols.E0[i]=" + fr.toString(pols.E0[i], 16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
            }
            if (rom.line[zkPC].indRR == 1)
            {
                if (!fr.toS32(addrRel, pols.RR[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_TOS32;
                    logError(ctx, "Failed calling fr.toS32() with pols.RR[i]=" + fr.toString(pols.RR[i], 16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
            }
            if (rom.line[zkPC].bOffsetPresent && rom.line[zkPC].offset!=0)
            {
                addrRel += rom.line[zkPC].offset;
            }
            if (rom.line[zkPC].isStack == 1)
            {
                int32_t sp;
                if (!fr.toS32(sp, pols.SP[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_TOS32;
                    logError(ctx, "failed calling fr.toS32(sp, pols.SP[i])=" + fr.toString(pols.SP[i], 16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                addrRel += sp;
            }
            // Check addrRel is not too big
            if ( addrRel >= ( ( (rom.line[zkPC].isMem==1) ? 0x20000 : 0x10000) - 2048 ) )

            {
                proverRequest.result = ZKR_SM_MAIN_ADDRESS_OUT_OF_RANGE;
                logError(ctx, "addrRel too big addrRel=" + to_string(addrRel));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            // If addrRel is negative, fail
            if (addrRel < 0)
            {
                proverRequest.result = ZKR_SM_MAIN_ADDRESS_NEGATIVE;
                logError(ctx, "addrRel<0 addrRel=" + to_string(addrRel));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            addr = addrRel;
#ifdef LOG_ADDR
            zklog.info("Any addr=" + to_string(addr));
#endif
        }

        // If useCTX, addr = addr + CTX*CTX_OFFSET
        if (rom.line[zkPC].useCTX == 1)
        {
            addr += fr.toU64(pols.CTX[i])*CTX_OFFSET;
            pols.useCTX[i] = fr.one();
#ifdef LOG_ADDR
            zklog.info("useCTX addr=" + to_string(addr));
#endif
        }

        // If isStack, addr = addr + STACK_OFFSET
        if (rom.line[zkPC].isStack == 1)
        {
            addr += STACK_OFFSET;
            pols.isStack[i] = fr.one();
#ifdef LOG_ADDR
            zklog.info("isStack addr=" + to_string(addr));
#endif
        }

        // If isMem, addr = addr + MEM_OFFSET
        if (rom.line[zkPC].isMem == 1)
        {
            addr += MEM_OFFSET;
            pols.isMem[i] = fr.one();
#ifdef LOG_ADDR
            zklog.info("isMem addr=" + to_string(addr));
#endif
        }

        // Copy ROM flags into the polynomials
        if (rom.line[zkPC].incStack != 0)
        {
            pols.incStack[i] = fr.fromS32(rom.line[zkPC].incStack);
        }
        if (rom.line[zkPC].ind == 1)
        {
            pols.ind[i] = fr.one();
        }
        if (rom.line[zkPC].indRR == 1)
        {
            pols.indRR[i] = fr.one();
        }

        // If offset, record it the committed polynomial
        if (rom.line[zkPC].bOffsetPresent && (rom.line[zkPC].offset!=0))
        {
            pols.offset[i] = fr.fromS32(rom.line[zkPC].offset);
        }

        /**************/
        /* FREE INPUT */
        /**************/

        // If inFREE or inFREE0, calculate the free input value, and add it to op
        if (!fr.isZero(rom.line[zkPC].inFREE) || !fr.isZero(rom.line[zkPC].inFREE0))
        {
            // freeInTag must be present
            if (rom.line[zkPC].freeInTag.isPresent == false)
            {
                logError(ctx, "Instruction with freeIn without freeInTag");
                exitProcess();
            }

            // Store free value here, and add it to op later
            Goldilocks::Element fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7;

            // If there is no operation specified in freeInTag.op, then get the free value directly from the corresponding source
            if (rom.line[zkPC].freeInTag.op == op_empty) {
                uint64_t nHits = 0;

                // Memory read free in: get fi=mem[addr], if it exists
                if ( (rom.line[zkPC].mOp==1) && (rom.line[zkPC].mWR==0) )
                {
                    std::unordered_map<uint64_t, Fea>::iterator memIterator;
                    memIterator = ctx.mem.find(addr);
                    if (memIterator != ctx.mem.end()) {
#ifdef LOG_MEMORY
                        zklog.info("Memory read mRD: addr:" + to_string(addr) + " " + fea2string(fr, ctx.mem[addr].fe0, ctx.mem[addr].fe1, ctx.mem[addr].fe2, ctx.mem[addr].fe3, ctx.mem[addr].fe4, ctx.mem[addr].fe5, ctx.mem[addr].fe6, ctx.mem[addr].fe7));
#endif
                        fi0 = memIterator->second.fe0;
                        fi1 = memIterator->second.fe1;
                        fi2 = memIterator->second.fe2;
                        fi3 = memIterator->second.fe3;
                        fi4 = memIterator->second.fe4;
                        fi5 = memIterator->second.fe5;
                        fi6 = memIterator->second.fe6;
                        fi7 = memIterator->second.fe7;

                    } else {
                        fi0 = fr.zero();
                        fi1 = fr.zero();
                        fi2 = fr.zero();
                        fi3 = fr.zero();
                        fi4 = fr.zero();
                        fi5 = fr.zero();
                        fi6 = fr.zero();
                        fi7 = fr.zero();
                    }
                    nHits++;
                }

                // Storage read free in: get a poseidon hash, and read fi=sto[hash]
                if (rom.line[zkPC].sRD == 1)
                {
                    zkresult zkResult;

                    // Check the range of the registers A and B
                    if  ( !fr.isZero(pols.A5[i]) || !fr.isZero(pols.A6[i]) || !fr.isZero(pols.A7[i]) || !fr.isZero(pols.B2[i]) || !fr.isZero(pols.B3[i]) || !fr.isZero(pols.B4[i]) || !fr.isZero(pols.B5[i])|| !fr.isZero(pols.B6[i])|| !fr.isZero(pols.B7[i]) )
                    {
                        proverRequest.result = ZKR_SM_MAIN_STORAGE_INVALID_KEY;
                        logError(ctx, "Storage read free in found non-zero A-B storage registers");
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }

                    // Get old state root
                    Goldilocks::Element oldRoot[4];
                    sr8to4(fr, pols.SR0[i], pols.SR1[i], pols.SR2[i], pols.SR3[i], pols.SR4[i], pols.SR5[i], pols.SR6[i], pols.SR7[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);

                    // Track if we used the state override or not
                    bool bStateOverride = false;
                    mpz_class value;
                    SmtGetResult smtGetResult;

                    // If the input contains a state override section, then use it
                    if (!proverRequest.input.stateOverride.empty())
                    {
                        // Get the key address
                        mpz_class auxScalar;
                        if (!fea2scalar(fr, auxScalar, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar()");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        string keyAddress = NormalizeTo0xNFormat(auxScalar.get_str(16), 40);

                        // Get the key type
                        if (!fea2scalar(fr, auxScalar, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar()");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        uint64_t keyType = auxScalar.get_ui();

                        // Get the key storage
                        if (!fea2scalar(fr, auxScalar, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar()");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        string keyStorage = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

                        unordered_map<string, OverrideEntry>::const_iterator it;
                        it = proverRequest.input.stateOverride.find(keyAddress);
                        if (it != proverRequest.input.stateOverride.end())
                        {
                            if ((keyType == rom.constants.SMT_KEY_BALANCE) && it->second.bBalance)
                            {
                                value = it->second.balance;
                                bStateOverride = true;
                            }
                            else if ((keyType == rom.constants.SMT_KEY_NONCE) && (it->second.nonce > 0))
                            {
                                value = it->second.nonce;
                                bStateOverride = true;
                            }
                            else if ((keyType == rom.constants.SMT_KEY_SC_CODE) && (it->second.code.size() > 0))
                            {
                                // Calculate the linear poseidon hash
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                                gettimeofday(&t, NULL);
#endif
                                Goldilocks::Element result[4];
                                linearPoseidon(ctx, it->second.code, result);
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                                mainMetrics.add("Poseidon", TimeDiff(t));
#endif
                                // Convert to scalar
                                fea2scalar(fr, value, result);

                                bStateOverride = true;
                            }
                            else if ((keyType == rom.constants.SMT_KEY_SC_LENGTH) && (it->second.code.size() > 0))
                            {
                                value = it->second.code.size();
                                bStateOverride = true;
                            }
                            else if ((keyType == rom.constants.SMT_KEY_SC_STORAGE) && (it->second.state.size() > 0))
                            {
                                unordered_map<string, mpz_class>::const_iterator itState;
                                itState = it->second.state.find(keyStorage);
                                if (itState != it->second.state.end())
                                {
                                    value = itState->second;
                                }
                                else
                                {
                                    value = 0;
                                }
                                bStateOverride = true;
                            }
                            else if ((keyType == rom.constants.SMT_KEY_SC_STORAGE) && (it->second.stateDiff.size() > 0))
                            {
                                unordered_map<string, mpz_class>::const_iterator itState;
                                itState = it->second.stateDiff.find(keyStorage);
                                if (itState != it->second.stateDiff.end())
                                {
                                    value = itState->second;
                                }
                                else
                                {
                                    value = 0;
                                }
                                bStateOverride = true;
                            }
                        }
                    }

                    if (bStateOverride)
                    {
                        smtGetResult.value = value;

#ifdef LOG_SMT_KEY_DETAILS
                    zklog.info("SMT get state override C=" + fea2string(fr, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]) +
                        " A=" + fea2string(fr, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]) +
                        " B=" + fea2string(fr, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]) +
                        " oldRoot=" + fea2string(fr, oldRoot) +
                        " value=" + value.get_str(10));
#endif
                    }
                    else
                    {
                        Goldilocks::Element Kin0[12];
                        Kin0[0] = pols.C0[i];
                        Kin0[1] = pols.C1[i];
                        Kin0[2] = pols.C2[i];
                        Kin0[3] = pols.C3[i];
                        Kin0[4] = pols.C4[i];
                        Kin0[5] = pols.C5[i];
                        Kin0[6] = pols.C6[i];
                        Kin0[7] = pols.C7[i];
                        Kin0[8] = fr.zero();
                        Kin0[9] = fr.zero();
                        Kin0[10] = fr.zero();
                        Kin0[11] = fr.zero();

                        Goldilocks::Element Kin1[12];
                        Kin1[0] = pols.A0[i];
                        Kin1[1] = pols.A1[i];
                        Kin1[2] = pols.A2[i];
                        Kin1[3] = pols.A3[i];
                        Kin1[4] = pols.A4[i];
                        Kin1[5] = pols.A5[i];
                        Kin1[6] = pols.B0[i];
                        Kin1[7] = pols.B1[i];

                        uint64_t b0 = fr.toU64(pols.B0[i]);
                        bool bIsTouchedAddressTree = (b0 == 5) || (b0 == 6);

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                        gettimeofday(&t, NULL);
#endif
                        // Call poseidon and get the hash key
                        Goldilocks::Element Kin0Hash[4];
                        poseidon.hash(Kin0Hash, Kin0);

                        // Reinject the first resulting hash as the capacity for the next poseidon hash
                        Kin1[8] = Kin0Hash[0];
                        Kin1[9] = Kin0Hash[1];
                        Kin1[10] = Kin0Hash[2];
                        Kin1[11] = Kin0Hash[3];

                        // Call poseidon hash
                        Goldilocks::Element Kin1Hash[4];
                        poseidon.hash(Kin1Hash, Kin1);

                        Goldilocks::Element key[4];
                        key[0] = Kin1Hash[0];
                        key[1] = Kin1Hash[1];
                        key[2] = Kin1Hash[2];
                        key[3] = Kin1Hash[3];
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                        mainMetrics.add("Poseidon", TimeDiff(t), 3);
#endif

#ifdef LOG_STORAGE
                        zklog.info("Storage read sRD got poseidon key: " + ctx.fr.toString(ctx.lastSWrite.key, 16));
#endif

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                        gettimeofday(&t, NULL);
#endif
                        // Collect the keys used to read or write store data
                        if (proverRequest.input.bGetKeys && !bIsTouchedAddressTree)
                        {
                            proverRequest.nodesKeys.insert(fea2string(fr, key));
                        }

                        zkResult = pHashDB->get(proverRequest.uuid, oldRoot, key, value, &smtGetResult, proverRequest.dbReadLog);
                        if (zkResult != ZKR_SUCCESS)
                        {
                            proverRequest.result = zkResult;
                            logError(ctx, string("Failed calling pHashDB->get() result=") + zkresult2string(zkResult) + " key=" + fea2string(fr, key));
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        incCounter = smtGetResult.proofHashCounter + 2;

#ifdef LOG_SMT_KEY_DETAILS
                        zklog.info("SMT get C=" + fea2string(fr, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]) +
                            " A=" + fea2string(fr, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]) +
                            " B=" + fea2string(fr, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]) +
                            " Kin0Hash=" + fea2string(fr, Kin0Hash) +
                            " Kin1Hash=" + fea2string(fr, Kin1Hash) +
                            " oldRoot=" + fea2string(fr, oldRoot) +
                            " value=" + value.get_str(10));
#endif

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                        mainMetrics.add("SMT Get", TimeDiff(t));
#endif

                        if (bProcessBatch)
                        {
                            zkResult = eval_addReadWriteAddress(ctx, smtGetResult.value, key);
                            if (zkResult != ZKR_SUCCESS)
                            {
                                proverRequest.result = zkResult;
                                logError(ctx, string("Failed calling eval_addReadWriteAddress() 1 result=") + zkresult2string(zkResult));
                                pHashDB->cancelBatch(proverRequest.uuid);
                                return;
                            }
                        }
                    }

                    scalar2fea(fr, smtGetResult.value, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);

                    nHits++;
#ifdef LOG_STORAGE
                    zklog.info("Storage read sRD read from key: " + ctx.fr.toString(ctx.lastSWrite.key, 16) + " value:" + fr.toString(fi3, 16) + ":" + fr.toString(fi2, 16) + ":" + fr.toString(fi1, 16) + ":" + fr.toString(fi0, 16));
#endif
                }

                // Storage write free in: calculate the poseidon hash key, check its entry exists in storage, and update new root hash
                if (rom.line[zkPC].sWR == 1)
                {
                    zkresult zkResult;

                    // Check the range of the registers A and B
                    if  ( !fr.isZero(pols.A5[i]) || !fr.isZero(pols.A6[i]) || !fr.isZero(pols.A7[i]) || !fr.isZero(pols.B2[i]) || !fr.isZero(pols.B3[i]) || !fr.isZero(pols.B4[i]) || !fr.isZero(pols.B5[i])|| !fr.isZero(pols.B6[i])|| !fr.isZero(pols.B7[i]) )
                    {
                        proverRequest.result = ZKR_SM_MAIN_STORAGE_INVALID_KEY;
                        logError(ctx, "Storage write free in found non-zero A-B registers");
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }
                    
                    // Get old state root
                    Goldilocks::Element oldRoot[4];
                    sr8to4(fr, pols.SR0[i], pols.SR1[i], pols.SR2[i], pols.SR3[i], pols.SR4[i], pols.SR5[i], pols.SR6[i], pols.SR7[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);

                    // Get the value to write
                    mpz_class value;
                    if (!fea2scalar(fr, value, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]))
                    {
                        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                        logError(ctx, "Failed calling fea2scalar()");
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }

                    // Track if we used the state override or not
                    bool bStatusOverride = false;

                    // If the input contains a state override section, then use it
                    if (!proverRequest.input.stateOverride.empty())
                    {
                        // Get the key address
                        mpz_class auxScalar;
                        if (!fea2scalar(fr, auxScalar, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar()");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        string keyAddress = NormalizeTo0xNFormat(auxScalar.get_str(16), 40);

                        // Get the key type
                        if (!fea2scalar(fr, auxScalar, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar()");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        uint64_t keyType = auxScalar.get_ui();

                        // Get the key storage
                        if (!fea2scalar(fr, auxScalar, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar()");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        string keyStorage = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

                        unordered_map<string, OverrideEntry>::iterator it;
                        it = proverRequest.input.stateOverride.find(keyAddress);
                        if (it != proverRequest.input.stateOverride.end())
                        {
                            if ((keyType == rom.constants.SMT_KEY_BALANCE) && it->second.bBalance)
                            {
                                it->second.balance = value;
                                it->second.bBalance = true;
                                bStatusOverride = true;
                            }
                            else if ((keyType == rom.constants.SMT_KEY_NONCE) && (it->second.nonce > 0))
                            {
                                it->second.nonce = value.get_ui();
                                bStatusOverride = true;
                            }
                            else if ((keyType == rom.constants.SMT_KEY_SC_CODE) && (it->second.code.size() > 0))
                            {
                                it->second.code = proverRequest.input.contractsBytecode[NormalizeTo0xNFormat(value.get_str(16), 64)];
                                bStatusOverride = true;
                            }
                            else if ((keyType == rom.constants.SMT_KEY_SC_STORAGE) && (it->second.state.size() > 0))
                            {
                                it->second.state[keyStorage] = value;
                                bStatusOverride = true;
                            }
                            else if ((keyType == rom.constants.SMT_KEY_SC_STORAGE) && (it->second.stateDiff.size() > 0))
                            {
                                it->second.stateDiff[keyStorage] = value;
                                bStatusOverride = true;
                            }
                        }
                    }

                    if (bStatusOverride)
                    {

#ifdef LOG_SMT_KEY_DETAILS
                    zklog.info("SMT set state override C=" + fea2string(fr, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]) +
                        " A=" + fea2string(fr, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]) +
                        " B=" + fea2string(fr, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]) +
                        " oldRoot=" + fea2string(fr, oldRoot) +
                        " value=" + value.get_str(10));
#endif
                    }
                    else
                    {
                        // reset lastSWrite
                        ctx.lastSWrite.reset();
                        Goldilocks::Element Kin0[12];
                        Kin0[0] = pols.C0[i];
                        Kin0[1] = pols.C1[i];
                        Kin0[2] = pols.C2[i];
                        Kin0[3] = pols.C3[i];
                        Kin0[4] = pols.C4[i];
                        Kin0[5] = pols.C5[i];
                        Kin0[6] = pols.C6[i];
                        Kin0[7] = pols.C7[i];
                        Kin0[8] = fr.zero();
                        Kin0[9] = fr.zero();
                        Kin0[10] = fr.zero();
                        Kin0[11] = fr.zero();

                        Goldilocks::Element Kin1[12];
                        Kin1[0] = pols.A0[i];
                        Kin1[1] = pols.A1[i];
                        Kin1[2] = pols.A2[i];
                        Kin1[3] = pols.A3[i];
                        Kin1[4] = pols.A4[i];
                        Kin1[5] = pols.A5[i];
                        Kin1[6] = pols.B0[i];
                        Kin1[7] = pols.B1[i];

                        uint64_t b0 = fr.toU64(pols.B0[i]);
                        bool bIsTouchedAddressTree = (b0 == 5) || (b0 == 6);
                        bool bIsBlockL2Hash = (b0 > 6);

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                        gettimeofday(&t, NULL);
#endif
                        // Call poseidon and get the hash key
                        Goldilocks::Element Kin0Hash[4];
                        poseidon.hash(Kin0Hash, Kin0);

                        Kin1[8] = Kin0Hash[0];
                        Kin1[9] = Kin0Hash[1];
                        Kin1[10] = Kin0Hash[2];
                        Kin1[11] = Kin0Hash[3];

                        // Call poseidon hash
                        Goldilocks::Element Kin1Hash[4];
                        poseidon.hash(Kin1Hash, Kin1);

                        // Store a copy of the data in ctx.lastSWrite
                        if (!bProcessBatch)
                        {
                            for (uint64_t j=0; j<12; j++)
                            {
                                ctx.lastSWrite.Kin0[j] = Kin0[j];
                            }
                            for (uint64_t j=0; j<12; j++)
                            {
                                ctx.lastSWrite.Kin1[j] = Kin1[j];
                            }
                        }
                        for (uint64_t j=0; j<4; j++)
                        {
                            ctx.lastSWrite.keyI[j] = Kin0Hash[j];
                        }
                        for (uint64_t j=0; j<4; j++)
                        {
                            ctx.lastSWrite.key[j] = Kin1Hash[j];
                        }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                        mainMetrics.add("Poseidon", TimeDiff(t));
#endif

#ifdef LOG_STORAGE
                        zklog.info("Storage write sWR got poseidon key: " + ctx.fr.toString(ctx.lastSWrite.key, 16));
#endif
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                        gettimeofday(&t, NULL);
#endif
                        // Collect the keys used to read or write store data
                        if (proverRequest.input.bGetKeys && !bIsTouchedAddressTree)
                        {
                            proverRequest.nodesKeys.insert(fea2string(fr, ctx.lastSWrite.key));
                        }

                        zkResult = pHashDB->set(proverRequest.uuid, proverRequest.pFullTracer->get_block_number(), proverRequest.pFullTracer->get_tx_number(), oldRoot, ctx.lastSWrite.key, value, bIsTouchedAddressTree ? PERSISTENCE_TEMPORARY : bIsBlockL2Hash ? PERSISTENCE_TEMPORARY_HASH : proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, ctx.lastSWrite.newRoot, &ctx.lastSWrite.res, proverRequest.dbReadLog);
                        if (zkResult != ZKR_SUCCESS)
                        {
                            proverRequest.result = zkResult;
                            logError(ctx, string("Failed calling pHashDB->set() result=") + zkresult2string(zkResult) + " key=" + fea2string(fr, ctx.lastSWrite.key));
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        incCounter = ctx.lastSWrite.res.proofHashCounter + 2;

#ifdef LOG_SMT_KEY_DETAILS
                        zklog.info("SMT set C=" + fea2string(fr, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]) +
                            " A=" + fea2string(fr, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]) +
                            " B=" + fea2string(fr, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]) +
                            " Kin0Hash=" + fea2string(fr, Kin0Hash) +
                            " Kin1Hash=" + fea2string(fr, Kin1Hash) +
                            " oldRoot=" + fea2string(fr, oldRoot) +
                            " value=" + value.get_str(10) +
                            " newRoot=" + fea2string(fr, ctx.lastSWrite.newRoot) +
                            " siblingLeftChild=" + fea2string(fr, ctx.lastSWrite.res.siblingLeftChild) +
                            " siblingRightChild=" + fea2string(fr, ctx.lastSWrite.res.siblingRightChild));
#endif
                    }
                    if (bProcessBatch)
                    {
                        zkResult = eval_addReadWriteAddress(ctx, value, ctx.lastSWrite.key);
                        if (zkResult != ZKR_SUCCESS)
                        {
                            proverRequest.result = zkResult;
                            logError(ctx, string("Failed calling eval_addReadWriteAddress() 2 result=") + zkresult2string(zkResult));
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                    }

                    // If we just modified a balance
                    if ( fr.isZero(pols.B0[i]) && fr.isZero(pols.B1[i]) )
                    {
                        mpz_class balanceDifference = ctx.lastSWrite.res.newValue - ctx.lastSWrite.res.oldValue;
                        ctx.totalTransferredBalance += balanceDifference;
                        //cout << "Set balance: oldValue=" << ctx.lastSWrite.res.oldValue.get_str(10) <<
                        //        " newValue=" << ctx.lastSWrite.res.newValue.get_str(10) <<
                        //        " difference=" << balanceDifference.get_str(10) <<
                        //        " total=" << ctx.totalTransferredBalance.get_str(10) << endl;
                    }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                    mainMetrics.add("SMT Set", TimeDiff(t));
#endif
                    ctx.lastSWrite.step = i;

                    sr4to8(fr, ctx.lastSWrite.newRoot[0], ctx.lastSWrite.newRoot[1], ctx.lastSWrite.newRoot[2], ctx.lastSWrite.newRoot[3], fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                    nHits++;
#ifdef LOG_STORAGE
                    zklog.info("Storage write sWR stored at key: " + ctx.fr.toString(ctx.lastSWrite.key, 16) + " newRoot: " + fr.toString(ctx.lastSWrite.res.newRoot, 16));
#endif
                }

                // HashK free in
                if ( (rom.line[zkPC].hashK == 1) || (rom.line[zkPC].hashK1 == 1) )
                {
                    unordered_map< uint64_t, HashValue >::iterator hashKIterator;

                    // If there is no entry in the hash database for this address, then create a new one
                    hashKIterator = ctx.hashK.find(addr);
                    if (hashKIterator == ctx.hashK.end())
                    {
                        HashValue hashValue;
                        ctx.hashK[addr] = hashValue;
                        hashKIterator = ctx.hashK.find(addr);
                        zkassert(hashKIterator != ctx.hashK.end());
                    }

                    // Get the size of the hash from D0
                    uint64_t size = 1;
                    if (rom.line[zkPC].hashK == 1)
                    {
                        size = fr.toU64(pols.D0[i]);
                        if (size>32)
                        {
                            proverRequest.result = ZKR_SM_MAIN_HASHK_SIZE_OUT_OF_RANGE;
                            logError(ctx, "Invalid size>32 for hashK 1: pols.D0[i]=" + fr.toString(pols.D0[i], 16) + " size=" + to_string(size));
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                    }

                    // Get the positon of the hash from HASHPOS
                    int64_t iPos;
                    fr.toS64(iPos, pols.HASHPOS[i]);
                    if (iPos < 0)
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHK_POSITION_NEGATIVE;
                        logError(ctx, "Invalid pos<0 for HashK 1: pols.HASHPOS[i]=" + fr.toString(pols.HASHPOS[i], 16) + " pos=" + to_string(iPos));
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }
                    uint64_t pos = iPos;

                    // Check that pos+size do not exceed data size
                    if ( (pos+size) > hashKIterator->second.data.size())
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHK_POSITION_PLUS_SIZE_OUT_OF_RANGE;
                        logError(ctx, "HashK 1 invalid size of hash: pos=" + to_string(pos) + " + size=" + to_string(size) + " > data.size=" + to_string(hashKIterator->second.data.size()));
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }

                    // Copy data into fi
                    mpz_class s;
                    for (uint64_t j=0; j<size; j++)
                    {
                        uint8_t data = hashKIterator->second.data[pos+j];
                        s = (s<<uint64_t(8)) + mpz_class(data);
                    }
                    scalar2fea(fr, s, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

                    nHits++;

#ifdef LOG_HASHK
                    zklog.info("hashK 1 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " addr=" + to_string(addr) + " pos=" + to_string(pos) + " size=" + to_string(size) + " data=" + s.get_str(16));
#endif
                }

                // HashKDigest free in
                if (rom.line[zkPC].hashKDigest == 1)
                {
                    unordered_map< uint64_t, HashValue >::iterator hashKIterator;

                    // If there is no entry in the hash database for this address, this is an error
                    hashKIterator = ctx.hashK.find(addr);
                    if (hashKIterator == ctx.hashK.end())
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHKDIGEST_ADDRESS_NOT_FOUND;
                        logError(ctx, "HashKDigest 1: digest not defined for addr=" + to_string(addr));
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }

                    // If digest was not calculated, this is an error
                    if (!hashKIterator->second.lenCalled)
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHKDIGEST_NOT_COMPLETED;
                        logError(ctx, "HashKDigest 1: digest not calculated for addr=" + to_string(addr) + ".  Call hashKLen to finish digest.");
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }

                    // Copy digest into fi
                    scalar2fea(fr, hashKIterator->second.digest, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

                    nHits++;

#ifdef LOG_HASHK
                    zklog.info("hashKDigest 1 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " addr=" + to_string(addr) + " digest=" + ctx.hashK[addr].digest.get_str(16));
#endif
                }

                // HashP free in
                if ( (rom.line[zkPC].hashP == 1) || (rom.line[zkPC].hashP1 == 1) )
                {
                    unordered_map< uint64_t, HashValue >::iterator hashPIterator;

                    // If there is no entry in the hash database for this address, then create a new one
                    hashPIterator = ctx.hashP.find(addr);
                    if (hashPIterator == ctx.hashP.end())
                    {
                        HashValue hashValue;
                        ctx.hashP[addr] = hashValue;
                        hashPIterator = ctx.hashP.find(addr);
                        zkassert(hashPIterator != ctx.hashP.end());
                    }

                    // Get the size of the hash from D0
                    uint64_t size = 1;
                    if (rom.line[zkPC].hashP == 1)
                    {
                        size = fr.toU64(pols.D0[i]);
                        if (size>32)
                        {
                            proverRequest.result = ZKR_SM_MAIN_HASHP_SIZE_OUT_OF_RANGE;
                            logError(ctx, "Invalid size>32 for hashP 1: pols.D0[i]=" + fr.toString(pols.D0[i], 16) + " size=" + to_string(size));
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                    }

                    // Get the positon of the hash from HASHPOS
                    int64_t iPos;
                    fr.toS64(iPos, pols.HASHPOS[i]);
                    if (iPos < 0)
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHP_POSITION_NEGATIVE;
                        logError(ctx, "Invalid pos<0 for HashP 1: pols.HASHPOS[i]=" + fr.toString(pols.HASHPOS[i], 16) + " pos=" + to_string(iPos));
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }
                    uint64_t pos = iPos;

                    // Check that pos+size do not exceed data size
                    if ( (pos+size) > hashPIterator->second.data.size())
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHP_POSITION_PLUS_SIZE_OUT_OF_RANGE;
                        logError(ctx, "HashP 1 invalid size of hash: pos=" + to_string(pos) + " size=" + to_string(size) + " data.size=" + to_string(ctx.hashP[addr].data.size()));
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }

                    // Copy data into fi
                    mpz_class s;
                    for (uint64_t j=0; j<size; j++)
                    {
                        uint8_t data = hashPIterator->second.data[pos+j];
                        s = (s<<uint64_t(8)) + mpz_class(data);
                    }
                    scalar2fea(fr, s, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

                    nHits++;
                }

                // HashPDigest free in
                if (rom.line[zkPC].hashPDigest == 1)
                {
                    unordered_map< uint64_t, HashValue >::iterator hashPIterator;

                    // If there is no entry in the hash database for this address, this is an error
                    hashPIterator = ctx.hashP.find(addr);
                    if (hashPIterator == ctx.hashP.end())
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHPDIGEST_ADDRESS_NOT_FOUND;
                        logError(ctx, "HashPDigest 1: digest not defined addr=" + to_string(addr));
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }

                    // If digest was not calculated, this is an error
                    if (!hashPIterator->second.lenCalled)
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHPDIGEST_NOT_COMPLETED;
                        logError(ctx, "HashPDigest 1: digest not calculated.  Call hashPLen to finish digest.");
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }

                    // Copy digest into fi
                    scalar2fea(fr, hashPIterator->second.digest, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

                    nHits++;
                }

                // HashS free in
                if ( (rom.line[zkPC].hashS == 1) || (rom.line[zkPC].hashS1 == 1) )
                {
                    unordered_map< uint64_t, HashValue >::iterator hashSIterator;

                    // If there is no entry in the hash database for this address, then create a new one
                    hashSIterator = ctx.hashS.find(addr);
                    if (hashSIterator == ctx.hashS.end())
                    {
                        HashValue hashValue;
                        ctx.hashS[addr] = hashValue;
                        hashSIterator = ctx.hashS.find(addr);
                        zkassert(hashSIterator != ctx.hashS.end());
                    }

                    // Get the size of the hash from D0
                    uint64_t size = 1;
                    if (rom.line[zkPC].hashS == 1)
                    {
                        size = fr.toU64(pols.D0[i]);
                        if (size>32)
                        {
                            proverRequest.result = ZKR_SM_MAIN_HASHS_SIZE_OUT_OF_RANGE;
                            logError(ctx, "Invalid size>32 for hashS 1: pols.D0[i]=" + fr.toString(pols.D0[i], 16) + " size=" + to_string(size));
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                    }

                    // Get the positon of the hash from HASHPOS
                    int64_t iPos;
                    fr.toS64(iPos, pols.HASHPOS[i]);
                    if (iPos < 0)
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHS_POSITION_NEGATIVE;
                        logError(ctx, "Invalid pos<0 for HashS 1: pols.HASHPOS[i]=" + fr.toString(pols.HASHPOS[i], 16) + " pos=" + to_string(iPos));
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }
                    uint64_t pos = iPos;

                    // Check that pos+size do not exceed data size
                    if ( (pos+size) > hashSIterator->second.data.size())
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHS_POSITION_PLUS_SIZE_OUT_OF_RANGE;
                        logError(ctx, "HashS 1 invalid size of hash: pos=" + to_string(pos) + " + size=" + to_string(size) + " > data.size=" + to_string(hashSIterator->second.data.size()));
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }

                    // Copy data into fi
                    mpz_class s;
                    for (uint64_t j=0; j<size; j++)
                    {
                        uint8_t data = hashSIterator->second.data[pos+j];
                        s = (s<<uint64_t(8)) + mpz_class(data);
                    }
                    scalar2fea(fr, s, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

                    nHits++;

#ifdef LOG_HASHS
                    zklog.info("hashS 1 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " addr=" + to_string(addr) + " pos=" + to_string(pos) + " size=" + to_string(size) + " data=" + s.get_str(16));
#endif
                }

                // HashSDigest free in
                if (rom.line[zkPC].hashSDigest == 1)
                {
                    unordered_map< uint64_t, HashValue >::iterator hashSIterator;

                    // If there is no entry in the hash database for this address, this is an error
                    hashSIterator = ctx.hashS.find(addr);
                    if (hashSIterator == ctx.hashS.end())
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHSDIGEST_ADDRESS_NOT_FOUND;
                        logError(ctx, "HashSDigest 1: digest not defined for addr=" + to_string(addr));
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }

                    // If digest was not calculated, this is an error
                    if (!hashSIterator->second.lenCalled)
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHSDIGEST_NOT_COMPLETED;
                        logError(ctx, "HashSDigest 1: digest not calculated for addr=" + to_string(addr) + ".  Call hashSLen to finish digest.");
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }

                    // Copy digest into fi
                    scalar2fea(fr, hashSIterator->second.digest, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

                    nHits++;

#ifdef LOG_HASHS
                    zklog.info("hashSDigest 1 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " addr=" + to_string(addr) + " digest=" + ctx.hashS[addr].digest.get_str(16));
#endif
                }

                // Binary free in
                if (rom.line[zkPC].bin == 1)
                {
                    if (rom.line[zkPC].binOpcode == 0) // ADD
                    {
                        mpz_class a, b, c;
                        if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.A)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.B)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        c = (a + b) & ScalarMask256;
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 1) // SUB
                    {
                        mpz_class a, b, c;
                        if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.A)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.B)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        c = (a - b + ScalarTwoTo256) & ScalarMask256;
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 2) // LT
                    {
                        mpz_class a, b, c;
                        if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.A)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.B)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        c = (a < b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 3) // SLT
                    {
                        mpz_class a, b, c;
                        if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.A)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.B)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        if (a >= ScalarTwoTo255) a = a - ScalarTwoTo256;
                        if (b >= ScalarTwoTo255) b = b - ScalarTwoTo256;
                        c = (a < b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 4) // EQ
                    {
                        mpz_class a, b, c;
                        if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.A)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.B)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        c = (a == b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 5) // AND
                    {
                        mpz_class a, b, c;
                        if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.A)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.B)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        c = (a & b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 6) // OR
                    {
                        mpz_class a, b, c;
                        if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.A)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.B)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        c = (a | b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 7) // XOR
                    {
                        mpz_class a, b, c;
                        if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.A)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.B)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        c = (a ^ b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    } else if ( rom.line[zkPC].binOpcode == 8 ) // LT4
                    {
                        mpz_class a, b, c;
                        if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.A)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                        {
                            proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                            logError(ctx, "Failed calling fea2scalar(pols.B)");
                            pHashDB->cancelBatch(proverRequest.uuid);
                            return;
                        }
                        c = lt4(a, b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nHits++;
                    }
                    else
                    {
                        logError(ctx, "Invalid binary operation: opcode=" + to_string(rom.line[zkPC].binOpcode));
                        exitProcess();
                    }
                }

                // Mem allign read free in
                if (rom.line[zkPC].memAlignRD==1)
                {
                    mpz_class m0;
                    if (!fea2scalar(fr, m0, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                    {
                        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                        logError(ctx, "Failed calling fea2scalar(pols.A)");
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }
                    mpz_class m1;
                    if (!fea2scalar(fr, m1, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                    {
                        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                        logError(ctx, "Failed calling fea2scalar(pols.B)");
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }
                    mpz_class offsetScalar;
                    if (!fea2scalar(fr, offsetScalar, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]))
                    {
                        proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                        logError(ctx, "Failed calling fea2scalar(pols.C)");
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }
                    if (offsetScalar<0 || offsetScalar>32)
                    {
                        proverRequest.result = ZKR_SM_MAIN_MEMALIGN_OFFSET_OUT_OF_RANGE;
                        logError(ctx, "MemAlign out of range offset=" + offsetScalar.get_str());
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }
                    uint64_t offset = offsetScalar.get_ui();
                    mpz_class leftV;
                    leftV = (m0 << (offset*8)) & ScalarMask256;
                    mpz_class rightV;
                    rightV = (m1 >> (256 - offset*8)) & (ScalarMask256 >> (256 - offset*8));
                    mpz_class _V;
                    _V = leftV | rightV;
                    scalar2fea(fr, _V, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                    nHits++;
                }

                // Check that one and only one instruction has been requested
                if (nHits != 1)
                {
                    proverRequest.result = ZKR_SM_MAIN_MULTIPLE_FREEIN;
                    logError(ctx, "Empty freeIn without just one instruction: nHits=" + to_string(nHits));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
            }
            // If freeInTag.op!="", then evaluate the requested command (recursively)
            else
            {
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                gettimeofday(&t, NULL);
#endif
                // Call evalCommand()
                CommandResult cr;
                evalCommand(ctx, rom.line[zkPC].freeInTag, cr);

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                mainMetrics.add("Eval command", TimeDiff(t));
                evalCommandMetrics.add(rom.line[zkPC].freeInTag.opAndFunction, TimeDiff(t));
#endif
                // In case of an external error, return it
                if (cr.zkResult != ZKR_SUCCESS)
                {
                    proverRequest.result = cr.zkResult;
                    logError(ctx, string("Main exec failed calling evalCommand() result=") + zkresult2string(proverRequest.result));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                // Copy fi=command result, depending on its type
                if (cr.type == crt_fea)
                {
                    fi0 = cr.fea0;
                    fi1 = cr.fea1;
                    fi2 = cr.fea2;
                    fi3 = cr.fea3;
                    fi4 = cr.fea4;
                    fi5 = cr.fea5;
                    fi6 = cr.fea6;
                    fi7 = cr.fea7;
                }
                else if (cr.type == crt_fe)
                {
                    fi0 = cr.fe;
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                    fi4 = fr.zero();
                    fi5 = fr.zero();
                    fi6 = fr.zero();
                    fi7 = fr.zero();
                }
                else if (cr.type == crt_scalar)
                {
                    scalar2fea(fr, cr.scalar, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                }
                else if (cr.type == crt_u16)
                {
                    fi0 = fr.fromU64(cr.u16);
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                    fi4 = fr.zero();
                    fi5 = fr.zero();
                    fi6 = fr.zero();
                    fi7 = fr.zero();
                }
                else if (cr.type == crt_u32)
                {
                    fi0 = fr.fromU64(cr.u32);
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                    fi4 = fr.zero();
                    fi5 = fr.zero();
                    fi6 = fr.zero();
                    fi7 = fr.zero();
                }
                else if (cr.type == crt_u64)
                {
                    fi0 = fr.fromU64(cr.u64);
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                    fi4 = fr.zero();
                    fi5 = fr.zero();
                    fi6 = fr.zero();
                    fi7 = fr.zero();
                }
                else
                {
                    logError(ctx, "Unexpected command result type: " + to_string(cr.type));
                    exitProcess();
                }
            }

            // Store polynomial FREE=fi
            pols.FREE0[i] = fi0;
            pols.FREE1[i] = fi1;
            pols.FREE2[i] = fi2;
            pols.FREE3[i] = fi3;
            pols.FREE4[i] = fi4;
            pols.FREE5[i] = fi5;
            pols.FREE6[i] = fi6;
            pols.FREE7[i] = fi7;

            // op = op + inFREE*fi
            op0 = fr.add(op0, fr.mul(fr.add(rom.line[zkPC].inFREE, rom.line[zkPC].inFREE0), fi0));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inFREE, fi1));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inFREE, fi2));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inFREE, fi3));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inFREE, fi4));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inFREE, fi5));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inFREE, fi6));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inFREE, fi7));

            // Copy ROM flags into the polynomials
            pols.inFREE[i] = rom.line[zkPC].inFREE;
            pols.inFREE0[i] = rom.line[zkPC].inFREE0;
        }

        if (!fr.isZero(op0) && !bProcessBatch)
        {
            pols.op0Inv[i] = glp.inv(op0);
        }

        /****************/
        /* INSTRUCTIONS */
        /****************/

        // If assert, check that A=op
        if (rom.line[zkPC].assert == 1)
        {
            if ( (!fr.equal(pols.A0[i], op0)) ||
                 (!fr.equal(pols.A1[i], op1)) ||
                 (!fr.equal(pols.A2[i], op2)) ||
                 (!fr.equal(pols.A3[i], op3)) ||
                 (!fr.equal(pols.A4[i], op4)) ||
                 (!fr.equal(pols.A5[i], op5)) ||
                 (!fr.equal(pols.A6[i], op6)) ||
                 (!fr.equal(pols.A7[i], op7)) )
            {
                proverRequest.result = ZKR_SM_MAIN_ASSERT;
                logError(ctx, string("ROM assert failed: AN!=opN") +
                " A:" + fr.toString(pols.A7[i], 16) + ":" + fr.toString(pols.A6[i], 16) + ":" + fr.toString(pols.A5[i], 16) + ":" + fr.toString(pols.A4[i], 16) + ":" + fr.toString(pols.A3[i], 16) + ":" + fr.toString(pols.A2[i], 16) + ":" + fr.toString(pols.A1[i], 16) + ":" + fr.toString(pols.A0[i], 16) +
                " OP:" + fr.toString(op7, 16) + ":" + fr.toString(op6, 16) + ":" + fr.toString(op5, 16) + ":" + fr.toString(op4,16) + ":" + fr.toString(op3, 16) + ":" + fr.toString(op2, 16) + ":" + fr.toString(op1, 16) + ":" + fr.toString(op0,16));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            pols.assert_pol[i] = fr.one();
#ifdef LOG_ASSERT
            zklog.info("assert");
#endif
        }

        // Memory operation instruction
        if (rom.line[zkPC].mOp == 1)
        {
            pols.mOp[i] = fr.one();

            // If mWR, mem[addr]=op
            if (rom.line[zkPC].mWR == 1)
            {
                pols.mWR[i] = fr.one();

                ctx.mem[addr].fe0 = op0;
                ctx.mem[addr].fe1 = op1;
                ctx.mem[addr].fe2 = op2;
                ctx.mem[addr].fe3 = op3;
                ctx.mem[addr].fe4 = op4;
                ctx.mem[addr].fe5 = op5;
                ctx.mem[addr].fe6 = op6;
                ctx.mem[addr].fe7 = op7;

                if (!bProcessBatch)
                {
                    MemoryAccess memoryAccess;
                    memoryAccess.bIsWrite = true;
                    memoryAccess.address = addr;
                    memoryAccess.pc = i;
                    memoryAccess.fe0 = op0;
                    memoryAccess.fe1 = op1;
                    memoryAccess.fe2 = op2;
                    memoryAccess.fe3 = op3;
                    memoryAccess.fe4 = op4;
                    memoryAccess.fe5 = op5;
                    memoryAccess.fe6 = op6;
                    memoryAccess.fe7 = op7;
                    required.Memory.push_back(memoryAccess);
                }

#ifdef LOG_MEMORY
                zklog.info("Memory write mWR: addr:" + to_string(addr) + " " + fea2string(fr, ctx.mem[addr].fe0, ctx.mem[addr].fe1, ctx.mem[addr].fe2, ctx.mem[addr].fe3, ctx.mem[addr].fe4, ctx.mem[addr].fe5, ctx.mem[addr].fe6, ctx.mem[addr].fe7));
#endif
            }
            else
            {
                if (!bProcessBatch)
                {
                    MemoryAccess memoryAccess;
                    memoryAccess.bIsWrite = false;
                    memoryAccess.address = addr;
                    memoryAccess.pc = i;
                    memoryAccess.fe0 = op0;
                    memoryAccess.fe1 = op1;
                    memoryAccess.fe2 = op2;
                    memoryAccess.fe3 = op3;
                    memoryAccess.fe4 = op4;
                    memoryAccess.fe5 = op5;
                    memoryAccess.fe6 = op6;
                    memoryAccess.fe7 = op7;
                    required.Memory.push_back(memoryAccess);
                }

                if (ctx.mem.find(addr) != ctx.mem.end())
                {
                    if ( (!fr.equal(ctx.mem[addr].fe0, op0)) ||
                         (!fr.equal(ctx.mem[addr].fe1, op1)) ||
                         (!fr.equal(ctx.mem[addr].fe2, op2)) ||
                         (!fr.equal(ctx.mem[addr].fe3, op3)) ||
                         (!fr.equal(ctx.mem[addr].fe4, op4)) ||
                         (!fr.equal(ctx.mem[addr].fe5, op5)) ||
                         (!fr.equal(ctx.mem[addr].fe6, op6)) ||
                         (!fr.equal(ctx.mem[addr].fe7, op7)) )
                    {
                        proverRequest.result = ZKR_SM_MAIN_MEMORY;
                        logError(ctx, "Memory Read does not match op=" + fea2string(fr, op0, op1, op2, op3, op4, op5, op6, op7) +
                            " mem=" + fea2string(fr, ctx.mem[addr].fe0, ctx.mem[addr].fe1, ctx.mem[addr].fe2, ctx.mem[addr].fe3, ctx.mem[addr].fe4, ctx.mem[addr].fe5, ctx.mem[addr].fe6, ctx.mem[addr].fe7));
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }
                }
                else
                {
                    if ( (!fr.isZero(op0)) ||
                         (!fr.isZero(op1)) ||
                         (!fr.isZero(op2)) ||
                         (!fr.isZero(op3)) ||
                         (!fr.isZero(op4)) ||
                         (!fr.isZero(op5)) ||
                         (!fr.isZero(op6)) ||
                         (!fr.isZero(op7)) )
                    {
                        proverRequest.result = ZKR_SM_MAIN_MEMORY;
                        logError(ctx, "Memory Read does not match (op!=0)");
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }
                }
            }
        }

        // overwrite 'op' when hiting 'checkFirstTxType' label
        if ((zkPC == checkFirstTxTypeLabel) && proverRequest.input.bSkipFirstChangeL2Block)
        {
            op0 = fr.one();
            op1 = fr.one();
            op2 = fr.one();
            op3 = fr.one();
            op4 = fr.one();
            op5 = fr.one();
            op6 = fr.one();
            op7 = fr.one();
        }

        // overwrite 'op' when hiting 'writeBlockInfoRoot' label
        if ((zkPC == writeBlockInfoRootLabel) && proverRequest.input.bSkipWriteBlockInfoRoot)
        {
            op0 = fr.zero();
            op1 = fr.zero();
            op2 = fr.zero();
            op3 = fr.zero();
            op4 = fr.zero();
            op5 = fr.zero();
            op6 = fr.zero();
            op7 = fr.zero();
        }

        // Storage read instruction
        if (rom.line[zkPC].sRD == 1)
        {
            if (!bProcessBatch) pols.sRD[i] = fr.one();

            Goldilocks::Element Kin0[12];
            Kin0[0] = pols.C0[i];
            Kin0[1] = pols.C1[i];
            Kin0[2] = pols.C2[i];
            Kin0[3] = pols.C3[i];
            Kin0[4] = pols.C4[i];
            Kin0[5] = pols.C5[i];
            Kin0[6] = pols.C6[i];
            Kin0[7] = pols.C7[i];
            Kin0[8] = fr.zero();
            Kin0[9] = fr.zero();
            Kin0[10] = fr.zero();
            Kin0[11] = fr.zero();

            Goldilocks::Element Kin1[12];
            Kin1[0] = pols.A0[i];
            Kin1[1] = pols.A1[i];
            Kin1[2] = pols.A2[i];
            Kin1[3] = pols.A3[i];
            Kin1[4] = pols.A4[i];
            Kin1[5] = pols.A5[i];
            Kin1[6] = pols.B0[i];
            Kin1[7] = pols.B1[i];

            uint64_t b0 = fr.toU64(pols.B0[i]);
            bool bIsTouchedAddressTree = (b0 == 5) || (b0 == 6);

            if  ( !fr.isZero(pols.A5[i]) || !fr.isZero(pols.A6[i]) || !fr.isZero(pols.A7[i]) || !fr.isZero(pols.B2[i]) || !fr.isZero(pols.B3[i]) || !fr.isZero(pols.B4[i]) || !fr.isZero(pols.B5[i])|| !fr.isZero(pols.B6[i])|| !fr.isZero(pols.B7[i]) )
            {
                proverRequest.result = ZKR_SM_MAIN_STORAGE_INVALID_KEY;
                logError(ctx, "Storage read instruction found non-zero A-B registers");
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
            gettimeofday(&t, NULL);
#endif
            // Call poseidon and get the hash key
            Goldilocks::Element Kin0Hash[4];
            poseidon.hash(Kin0Hash, Kin0);

            Goldilocks::Element keyI[4];
            keyI[0] = Kin0Hash[0];
            keyI[1] = Kin0Hash[1];
            keyI[2] = Kin0Hash[2];
            keyI[3] = Kin0Hash[3];

            Kin1[8] = Kin0Hash[0];
            Kin1[9] = Kin0Hash[1];
            Kin1[10] = Kin0Hash[2];
            Kin1[11] = Kin0Hash[3];

            Goldilocks::Element Kin1Hash[4];
            poseidon.hash(Kin1Hash, Kin1);

            // Store PoseidonG required data
            if (!bProcessBatch)
            {
                // Declare PoseidonG required data
                array<Goldilocks::Element,17> pg;

                // Store PoseidonG required data
                for (uint64_t j=0; j<12; j++)
                {
                    pg[j] = Kin0[j];
                }
                for (uint64_t j=0; j<4; j++)
                {
                    pg[12+j] = Kin0Hash[j];
                }
                pg[16] = fr.fromU64(POSEIDONG_PERMUTATION1_ID);
                required.PoseidonG.push_back(pg);

                // Store PoseidonG required data
                for (uint64_t j=0; j<12; j++)
                {
                    pg[j] = Kin1[j];
                }
                for (uint64_t j=0; j<4; j++)
                {
                    pg[12+j] = Kin1Hash[j];
                }
                pg[16] = fr.fromU64(POSEIDONG_PERMUTATION2_ID);
                required.PoseidonG.push_back(pg);
            }

            Goldilocks::Element key[4];
            key[0] = Kin1Hash[0];
            key[1] = Kin1Hash[1];
            key[2] = Kin1Hash[2];
            key[3] = Kin1Hash[3];

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
            mainMetrics.add("Poseidon", TimeDiff(t), 3);
#endif

#ifdef LOG_STORAGE
            zklog.info("Storage read sRD got poseidon key: " + ctx.fr.toString(ctx.lastSWrite.key, 16));
#endif
            Goldilocks::Element oldRoot[4];
            sr8to4(fr, pols.SR0[i], pols.SR1[i], pols.SR2[i], pols.SR3[i], pols.SR4[i], pols.SR5[i], pols.SR6[i], pols.SR7[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
            gettimeofday(&t, NULL);
#endif

            // Collect the keys used to read or write store data
            if (proverRequest.input.bGetKeys && !bIsTouchedAddressTree)
            {
                proverRequest.nodesKeys.insert(fea2string(fr, key));
            }

            SmtGetResult smtGetResult;
            mpz_class value;
            zkresult zkResult = pHashDB->get(proverRequest.uuid, oldRoot, key, value, &smtGetResult, proverRequest.dbReadLog);
            if (zkResult != ZKR_SUCCESS)
            {
                proverRequest.result = zkResult;
                logError(ctx, string("Failed calling pHashDB->get() result=") + zkresult2string(zkResult) + " key=" + fea2string(fr, key));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            incCounter = smtGetResult.proofHashCounter + 2;
            //cout << "smt.get() returns value=" << smtGetResult.value.get_str(16) << endl;

            if (bProcessBatch)
            {
                zkResult = eval_addReadWriteAddress(ctx, smtGetResult.value, key);
                if (zkResult != ZKR_SUCCESS)
                {
                    proverRequest.result = zkResult;
                    logError(ctx, string("Failed calling eval_addReadWriteAddress() 3 result=") + zkresult2string(zkResult));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
            }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
            mainMetrics.add("SMT Get", TimeDiff(t));
#endif
            if (!bProcessBatch)
            {
                SmtAction smtAction;
                smtAction.bIsSet = false;
                smtAction.getResult = smtGetResult;
                required.Storage.push_back(smtAction);
            }

#ifdef LOG_STORAGE
            zklog.info("Storage read sRD read from key: " + ctx.fr.toString(ctx.lastSWrite.key, 16) + " value:" + value.get_str(16));
#endif
            mpz_class opScalar;
            if (!fea2scalar(fr, opScalar, op0, op1, op2, op3, op4, op5, op6, op7))
            {
                proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                logError(ctx, "Failed calling fea2scalar(op)");
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            if (smtGetResult.value != opScalar)
            {
                proverRequest.result = ZKR_SM_MAIN_STORAGE_READ_MISMATCH;
                logError(ctx, "Storage read does not match: smtGetResult.value=" + smtGetResult.value.get_str() + " opScalar=" + opScalar.get_str());
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            for (uint64_t k=0; k<4; k++)
            {
                pols.sKeyI[k][i] = keyI[k];
                pols.sKey[k][i] = key[k];
            }
        }

        // Storage write instruction
        if (rom.line[zkPC].sWR == 1)
        {
            // Copy ROM flags into the polynomials
            if (!bProcessBatch) pols.sWR[i] = fr.one();

            if ( (!bProcessBatch && (ctx.lastSWrite.step == 0)) || (ctx.lastSWrite.step != i) )
            {
                // Reset lastSWrite
                ctx.lastSWrite.reset();

                Goldilocks::Element Kin0[12];
                Kin0[0] = pols.C0[i];
                Kin0[1] = pols.C1[i];
                Kin0[2] = pols.C2[i];
                Kin0[3] = pols.C3[i];
                Kin0[4] = pols.C4[i];
                Kin0[5] = pols.C5[i];
                Kin0[6] = pols.C6[i];
                Kin0[7] = pols.C7[i];
                Kin0[8] = fr.zero();
                Kin0[9] = fr.zero();
                Kin0[10] = fr.zero();
                Kin0[11] = fr.zero();

                Goldilocks::Element Kin1[12];
                Kin1[0] = pols.A0[i];
                Kin1[1] = pols.A1[i];
                Kin1[2] = pols.A2[i];
                Kin1[3] = pols.A3[i];
                Kin1[4] = pols.A4[i];
                Kin1[5] = pols.A5[i];
                Kin1[6] = pols.B0[i];
                Kin1[7] = pols.B1[i];

                uint64_t b0 = fr.toU64(pols.B0[i]);
                bool bIsTouchedAddressTree = (b0 == 5) || (b0 == 6);
                bool bIsBlockL2Hash = (b0 > 6);

                if  ( !fr.isZero(pols.A5[i]) || !fr.isZero(pols.A6[i]) || !fr.isZero(pols.A7[i]) || !fr.isZero(pols.B2[i]) || !fr.isZero(pols.B3[i]) || !fr.isZero(pols.B4[i]) || !fr.isZero(pols.B5[i])|| !fr.isZero(pols.B6[i])|| !fr.isZero(pols.B7[i]) )
                {
                    proverRequest.result = ZKR_SM_MAIN_STORAGE_INVALID_KEY;
                    logError(ctx, "Storage write instruction found non-zero A-B registers");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                gettimeofday(&t, NULL);
#endif
                // Call poseidon and get the hash key
                Goldilocks::Element Kin0Hash[4];
                poseidon.hash(Kin0Hash, Kin0);

                Kin1[8] = Kin0Hash[0];
                Kin1[9] = Kin0Hash[1];
                Kin1[10] = Kin0Hash[2];
                Kin1[11] = Kin0Hash[3];

                Goldilocks::Element Kin1Hash[4];
                poseidon.hash(Kin1Hash, Kin1);

                // Store a copy of the data in ctx.lastSWrite
                if (!bProcessBatch)
                {
                    for (uint64_t j=0; j<12; j++)
                    {
                        ctx.lastSWrite.Kin0[j] = Kin0[j];
                    }
                    for (uint64_t j=0; j<12; j++)
                    {
                        ctx.lastSWrite.Kin1[j] = Kin1[j];
                    }
                }
                for (uint64_t j=0; j<4; j++)
                {
                    ctx.lastSWrite.keyI[j] = Kin0Hash[j];
                }
                for (uint64_t j=0; j<4; j++)
                {
                    ctx.lastSWrite.key[j] = Kin1Hash[j];
                }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                mainMetrics.add("Poseidon", TimeDiff(t));
#endif
                // Call SMT to get the new Merkel Tree root hash
                mpz_class scalarD;
                if (!fea2scalar(fr, scalarD, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.D)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                gettimeofday(&t, NULL);
#endif
                Goldilocks::Element oldRoot[4];
                sr8to4(fr, pols.SR0[i], pols.SR1[i], pols.SR2[i], pols.SR3[i], pols.SR4[i], pols.SR5[i], pols.SR6[i], pols.SR7[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);

                // Collect the keys used to read or write store data
                if (proverRequest.input.bGetKeys && !bIsTouchedAddressTree)
                {
                    proverRequest.nodesKeys.insert(fea2string(fr, ctx.lastSWrite.key));
                }

                zkresult zkResult = pHashDB->set(proverRequest.uuid, proverRequest.pFullTracer->get_block_number(), proverRequest.pFullTracer->get_tx_number(), oldRoot, ctx.lastSWrite.key, scalarD, bIsTouchedAddressTree ? PERSISTENCE_TEMPORARY : bIsBlockL2Hash ? PERSISTENCE_TEMPORARY_HASH : proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, ctx.lastSWrite.newRoot, &ctx.lastSWrite.res, proverRequest.dbReadLog);
                if (zkResult != ZKR_SUCCESS)
                {
                    proverRequest.result = zkResult;
                    logError(ctx, string("Failed calling pHashDB->set() result=") + zkresult2string(zkResult) + " key=" + fea2string(fr, ctx.lastSWrite.key));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                incCounter = ctx.lastSWrite.res.proofHashCounter + 2;

                if (bProcessBatch)
                {
                    zkResult = eval_addReadWriteAddress(ctx, scalarD, ctx.lastSWrite.key);
                    if (zkResult != ZKR_SUCCESS)
                    {
                        proverRequest.result = zkResult;
                        logError(ctx, string("Failed calling eval_addReadWriteAddress() 4 result=") + zkresult2string(zkResult));
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }
                }

                // If we just modified a balance
                if ( fr.isZero(pols.B0[i]) && fr.isZero(pols.B1[i]) )
                {
                    mpz_class balanceDifference = ctx.lastSWrite.res.newValue - ctx.lastSWrite.res.oldValue;
                    ctx.totalTransferredBalance += balanceDifference;
                }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                mainMetrics.add("SMT Set", TimeDiff(t));
#endif
                ctx.lastSWrite.step = i;
            }

            // Store PoseidonG required data
            if (!bProcessBatch)
            {
                // Declare PoseidonG required data
                array<Goldilocks::Element,17> pg;

                // Store PoseidonG required data
                for (uint64_t j=0; j<12; j++)
                {
                    pg[j] = ctx.lastSWrite.Kin0[j];
                }
                for (uint64_t j=0; j<4; j++)
                {
                    pg[12+j] = ctx.lastSWrite.keyI[j];
                }
                pg[16] = fr.fromU64(POSEIDONG_PERMUTATION1_ID);
                required.PoseidonG.push_back(pg);

                // Store PoseidonG required data
                for (uint64_t j=0; j<12; j++)
                {
                    pg[j] = ctx.lastSWrite.Kin1[j];
                }
                for (uint64_t j=0; j<4; j++)
                {
                    pg[12+j] = ctx.lastSWrite.key[j];
                }
                pg[16] = fr.fromU64(POSEIDONG_PERMUTATION2_ID);
                required.PoseidonG.push_back(pg);
            }

            if (!bProcessBatch)
            {
                SmtAction smtAction;
                smtAction.bIsSet = true;
                smtAction.setResult = ctx.lastSWrite.res;
                required.Storage.push_back(smtAction);
            }

            // Check that the new root hash equals op0
            Goldilocks::Element oldRoot[4];
            sr8to4(fr, op0, op1, op2, op3, op4, op5, op6, op7, oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);

            if ( !fr.equal(ctx.lastSWrite.newRoot[0], oldRoot[0]) ||
                 !fr.equal(ctx.lastSWrite.newRoot[1], oldRoot[1]) ||
                 !fr.equal(ctx.lastSWrite.newRoot[2], oldRoot[2]) ||
                 !fr.equal(ctx.lastSWrite.newRoot[3], oldRoot[3]) )
            {
                proverRequest.result = ZKR_SM_MAIN_STORAGE_WRITE_MISMATCH;
                logError(ctx, "Storage write does not match: ctx.lastSWrite.newRoot: " + fr.toString(ctx.lastSWrite.newRoot[3], 16) + ":" + fr.toString(ctx.lastSWrite.newRoot[2], 16) + ":" + fr.toString(ctx.lastSWrite.newRoot[1], 16) + ":" + fr.toString(ctx.lastSWrite.newRoot[0], 16) +
                    " oldRoot: " + fr.toString(oldRoot[3], 16) + ":" + fr.toString(oldRoot[2], 16) + ":" + fr.toString(oldRoot[1], 16) + ":" + fr.toString(oldRoot[0], 16));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            Goldilocks::Element fea[4];
            sr8to4(fr, op0, op1, op2, op3, op4, op5, op6, op7, fea[0], fea[1], fea[2], fea[3]);
            if ( !fr.equal(ctx.lastSWrite.newRoot[0], fea[0]) ||
                 !fr.equal(ctx.lastSWrite.newRoot[1], fea[1]) ||
                 !fr.equal(ctx.lastSWrite.newRoot[2], fea[2]) ||
                 !fr.equal(ctx.lastSWrite.newRoot[3], fea[3]) )
            {
                proverRequest.result = ZKR_SM_MAIN_STORAGE_WRITE_MISMATCH;
                logError(ctx, "Storage write does not match: ctx.lastSWrite.newRoot=" + fea2string(fr, ctx.lastSWrite.newRoot) + " op=" + fea2string(fr, fea));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            for (uint64_t k=0; k<4; k++)
            {
                pols.sKeyI[k][i] =  ctx.lastSWrite.keyI[k];
                pols.sKey[k][i] = ctx.lastSWrite.key[k];
            }
        }

        // HashK instruction
        if ( (rom.line[zkPC].hashK == 1) || (rom.line[zkPC].hashK1 == 1) )
        {
            if (!bProcessBatch)
            {
                if (rom.line[zkPC].hashK == 1)
                {
                    pols.hashK[i] = fr.one();
                }
                else
                {
                    pols.hashK1[i] = fr.one();
                }
            }

            unordered_map< uint64_t, HashValue >::iterator hashKIterator;

            // If there is no entry in the hash database for this address, then create a new one
            hashKIterator = ctx.hashK.find(addr);
            if (hashKIterator == ctx.hashK.end())
            {
                HashValue hashValue;
                ctx.hashK[addr] = hashValue;
                hashKIterator = ctx.hashK.find(addr);
                zkassert(hashKIterator != ctx.hashK.end());
            }

            // Get the size of the hash from D0
            uint64_t size = 1;
            if (rom.line[zkPC].hashK == 1)
            {
                size = fr.toU64(pols.D0[i]);
                if (size>32)
                {
                    proverRequest.result = ZKR_SM_MAIN_HASHK_SIZE_OUT_OF_RANGE;
                    logError(ctx, "Invalid size>32 for hashK 2: pols.D0[i]=" + fr.toString(pols.D0[i], 16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
            }

            // Get the position of the hash from HASHPOS
            int64_t iPos;
            fr.toS64(iPos, pols.HASHPOS[i]);
            if (iPos < 0)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHK_POSITION_NEGATIVE;
                logError(ctx, string("Invalid pos<0 for HashK 2: pols.HASHPOS[i]=") + fr.toString(pols.HASHPOS[i], 16) + " pos=" + to_string(iPos));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            uint64_t pos = iPos;

            // Get contents of opN into a
            mpz_class a;
            if (!fea2scalar(fr, a, op0, op1, op2, op3, op4, op5, op6, op7))
            {
                proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                logError(ctx, "Failed calling fea2scalar(op)");
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            // Fill the hash data vector with chunks of the scalar value
            mpz_class result;
            for (uint64_t j=0; j<size; j++)
            {
                result = (a >> ((size-j-1)*8)) & ScalarMask8;
                uint8_t bm = result.get_ui();
                if (hashKIterator->second.data.size() == (pos+j))
                {
                    hashKIterator->second.data.push_back(bm);
                }
                else if (hashKIterator->second.data.size() < (pos+j))
                {
                    proverRequest.result = ZKR_SM_MAIN_HASHK_POSITION_PLUS_SIZE_OUT_OF_RANGE;
                    logError(ctx, "HashK 2: trying to insert data in a position:" + to_string(pos+j) + " higher than current data size:" + to_string(ctx.hashK[addr].data.size()));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                else
                {
                    uint8_t bh;
                    bh = hashKIterator->second.data[pos+j];
                    if (bm != bh)
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHK_VALUE_MISMATCH;
                        logError(ctx, "HashK 2 bytes do not match: addr=" + to_string(addr) + " pos+j=" + to_string(pos+j) + " is bm=" + to_string(bm) + " and it should be bh=" + to_string(bh));
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }
                }
            }

            // Check that the remaining of a (op) is zero, i.e. no more data exists beyond size
            mpz_class paddingA = a >> (size*8);
            if (paddingA != 0)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHK_PADDING_MISMATCH;
                logError(ctx, "HashK 2 incoherent size=" + to_string(size) + " a=" + a.get_str(16) + " paddingA=" + paddingA.get_str(16));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            // Record the read operation
            unordered_map<uint64_t, uint64_t>::iterator readsIterator;
            readsIterator = hashKIterator->second.reads.find(pos);
            if ( readsIterator != hashKIterator->second.reads.end() )
            {
                if (readsIterator->second != size)
                {
                    proverRequest.result = ZKR_SM_MAIN_HASHK_SIZE_MISMATCH;
                    logError(ctx, "HashK 2 different read sizes in the same position addr=" + to_string(addr) + " pos=" + to_string(pos) + " ctx.hashK[addr].reads[pos]=" + to_string(ctx.hashK[addr].reads[pos]) + " size=" + to_string(size));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
            }
            else
            {
                ctx.hashK[addr].reads[pos] = size;
            }

            // Store the size
            incHashPos = size;

#ifdef LOG_HASHK
            zklog.info("hashK 2 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " addr=" + to_string(addr) + " pos=" + to_string(pos) + " size=" + to_string(size) + " data=" + a.get_str(16));
#endif
        }

        // HashKLen instruction
        if (rom.line[zkPC].hashKLen == 1)
        {
            if (!bProcessBatch) pols.hashKLen[i] = fr.one();

            unordered_map< uint64_t, HashValue >::iterator hashKIterator;

            // Get the length
            uint64_t lm = fr.toU64(op0);

            // Find the entry in the hash database for this address
            hashKIterator = ctx.hashK.find(addr);

            // If it's undefined, compute a hash of 0 bytes
            if (hashKIterator == ctx.hashK.end())
            {
                // Check that length = 0
                if (lm != 0)
                {
                    proverRequest.result = ZKR_SM_MAIN_HASHKLEN_LENGTH_MISMATCH;
                    logError(ctx, "HashKLen 2 hashK[addr] is empty but lm is not 0 addr=" + to_string(addr) + " lm=" + to_string(lm));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                // Create an empty entry in this address slot
                HashValue hashValue;
                ctx.hashK[addr] = hashValue;
                hashKIterator = ctx.hashK.find(addr);
                zkassert(hashKIterator != ctx.hashK.end());
            }

            if (ctx.hashK[addr].lenCalled)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHKLEN_CALLED_TWICE;
                logError(ctx, "HashKLen 2 called more than once addr=" + to_string(addr));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            ctx.hashK[addr].lenCalled = true;

            uint64_t lh = hashKIterator->second.data.size();
            if (lm != lh)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHKLEN_LENGTH_MISMATCH;
                logError(ctx, "HashKLen 2 length does not match addr=" + to_string(addr) + " is lm=" + to_string(lm) + " and it should be lh=" + to_string(lh));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            if (!hashKIterator->second.digestCalled)
            {
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                gettimeofday(&t, NULL);
#endif
                keccak256(hashKIterator->second.data.data(), hashKIterator->second.data.size(), hashKIterator->second.digest);
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                mainMetrics.add("Keccak", TimeDiff(t));
#endif

#ifdef LOG_HASHK
                {
                    string s = "hashKLen 2 calculate hashKLen: addr:" + to_string(addr) + " hash:" + ctx.hashK[addr].digest.get_str(16) + " size:" + to_string(ctx.hashK[addr].data.size()) + " data:";
                    for (uint64_t k=0; k<ctx.hashK[addr].data.size(); k++) s += byte2string(ctx.hashK[addr].data[k]) + ":";
                    zklog.info(s);
                }
#endif
            }

#ifdef LOG_HASHK
            zklog.info("hashKLen 2 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " addr=" + to_string(addr));
#endif
        }

        // HashKDigest instruction
        if (rom.line[zkPC].hashKDigest == 1)
        {
            if (!bProcessBatch) pols.hashKDigest[i] = fr.one();

            unordered_map< uint64_t, HashValue >::iterator hashKIterator;

            // Find the entry in the hash database for this address
            hashKIterator = ctx.hashK.find(addr);
            if (hashKIterator == ctx.hashK.end())
            {
                proverRequest.result = ZKR_SM_MAIN_HASHKDIGEST_NOT_FOUND;
                logError(ctx, "HashKDigest 2 could not find entry for addr=" + to_string(addr));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            // Get contents of op into dg
            mpz_class dg;
            if (!fea2scalar(fr, dg, op0, op1, op2, op3, op4, op5, op6, op7))
            {
                proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                logError(ctx, "Failed calling fea2scalar(op)");
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            if (dg != hashKIterator->second.digest)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHKDIGEST_DIGEST_MISMATCH;
                logError(ctx, "HashKDigest 2: Digest does not match op");
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            if (ctx.hashK[addr].digestCalled)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHKDIGEST_CALLED_TWICE;
                logError(ctx, "HashKDigest 2 called more than once addr=" + to_string(addr));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            ctx.hashK[addr].digestCalled = true;

            incCounter = ceil((double(hashKIterator->second.data.size()) + double(1)) / double(136));

#ifdef LOG_HASHK
            zklog.info("hashKDigest 2 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " addr=" + to_string(addr) + " digest=" + ctx.hashK[addr].digest.get_str(16));
#endif
        }

        // HashP instruction
        if ( (rom.line[zkPC].hashP == 1) || (rom.line[zkPC].hashP1 == 1) )
        {
            if (!bProcessBatch)
            {
                if (rom.line[zkPC].hashP == 1)
                {
                    pols.hashP[i] = fr.one();
                }
                else
                {
                    pols.hashP1[i] = fr.one();
                }
            }

            unordered_map< uint64_t, HashValue >::iterator hashPIterator;

            // If there is no entry in the hash database for this address, then create a new one
            hashPIterator = ctx.hashP.find(addr);
            if (hashPIterator == ctx.hashP.end())
            {
                HashValue hashValue;
                ctx.hashP[addr] = hashValue;
                hashPIterator = ctx.hashP.find(addr);
                zkassert(hashPIterator != ctx.hashP.end());
            }

            // Get the size of the hash from D0
            uint64_t size = 1;
            if (rom.line[zkPC].hashP == 1)
            {
                size = fr.toU64(pols.D0[i]);
                if (size>32)
                {
                    proverRequest.result = ZKR_SM_MAIN_HASHP_SIZE_OUT_OF_RANGE;
                    logError(ctx, "Invalid size>32 for hashP 2: pols.D0[i]=" + fr.toString(pols.D0[i], 16) + " size=" + to_string(size));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
            }

            // Get the positon of the hash from HASHPOS
            int64_t iPos;
            fr.toS64(iPos, pols.HASHPOS[i]);
            if (iPos < 0)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHP_POSITION_NEGATIVE;
                logError(ctx, "Invalid pos<0 for HashP 2: pols.HASHPOS[i]=" + fr.toString(pols.HASHPOS[i], 16) + " pos=" + to_string(iPos));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            uint64_t pos = iPos;

            // Get contents of opN into a
            mpz_class a;
            if (!fea2scalar(fr, a, op0, op1, op2, op3, op4, op5, op6, op7))
            {
                proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                logError(ctx, "Failed calling fea2scalar(op)");
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            // Fill the hash data vector with chunks of the scalar value
            mpz_class result;
            for (uint64_t j=0; j<size; j++) {
                result = (a >> (size-j-1)*8) & ScalarMask8;
                uint8_t bm = result.get_ui();

                // Allow to fill the first byte with a zero
                if (((pos+j) == 1) && hashPIterator->second.data.empty() && !hashPIterator->second.firstByteWritten)
                {
                    // Fill a zero
                    hashPIterator->second.data.push_back(0);
                    
                    // Record the read operation
                    unordered_map<uint64_t, uint64_t>::iterator readsIterator;
                    readsIterator = hashPIterator->second.reads.find(0);
                    if ( readsIterator != hashPIterator->second.reads.end() )
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHP_SIZE_MISMATCH;
                        logError(ctx, "HashP 2 zero position already existed addr=" + to_string(addr) + " pos=" + to_string(pos));
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }
                    else
                    {
                        ctx.hashP[addr].reads[0] = 1;
                    }
                }

                // Allow to overwrite the first byte
                if (((pos+j) == 0) && (size==1) && !hashPIterator->second.data.empty() && !hashPIterator->second.firstByteWritten)
                {
                    hashPIterator->second.data[0] = bm;
                    hashPIterator->second.firstByteWritten = true;
                }
                else if (hashPIterator->second.data.size() == (pos+j))
                {
                    hashPIterator->second.data.push_back(bm);
                }
                else if (hashPIterator->second.data.size() < (pos+j))
                {
                    proverRequest.result = ZKR_SM_MAIN_HASHP_POSITION_PLUS_SIZE_OUT_OF_RANGE;
                    logError(ctx, "HashP 2: trying to insert data in a position:" + to_string(pos+j) + " higher than current data size:" + to_string(ctx.hashP[addr].data.size()));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                else
                {
                    uint8_t bh;
                    bh = hashPIterator->second.data[pos+j];
                    if (bm != bh)
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHP_VALUE_MISMATCH;
                        logError(ctx, "HashP 2 bytes do not match: addr=" + to_string(addr) + " pos+j=" + to_string(pos+j) + " is bm=" + to_string(bm) + " and it should be bh=" + to_string(bh));
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }
                }
            }

            // Check that the remaining of a (op) is zero, i.e. no more data exists beyond size
            mpz_class paddingA = a >> (size*8);
            if (paddingA != 0)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHP_PADDING_MISMATCH;
                logError(ctx, "HashP2 incoherent size=" + to_string(size) + " a=" + a.get_str(16) + " paddingA=" + paddingA.get_str(16));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            // Record the read operation
            unordered_map<uint64_t, uint64_t>::iterator readsIterator;
            readsIterator = hashPIterator->second.reads.find(pos);
            if ( readsIterator != hashPIterator->second.reads.end() )
            {
                if (readsIterator->second != size)
                {
                    proverRequest.result = ZKR_SM_MAIN_HASHP_SIZE_MISMATCH;
                    logError(ctx, "HashP 2 diferent read sizes in the same position addr=" + to_string(addr) + " pos=" + to_string(pos));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
            }
            else
            {
                ctx.hashP[addr].reads[pos] = size;
            }

            // Store the size
            incHashPos = size;
        }

        // HashPLen instruction
        if (rom.line[zkPC].hashPLen == 1)
        {
            if (!bProcessBatch) pols.hashPLen[i] = fr.one();

            unordered_map< uint64_t, HashValue >::iterator hashPIterator;

            // Get the length
            uint64_t lm = fr.toU64(op0);

            // Find the entry in the hash database for this address
            hashPIterator = ctx.hashP.find(addr);

            // If it's undefined, compute a hash of 0 bytes
            if (hashPIterator == ctx.hashP.end())
            {
                // Check that length = 0
                if (lm != 0)
                {
                    proverRequest.result = ZKR_SM_MAIN_HASHPLEN_LENGTH_MISMATCH;
                    logError(ctx, "HashPLen 2 hashP[addr] is empty but lm is not 0 addr=" + to_string(addr) + " lm=" + to_string(lm));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                // Create an empty entry in this address slot
                HashValue hashValue;
                ctx.hashP[addr] = hashValue;
                hashPIterator = ctx.hashP.find(addr);
                zkassert(hashPIterator != ctx.hashP.end());
            }

            if (ctx.hashP[addr].lenCalled)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHPLEN_CALLED_TWICE;
                logError(ctx, "HashPLen 2 called more than once addr=" + to_string(addr));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            ctx.hashP[addr].lenCalled = true;

            uint64_t lh = hashPIterator->second.data.size();
            if (lm != lh)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHPLEN_LENGTH_MISMATCH;
                logError(ctx, "HashPLen 2 does not match match addr=" + to_string(addr) + " is lm=" + to_string(lm) + " and it should be lh=" + to_string(lh));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            if (!hashPIterator->second.digestCalled)
            {
                // Calculate the linear poseidon hash
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                gettimeofday(&t, NULL);
#endif
                Goldilocks::Element result[4];
                linearPoseidon(ctx, hashPIterator->second.data, result);
                fea2scalar(fr, hashPIterator->second.digest, result);
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                mainMetrics.add("Poseidon", TimeDiff(t));
#endif

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                gettimeofday(&t, NULL);
#endif
                // Collect the keys used to read or write program data
                if (proverRequest.input.bGetKeys)
                {
                    proverRequest.programKeys.insert(fea2string(fr, result));
                }

                zkresult zkResult = pHashDB->setProgram(proverRequest.uuid, proverRequest.pFullTracer->get_block_number(), proverRequest.pFullTracer->get_tx_number(), result, hashPIterator->second.data, proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE);
                if (zkResult != ZKR_SUCCESS)
                {
                    proverRequest.result = zkResult;
                    logError(ctx, string("Failed calling pHashDB->setProgram() result=") + zkresult2string(zkResult) + " key=" + fea2string(fr, result));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                mainMetrics.add("Set program", TimeDiff(t));
#endif

#ifdef LOG_HASH
                {
                    string s = "Hash calculate hashPLen 2: addr:" + to_string(addr) + " hash:" + ctx.hashP[addr].digest.get_str(16) + " size:" + to_string(ctx.hashP[addr].data.size()) + " data:";
                    for (uint64_t k=0; k<ctx.hashP[addr].data.size(); k++) s += byte2string(ctx.hashP[addr].data[k]) + ":";
                    zklog.info(s);
                }
#endif
            }
        }

        // HashPDigest instruction
        if (rom.line[zkPC].hashPDigest == 1)
        {
            if (!bProcessBatch) pols.hashPDigest[i] = fr.one();

            // Get contents of op into dg
            mpz_class dg;
            if (!fea2scalar(fr, dg, op0, op1, op2, op3, op4, op5, op6, op7))
            {
                proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                logError(ctx, "Failed calling fea2scalar(op)");
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            unordered_map< uint64_t, HashValue >::iterator hashPIterator;
            hashPIterator = ctx.hashP.find(addr);
            if (hashPIterator == ctx.hashP.end())
            {
                HashValue hashValue;
                hashValue.digest = dg;
                Goldilocks::Element aux[4];
                scalar2fea(fr, dg, aux);
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                gettimeofday(&t, NULL);
#endif
                // Collect the keys used to read or write store data
                if (proverRequest.input.bGetKeys)
                {
                    proverRequest.programKeys.insert(fea2string(fr, aux));
                }

                zkresult zkResult = pHashDB->getProgram(proverRequest.uuid, aux, hashValue.data, proverRequest.dbReadLog);
                if (zkResult != ZKR_SUCCESS)
                {
                    proverRequest.result = zkResult;
                    logError(ctx, string("Failed calling pHashDB->getProgram() result=") + zkresult2string(zkResult) + " key=" + fea2string(fr, aux));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                mainMetrics.add("Get program", TimeDiff(t));
#endif
                ctx.hashP[addr] = hashValue;
                hashPIterator = ctx.hashP.find(addr);
                zkassert(hashPIterator != ctx.hashP.end());
            }

            if (ctx.hashP[addr].digestCalled)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHPDIGEST_CALLED_TWICE;
                logError(ctx, "HashPDigest 2 called more than once addr=" + to_string(addr));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            ctx.hashP[addr].digestCalled = true;

            incCounter = ceil((double(hashPIterator->second.data.size()) + double(1)) / double(56));

            // Check that digest equals op
            if (dg != hashPIterator->second.digest)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHPDIGEST_DIGEST_MISMATCH;
                logError(ctx, "HashPDigest 2: ctx.hashP[addr].digest=" + ctx.hashP[addr].digest.get_str(16) + " does not match op=" + dg.get_str(16));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
        }

        // HashS instruction
        if ( (rom.line[zkPC].hashS == 1) || (rom.line[zkPC].hashS1 == 1) )
        {
            if (!bProcessBatch)
            {
                if (rom.line[zkPC].hashS == 1)
                {
                    pols.hashS[i] = fr.one();
                }
                else
                {
                    pols.hashS1[i] = fr.one();
                }
            }

            unordered_map< uint64_t, HashValue >::iterator hashSIterator;

            // If there is no entry in the hash database for this address, then create a new one
            hashSIterator = ctx.hashS.find(addr);
            if (hashSIterator == ctx.hashS.end())
            {
                HashValue hashValue;
                ctx.hashS[addr] = hashValue;
                hashSIterator = ctx.hashS.find(addr);
                zkassert(hashSIterator != ctx.hashS.end());
            }

            // Get the size of the hash from D0
            uint64_t size = 1;
            if (rom.line[zkPC].hashS == 1)
            {
                size = fr.toU64(pols.D0[i]);
                if (size>32)
                {
                    proverRequest.result = ZKR_SM_MAIN_HASHS_SIZE_OUT_OF_RANGE;
                    logError(ctx, "Invalid size>32 for hashS 2: pols.D0[i]=" + fr.toString(pols.D0[i], 16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
            }

            // Get the position of the hash from HASHPOS
            int64_t iPos;
            fr.toS64(iPos, pols.HASHPOS[i]);
            if (iPos < 0)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHS_POSITION_NEGATIVE;
                logError(ctx, string("Invalid pos<0 for HashS 2: pols.HASHPOS[i]=") + fr.toString(pols.HASHPOS[i], 16) + " pos=" + to_string(iPos));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            uint64_t pos = iPos;

            // Get contents of opN into a
            mpz_class a;
            if (!fea2scalar(fr, a, op0, op1, op2, op3, op4, op5, op6, op7))
            {
                proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                logError(ctx, "Failed calling fea2scalar(op)");
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            // Fill the hash data vector with chunks of the scalar value
            mpz_class result;
            for (uint64_t j=0; j<size; j++)
            {
                result = (a >> ((size-j-1)*8)) & ScalarMask8;
                uint8_t bm = result.get_ui();
                if (hashSIterator->second.data.size() == (pos+j))
                {
                    hashSIterator->second.data.push_back(bm);
                }
                else if (hashSIterator->second.data.size() < (pos+j))
                {
                    proverRequest.result = ZKR_SM_MAIN_HASHS_POSITION_PLUS_SIZE_OUT_OF_RANGE;
                    logError(ctx, "HashS 2: trying to insert data in a position:" + to_string(pos+j) + " higher than current data size:" + to_string(ctx.hashS[addr].data.size()));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                else
                {
                    uint8_t bh;
                    bh = hashSIterator->second.data[pos+j];
                    if (bm != bh)
                    {
                        proverRequest.result = ZKR_SM_MAIN_HASHS_VALUE_MISMATCH;
                        logError(ctx, "HashS 2 bytes do not match: addr=" + to_string(addr) + " pos+j=" + to_string(pos+j) + " is bm=" + to_string(bm) + " and it should be bh=" + to_string(bh));
                        pHashDB->cancelBatch(proverRequest.uuid);
                        return;
                    }
                }
            }

            // Check that the remaining of a (op) is zero, i.e. no more data exists beyond size
            mpz_class paddingA = a >> (size*8);
            if (paddingA != 0)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHS_PADDING_MISMATCH;
                logError(ctx, "HashS 2 incoherent size=" + to_string(size) + " a=" + a.get_str(16) + " paddingA=" + paddingA.get_str(16));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            // Record the read operation
            unordered_map<uint64_t, uint64_t>::iterator readsIterator;
            readsIterator = hashSIterator->second.reads.find(pos);
            if ( readsIterator != hashSIterator->second.reads.end() )
            {
                if (readsIterator->second != size)
                {
                    proverRequest.result = ZKR_SM_MAIN_HASHS_SIZE_MISMATCH;
                    logError(ctx, "HashS 2 different read sizes in the same position addr=" + to_string(addr) + " pos=" + to_string(pos) + " ctx.hashS[addr].reads[pos]=" + to_string(ctx.hashS[addr].reads[pos]) + " size=" + to_string(size));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
            }
            else
            {
                ctx.hashS[addr].reads[pos] = size;
            }

            // Store the size
            incHashPos = size;

#ifdef LOG_HASHS
            zklog.info("hashS 2 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " addr=" + to_string(addr) + " pos=" + to_string(pos) + " size=" + to_string(size) + " data=" + a.get_str(16));
#endif
        }

        // HashSLen instruction
        if (rom.line[zkPC].hashSLen == 1)
        {
            if (!bProcessBatch) pols.hashSLen[i] = fr.one();

            unordered_map< uint64_t, HashValue >::iterator hashSIterator;

            // Get the length
            uint64_t lm = fr.toU64(op0);

            // Find the entry in the hash database for this address
            hashSIterator = ctx.hashS.find(addr);

            // If it's undefined, compute a hash of 0 bytes
            if (hashSIterator == ctx.hashS.end())
            {
                // Check that length = 0
                if (lm != 0)
                {
                    proverRequest.result = ZKR_SM_MAIN_HASHSLEN_LENGTH_MISMATCH;
                    logError(ctx, "HashSLen 2 hashS[addr] is empty but lm is not 0 addr=" + to_string(addr) + " lm=" + to_string(lm));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                // Create an empty entry in this address slot
                HashValue hashValue;
                ctx.hashS[addr] = hashValue;
                hashSIterator = ctx.hashS.find(addr);
                zkassert(hashSIterator != ctx.hashS.end());
            }

            if (ctx.hashS[addr].lenCalled)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHSLEN_CALLED_TWICE;
                logError(ctx, "HashSLen 2 called more than once addr=" + to_string(addr));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            ctx.hashS[addr].lenCalled = true;

            uint64_t lh = hashSIterator->second.data.size();
            if (lm != lh)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHSLEN_LENGTH_MISMATCH;
                logError(ctx, "HashSLen 2 length does not match addr=" + to_string(addr) + " is lm=" + to_string(lm) + " and it should be lh=" + to_string(lh));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            if (!hashSIterator->second.digestCalled)
            {
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                gettimeofday(&t, NULL);
#endif
                SHA256(hashSIterator->second.data.data(), hashSIterator->second.data.size(), hashSIterator->second.digest);
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                mainMetrics.add("SHA256", TimeDiff(t));
#endif

#ifdef LOG_HASHS
                {
                    string s = "hashSLen 2 calculate hashSLen: addr:" + to_string(addr) + " hash:" + ctx.hashS[addr].digest.get_str(16) + " size:" + to_string(ctx.hashS[addr].data.size()) + " data:";
                    for (uint64_t k=0; k<ctx.hashS[addr].data.size(); k++) s += byte2string(ctx.hashS[addr].data[k]) + ":";
                    zklog.info(s);
                }
#endif
            }

#ifdef LOG_HASHS
            zklog.info("hashSLen 2 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " addr=" + to_string(addr));
#endif
        }

        // HashSDigest instruction
        if (rom.line[zkPC].hashSDigest == 1)
        {
            if (!bProcessBatch) pols.hashSDigest[i] = fr.one();

            unordered_map< uint64_t, HashValue >::iterator hashSIterator;

            // Find the entry in the hash database for this address
            hashSIterator = ctx.hashS.find(addr);
            if (hashSIterator == ctx.hashS.end())
            {
                proverRequest.result = ZKR_SM_MAIN_HASHSDIGEST_NOT_FOUND;
                logError(ctx, "HashSDigest 2 could not find entry for addr=" + to_string(addr));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            // Get contents of op into dg
            mpz_class dg;
            if (!fea2scalar(fr, dg, op0, op1, op2, op3, op4, op5, op6, op7))
            {
                proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                logError(ctx, "Failed calling fea2scalar(op)");
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            if (dg != hashSIterator->second.digest)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHSDIGEST_DIGEST_MISMATCH;
                logError(ctx, "HashSDigest 2: Digest does not match op");
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            if (ctx.hashS[addr].digestCalled)
            {
                proverRequest.result = ZKR_SM_MAIN_HASHSDIGEST_CALLED_TWICE;
                logError(ctx, "HashSDigest 2 called more than once addr=" + to_string(addr));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            ctx.hashS[addr].digestCalled = true;

            incCounter = ceil((double(hashSIterator->second.data.size()) + double(1+8)) / double(64));

#ifdef LOG_HASHS
            zklog.info("hashSDigest 2 i=" + to_string(i) + " zkPC=" + to_string(zkPC) + " addr=" + to_string(addr) + " digest=" + ctx.hashS[addr].digest.get_str(16));
#endif
        }

        // HashP or Storage write instructions, required data
        if (!bProcessBatch && (rom.line[zkPC].hashPDigest || rom.line[zkPC].sWR))
        {
            mpz_class op;
            if (!fea2scalar(fr, op, op0, op1, op2, op3, op4, op5, op6, op7))
            {
                proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                logError(ctx, "Failed calling fea2scalar(op)");
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }

            // Store the binary action to execute it later with the binary SM
            BinaryAction binaryAction;
            binaryAction.a = op;
            binaryAction.b = Scalar4xGoldilocksPrime;
            binaryAction.c = 1;
            binaryAction.opcode = 8;
            binaryAction.type = 2;
            required.Binary.push_back(binaryAction);
        }

        // Arith instruction
        if (rom.line[zkPC].arithEq0==1 || rom.line[zkPC].arithEq1==1 || rom.line[zkPC].arithEq2==1 || rom.line[zkPC].arithEq3==1 || rom.line[zkPC].arithEq4==1 || rom.line[zkPC].arithEq5==1)
        {
            // Arith instruction: check that A*B + C = D<<256 + op, using scalars (result can be a big number)
            if (rom.line[zkPC].arithEq0==1 && rom.line[zkPC].arithEq1==0 && rom.line[zkPC].arithEq2==0 && rom.line[zkPC].arithEq3==0 && rom.line[zkPC].arithEq4==0 && rom.line[zkPC].arithEq5==0)
            {
                // Convert to scalar
                mpz_class A, B, C, D, op;
                if (!fea2scalar(fr, A, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.A)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, B, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.B)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, C, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.C)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, D, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.D)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, op, op0, op1, op2, op3, op4, op5, op6, op7))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(op)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                // Check the condition
                if ( (A*B) + C != (D<<256) + op )
                {
                    proverRequest.result = ZKR_SM_MAIN_ARITH_MISMATCH;
                    mpz_class left = (A*B) + C;
                    mpz_class right = (D<<256) + op;
                    logError(ctx, "Arithmetic does not match: (A*B) + C = " + left.get_str(16) + ", (D<<256) + op = " + right.get_str(16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                // Store the arith action to execute it later with the arith SM
                if (!bProcessBatch)
                {
                    // Copy ROM flags into the polynomials
                    pols.arithEq0[i] = fr.one();

                    ArithAction arithAction;
                    arithAction.x1 = A;
                    arithAction.y1 = B;
                    arithAction.x2 = C;
                    arithAction.y2 = D;
                    arithAction.x3 = 0;
                    arithAction.y3 = op;
                    arithAction.selEq0 = 1;
                    arithAction.selEq1 = 0;
                    arithAction.selEq2 = 0;
                    arithAction.selEq3 = 0;
                    arithAction.selEq4 = 0;
                    arithAction.selEq5 = 0;
                    arithAction.selEq6 = 0;
                    required.Arith.push_back(arithAction);
                }
            }
            // Arithmetic FP2 multiplication
            else if (rom.line[zkPC].arithEq0==0 && rom.line[zkPC].arithEq1==0 && rom.line[zkPC].arithEq2==0 && rom.line[zkPC].arithEq3==1 && rom.line[zkPC].arithEq4==0 && rom.line[zkPC].arithEq5==0)
            {
                // Convert to scalar
                mpz_class x1, y1, x2, y2, x3, y3;
                if (!fea2scalar(fr, x1, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.A)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, y1, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.B)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, x2, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.C)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, y2, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.D)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, x3, pols.E0[i], pols.E1[i], pols.E2[i], pols.E3[i], pols.E4[i], pols.E5[i], pols.E6[i], pols.E7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.E)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, y3, op0, op1, op2, op3, op4, op5, op6, op7))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(op)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
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
                    proverRequest.result = ZKR_SM_MAIN_ARITH_MISMATCH;
                    logError(ctx, "Arithmetic FP2 multiplication point does not match: x3=" + fq.toString(x3fe, 16) + " _x3=" + fq.toString(_x3fe, 16) + " y3=" + fq.toString(y3fe, 16) + " _y3=" + fq.toString(_y3fe, 16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                // Store the arith action to execute it later with the arith SM
                if (!bProcessBatch)
                {
                    // Copy ROM flags into the polynomials
                    pols.arithEq3[i] = fr.one();

                    ArithAction arithAction;
                    arithAction.x1 = x1;
                    arithAction.y1 = y1;
                    arithAction.x2 = x2;
                    arithAction.y2 = y2;
                    arithAction.x3 = x3;
                    arithAction.y3 = y3;
                    arithAction.selEq0 = 0;
                    arithAction.selEq1 = 0;
                    arithAction.selEq2 = 0;
                    arithAction.selEq3 = 0;
                    arithAction.selEq4 = 1;
                    arithAction.selEq5 = 0;
                    arithAction.selEq6 = 0;
                    required.Arith.push_back(arithAction);
                }
            }
            // Arithmetic FP2 addition
            else if (rom.line[zkPC].arithEq0==0 && rom.line[zkPC].arithEq1==0 && rom.line[zkPC].arithEq2==0 && rom.line[zkPC].arithEq3==0 && rom.line[zkPC].arithEq4==1 && rom.line[zkPC].arithEq5==0)
            {
                // Convert to scalar
                mpz_class x1, y1, x2, y2, x3, y3;
                if (!fea2scalar(fr, x1, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.A)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, y1, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.B)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, x2, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.C)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, y2, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.D)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, x3, pols.E0[i], pols.E1[i], pols.E2[i], pols.E3[i], pols.E4[i], pols.E5[i], pols.E6[i], pols.E7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.E)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, y3, op0, op1, op2, op3, op4, op5, op6, op7))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(op)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
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
                    proverRequest.result = ZKR_SM_MAIN_ARITH_MISMATCH;
                    logError(ctx, "Arithmetic FP2 addition point does not match: x3=" + fq.toString(x3fe, 16) + " _x3=" + fq.toString(_x3fe, 16) + " y3=" + fq.toString(y3fe, 16) + " _y3=" + fq.toString(_y3fe, 16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                // Store the arith action to execute it later with the arith SM
                if (!bProcessBatch)
                {
                    // Copy ROM flags into the polynomials
                    pols.arithEq4[i] = fr.one();

                    ArithAction arithAction;
                    arithAction.x1 = x1;
                    arithAction.y1 = y1;
                    arithAction.x2 = x2;
                    arithAction.y2 = y2;
                    arithAction.x3 = x3;
                    arithAction.y3 = y3;
                    arithAction.selEq0 = 0;
                    arithAction.selEq1 = 0;
                    arithAction.selEq2 = 0;
                    arithAction.selEq3 = 0;
                    arithAction.selEq4 = 0;
                    arithAction.selEq5 = 1;
                    arithAction.selEq6 = 0;
                    required.Arith.push_back(arithAction);
                }
            }
            // Arithmetic FP2 subtraction
            else if (rom.line[zkPC].arithEq0==0 && rom.line[zkPC].arithEq1==0 && rom.line[zkPC].arithEq2==0 && rom.line[zkPC].arithEq3==0 && rom.line[zkPC].arithEq4==0 && rom.line[zkPC].arithEq5==1)
            {
                // Convert to scalar
                mpz_class x1, y1, x2, y2, x3, y3;
                if (!fea2scalar(fr, x1, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.A)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, y1, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.B)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, x2, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.C)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, y2, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.D)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, x3, pols.E0[i], pols.E1[i], pols.E2[i], pols.E3[i], pols.E4[i], pols.E5[i], pols.E6[i], pols.E7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.E)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, y3, op0, op1, op2, op3, op4, op5, op6, op7))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(op)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
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
                    proverRequest.result = ZKR_SM_MAIN_ARITH_MISMATCH;
                    logError(ctx, "Arithmetic FP2 subtraction point does not match: x3=" + fq.toString(x3fe, 16) + " _x3=" + fq.toString(_x3fe, 16) + " y3=" + fq.toString(y3fe, 16) + " _y3=" + fq.toString(_y3fe, 16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                // Store the arith action to execute it later with the arith SM
                if (!bProcessBatch)
                {
                    // Copy ROM flags into the polynomials
                    pols.arithEq5[i] = fr.one();

                    ArithAction arithAction;
                    arithAction.x1 = x1;
                    arithAction.y1 = y1;
                    arithAction.x2 = x2;
                    arithAction.y2 = y2;
                    arithAction.x3 = x3;
                    arithAction.y3 = y3;
                    arithAction.selEq0 = 0;
                    arithAction.selEq1 = 0;
                    arithAction.selEq2 = 0;
                    arithAction.selEq3 = 0;
                    arithAction.selEq4 = 0;
                    arithAction.selEq5 = 0;
                    arithAction.selEq6 = 1;
                    required.Arith.push_back(arithAction);
                }
            }
            // Arith instruction: check curve points
            else
            {
                // Convert to scalar
                mpz_class x1, y1, x2, y2, x3, y3;
                if (!fea2scalar(fr, x1, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.A)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, y1, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.B)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, x2, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.C)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, y2, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.D)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, x3, pols.E0[i], pols.E1[i], pols.E2[i], pols.E3[i], pols.E4[i], pols.E5[i], pols.E6[i], pols.E7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.E)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, y3, op0, op1, op2, op3, op4, op5, op6, op7))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(op)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                // Convert to RawFec::Element
                RawFec::Element fecX1, fecY1, fecX2, fecY2;
                fec.fromMpz(fecX1, x1.get_mpz_t());
                fec.fromMpz(fecY1, y1.get_mpz_t());
                fec.fromMpz(fecX2, x2.get_mpz_t());
                fec.fromMpz(fecY2, y2.get_mpz_t());

                // Check if this is a double operation
                bool dbl = false;
                if (rom.line[zkPC].arithEq0==0 && rom.line[zkPC].arithEq1==1 && rom.line[zkPC].arithEq2==0 && rom.line[zkPC].arithEq3==0 && rom.line[zkPC].arithEq4==0 && rom.line[zkPC].arithEq5==0)
                {
                    dbl = false;
                }
                else if (rom.line[zkPC].arithEq0==0 && rom.line[zkPC].arithEq1==0 && rom.line[zkPC].arithEq2==1 && rom.line[zkPC].arithEq3==0 && rom.line[zkPC].arithEq4==0 && rom.line[zkPC].arithEq5==0)
                {
                    dbl = true;
                }
                else
                {
                    logError(ctx, "Invalid arithmetic op");
                    exitProcess();
                }

                // Add the elliptic curve points
                RawFec::Element fecX3, fecY3;
                zkresult r = AddPointEc(ctx, dbl, fecX1, fecY1, dbl?fecX1:fecX2, dbl?fecY1:fecY2, fecX3, fecY3);
                if (r != ZKR_SUCCESS)
                {
                    proverRequest.result = r;
                    logError(ctx, "Failed calling AddPointEc() in arith operation");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                // Convert to scalar
                mpz_class _x3, _y3;
                fec.toMpz(_x3.get_mpz_t(), fecX3);
                fec.toMpz(_y3.get_mpz_t(), fecY3);

                // Compare
                bool x3eq = (x3 == _x3);
                bool y3eq = (y3 == _y3);

                if (!x3eq || !y3eq)
                {
                    proverRequest.result = ZKR_SM_MAIN_ARITH_ECRECOVER_MISMATCH;
                    logError(ctx, string("Arithmetic curve ") + (dbl?"dbl":"add") + " point does not match" +
                        " x1=" + x1.get_str() +
                        " y1=" + y1.get_str() +
                        " x2=" + x2.get_str() +
                        " y2=" + y2.get_str() +
                        " x3=" + x3.get_str() +
                        " y3=" + y3.get_str() +
                        "_x3=" + _x3.get_str() +
                        "_y3=" + _y3.get_str());
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                if (!bProcessBatch)
                {
                    pols.arithEq0[i] = fr.fromU64(rom.line[zkPC].arithEq0);
                    pols.arithEq1[i] = fr.fromU64(rom.line[zkPC].arithEq1);
                    pols.arithEq2[i] = fr.fromU64(rom.line[zkPC].arithEq2);

                    // Store the arith action to execute it later with the arith SM
                    ArithAction arithAction;
                    arithAction.x1 = x1;
                    arithAction.y1 = y1;
                    arithAction.x2 = dbl ? x1 : x2;
                    arithAction.y2 = dbl ? y1 : y2;
                    arithAction.x3 = x3;
                    arithAction.y3 = y3;
                    arithAction.selEq0 = 0;
                    arithAction.selEq1 = dbl ? 0 : 1;
                    arithAction.selEq2 = dbl ? 1 : 0;
                    arithAction.selEq3 = 1;
                    arithAction.selEq4 = 0;
                    arithAction.selEq5 = 0;
                    arithAction.selEq6 = 0;
                    required.Arith.push_back(arithAction);
                }
            }
        }

        // Binary instruction
        if (bProcessBatch) pols.carry[i] = fr.zero();
        if (rom.line[zkPC].bin == 1)
        {
            if (rom.line[zkPC].binOpcode == 0) // ADD
            {
                mpz_class a, b, c;
                if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.A)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.B)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(op)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                mpz_class expectedC;
                expectedC = (a + b) & ScalarMask256;
                if (c != expectedC)
                {
                    proverRequest.result = ZKR_SM_MAIN_BINARY_ADD_MISMATCH;
                    logError(ctx, "Binary ADD operation does not match c=op=" + c.get_str(16) + " expectedC=(a + b) & ScalarMask256=" + expectedC.get_str(16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                pols.carry[i] = fr.fromU64(((a + b) >> 256) > 0);

                if (!bProcessBatch)
                {
                    pols.binOpcode[i] = fr.zero();

                    // Store the binary action to execute it later with the binary SM
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 0;
                    binaryAction.type = 1;
                    required.Binary.push_back(binaryAction);
                }
            }
            else if (rom.line[zkPC].binOpcode == 1) // SUB
            {
                mpz_class a, b, c;
                if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.A)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.B)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(op)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                mpz_class expectedC;
                expectedC = (a - b + ScalarTwoTo256) & ScalarMask256;
                if (c != expectedC)
                {
                    proverRequest.result = ZKR_SM_MAIN_BINARY_SUB_MISMATCH;
                    logError(ctx, "Binary SUB operation does not match c=op=" + c.get_str(16) + " expectedC=(a - b + ScalarTwoTo256) & ScalarMask256=" + expectedC.get_str(16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                pols.carry[i] = fr.fromU64((a - b) < 0);

                if (!bProcessBatch)
                {
                    pols.binOpcode[i] = fr.one();

                    // Store the binary action to execute it later with the binary SM
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 1;
                    binaryAction.type = 1;
                    required.Binary.push_back(binaryAction);
                }
            }
            else if (rom.line[zkPC].binOpcode == 2) // LT
            {
                mpz_class a, b, c;
                if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.A)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.B)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(op)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                mpz_class expectedC;
                expectedC = (a < b);
                if (c != expectedC)
                {
                    proverRequest.result = ZKR_SM_MAIN_BINARY_LT_MISMATCH;
                    logError(ctx, "Binary LY operation does not match c=op=" + c.get_str(16) + " expectedC=(a < b)=" + expectedC.get_str(16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                pols.carry[i] = fr.fromU64(a < b);

                if (!bProcessBatch)
                {
                    pols.binOpcode[i] = fr.fromU64(2);

                    // Store the binary action to execute it later with the binary SM
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 2;
                    binaryAction.type = 1;
                    required.Binary.push_back(binaryAction);
                }
            }
            else if (rom.line[zkPC].binOpcode == 3) // SLT
            {
                mpz_class a, b, c, _a, _b;
                if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.A)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.B)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(op)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                _a = a;
                _b = b;

                if (a >= ScalarTwoTo255) _a = a - ScalarTwoTo256;
                if (b >= ScalarTwoTo255) _b = b - ScalarTwoTo256;


                mpz_class expectedC;
                expectedC = (_a < _b);
                if (c != expectedC)
                {
                    proverRequest.result = ZKR_SM_MAIN_BINARY_SLT_MISMATCH;
                    logError(ctx, "Binary SLT operation does not match a=" + a.get_str(16) + " b=" + b.get_str(16) + " c=" + c.get_str(16) + " _a=" + _a.get_str(16) + " _b=" + _b.get_str(16) + " expectedC=" + expectedC.get_str(16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                pols.carry[i] = fr.fromU64(_a < _b);

                if (!bProcessBatch)
                {
                    pols.binOpcode[i] = fr.fromU64(3);

                    // Store the binary action to execute it later with the binary SM
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 3;
                    binaryAction.type = 1;
                    required.Binary.push_back(binaryAction);
                }
            }
            else if (rom.line[zkPC].binOpcode == 4) // EQ
            {
                mpz_class a, b, c;
                if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.A)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.B)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(op)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                mpz_class expectedC;
                expectedC = (a == b);
                if (c != expectedC)
                {
                    proverRequest.result = ZKR_SM_MAIN_BINARY_EQ_MISMATCH;
                    logError( ctx, "Binary EQ operation does not match c=op=" + c.get_str(16) + " expectedC=(a==b)=" + expectedC.get_str(16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                pols.carry[i] = fr.fromU64((a == b));

                if (!bProcessBatch)
                {
                    pols.binOpcode[i] = fr.fromU64(4);

                    // Store the binary action to execute it later with the binary SM
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 4;
                    binaryAction.type = 1;
                    required.Binary.push_back(binaryAction);
                }
            }
            else if (rom.line[zkPC].binOpcode == 5) // AND
            {
                mpz_class a, b, c;
                if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.A)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.B)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(op)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                mpz_class expectedC;
                expectedC = (a & b);
                if (c != expectedC)
                {
                    proverRequest.result = ZKR_SM_MAIN_BINARY_AND_MISMATCH;
                    logError(ctx, "Binary AND operation does not match c=op=" + c.get_str(16) + " expectedC=(a&b)=" + expectedC.get_str(16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                if (c != 0)
                {
                    pols.carry[i] = fr.one();
                }

                if (!bProcessBatch)
                {
                    pols.binOpcode[i] = fr.fromU64(5);

                    // Store the binary action to execute it later with the binary SM
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 5;
                    binaryAction.type = 1;
                    required.Binary.push_back(binaryAction);
                }
            }
            else if (rom.line[zkPC].binOpcode == 6) // OR
            {
                mpz_class a, b, c;
                if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.A)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.B)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(op)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                mpz_class expectedC;
                expectedC = (a | b);
                if (c != expectedC)
                {
                    proverRequest.result = ZKR_SM_MAIN_BINARY_OR_MISMATCH;
                    logError(ctx, "Binary OR operation does not match c=op=" + c.get_str(16) + " expectedC=(a|b)=" + expectedC.get_str(16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                if (!bProcessBatch)
                {
                    pols.binOpcode[i] = fr.fromU64(6);

                    // Store the binary action to execute it later with the binary SM
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 6;
                    binaryAction.type = 1;
                    required.Binary.push_back(binaryAction);
                }
            }
            else if (rom.line[zkPC].binOpcode == 7) // XOR
            {
                mpz_class a, b, c;
                if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.A)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.B)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(op)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                mpz_class expectedC;
                expectedC = (a ^ b);
                if (c != expectedC)
                {
                    proverRequest.result = ZKR_SM_MAIN_BINARY_XOR_MISMATCH;
                    logError(ctx, "Binary XOR operation does not match c=op=" + c.get_str(16) + " expectedC=(a^b)=" + expectedC.get_str(16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                if (!bProcessBatch)
                {
                    pols.binOpcode[i] = fr.fromU64(7);

                    // Store the binary action to execute it later with the binary SM
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 7;
                    binaryAction.type = 1;
                    required.Binary.push_back(binaryAction);
                }
            } else if (rom.line[zkPC].binOpcode == 8) // LT4
            {
                mpz_class a, b, c;
                if (!fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.A)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                if (!fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.B)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                }
                if (!fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError( ctx, "Failed calling fea2scalar(op)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                   
                mpz_class expectedC;
                expectedC = lt4(a,b); 
                if (c != expectedC)
                {
                    proverRequest.result = ZKR_SM_MAIN_BINARY_LT4_MISMATCH; 
                    logError(ctx, "Binary LT4 operation does not match c=op=" + c.get_str(16) + " expectedC=(a LT4 b)=" + expectedC.get_str(16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                pols.carry[i] = fr.fromScalar(c);

                 if (!bProcessBatch)
                {
                    pols.binOpcode[i] = fr.fromU64(8);

                    // Store the binary action to execute it later with the binary SM
                    BinaryAction binaryAction;
                    binaryAction.a = a;
                    binaryAction.b = b;
                    binaryAction.c = c;
                    binaryAction.opcode = 8;
                    binaryAction.type = 1;
                    required.Binary.push_back(binaryAction);
                }
            }
            else
            {
                logError(ctx, "Invalid binary operation opcode=" + rom.line[zkPC].binOpcode);
                exitProcess();
            }
            pols.bin[i] = fr.one();
        }

        // MemAlign instruction
        if ( (rom.line[zkPC].memAlignRD==1) || (rom.line[zkPC].memAlignWR==1) || (rom.line[zkPC].memAlignWR8==1) )
        {
            mpz_class m0;
            if (!fea2scalar(fr, m0, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]))
            {
                proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                logError(ctx, "Failed calling fea2scalar(pols.A)");
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            mpz_class m1;
            if (!fea2scalar(fr, m1, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]))
            {
                proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                logError(ctx, "Failed calling fea2scalar(pols.B)");
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            mpz_class v;
            if (!fea2scalar(fr, v, op0, op1, op2, op3, op4, op5, op6, op7))
            {
                proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                logError(ctx, "Failed calling fea2scalar(op)");
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            mpz_class offsetScalar;
            if (!fea2scalar(fr, offsetScalar, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]))
            {
                proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                logError(ctx, "Failed calling fea2scalar(pols.C)");
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            if (offsetScalar<0 || offsetScalar>32)
            {
                proverRequest.result = ZKR_SM_MAIN_MEMALIGN_OFFSET_OUT_OF_RANGE;
                logError(ctx, "MemAlign out of range offset=" + offsetScalar.get_str());
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            uint64_t offset = offsetScalar.get_ui();

            if (rom.line[zkPC].memAlignRD==0 && rom.line[zkPC].memAlignWR==1 && rom.line[zkPC].memAlignWR8==0)
            {
                pols.memAlignWR[i] = fr.one();

                mpz_class w0;
                if (!fea2scalar(fr, w0, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.D)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                mpz_class w1;
                if (!fea2scalar(fr, w1, pols.E0[i], pols.E1[i], pols.E2[i], pols.E3[i], pols.E4[i], pols.E5[i], pols.E6[i], pols.E7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.E)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                mpz_class _W0;
                _W0 = (m0 & (ScalarTwoTo256 - (ScalarOne << (256-offset*8)))) | (v >> offset*8);
                mpz_class _W1;
                _W1 = (m1 & (ScalarMask256 >> offset*8)) | ((v << (256 - offset*8)) & ScalarMask256);
                if ( (w0 != _W0) || (w1 != _W1) )
                {
                    proverRequest.result = ZKR_SM_MAIN_MEMALIGN_WRITE_MISMATCH;
                    logError(ctx, "MemAlign w0, w1 invalid: w0=" + w0.get_str(16) + " w1=" + w1.get_str(16) + " _W0=" + _W0.get_str(16) + " _W1=" + _W1.get_str(16) + " m0=" + m0.get_str(16) + " m1=" + m1.get_str(16) + " offset=" + to_string(offset) + " v=" + v.get_str(16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                if (!bProcessBatch)
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
                    required.MemAlign.push_back(memAlignAction);
                }
            }
            else if (rom.line[zkPC].memAlignRD==0 && rom.line[zkPC].memAlignWR==0 && rom.line[zkPC].memAlignWR8==1)
            {
                pols.memAlignWR8[i] = fr.one();

                mpz_class w0;
                if (!fea2scalar(fr, w0, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]))
                {
                    proverRequest.result = ZKR_SM_MAIN_FEA2SCALAR;
                    logError(ctx, "Failed calling fea2scalar(pols.D)");
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                mpz_class _W0;
                mpz_class byteMaskOn256("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
                _W0 = (m0 & (byteMaskOn256 >> (offset*8))) | ((v & 0xFF) << ((31-offset)*8));
                if (w0 != _W0)
                {
                    proverRequest.result = ZKR_SM_MAIN_MEMALIGN_WRITE8_MISMATCH;
                    logError(ctx, "MemAlign w0 invalid: w0=" + w0.get_str(16) + " _W0=" + _W0.get_str(16) + " m0=" + m0.get_str(16) + " offset=" + to_string(offset) + " v=" + v.get_str(16));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                if (!bProcessBatch)
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
                    required.MemAlign.push_back(memAlignAction);
                }
            }
            else if (rom.line[zkPC].memAlignRD==1 && rom.line[zkPC].memAlignWR==0 && rom.line[zkPC].memAlignWR8==0)
            {
                pols.memAlignRD[i] = fr.one();

                mpz_class leftV;
                leftV = (m0 << offset*8) & ScalarMask256;
                mpz_class rightV;
                rightV = (m1 >> (256 - offset*8)) & (ScalarMask256 >> (256 - offset*8));
                mpz_class _V;
                _V = leftV | rightV;
                if (v != _V)
                {
                    proverRequest.result = ZKR_SM_MAIN_MEMALIGN_READ_MISMATCH;
                    logError(ctx, "MemAlign v invalid: v=" + v.get_str(16) + " _V=" + _V.get_str(16) + " m0=" + m0.get_str(16) + " m1=" + m1.get_str(16) + " offset=" + to_string(offset));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }

                if (!bProcessBatch)
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
                    required.MemAlign.push_back(memAlignAction);
                }
            }
            else
            {
                logError(ctx, "Invalid memAlign operation");
                exitProcess();
            }
        }

        // Repeat instruction
        if ((rom.line[zkPC].repeat == 1) && (!bProcessBatch))
        {
            pols.repeat[i] = fr.one();
        }

        /***********/
        /* SETTERS */
        /***********/

        // If setA, A'=op
        if (rom.line[zkPC].setA == 1) {
            pols.A0[nexti] = op0;
            pols.A1[nexti] = op1;
            pols.A2[nexti] = op2;
            pols.A3[nexti] = op3;
            pols.A4[nexti] = op4;
            pols.A5[nexti] = op5;
            pols.A6[nexti] = op6;
            pols.A7[nexti] = op7;
            pols.setA[i] = fr.one();
#ifdef LOG_SETX
            zklog.info("setA A[nexti]=" + fea2string(fr, pols.A0[nexti], pols.A1[nexti], pols.A2[nexti], pols.A3[nexti], pols.A4[nexti], pols.A5[nexti], pols.A6[nexti], pols.A7[nexti]));
#endif
        } else if (bUnsignedTransaction && (zkPC == checkAndSaveFromLabel)) {
            // Set A register with input.from to process unsigned transactions
            mpz_class from(proverRequest.input.from);
            scalar2fea(fr, from, pols.A0[nexti], pols.A1[nexti], pols.A2[nexti], pols.A3[nexti], pols.A4[nexti], pols.A5[nexti], pols.A6[nexti], pols.A7[nexti] );
        } else {
            pols.A0[nexti] = pols.A0[i];
            pols.A1[nexti] = pols.A1[i];
            pols.A2[nexti] = pols.A2[i];
            pols.A3[nexti] = pols.A3[i];
            pols.A4[nexti] = pols.A4[i];
            pols.A5[nexti] = pols.A5[i];
            pols.A6[nexti] = pols.A6[i];
            pols.A7[nexti] = pols.A7[i];
        }

        // If setB, B'=op
        if (rom.line[zkPC].setB == 1) {
            pols.B0[nexti] = op0;
            pols.B1[nexti] = op1;
            pols.B2[nexti] = op2;
            pols.B3[nexti] = op3;
            pols.B4[nexti] = op4;
            pols.B5[nexti] = op5;
            pols.B6[nexti] = op6;
            pols.B7[nexti] = op7;
            pols.setB[i] = fr.one();
#ifdef LOG_SETX
            zklog.info("setB B[nexti]=" + fea2string(fr, pols.B0[nexti], pols.B1[nexti], pols.B2[nexti], pols.B3[nexti], pols.B4[nexti], pols.B5[nexti], pols.B6[nexti], pols.B7[nexti]));
#endif
        } else {
            pols.B0[nexti] = pols.B0[i];
            pols.B1[nexti] = pols.B1[i];
            pols.B2[nexti] = pols.B2[i];
            pols.B3[nexti] = pols.B3[i];
            pols.B4[nexti] = pols.B4[i];
            pols.B5[nexti] = pols.B5[i];
            pols.B6[nexti] = pols.B6[i];
            pols.B7[nexti] = pols.B7[i];
        }

        // If setC, C'=op
        if (rom.line[zkPC].setC == 1) {
            pols.C0[nexti] = op0;
            pols.C1[nexti] = op1;
            pols.C2[nexti] = op2;
            pols.C3[nexti] = op3;
            pols.C4[nexti] = op4;
            pols.C5[nexti] = op5;
            pols.C6[nexti] = op6;
            pols.C7[nexti] = op7;
            pols.setC[i] = fr.one();
#ifdef LOG_SETX
            zklog.info("setC C[nexti]=" + fea2string(fr, pols.C0[nexti], pols.C1[nexti], pols.C2[nexti], pols.C3[nexti], pols.C4[nexti], pols.C5[nexti], pols.C6[nexti], pols.C7[nexti]));
#endif
        }            
        else if ((zkPC == verifyMerkleProofEndLabel) && proverRequest.input.bSkipVerifyL1InfoRoot)
        {
            // Set C register with input.l1InfoRoot to process unsigned transactions
            scalar2fea(fr, proverRequest.input.publicInputsExtended.publicInputs.l1InfoRoot,
                pols.C0[nexti],
                pols.C1[nexti],
                pols.C2[nexti],
                pols.C3[nexti],
                pols.C4[nexti],
                pols.C5[nexti],
                pols.C6[nexti],
                pols.C7[nexti]);
        }
        else
        {
            pols.C0[nexti] = pols.C0[i];
            pols.C1[nexti] = pols.C1[i];
            pols.C2[nexti] = pols.C2[i];
            pols.C3[nexti] = pols.C3[i];
            pols.C4[nexti] = pols.C4[i];
            pols.C5[nexti] = pols.C5[i];
            pols.C6[nexti] = pols.C6[i];
            pols.C7[nexti] = pols.C7[i];
        }

        // If setD, D'=op
        if (rom.line[zkPC].setD == 1) {
            pols.D0[nexti] = op0;
            pols.D1[nexti] = op1;
            pols.D2[nexti] = op2;
            pols.D3[nexti] = op3;
            pols.D4[nexti] = op4;
            pols.D5[nexti] = op5;
            pols.D6[nexti] = op6;
            pols.D7[nexti] = op7;
            pols.setD[i] = fr.one();
#ifdef LOG_SETX
            zklog.info("setD D[nexti]=" + fea2string(fr, pols.D0[nexti], pols.D1[nexti], pols.D2[nexti], pols.D3[nexti], pols.D4[nexti], pols.D5[nexti], pols.D6[nexti], pols.D7[nexti]));
#endif
        } else {
            pols.D0[nexti] = pols.D0[i];
            pols.D1[nexti] = pols.D1[i];
            pols.D2[nexti] = pols.D2[i];
            pols.D3[nexti] = pols.D3[i];
            pols.D4[nexti] = pols.D4[i];
            pols.D5[nexti] = pols.D5[i];
            pols.D6[nexti] = pols.D6[i];
            pols.D7[nexti] = pols.D7[i];
        }

        // If setE, E'=op
        if (rom.line[zkPC].setE == 1) {
            pols.E0[nexti] = op0;
            pols.E1[nexti] = op1;
            pols.E2[nexti] = op2;
            pols.E3[nexti] = op3;
            pols.E4[nexti] = op4;
            pols.E5[nexti] = op5;
            pols.E6[nexti] = op6;
            pols.E7[nexti] = op7;
            pols.setE[i] = fr.one();
#ifdef LOG_SETX
            zklog.info("setE E[nexti]=" + fea2string(fr, pols.E0[nexti], pols.E1[nexti], pols.E2[nexti], pols.E3[nexti], pols.E4[nexti], pols.E5[nexti], pols.E6[nexti], pols.E7[nexti]));
#endif
        } else {
            pols.E0[nexti] = pols.E0[i];
            pols.E1[nexti] = pols.E1[i];
            pols.E2[nexti] = pols.E2[i];
            pols.E3[nexti] = pols.E3[i];
            pols.E4[nexti] = pols.E4[i];
            pols.E5[nexti] = pols.E5[i];
            pols.E6[nexti] = pols.E6[i];
            pols.E7[nexti] = pols.E7[i];
        }

        // If setSR, SR'=op
        if (rom.line[zkPC].setSR == 1) {
            pols.SR0[nexti] = op0;
            pols.SR1[nexti] = op1;
            pols.SR2[nexti] = op2;
            pols.SR3[nexti] = op3;
            pols.SR4[nexti] = op4;
            pols.SR5[nexti] = op5;
            pols.SR6[nexti] = op6;
            pols.SR7[nexti] = op7;
            pols.setSR[i] = fr.one();
#ifdef LOG_SETX
            zklog.info("setSR SR[nexti]=" + fea2string(fr, pols.SR0[nexti], pols.SR1[nexti], pols.SR2[nexti], pols.SR3[nexti], pols.SR4[nexti], pols.SR5[nexti], pols.SR6[nexti], pols.SR7[nexti]));
#endif
        } else {
            pols.SR0[nexti] = pols.SR0[i];
            pols.SR1[nexti] = pols.SR1[i];
            pols.SR2[nexti] = pols.SR2[i];
            pols.SR3[nexti] = pols.SR3[i];
            pols.SR4[nexti] = pols.SR4[i];
            pols.SR5[nexti] = pols.SR5[i];
            pols.SR6[nexti] = pols.SR6[i];
            pols.SR7[nexti] = pols.SR7[i];
        }

        // If setCTX, CTX'=op
        if (rom.line[zkPC].setCTX == 1) {
            pols.CTX[nexti] = op0;
            pols.setCTX[i] = fr.one();
#ifdef LOG_SETX
            zklog.info("setCTX CTX[nexti]=" + fr.toString(pols.CTX[nexti], 16));
#endif
        } else {
            pols.CTX[nexti] = pols.CTX[i];
        }

        // If setSP, SP'=op
        if (rom.line[zkPC].setSP == 1) {
            pols.SP[nexti] = op0;
            pols.setSP[i] = fr.one();
#ifdef LOG_SETX
            zklog.info("setSP SP[nexti]=" + fr.toString(pols.SP[nexti], 16));
#endif
        } else {
            // SP' = SP + incStack
            pols.SP[nexti] = fr.add(pols.SP[i], fr.fromS32(rom.line[zkPC].incStack));
        }

        // If setPC, PC'=op
        if (rom.line[zkPC].setPC == 1) {
            pols.PC[nexti] = op0;
            pols.setPC[i] = fr.one();
#ifdef LOG_SETX
            zklog.info("setPC PC[nexti]=" + fr.toString(pols.PC[nexti], 16));
#endif
        } else {
            // PC' = PC
            pols.PC[nexti] = pols.PC[i];
        }

        // If setRR, RR'=op0
        if (rom.line[zkPC].setRR == 1)
        {
            pols.RR[nexti] = op0;
            if (!bProcessBatch) pols.setRR[i] = fr.one();
        }
        else if (rom.line[zkPC].call == 1)
        {
            pols.RR[nexti] = fr.fromU64(zkPC + 1);
        }
        else
        {
            pols.RR[nexti] = pols.RR[i];
        }

        // If arith, increment pols.cntArith
        if (!proverRequest.input.bNoCounters && (rom.line[zkPC].arithEq0==1 || rom.line[zkPC].arithEq1==1 || rom.line[zkPC].arithEq2==1 || rom.line[zkPC].arithEq3==1 || rom.line[zkPC].arithEq4==1 || rom.line[zkPC].arithEq5==1) ) {
            pols.cntArith[nexti] = fr.inc(pols.cntArith[i]);
#ifdef CHECK_MAX_CNT_ASAP
            if (fr.toU64(pols.cntArith[nexti]) > rom.constants.MAX_CNT_ARITH_LIMIT)
            {
                logError(ctx, "Main Executor found pols.cntArith[nexti]=" + fr.toString(pols.cntArith[nexti], 10) + " > MAX_CNT_ARITH_LIMIT_LIMIT=" + to_string(rom.constants.MAX_CNT_ARITH_LIMIT));
                if (bProcessBatch)
                {
                    proverRequest.result = ZKR_SM_MAIN_OOC_ARITH;
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                exitProcess();
            }
#endif
        } else {
            pols.cntArith[nexti] = pols.cntArith[i];
        }

        // If bin, increment pols.cntBinary
        if ((rom.line[zkPC].bin || rom.line[zkPC].sWR || rom.line[zkPC].hashPDigest ) && !proverRequest.input.bNoCounters) {
            pols.cntBinary[nexti] = fr.inc(pols.cntBinary[i]);
#ifdef CHECK_MAX_CNT_ASAP
            if (fr.toU64(pols.cntBinary[nexti]) > rom.constants.MAX_CNT_BINARY_LIMIT)
            {
                logError(ctx, "Main Executor found pols.cntBinary[nexti]=" + fr.toString(pols.cntBinary[nexti], 10) + " > MAX_CNT_BINARY_LIMIT=" + to_string(rom.constants.MAX_CNT_BINARY_LIMIT));
                if (bProcessBatch)
                {
                    proverRequest.result = ZKR_SM_MAIN_OOC_BINARY;
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                exitProcess();
            }
#endif
        } else {
            pols.cntBinary[nexti] = pols.cntBinary[i];
        }

        // If memAlign, increment pols.cntMemAlign
        if ( (rom.line[zkPC].memAlignRD || rom.line[zkPC].memAlignWR || rom.line[zkPC].memAlignWR8) && !proverRequest.input.bNoCounters) {
            pols.cntMemAlign[nexti] = fr.inc(pols.cntMemAlign[i]);
#ifdef CHECK_MAX_CNT_ASAP
            if (fr.toU64(pols.cntMemAlign[nexti]) > rom.constants.MAX_CNT_MEM_ALIGN_LIMIT)
            {
                logError(ctx, "Main Executor found pols.cntMemAlign[nexti]=" + fr.toString(pols.cntMemAlign[nexti], 10) + " > MAX_CNT_MEM_ALIGN_LIMIT=" + to_string(rom.constants.MAX_CNT_MEM_ALIGN_LIMIT));
                if (bProcessBatch)
                {
                    proverRequest.result = ZKR_SM_MAIN_OOC_MEM_ALIGN;
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                exitProcess();
            }
#endif
        } else {
            pols.cntMemAlign[nexti] = pols.cntMemAlign[i];
        }

        // If setRCX, RCX=op, else if RCX>0, RCX--
        if (rom.line[zkPC].setRCX)
        {
            pols.RCX[nexti] = op0;
            if (!bProcessBatch)
                pols.setRCX[i] = fr.one();
        }
        else if (rom.line[zkPC].repeat)
        {
            currentRCX = pols.RCX[i];
            if (!fr.isZero(pols.RCX[i]))
            {
                pols.RCX[nexti] = fr.dec(pols.RCX[i]);
            }
        }
        else
        {
            pols.RCX[nexti] = pols.RCX[i];
        }

        // Calculate the inverse of RCX (if not zero)
        if (!bProcessBatch)
        {
            if (!fr.isZero(pols.RCX[nexti]))
            {
                pols.RCXInv[nexti] = fr.inv(pols.RCX[nexti]);
            }
        }

        if (rom.line[zkPC].bJmpAddrPresent && !bProcessBatch)
        {
            pols.jmpAddr[i] = rom.line[zkPC].jmpAddr;
        }
        if (rom.line[zkPC].useJmpAddr == 1 && !bProcessBatch)
        {
            pols.useJmpAddr[i] = fr.one();
        }
        if (rom.line[zkPC].useElseAddr == 1 && !bProcessBatch)
        {
            pols.useElseAddr[i] = fr.one();
        }

        if (!bProcessBatch)
        {
            if (rom.line[zkPC].useElseAddr == 1)
            {
                zkassert(rom.line[zkPC].bElseAddrPresent);
                pols.elseAddr[i] = rom.line[zkPC].elseAddr;
            }
        }

        /*********/
        /* JUMPS */
        /*********/

        // If JMPN, jump conditionally if op0<0
        if (rom.line[zkPC].JMPN == 1)
        {
#ifdef LOG_JMP
            zklog.info("JMPN: op0=" + fr.toString(op0));
#endif
            if (rom.line[zkPC].jmpAddr == fr.fromU64(outOfCountersStepLabel))
            {
                int64_t reserve = int64_t(rom.constants.MAX_CNT_STEPS) - fr.toS64(op0);
                if (reserve < 0)
                {
                    reserve = 0;
                }
                proverRequest.counters_reserve.steps = zkmax(proverRequest.counters_reserve.steps, uint64_t(reserve));
            }
            else if (rom.line[zkPC].jmpAddr == fr.fromU64(outOfCountersArithLabel))
            {
                int64_t reserve = int64_t(rom.constants.MAX_CNT_ARITH) - fr.toS64(op0);
                if (reserve < 0)
                {
                    reserve = 0;
                }
                proverRequest.counters_reserve.arith = zkmax(proverRequest.counters_reserve.arith, uint64_t(reserve));
            }
            else if (rom.line[zkPC].jmpAddr == fr.fromU64(outOfCountersBinaryLabel))
            {
                int64_t reserve = int64_t(rom.constants.MAX_CNT_BINARY) - fr.toS64(op0);
                if (reserve < 0)
                {
                    reserve = 0;
                }
                proverRequest.counters_reserve.binary = zkmax(proverRequest.counters_reserve.binary, uint64_t(reserve));
            }
            else if (rom.line[zkPC].jmpAddr == fr.fromU64(outOfCountersKeccakLabel))
            {
                int64_t reserve = int64_t(rom.constants.MAX_CNT_KECCAK_F) - fr.toS64(op0);
                if (reserve < 0)
                {
                    reserve = 0;
                }
                proverRequest.counters_reserve.keccakF = zkmax(proverRequest.counters_reserve.keccakF, uint64_t(reserve));
            }
            else if (rom.line[zkPC].jmpAddr == fr.fromU64(outOfCountersSha256Label))
            {
                int64_t reserve = int64_t(rom.constants.MAX_CNT_SHA256_F) - fr.toS64(op0);
                if (reserve < 0)
                {
                    reserve = 0;
                }
                proverRequest.counters_reserve.sha256F = zkmax(proverRequest.counters_reserve.sha256F, uint64_t(reserve));
            }
            else if (rom.line[zkPC].jmpAddr == fr.fromU64(outOfCountersMemalignLabel))
            {
                int64_t reserve = int64_t(rom.constants.MAX_CNT_MEM_ALIGN) - fr.toS64(op0);
                if (reserve < 0)
                {
                    reserve = 0;
                }
                proverRequest.counters_reserve.memAlign = zkmax(proverRequest.counters_reserve.memAlign, uint64_t(reserve));
            }
            else if (rom.line[zkPC].jmpAddr == fr.fromU64(outOfCountersPoseidonLabel))
            {
                int64_t reserve = int64_t(rom.constants.MAX_CNT_POSEIDON_G) - fr.toS64(op0);
                if (reserve < 0)
                {
                    reserve = 0;
                }
                proverRequest.counters_reserve.poseidonG = zkmax(proverRequest.counters_reserve.poseidonG, uint64_t(reserve));
            }
            else if (rom.line[zkPC].jmpAddr == fr.fromU64(outOfCountersPaddingLabel))
            {
                int64_t reserve = int64_t(rom.constants.MAX_CNT_PADDING_PG) - fr.toS64(op0);
                if (reserve < 0)
                {
                    reserve = 0;
                }
                proverRequest.counters_reserve.paddingPG = zkmax(proverRequest.counters_reserve.paddingPG, uint64_t(reserve));
            }

            uint64_t jmpnCondValue = fr.toU64(op0);

            // If op<0, jump to addr: zkPC'=addr
            if (jmpnCondValue >= FrFirst32Negative)
            {
                pols.isNeg[i] = fr.one();
                if (rom.line[zkPC].useJmpAddr)
                    pols.zkPC[nexti] = rom.line[zkPC].jmpAddr;
                else
                    pols.zkPC[nexti] = fr.fromU64(addr);
                jmpnCondValue = fr.toU64(fr.add(op0, fr.fromU64(0x100000000)));
#ifdef LOG_JMP
                zklog.info("JMPN next zkPC(1)=" + fr.toString(pols.zkPC[nexti]));
#endif
            }
            // If op>=0, simply increase zkPC'=zkPC+1
            else if (jmpnCondValue <= FrLast32Positive)
            {
                if (rom.line[zkPC].useElseAddr)
                {
                    if (bUnsignedTransaction && (rom.line[zkPC].elseAddrLabel == "invalidIntrinsicTxSenderCode"))
                    {
                        pols.zkPC[nexti] = rom.line[zkPC].useJmpAddr ? rom.line[zkPC].jmpAddr : fr.fromU64(addr);
                    }
                    else
                    {
                        pols.zkPC[nexti] = rom.line[zkPC].elseAddr;
                    }
                }
                else
                {
                    pols.zkPC[nexti] = fr.inc(pols.zkPC[i]);
                }
#ifdef LOG_JMP
                zklog.info("JMPN next zkPC(2)=" + fr.toString(pols.zkPC[nexti]));
#endif
            }
            else
            {
                proverRequest.result = ZKR_SM_MAIN_S33;
                logError(ctx, "JMPN invalid S33 value op0=" + to_string(jmpnCondValue));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            pols.lJmpnCondValue[i] = fr.fromU64(jmpnCondValue & 0x7FFFFF);
            jmpnCondValue = jmpnCondValue >> 23;
            for (uint64_t index = 0; index < 9; ++index)
            {
                pols.hJmpnCondValueBit[index][i] = fr.fromU64(jmpnCondValue & 0x01);
                jmpnCondValue = jmpnCondValue >> 1;
            }
            pols.JMPN[i] = fr.one();
        }
        // If JMPC, jump conditionally if carry
        else if (rom.line[zkPC].JMPC == 1)
        {
            // If carry, jump to addr: zkPC'=addr
            if (!fr.isZero(pols.carry[i]))
            {
                if (rom.line[zkPC].useJmpAddr)
                    pols.zkPC[nexti] = rom.line[zkPC].jmpAddr;
                else
                    pols.zkPC[nexti] = fr.fromU64(addr);
#ifdef LOG_JMP
               zklog.info("JMPC next zkPC(3)=" + fr.toString(pols.zkPC[nexti]));
#endif
            }
            // If not carry, simply increase zkPC'=zkPC+1
            else
            {
                if (rom.line[zkPC].useElseAddr)
                {
                    if (bUnsignedTransaction && (rom.line[zkPC].elseAddrLabel == "invalidIntrinsicTxSenderCode"))
                    {
                        pols.zkPC[nexti] = rom.line[zkPC].useJmpAddr ? rom.line[zkPC].jmpAddr : fr.fromU64(addr);
                    }
                    else
                    {
                        pols.zkPC[nexti] = rom.line[zkPC].elseAddr;
                    }
                }
                else
                {
                    pols.zkPC[nexti] = fr.inc(pols.zkPC[i]);
                }
#ifdef LOG_JMP
                zklog.info("JMPC next zkPC(4)=" + fr.toString(pols.zkPC[nexti]));
#endif
            }
            pols.JMPC[i] = fr.one();
        }
        // If JMPZ, jump
        else if (rom.line[zkPC].JMPZ)
        {
            if (fr.isZero(op0))
            {
                if (rom.line[zkPC].useJmpAddr)
                    pols.zkPC[nexti] = rom.line[zkPC].jmpAddr;
                else
                    pols.zkPC[nexti] = fr.fromU64(addr);
            }
            else
            {
                if (rom.line[zkPC].useElseAddr)
                {
                    if (bUnsignedTransaction && (rom.line[zkPC].elseAddrLabel == "invalidIntrinsicTxSenderCode"))
                    {
                        pols.zkPC[nexti] = rom.line[zkPC].useJmpAddr ? rom.line[zkPC].jmpAddr : fr.fromU64(addr);
                    }
                    else
                    {
                        pols.zkPC[nexti] = rom.line[zkPC].elseAddr;
                    }
                }
                else
                {
                    pols.zkPC[nexti] = fr.inc(pols.zkPC[i]);
                }
            }
            pols.JMPZ[i] = fr.one();
        }
        // If JMP, directly jump zkPC'=addr
        else if (rom.line[zkPC].JMP == 1)
        {
            if (rom.line[zkPC].useJmpAddr)
                pols.zkPC[nexti] = rom.line[zkPC].jmpAddr;
            else
                pols.zkPC[nexti] = fr.fromU64(addr);
#ifdef LOG_JMP
            zklog.info("JMP next zkPC(5)=" + fr.toString(pols.zkPC[nexti]));
#endif
            pols.JMP[i] = fr.one();
        }
        // If call, jump to finalJmpAddr
        else if (rom.line[zkPC].call == 1)
        {
            if (rom.line[zkPC].useJmpAddr)
                pols.zkPC[nexti] = rom.line[zkPC].jmpAddr;
            else
                pols.zkPC[nexti] = fr.fromU64(addr);
            pols.call[i] = fr.one();
        }
        // If return, jump back to RR
        else if (rom.line[zkPC].return_ == 1)
        {
            pols.zkPC[nexti] = pols.RR[i];
            pols.return_pol[i] = fr.one();
        }
        // Else, repeat, leave the same zkPC
        else if (rom.line[zkPC].repeat && !fr.isZero(currentRCX))
        {
            pols.zkPC[nexti] = pols.zkPC[i];
        }
        // Else, simply increase zkPC'=zkPC+1
        else
        {
            pols.zkPC[nexti] = fr.inc(pols.zkPC[i]);
        }

        // If setGAS, GAS'=op
        if (rom.line[zkPC].setGAS == 1) {
            pols.GAS[nexti] = op0;
            pols.setGAS[i] = fr.one();
#ifdef LOG_SETX
            zklog.info("setGAS GAS[nexti]=" + fr.toString(pols.GAS[nexti]));
#endif
        } else {
            pols.GAS[nexti] = pols.GAS[i];
        }

        // If setHASHPOS, HASHPOS' = op0 + incHashPos
        if (rom.line[zkPC].setHASHPOS == 1) {
            pols.HASHPOS[nexti] = fr.add(op0, fr.fromU64(incHashPos));
            pols.setHASHPOS[i] = fr.one();
        } else {
            pols.HASHPOS[nexti] = fr.add( pols.HASHPOS[i], fr.fromU64(incHashPos) );
        }

        if (rom.line[zkPC].sRD || rom.line[zkPC].sWR || rom.line[zkPC].hashKDigest || rom.line[zkPC].hashPDigest || rom.line[zkPC].hashSDigest)
        {
            pols.incCounter[i] = fr.fromU64(incCounter);
        }

        if (rom.line[zkPC].hashKDigest && !proverRequest.input.bNoCounters)
        {
            pols.cntKeccakF[nexti] = fr.add(pols.cntKeccakF[i], fr.fromU64(incCounter));
#ifdef CHECK_MAX_CNT_ASAP
            if (fr.toU64(pols.cntKeccakF[nexti]) > rom.constants.MAX_CNT_KECCAK_F_LIMIT)
            {
                logError(ctx, "Main Executor found pols.cntKeccakF[nexti]=" + fr.toString(pols.cntKeccakF[nexti], 10) + " > MAX_CNT_KECCAK_F_LIMIT=" + to_string(rom.constants.MAX_CNT_KECCAK_F_LIMIT));
                if (bProcessBatch)
                {
                    proverRequest.result = ZKR_SM_MAIN_OOC_KECCAK_F;
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                exitProcess();
            }
#endif
        }
        else
        {
            pols.cntKeccakF[nexti] = pols.cntKeccakF[i];
        }

        if (rom.line[zkPC].hashPDigest && !proverRequest.input.bNoCounters)
        {
            pols.cntPaddingPG[nexti] = fr.add(pols.cntPaddingPG[i], fr.fromU64(incCounter));
#ifdef CHECK_MAX_CNT_ASAP
            if (fr.toU64(pols.cntPaddingPG[nexti]) > rom.constants.MAX_CNT_PADDING_PG_LIMIT)
            {
                logError(ctx, "Main Executor found pols.cntPaddingPG[nexti]=" + fr.toString(pols.cntPaddingPG[nexti], 10) + " > MAX_CNT_PADDING_PG_LIMIT=" + to_string(rom.constants.MAX_CNT_PADDING_PG_LIMIT));
                if (bProcessBatch)
                {
                    proverRequest.result = ZKR_SM_MAIN_OOC_PADDING_PG;
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                exitProcess();
            }
#endif
        }
        else
        {
            pols.cntPaddingPG[nexti] = pols.cntPaddingPG[i];
        }

        if (rom.line[zkPC].hashSDigest && !proverRequest.input.bNoCounters)
        {
            pols.cntSha256F[nexti] = fr.add(pols.cntSha256F[i], fr.fromU64(incCounter));
#ifdef CHECK_MAX_CNT_ASAP
            if (fr.toU64(pols.cntSha256F[nexti]) > rom.constants.MAX_CNT_SHA256_F_LIMIT)
            {
                logError(ctx, "Main Executor found pols.cntSha256F[nexti]=" + fr.toString(pols.cntSha256F[nexti], 10) + " > MAX_CNT_SHA256_F_LIMIT=" + to_string(rom.constants.MAX_CNT_SHA256_F_LIMIT));
                if (bProcessBatch)
                {
                    proverRequest.result = ZKR_SM_MAIN_OOC_SHA256_F;
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                exitProcess();
            }
#endif
        }
        else
        {
            pols.cntSha256F[nexti] = pols.cntSha256F[i];
        }

        if ((rom.line[zkPC].sRD || rom.line[zkPC].sWR || rom.line[zkPC].hashPDigest) && !proverRequest.input.bNoCounters)
        {
            pols.cntPoseidonG[nexti] = fr.add(pols.cntPoseidonG[i], fr.fromU64(incCounter));
#ifdef CHECK_MAX_CNT_ASAP
            if (fr.toU64(pols.cntPoseidonG[nexti]) > rom.constants.MAX_CNT_POSEIDON_G_LIMIT)
            {
                logError(ctx, "Main Executor found pols.cntPoseidonG[nexti]=" + fr.toString(pols.cntPoseidonG[nexti], 10) + " > MAX_CNT_POSEIDON_G_LIMIT=" + to_string(rom.constants.MAX_CNT_POSEIDON_G_LIMIT));
                if (bProcessBatch)
                {
                    proverRequest.result = ZKR_SM_MAIN_OOC_POSEIDON_G;
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
                exitProcess();
            }
#endif
        }
        else
        {
            pols.cntPoseidonG[nexti] = pols.cntPoseidonG[i];
        }

        // Evaluate the list cmdAfter commands of the previous ROM line,
        // and any children command, recursively
        if ( (rom.line[zkPC].cmdAfter.size() > 0) && (step < (N_Max - 1)) )
        {
            if (!bProcessBatch) i++;
            for (uint64_t j=0; j<rom.line[zkPC].cmdAfter.size(); j++)
            {
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                gettimeofday(&t, NULL);
#endif
                CommandResult cr;
                evalCommand(ctx, *rom.line[zkPC].cmdAfter[j], cr);

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                mainMetrics.add("Eval command", TimeDiff(t));
                evalCommandMetrics.add(rom.line[zkPC].cmdAfter[j]->opAndFunction, TimeDiff(t));
#endif
                // In case of an external error, return it
                if (cr.zkResult != ZKR_SUCCESS)
                {
                    proverRequest.result = cr.zkResult;
                    logError(ctx, string("Failed calling evalCommand() after result=") + zkresult2string(proverRequest.result));
                    pHashDB->cancelBatch(proverRequest.uuid);
                    return;
                }
            }
            if (!bProcessBatch) i--;
        }

#ifdef LOG_COMPLETED_STEPS
        zklog.info("<-- Completed step=" + to_string(step) +
            " zkPC=" + to_string(zkPC) +
            " op=" + fr.toString(op7,16) + ":" + fr.toString(op6,16) + ":" + fr.toString(op5,16) + ":" + fr.toString(op4,16) + ":" + fr.toString(op3,16) + ":" + fr.toString(op2,16) + ":" + fr.toString(op1,16) + ":" + fr.toString(op0,16) +
            " ABCDE0=" + fr.toString(pols.A0[nexti],16) + ":" + fr.toString(pols.B0[nexti],16) + ":" + fr.toString(pols.C0[nexti],16) + ":" + fr.toString(pols.D0[nexti],16) + ":" + fr.toString(pols.E0[nexti],16) +
            " FREE0:7=" + fr.toString(pols.FREE0[i],16) + ":" + fr.toString(pols.FREE7[i],16) +
            " addr=" + to_string(addr));
#endif
#ifdef LOG_COMPLETED_STEPS_TO_FILE
        std::ofstream outfile;
        outfile.open("c.txt", std::ios_base::app); // append instead of overwrite
        outfile << "<-- Completed step=" << step << " zkPC=" << zkPC << " op=" << fr.toString(op7,16) << ":" << fr.toString(op6,16) << ":" << fr.toString(op5,16) << ":" << fr.toString(op4,16) << ":" << fr.toString(op3,16) << ":" << fr.toString(op2,16) << ":" << fr.toString(op1,16) << ":" << fr.toString(op0,16) << " ABCDE0=" << fr.toString(pols.A0[nexti],16) << ":" << fr.toString(pols.B0[nexti],16) << ":" << fr.toString(pols.C0[nexti],16) << ":" << fr.toString(pols.D0[nexti],16) << ":" << fr.toString(pols.E0[nexti],16) << " FREE0:7=" << fr.toString(pols.FREE0[i],16) << ":" << fr.toString(pols.FREE7[i],16) << " addr=" << addr << endl;
        /*outfile << "<-- Completed step=" << step << " zkPC=" << zkPC <<
                   " op=" << fr.toString(op7,16) << ":" << fr.toString(op6,16) << ":" << fr.toString(op5,16) << ":" << fr.toString(op4,16) << ":" << fr.toString(op3,16) << ":" << fr.toString(op2,16) << ":" << fr.toString(op1,16) << ":" << fr.toString(op0,16) <<
                   " A=" << fr.toString(pols.A7[nexti],16) << ":" << fr.toString(pols.A6[nexti],16) << ":" << fr.toString(pols.A5[nexti],16) << ":" << fr.toString(pols.A4[nexti],16) << ":" << fr.toString(pols.A3[nexti],16) << ":" << fr.toString(pols.A2[nexti],16) << ":" << fr.toString(pols.A1[nexti],16) << ":" << fr.toString(pols.A0[nexti],16) <<
                   " B=" << fr.toString(pols.B7[nexti],16) << ":" << fr.toString(pols.B6[nexti],16) << ":" << fr.toString(pols.B5[nexti],16) << ":" << fr.toString(pols.B4[nexti],16) << ":" << fr.toString(pols.B3[nexti],16) << ":" << fr.toString(pols.B2[nexti],16) << ":" << fr.toString(pols.B1[nexti],16) << ":" << fr.toString(pols.B0[nexti],16) <<
                   " C=" << fr.toString(pols.C7[nexti],16) << ":" << fr.toString(pols.C6[nexti],16) << ":" << fr.toString(pols.C5[nexti],16) << ":" << fr.toString(pols.C4[nexti],16) << ":" << fr.toString(pols.C3[nexti],16) << ":" << fr.toString(pols.C2[nexti],16) << ":" << fr.toString(pols.C1[nexti],16) << ":" << fr.toString(pols.C0[nexti],16) <<
                   " D=" << fr.toString(pols.D7[nexti],16) << ":" << fr.toString(pols.D6[nexti],16) << ":" << fr.toString(pols.D5[nexti],16) << ":" << fr.toString(pols.D4[nexti],16) << ":" << fr.toString(pols.D3[nexti],16) << ":" << fr.toString(pols.D2[nexti],16) << ":" << fr.toString(pols.D1[nexti],16) << ":" << fr.toString(pols.D0[nexti],16) <<
                   " E=" << fr.toString(pols.E7[nexti],16) << ":" << fr.toString(pols.E6[nexti],16) << ":" << fr.toString(pols.E5[nexti],16) << ":" << fr.toString(pols.E4[nexti],16) << ":" << fr.toString(pols.E3[nexti],16) << ":" << fr.toString(pols.E2[nexti],16) << ":" << fr.toString(pols.E1[nexti],16) << ":" << fr.toString(pols.E0[nexti],16) <<
                   " FREE0:7=" << fr.toString(pols.FREE0[i],16) << ":" << fr.toString(pols.FREE7[i],16) <<
                   " addr=" << addr << endl;*/
        outfile.close();
        //if (i==1000) break;
#endif

        // When processing a txs batch, break the loop when done to complete the execution faster
        if ( zkPC == finalizeExecutionLabel )
        {
            if (ctx.lastStep != 0)
            {
                logError(ctx, "Called finalizeExecutionLabel with a non-zero ctx.lastStep=" + to_string(ctx.lastStep));
                exitProcess();
            }
            ctx.lastStep = step;
            if (bProcessBatch)
            {
                break;
            }
        }

    } // End of main executor loop, for all evaluations

    // Copy the counters
    proverRequest.counters.arith = fr.toU64(pols.cntArith[0]);
    proverRequest.counters.binary = fr.toU64(pols.cntBinary[0]);
    proverRequest.counters.keccakF = fr.toU64(pols.cntKeccakF[0]);
    proverRequest.counters.memAlign = fr.toU64(pols.cntMemAlign[0]);
    proverRequest.counters.paddingPG = fr.toU64(pols.cntPaddingPG[0]);
    proverRequest.counters.poseidonG = fr.toU64(pols.cntPoseidonG[0]);
    proverRequest.counters.sha256F = fr.toU64(pols.cntSha256F[0]);
    proverRequest.counters.steps = ctx.lastStep;
    proverRequest.counters_reserve.arith = zkmax(proverRequest.counters_reserve.arith, proverRequest.counters.arith);
    proverRequest.counters_reserve.binary = zkmax(proverRequest.counters_reserve.binary, proverRequest.counters.binary);
    proverRequest.counters_reserve.keccakF = zkmax(proverRequest.counters_reserve.keccakF, proverRequest.counters.keccakF);
    proverRequest.counters_reserve.memAlign = zkmax(proverRequest.counters_reserve.memAlign, proverRequest.counters.memAlign);
    proverRequest.counters_reserve.paddingPG = zkmax(proverRequest.counters_reserve.paddingPG, proverRequest.counters.paddingPG);
    proverRequest.counters_reserve.poseidonG = zkmax(proverRequest.counters_reserve.poseidonG, proverRequest.counters.poseidonG);
    proverRequest.counters_reserve.sha256F = zkmax(proverRequest.counters_reserve.sha256F, proverRequest.counters.sha256F);
    proverRequest.counters_reserve.steps = zkmax(proverRequest.counters_reserve.steps, proverRequest.counters.steps);

    // Set the error (all previous errors generated a return)
    proverRequest.result = ZKR_SUCCESS;

    // Check that we did not run out of steps during the execution
    if (ctx.lastStep == 0)
    {
        proverRequest.result = ZKR_SM_MAIN_OUT_OF_STEPS;
        logError(ctx, "Found ctx.lastStep=0, so execution was not complete");
        if (!bProcessBatch)
        {
            exitProcess();
        }
    }
    if (!proverRequest.input.bNoCounters && (ctx.lastStep > rom.constants.MAX_CNT_STEPS_LIMIT))
    {
        proverRequest.result = ZKR_SM_MAIN_OUT_OF_STEPS;
        logError(ctx, "Found ctx.lastStep=" + to_string(ctx.lastStep) + " > MAX_CNT_STEPS_LIMIT=" + to_string(rom.constants.MAX_CNT_STEPS_LIMIT));
        if (!bProcessBatch)
        {
            exitProcess();
        }
    }

#ifdef CHECK_MAX_CNT_AT_THE_END
    if (!proverRequest.input.bNoCounters && (fr.toU64(pols.cntArith[0]) > rom.constants.MAX_CNT_ARITH_LIMIT))
    {
        proverRequest.result = ZKR_SM_MAIN_OOC_ARITH;
        logError(ctx, "Found pols.cntArith[0]=" + to_string(fr.toU64(pols.cntArith[0])) + " > MAX_CNT_ARITH_LIMIT=" + to_string(rom.constants.MAX_CNT_ARITH_LIMIT));
        if (!bProcessBatch)
        {
            exitProcess();
        }
    }
    if (!proverRequest.input.bNoCounters && (fr.toU64(pols.cntBinary[0]) > rom.constants.MAX_CNT_BINARY_LIMIT))
    {
        proverRequest.result = ZKR_SM_MAIN_OOC_BINARY;
        logError(ctx, "Found pols.cntBinary[0]=" + to_string(fr.toU64(pols.cntBinary[0])) + " > MAX_CNT_BINARY_LIMIT=" + to_string(rom.constants.MAX_CNT_BINARY_LIMIT));
        if (!bProcessBatch)
        {
            exitProcess();
        }
    }
    if (!proverRequest.input.bNoCounters && (fr.toU64(pols.cntMemAlign[0]) > rom.constants.MAX_CNT_MEM_ALIGN_LIMIT))
    {
        proverRequest.result = ZKR_SM_MAIN_OOC_MEM_ALIGN;
        logError(ctx, "Found pols.cntMemAlign[0]=" + to_string(fr.toU64(pols.cntMemAlign[0])) + " > MAX_CNT_MEM_ALIGN_LIMIT=" + to_string(rom.constants.MAX_CNT_MEM_ALIGN_LIMIT));
        if (!bProcessBatch)
        {
            exitProcess();
        }
    }
    if (!proverRequest.input.bNoCounters && (fr.toU64(pols.cntKeccakF[0]) > rom.constants.MAX_CNT_KECCAK_F_LIMIT))
    {
        proverRequest.result = ZKR_SM_MAIN_OOC_KECCAK_F;
        logError(ctx, "Found pols.cntKeccakF[0]=" + to_string(fr.toU64(pols.cntKeccakF[0])) + " > MAX_CNT_KECCAK_F_LIMIT=" + to_string(rom.constants.MAX_CNT_KECCAK_F_LIMIT));
        if (!bProcessBatch)
        {
            exitProcess();
        }
    }
    if (!proverRequest.input.bNoCounters && (fr.toU64(pols.cntPaddingPG[0]) > rom.constants.MAX_CNT_PADDING_PG_LIMIT))
    {
        proverRequest.result = ZKR_SM_MAIN_OOC_PADDING_PG;
        logError(ctx, "Found pols.cntPaddingPG[0]=" + to_string(fr.toU64(pols.cntPaddingPG[0])) + " > MAX_CNT_PADDING_PG_LLIMIT=" + to_string(rom.constants.MAX_CNT_PADDING_PG_LIMIT));
        if (!bProcessBatch)
        {
            exitProcess();
        }
    }
    if (!proverRequest.input.bNoCounters && (fr.toU64(pols.cntPoseidonG[0]) > rom.constants.MAX_CNT_POSEIDON_G_LIMIT))
    {
        proverRequest.result = ZKR_SM_MAIN_OOC_POSEIDON_G;
        logError(ctx, "Found pols.cntPoseidonG[0]=" + to_string(fr.toU64(pols.cntPoseidonG[0])) + " > MAX_CNT_POSEIDON_G_LIMIT=" + to_string(rom.constants.MAX_CNT_POSEIDON_G_LIMIT));
        if (!bProcessBatch)
        {
            exitProcess();
        }
    }
    if (!proverRequest.input.bNoCounters && (fr.toU64(pols.cntSha256F[0]) > rom.constants.MAX_CNT_SHA256_F_LIMIT))
    {
        proverRequest.result = ZKR_SM_MAIN_OOC_SHA256_F;
        logError(ctx, "Found pols.cntSha256F[0]=" + to_string(fr.toU64(pols.cntSha256F[0])) + " > MAX_CNT_SHA256_F_LIMIT=" + to_string(rom.constants.MAX_CNT_SHA256_F_LIMIT));
        if (!bProcessBatch)
        {
            exitProcess();
        }
    }
#endif

    //printRegs(ctx);
    //printVars(ctx);
    //printMem(ctx);
    //printStorage(ctx);
    //printDb(ctx);

    if (!bProcessBatch) // In fast mode, last nexti was not 0 but 1, and pols have only 2 evaluations
    {
        // Check that all registers have the correct final state
        checkFinalState(ctx);
        assertOutputs(ctx);

        // Generate Padding KK required data
        for (uint64_t i=0; i<ctx.hashK.size(); i++)
        {
            PaddingKKExecutorInput h;
            h.dataBytes = ctx.hashK[i].data;
            uint64_t p = 0;
            while (p<ctx.hashK[i].data.size())
            {
                if (ctx.hashK[i].reads[p] != 0)
                {
                    h.reads.push_back(ctx.hashK[i].reads[p]);
                    p += ctx.hashK[i].reads[p];
                }
                else
                {
                    h.reads.push_back(1);
                    p++;
                }
            }
            if (p != ctx.hashK[i].data.size())
            {
                proverRequest.result = ZKR_SM_MAIN_HASHK_READ_OUT_OF_RANGE;
                logError(ctx, "Reading hashK out of limits: i=" + to_string(i) + " p=" + to_string(p) + " ctx.hashK[i].data.size()=" + to_string(ctx.hashK[i].data.size()));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            h.digestCalled = ctx.hashK[i].digestCalled;
            h.lenCalled = ctx.hashK[i].lenCalled;
            required.PaddingKK.push_back(h);
        }

        // Generate Padding PG required data
        for (uint64_t i=0; i<ctx.hashP.size(); i++)
        {
            PaddingPGExecutorInput h;
            h.dataBytes = ctx.hashP[i].data;
            uint64_t p = 0;
            while (p<ctx.hashP[i].data.size())
            {
                if (ctx.hashP[i].reads[p] != 0)
                {
                    h.reads.push_back(ctx.hashP[i].reads[p]);
                    p += ctx.hashP[i].reads[p];
                }
                else
                {
                    h.reads.push_back(1);
                    p++;
                }
            }
            if (p != ctx.hashP[i].data.size())
            {
                proverRequest.result = ZKR_SM_MAIN_HASHP_READ_OUT_OF_RANGE;
                logError(ctx, "Reading hashP out of limits: i=" + to_string(i) + " p=" + to_string(p) + " ctx.hashP[i].data.size()=" + to_string(ctx.hashP[i].data.size()));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            h.digestCalled = ctx.hashP[i].digestCalled;
            h.lenCalled = ctx.hashP[i].lenCalled;
            required.PaddingPG.push_back(h);
        }

        // Generate Padding SHA required data
        for (uint64_t i=0; i<ctx.hashS.size(); i++)
        {
            PaddingSha256ExecutorInput h;
            h.dataBytes = ctx.hashS[i].data;
            uint64_t p = 0;
            while (p<ctx.hashS[i].data.size())
            {
                if (ctx.hashS[i].reads[p] != 0)
                {
                    h.reads.push_back(ctx.hashS[i].reads[p]);
                    p += ctx.hashS[i].reads[p];
                }
                else
                {
                    h.reads.push_back(1);
                    p++;
                }
            }
            if (p != ctx.hashS[i].data.size())
            {
                proverRequest.result = ZKR_SM_MAIN_HASHS_READ_OUT_OF_RANGE;
                logError(ctx, "Reading hashS out of limits: i=" + to_string(i) + " p=" + to_string(p) + " ctx.hashS[i].data.size()=" + to_string(ctx.hashS[i].data.size()));
                pHashDB->cancelBatch(proverRequest.uuid);
                return;
            }
            h.digestCalled = ctx.hashS[i].digestCalled;
            h.lenCalled = ctx.hashS[i].lenCalled;
            required.PaddingSha256.push_back(h);
        }
    }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    gettimeofday(&t, NULL);
#endif

    if (config.hashDB64)
    {
        /*Goldilocks::Element newStateRoot[4];
        string2fea(fr, NormalizeToNFormat(proverRequest.pFullTracer->get_new_state_root(), 64), newStateRoot);
        zkresult zkr;
        zkr = pHashDB->purge(proverRequest.uuid, newStateRoot, proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE);
        if (zkr != ZKR_SUCCESS)
        {
            proverRequest.result = zkr;
            logError(ctx, string("Failed calling pHashDB->purge() result=") + zkresult2string(zkr));
            pHashDB->cancelBatch(proverRequest.uuid);
            return;
        }*/

        /*zkr = pHashDB->flush(proverRequest.uuid, proverRequest.pFullTracer->get_new_state_root(), proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, proverRequest.flushId, proverRequest.lastSentFlushId);
        if (zkr != ZKR_SUCCESS)
        {
            proverRequest.result = zkr;
            logError(ctx, string("Failed calling pHashDB->flush() result=") + zkresult2string(zkr));
            pHashDB->cancelBatch(proverRequest.uuid);
            return;
        }*/

        /*Goldilocks::Element consolidatedStateRoot[4];
        zkr = pHashDB->consolidateState(newStateRoot, proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, consolidatedStateRoot, proverRequest.flushId, proverRequest.lastSentFlushId);
        if (zkr != ZKR_SUCCESS)
        {
            proverRequest.result = zkr;
            logError(ctx, string("Failed calling pHashDB->consolidateState() result=") + zkresult2string(zkr));
            pHashDB->cancelBatch(proverRequest.uuid);
            return;
        }*/
    }
    else
    {
        zkresult zkr = pHashDB->flush(proverRequest.uuid, proverRequest.pFullTracer->get_new_state_root(), proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, proverRequest.flushId, proverRequest.lastSentFlushId);
        if (zkr != ZKR_SUCCESS)
        {
            proverRequest.result = zkr;
            logError(ctx, string("Failed calling pHashDB->flush() result=") + zkresult2string(zkr));
            pHashDB->cancelBatch(proverRequest.uuid);
            return;
        }
    }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    mainMetrics.add("Flush", TimeDiff(t));
#endif

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    if (config.executorTimeStatistics)
    {
        mainMetrics.print("Main Executor calls");
        evalCommandMetrics.print("Main Executor eval command calls");
    }
#endif

    if (config.dbMetrics)
    {
        proverRequest.dbReadLog->print();
    }

    zklog.info("MainExecutor::execute() done lastStep=" + to_string(ctx.lastStep) + " (" + to_string((double(ctx.lastStep)*100)/N) + "%)", &proverRequest.tags);

    TimerStopAndLog(MAIN_EXECUTOR_EXECUTE);
}

// Initialize the first evaluation
void MainExecutor::initState(Context &ctx)
{
    // Set oldStateRoot to register B
    scalar2fea(fr, ctx.proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot, ctx.pols.B0[0], ctx.pols.B1[0], ctx.pols.B2[0], ctx.pols.B3[0], ctx.pols.B4[0], ctx.pols.B5[0], ctx.pols.B6[0], ctx.pols.B7[0]);

    // Set oldAccInputHash to register C
    scalar2fea(fr, ctx.proverRequest.input.publicInputsExtended.publicInputs.oldAccInputHash, ctx.pols.C0[0], ctx.pols.C1[0], ctx.pols.C2[0], ctx.pols.C3[0], ctx.pols.C4[0], ctx.pols.C5[0], ctx.pols.C6[0], ctx.pols.C7[0]);

    // Set oldNumBatch to SP register
    ctx.pols.SP[0] = fr.fromU64(ctx.proverRequest.input.publicInputsExtended.publicInputs.oldBatchNum);

    // Set chainID to GAS register
    ctx.pols.GAS[0] = fr.fromU64(ctx.proverRequest.input.publicInputsExtended.publicInputs.chainID);

    // Set fork ID to CTX register
    ctx.pols.CTX[0] = fr.fromU64(ctx.proverRequest.input.publicInputsExtended.publicInputs.forkID);
}

// Check that last evaluation (which is in fact the first one) is zero
void MainExecutor::checkFinalState(Context &ctx)
{
    if (
        (!fr.isZero(ctx.pols.A0[0])) ||
        (!fr.isZero(ctx.pols.A1[0])) ||
        (!fr.isZero(ctx.pols.A2[0])) ||
        (!fr.isZero(ctx.pols.A3[0])) ||
        (!fr.isZero(ctx.pols.A4[0])) ||
        (!fr.isZero(ctx.pols.A5[0])) ||
        (!fr.isZero(ctx.pols.A6[0])) ||
        (!fr.isZero(ctx.pols.A7[0])) ||
        (!fr.isZero(ctx.pols.D0[0])) ||
        (!fr.isZero(ctx.pols.D1[0])) ||
        (!fr.isZero(ctx.pols.D2[0])) ||
        (!fr.isZero(ctx.pols.D3[0])) ||
        (!fr.isZero(ctx.pols.D4[0])) ||
        (!fr.isZero(ctx.pols.D5[0])) ||
        (!fr.isZero(ctx.pols.D6[0])) ||
        (!fr.isZero(ctx.pols.D7[0])) ||
        (!fr.isZero(ctx.pols.E0[0])) ||
        (!fr.isZero(ctx.pols.E1[0])) ||
        (!fr.isZero(ctx.pols.E2[0])) ||
        (!fr.isZero(ctx.pols.E3[0])) ||
        (!fr.isZero(ctx.pols.E4[0])) ||
        (!fr.isZero(ctx.pols.E5[0])) ||
        (!fr.isZero(ctx.pols.E6[0])) ||
        (!fr.isZero(ctx.pols.E7[0])) ||
        (!fr.isZero(ctx.pols.SR0[0])) ||
        (!fr.isZero(ctx.pols.SR1[0])) ||
        (!fr.isZero(ctx.pols.SR2[0])) ||
        (!fr.isZero(ctx.pols.SR3[0])) ||
        (!fr.isZero(ctx.pols.SR4[0])) ||
        (!fr.isZero(ctx.pols.SR5[0])) ||
        (!fr.isZero(ctx.pols.SR6[0])) ||
        (!fr.isZero(ctx.pols.SR7[0])) ||
        (!fr.isZero(ctx.pols.PC[0])) ||
        (!fr.isZero(ctx.pols.zkPC[0]))
    )
    {
        logError(ctx, "MainExecutor::checkFinalState() Program terminated with registers A, D, E, SR, CTX, PC, zkPC not set to zero");
        exitProcess();
    }

    Goldilocks::Element feaOldStateRoot[8];
    scalar2fea(fr, ctx.proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot, feaOldStateRoot);
    if (
        (!fr.equal(ctx.pols.B0[0], feaOldStateRoot[0])) ||
        (!fr.equal(ctx.pols.B1[0], feaOldStateRoot[1])) ||
        (!fr.equal(ctx.pols.B2[0], feaOldStateRoot[2])) ||
        (!fr.equal(ctx.pols.B3[0], feaOldStateRoot[3])) ||
        (!fr.equal(ctx.pols.B4[0], feaOldStateRoot[4])) ||
        (!fr.equal(ctx.pols.B5[0], feaOldStateRoot[5])) ||
        (!fr.equal(ctx.pols.B6[0], feaOldStateRoot[6])) ||
        (!fr.equal(ctx.pols.B7[0], feaOldStateRoot[7])) )
    {
        mpz_class bScalar;
        if (!fea2scalar(ctx.fr, bScalar, ctx.pols.B0[0], ctx.pols.B1[0], ctx.pols.B2[0], ctx.pols.B3[0], ctx.pols.B4[0], ctx.pols.B5[0], ctx.pols.B6[0], ctx.pols.B7[0]))
        {
            logError(ctx, "MainExecutor::checkFinalState() failed calling fea2scalar(pols.B)");
        }
        logError(ctx, "MainExecutor::checkFinalState() Register B=" + bScalar.get_str(16) + " not terminated equal as its initial value=" + ctx.proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot.get_str(16));
        exitProcess();
    }

    Goldilocks::Element feaOldAccInputHash[8];
    scalar2fea(fr, ctx.proverRequest.input.publicInputsExtended.publicInputs.oldAccInputHash, feaOldAccInputHash);
    if (
        (!fr.equal(ctx.pols.C0[0], feaOldAccInputHash[0])) ||
        (!fr.equal(ctx.pols.C1[0], feaOldAccInputHash[1])) ||
        (!fr.equal(ctx.pols.C2[0], feaOldAccInputHash[2])) ||
        (!fr.equal(ctx.pols.C3[0], feaOldAccInputHash[3])) ||
        (!fr.equal(ctx.pols.C4[0], feaOldAccInputHash[4])) ||
        (!fr.equal(ctx.pols.C5[0], feaOldAccInputHash[5])) ||
        (!fr.equal(ctx.pols.C6[0], feaOldAccInputHash[6])) ||
        (!fr.equal(ctx.pols.C7[0], feaOldAccInputHash[7])) )
    {
        mpz_class cScalar;
        if (!fea2scalar(ctx.fr, cScalar, ctx.pols.C0[0], ctx.pols.C1[0], ctx.pols.C2[0], ctx.pols.C3[0], ctx.pols.C4[0], ctx.pols.C5[0], ctx.pols.C6[0], ctx.pols.C7[0]))
        {
            logError(ctx, "MainExecutor::checkFinalState() failed calling fea2scalar(pols.C)");
        }
        logError(ctx, "MainExecutor::checkFinalState() Register C=" + cScalar.get_str(16) + " not terminated equal as its initial value=" + ctx.proverRequest.input.publicInputsExtended.publicInputs.oldAccInputHash.get_str(16));
        exitProcess();
    }

    if (!fr.equal(ctx.pols.SP[0], fr.fromU64(ctx.proverRequest.input.publicInputsExtended.publicInputs.oldBatchNum)))
    {
        logError(ctx, "MainExecutor::checkFinalState() Register SP not terminated equal as its initial value");
        exitProcess();
    }

    if (!fr.equal(ctx.pols.GAS[0], fr.fromU64(ctx.proverRequest.input.publicInputsExtended.publicInputs.chainID)))
    {
        logError(ctx, "MainExecutor::checkFinalState() Register GAS not terminated equal as its initial value");
        exitProcess();
    }

    if (!fr.equal(ctx.pols.CTX[0], fr.fromU64(ctx.proverRequest.input.publicInputsExtended.publicInputs.forkID)))
    {
        logError(ctx, "MainExecutor::checkFinalState() Register CTX not terminated equal as its initial value");
        exitProcess();
    }
}

void MainExecutor::assertOutputs(Context &ctx)
{
    uint64_t step = *ctx.pStep;

    if ( ctx.proverRequest.input.publicInputsExtended.newStateRoot != 0 )
    {
        Goldilocks::Element feaNewStateRoot[8];
        scalar2fea(fr, ctx.proverRequest.input.publicInputsExtended.newStateRoot, feaNewStateRoot);

        if (
            (!fr.equal(ctx.pols.SR0[step], feaNewStateRoot[0])) ||
            (!fr.equal(ctx.pols.SR1[step], feaNewStateRoot[1])) ||
            (!fr.equal(ctx.pols.SR2[step], feaNewStateRoot[2])) ||
            (!fr.equal(ctx.pols.SR3[step], feaNewStateRoot[3])) ||
            (!fr.equal(ctx.pols.SR4[step], feaNewStateRoot[4])) ||
            (!fr.equal(ctx.pols.SR5[step], feaNewStateRoot[5])) ||
            (!fr.equal(ctx.pols.SR6[step], feaNewStateRoot[6])) ||
            (!fr.equal(ctx.pols.SR7[step], feaNewStateRoot[7])) )
        {
            mpz_class auxScalar;
            if (!fea2scalar(fr, auxScalar, ctx.pols.SR0[step], ctx.pols.SR1[step], ctx.pols.SR2[step], ctx.pols.SR3[step], ctx.pols.SR4[step], ctx.pols.SR5[step], ctx.pols.SR6[step], ctx.pols.SR7[step]))
            {
                logError(ctx, "MainExecutor::assertOutputs() failed calling fea2scalar(pols.SR)");
            }
            logError(ctx, "MainExecutor::assertOutputs() Register SR=" + auxScalar.get_str(16) + " not terminated equal to newStateRoot=" + ctx.proverRequest.input.publicInputsExtended.newStateRoot.get_str(16));
            exitProcess();
        }
    }

    if ( ctx.proverRequest.input.publicInputsExtended.newAccInputHash != 0 )
    {
        Goldilocks::Element feaNewAccInputHash[8];
        scalar2fea(fr, ctx.proverRequest.input.publicInputsExtended.newAccInputHash, feaNewAccInputHash);

        if (
            (!fr.equal(ctx.pols.D0[step], feaNewAccInputHash[0])) ||
            (!fr.equal(ctx.pols.D1[step], feaNewAccInputHash[1])) ||
            (!fr.equal(ctx.pols.D2[step], feaNewAccInputHash[2])) ||
            (!fr.equal(ctx.pols.D3[step], feaNewAccInputHash[3])) ||
            (!fr.equal(ctx.pols.D4[step], feaNewAccInputHash[4])) ||
            (!fr.equal(ctx.pols.D5[step], feaNewAccInputHash[5])) ||
            (!fr.equal(ctx.pols.D6[step], feaNewAccInputHash[6])) ||
            (!fr.equal(ctx.pols.D7[step], feaNewAccInputHash[7])) )
        {
            mpz_class auxScalar;
            if (!fea2scalar(fr, auxScalar, ctx.pols.D0[step], ctx.pols.D1[step], ctx.pols.D2[step], ctx.pols.D3[step], ctx.pols.D4[step], ctx.pols.D5[step], ctx.pols.D6[step], ctx.pols.D7[step]))
            {
                logError(ctx, "MainExecutor::assertOutputs() failed calling fea2scalar(pols.D)");
            }
            logError(ctx, "MainExecutor::assertOutputs() Register D=" + auxScalar.get_str(16) + " not terminated equal to newAccInputHash=" + ctx.proverRequest.input.publicInputsExtended.newAccInputHash.get_str(16));
            exitProcess();
        }
    }

    if ( ctx.proverRequest.input.publicInputsExtended.newLocalExitRoot != 0 )
    {
        Goldilocks::Element feaNewLocalExitRoot[8];
        scalar2fea(fr, ctx.proverRequest.input.publicInputsExtended.newLocalExitRoot, feaNewLocalExitRoot);

        if (
            (!fr.equal(ctx.pols.E0[step], feaNewLocalExitRoot[0])) ||
            (!fr.equal(ctx.pols.E1[step], feaNewLocalExitRoot[1])) ||
            (!fr.equal(ctx.pols.E2[step], feaNewLocalExitRoot[2])) ||
            (!fr.equal(ctx.pols.E3[step], feaNewLocalExitRoot[3])) ||
            (!fr.equal(ctx.pols.E4[step], feaNewLocalExitRoot[4])) ||
            (!fr.equal(ctx.pols.E5[step], feaNewLocalExitRoot[5])) ||
            (!fr.equal(ctx.pols.E6[step], feaNewLocalExitRoot[6])) ||
            (!fr.equal(ctx.pols.E7[step], feaNewLocalExitRoot[7])) )
        {
            mpz_class auxScalar;
            if (!fea2scalar(fr, auxScalar, ctx.pols.E0[step], ctx.pols.E1[step], ctx.pols.E2[step], ctx.pols.E3[step], ctx.pols.E4[step], ctx.pols.E5[step], ctx.pols.E6[step], ctx.pols.E7[step]))
            {
                logError(ctx, "MainExecutor::assertOutputs() failed calling fea2scalar(pols.E)");
            }
            logError(ctx, "MainExecutor::assertOutputs() Register E=" + auxScalar.get_str(16) + " not terminated equal to newLocalExitRoot=" + ctx.proverRequest.input.publicInputsExtended.newLocalExitRoot.get_str(16));
            exitProcess();
        }
    }

    if (ctx.proverRequest.input.publicInputsExtended.newBatchNum != 0)
    {
        if (!fr.equal(ctx.pols.PC[step], fr.fromU64(ctx.proverRequest.input.publicInputsExtended.newBatchNum)))
        {
            logError(ctx, "MainExecutor::assertOutputs() Register PC=" + to_string(fr.toU64(ctx.pols.PC[step])) + " not terminated equal to newBatchNum=" + to_string(ctx.proverRequest.input.publicInputsExtended.newBatchNum));
            exitProcess();
        }
    }
}

void MainExecutor::logError (Context &ctx, const string &message)
{
    // Log the message, if provided
    string log0 = "MainExecutor::logError()";
    string log1;
    if (message.size() > 0)
    {
        log1 = message;
        zklog.error(log0 + " " + log1);
    }

    // Log details
#define INVALID_LOG_ERROR_VALUE 999999999
    uint64_t step = (ctx.pStep != NULL) ? *ctx.pStep : INVALID_LOG_ERROR_VALUE;
    uint64_t evaluation = (ctx.pEvaluation != NULL) ? *ctx.pEvaluation : INVALID_LOG_ERROR_VALUE;
    uint64_t zkpc = (ctx.pZKPC != NULL) ? *ctx.pZKPC : INVALID_LOG_ERROR_VALUE;
    string romLine = (ctx.pZKPC != NULL) ? rom.line[*ctx.pZKPC].toString(fr) : "INVALID_ZKPC";
    string log2 = string("proverRequest.result=") + zkresult2string(ctx.proverRequest.result) +
        " step=" + to_string(step) +
        " eval=" + to_string(evaluation) +
        " zkPC=" + to_string(zkpc) +
        " rom.line={" + romLine +
        "} uuid=" + ctx.proverRequest.uuid;
    zklog.error(log0 + " " + log2, &ctx.proverRequest.tags);

    // Log registers
    string log3;
    ctx.printRegs(log3);

    // Log the input file content
    json inputJson;
    ctx.proverRequest.input.save(inputJson);
    zklog.error("Input=" + inputJson.dump());

    ctx.proverRequest.errorLog = log0 + " " + log1 + " " + log2 + " " + log3;
}

void MainExecutor::linearPoseidon (Context &ctx, const vector<uint8_t> &data, Goldilocks::Element (&result)[4])
{
    poseidonLinearHash(data, result);
}

} // namespace