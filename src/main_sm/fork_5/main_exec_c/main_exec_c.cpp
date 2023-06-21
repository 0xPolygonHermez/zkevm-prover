#include "main_sm/fork_5/main_exec_c/main_exec_c.hpp"
#include "main_sm/fork_5/main_exec_c/context_c.hpp"
#include "main_sm/fork_5/main_exec_c/variables_c.hpp"
#include "main_sm/fork_5/main_exec_c/batch_decode.hpp"
#include "main_sm/fork_5/main/eval_command.hpp"
#include "main_sm/fork_5/main/context.hpp"
#include "scalar.hpp"
#include <fstream>
#include "utils.hpp"
#include "timer.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "poseidon_g_permutation.hpp"
#include "utils/time_metric.hpp"
#include "zklog.hpp"

namespace fork_5
{
void MainExecutorC::execute (ProverRequest &proverRequest)
{
    TimerStart(MAIN_EXEC_C);
    ContextC ctxc;

    uint8_t polsBuffer[CommitPols::numPols()*sizeof(Goldilocks::Element)] = { 0 };
    MainCommitPols pols((void *)polsBuffer, 1);

    // Get a HashDBInterface interface, according to the configuration
    HashDBInterface *pHashDB = HashDBClientFactory::createHashDBClient(mainExecutor.fr, mainExecutor.config);
    if (pHashDB == NULL)
    {
        zklog.error("main_exec_c() failed calling HashDBClientFactory::createHashDBClient() uuid=" + proverRequest.uuid);
        proverRequest.result = ZKR_DB_ERROR;
        return;
    }

    // Copy input database content into context database
    if (proverRequest.input.db.size() > 0)
    {
        pHashDB->loadDB(proverRequest.input.db, true);
        uint64_t flushId, lastSentFlushId;
        pHashDB->flush(flushId, lastSentFlushId);
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
        pHashDB->flush(flushId, lastSentFlushId);
        if (config.dbClearCache && (config.databaseURL != "local"))
        {
            pHashDB->clearCache();
        }
    }

    // Init execution flags
    bool bProcessBatch = (proverRequest.type == prt_processBatch);
    bool bUnsignedTransaction = (proverRequest.input.from != "") && (proverRequest.input.from != "0x");

    // Unsigned transactions (from!=empty) are intended to be used to "estimage gas" (or "call")
    // In prover mode, we cannot accept unsigned transactions, since the proof would not meet the PIL constrains
    if (bUnsignedTransaction && !bProcessBatch)
    {
        proverRequest.result = ZKR_SM_MAIN_INVALID_UNSIGNED_TX;
        zklog.error("main_exec_c) failed called with bUnsignedTransaction=true but bProcessBatch=false");
        HashDBClientFactory::freeHashDBClient(pHashDB);
        return;
    }

    Context ctx(mainExecutor.fr, mainExecutor.config, mainExecutor.fec, mainExecutor.fnec, pols, mainExecutor.rom, proverRequest, pHashDB);

    /****************************/
    /* A - Load input variables */
    /****************************/
 
    //    STEP => A
    //    0                                   :ASSERT ; Ensure it is the beginning of the execution
    // No need to check STEP
 
    //    CTX                                 :MSTORE(forkID)
    //    CTX - %FORK_ID                      :JMPNZ(failAssert)

    // Check that forkID is correct
    if (proverRequest.input.publicInputsExtended.publicInputs.forkID != 5) // fork_5
    {
        zklog.error("main_exec_c() called with invalid forkID=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.forkID));
        proverRequest.result = ZKR_SM_MAIN_INVALID_FORK_ID;
        HashDBClientFactory::freeHashDBClient(pHashDB);
        return;
    }

    /*
        B                                   :MSTORE(oldStateRoot)
        C                                   :MSTORE(oldAccInputHash)
        SP                                  :MSTORE(oldNumBatch)
        GAS                                 :MSTORE(chainID) ; assumed to be less than 32 bits

        ${getGlobalExitRoot()}              :MSTORE(globalExitRoot)
        ${getSequencerAddr()}               :MSTORE(sequencerAddr)
        ${getTimestamp()}                   :MSTORE(timestamp)
        ${getTxsLen()}                      :MSTORE(batchL2DataLength) ; less than 300.000 bytes. Enforced by the smart contract

        B => SR ;set initial state root

        ; Increase batch number
        SP + 1                              :MSTORE(newNumBatch)
    */

    // Set initial state root
    ctxc.regs.SR = proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot;

    // Set oldAccInputHash
    ctxc.globalVars.oldAccInputHash = proverRequest.input.publicInputsExtended.publicInputs.oldAccInputHash;

    // Set newNumBatch
    proverRequest.input.publicInputsExtended.newBatchNum = proverRequest.input.publicInputsExtended.publicInputs.oldBatchNum + 1;

    // Set chainID;
    ctxc.globalVars.chainID = proverRequest.input.publicInputsExtended.publicInputs.chainID;
    // assumed to be less than 32 bits  TODO: Should we check and return an error?

    // Set globalExitRoot
    ctxc.globalVars.globalExitRoot = proverRequest.input.publicInputsExtended.publicInputs.globalExitRoot;

    // Set sequencerAddr
    ctxc.globalVars.sequencerAddr = proverRequest.input.publicInputsExtended.publicInputs.sequencerAddr;

    // Set timestamp
    ctxc.globalVars.timestamp = proverRequest.input.publicInputsExtended.publicInputs.timestamp;

    // Set batchL2DataLength
    ctxc.globalVars.batchL2DataLength = proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.size();

    BatchData batchData;
    zkresult result = BatchDecode(proverRequest.input.publicInputsExtended.publicInputs.batchL2Data, batchData);
    if (result != ZKR_SUCCESS)
    {
        zklog.error("main_exec_c() failed calling BatchDecode()");
        proverRequest.result = result;
        HashDBClientFactory::freeHashDBClient(pHashDB);
        return;
    }

    /*RomCommand cmd;
    result = ((fork_5::FullTracer *)ctx.proverRequest.pFullTracer)->onStartBatch(ctx, cmd);
    if (result != ZKR_SUCCESS)
    {
        zklog.error("main_exec_c() failed calling onStartBatch()");
        proverRequest.result = result;
        HashDBClientFactory::freeHashDBClient(pHashDB);
        return;
    }*/

    for (uint64_t tx=0; tx<batchData.tx.size(); tx++)
    {
        // Log TX info
        zklog.info("main_exec_c() processing tx=" + to_string(tx));
        batchData.tx[tx].print();

        // calculate tx hash
        string signHash = batchData.tx[tx].signHash();
        zklog.info("signHash=" + signHash);

        // Verify signature and obtain public from key
        //ecRecover(r, s, v, hash) -> obtain public key

        // from.account.balance -= value + gas*gasPrice;
        //SMT.get -> balance > x
        //SMT.set -> balance = y = x - value - gas*gasPrice

        // to.account.balance += value
        //SMT.get -> balance = x
        //SMT.set -> balance = y = x + value
    }

    /*result = ((fork_5::FullTracer *)ctx.proverRequest.pFullTracer)->onFinishBatch(ctx, cmd);
    if (result != ZKR_SUCCESS)
    {
        zklog.error("main_exec_c() failed calling onFinishBatch()");
        proverRequest.result = result;
        HashDBClientFactory::freeHashDBClient(pHashDB);
        return;
    }*/

    proverRequest.result = ZKR_SUCCESS;
    TimerStopAndLog(MAIN_EXEC_C);
}

zkresult MainExecutorC::SLOAD(Context &ctx, ContextC &ctxc, const mpz_class &a, const mpz_class &b, const mpz_class &c, mpz_class &value)
{
    Goldilocks::Element A[8];
    scalar2fea(fr, a, A);
    Goldilocks::Element B[8];
    scalar2fea(fr, b, B);
    Goldilocks::Element C[8];
    scalar2fea(fr, c, C);

    Goldilocks::Element Kin0[12];
    Kin0[0] = C[0];
    Kin0[1] = C[1];
    Kin0[2] = C[2];
    Kin0[3] = C[3];
    Kin0[4] = C[4];
    Kin0[5] = C[5];
    Kin0[6] = C[6];
    Kin0[7] = C[7];
    Kin0[8] = fr.zero();
    Kin0[9] = fr.zero();
    Kin0[10] = fr.zero();
    Kin0[11] = fr.zero();

    Goldilocks::Element Kin1[12];
    Kin1[0] = A[0];
    Kin1[1] = A[1];
    Kin1[2] = A[2];
    Kin1[3] = A[3];
    Kin1[4] = A[4];
    Kin1[5] = A[5];
    Kin1[6] = B[0];
    Kin1[7] = B[1];

    if  ( !fr.isZero(A[5]) || !fr.isZero(A[6]) || !fr.isZero(A[7]) || !fr.isZero(B[2]) || !fr.isZero(B[3]) || !fr.isZero(B[4]) || !fr.isZero(B[5])|| !fr.isZero(B[6])|| !fr.isZero(B[7]) )
    {
        zklog.error("MainExecutorC::SLOAD() found non-zero A-B storage registers");
        return ZKR_SM_MAIN_STORAGE;
    }

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

    Goldilocks::Element oldRoot[4];
    scalar2fea(fr, ctxc.regs.SR, oldRoot);

    //SmtGetResult smtGetResult;
    //mpz_class value;
    zkresult zkResult = ctx.pHashDB->get(oldRoot, key, value, /*&smtGetResult*/NULL, ctx.proverRequest.dbReadLog);
    if (zkResult != ZKR_SUCCESS)
    {
        zklog.error("MainExecutorC::SLOAD() failed calling pHashDB->get() result=" + zkresult2string(zkResult));
        return zkResult;
    }
    //incCounter = smtGetResult.proofHashCounter + 2;
                    //cout << "smt.get() returns value=" << smtGetResult.value.get_str(16) << endl;

    /*if (bProcessBatch)
    {
        zkResult = eval_addReadWriteAddress(ctx, smtGetResult.value);
        if (zkResult != ZKR_SUCCESS)
        {
            proverRequest.result = zkResult;
            logError(ctx, string("Failed calling eval_addReadWriteAddress() 1 result=") + zkresult2string(zkResult));
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
    }*/
/*

                    SmtGetResult smtGetResult;
                    mpz_class value;
                    zkresult zkResult = pHashDB->get(oldRoot, key, value, &smtGetResult, proverRequest.dbReadLog);
                    if (zkResult != ZKR_SUCCESS)
                    {
                        proverRequest.result = zkResult;
                        logError(ctx, string("Failed calling pHashDB->get() result=") + zkresult2string(zkResult));
                        HashDBClientFactory::freeHashDBClient(pHashDB);
                        return;
                    }
                    incCounter = smtGetResult.proofHashCounter + 2;
                    //cout << "smt.get() returns value=" << smtGetResult.value.get_str(16) << endl;

                    if (bProcessBatch)
                    {
                        zkResult = eval_addReadWriteAddress(ctx, smtGetResult.value);
                        if (zkResult != ZKR_SUCCESS)
                        {
                            proverRequest.result = zkResult;
                            logError(ctx, string("Failed calling eval_addReadWriteAddress() 1 result=") + zkresult2string(zkResult));
                            HashDBClientFactory::freeHashDBClient(pHashDB);
                            return;
                        }
                    }

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
                    mainMetrics.add("SMT Get", TimeDiff(t));
#endif

                    scalar2fea(fr, smtGetResult.value, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);

                    nHits++;
#ifdef LOG_STORAGE
                    zklog.info("Storage read sRD read from key: " + ctx.fr.toString(ctx.lastSWrite.key, 16) + " value:" + fr.toString(fi3, 16) + ":" + fr.toString(fi2, 16) + ":" + fr.toString(fi1, 16) + ":" + fr.toString(fi0, 16));
#endif

*/
    return ZKR_SUCCESS;
}

}