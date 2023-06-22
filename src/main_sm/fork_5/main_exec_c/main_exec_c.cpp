#include "main_sm/fork_5/main_exec_c/main_exec_c.hpp"
#include "main_sm/fork_5/main_exec_c/context_c.hpp"
#include "main_sm/fork_5/main_exec_c/variables_c.hpp"
#include "main_sm/fork_5/main_exec_c/batch_decode.hpp"
#include "main_sm/fork_5/main_exec_c/account.hpp"
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
    Goldilocks::Element oldRoot[4];
    scalar2fea(fr, proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot, oldRoot); // TODO: Check range?

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
        mpz_class fromPublicKey;  // TODO: Call ecrecover()

        // from.account.balance -= value + gas*gasPrice;
        Account fromAccount(fr, poseidon, *pHashDB);
        result = fromAccount.Init(fromPublicKey);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling fromAccount.Init()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
        Account toAccount(fr, poseidon, *pHashDB);
        result = toAccount.Init(batchData.tx[tx].to);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling toAccount.Init()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }

        // Get balance of from account
        mpz_class fromBalance;
        result = fromAccount.GetBalance(oldRoot, fromBalance);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling fromAccount.GetBalance()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }

        // Get balance of to account
        mpz_class toBalance;
        result = toAccount.GetBalance(oldRoot, toBalance);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling toAccount.GetBalance()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }

        // Calculate gas
        mpz_class gas = 21000; // Transfer with no data, no CALLDATA cost, no deployment cost

        // Check that gas is not higher than gas limit
        if (gas > batchData.tx[tx].gasLimit)
        {
            zklog.error("main_exec_c() failed gas=" + gas.get_str(10) + " < gasLimit=" + to_string(batchData.tx[tx].gasLimit));
            proverRequest.result = ZKR_UNSPECIFIED;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }

        // Check that from account has enough balance to complete the transfer
        mpz_class fromAmount = batchData.tx[tx].value + batchData.tx[tx].gasPrice*gas;
        if (fromBalance < fromAmount)
        {
            zklog.error("main_exec_c() failed fromBalance=" + fromBalance.get_str(10) + " < fromAmount=" + fromAmount.get_str(10));
            proverRequest.result = ZKR_UNSPECIFIED;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }

        // Update from account balance = balance - value - gas*gasPrice
        fromBalance -= fromAmount;
        result = fromAccount.SetBalance(fromBalance);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling fromAccount.SetBalance()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }

        // Update to account balance = balance + value
        toBalance += batchData.tx[tx].value;
        result = toAccount.SetBalance(toBalance);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling toAccount.SetBalance()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
    }

    /*result = ((fork_5::FullTracer *)ctx.proverRequest.pFullTracer)->onFinishBatch(ctx, cmd);
    if (result != ZKR_SUCCESS)
    {
        zklog.error("main_exec_c() failed calling onFinishBatch()");
        proverRequest.result = result;
        HashDBClientFactory::freeHashDBClient(pHashDB);
        return;
    }*/

    HashDBClientFactory::freeHashDBClient(pHashDB);
    proverRequest.result = ZKR_SUCCESS;
    TimerStopAndLog(MAIN_EXEC_C);
}

}