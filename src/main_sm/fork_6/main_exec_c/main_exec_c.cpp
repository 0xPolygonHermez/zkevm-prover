#include "main_sm/fork_6/main_exec_c/main_exec_c.hpp"
#include "main_sm/fork_6/main_exec_c/context_c.hpp"
#include "main_sm/fork_6/main_exec_c/variables_c.hpp"
#include "main_sm/fork_6/main_exec_c/batch_decode.hpp"
#include "main_sm/fork_6/main_exec_c/account.hpp"
#include "main_sm/fork_6/main/eval_command.hpp"
#include "main_sm/fork_6/main/context.hpp"
#include "scalar.hpp"
#include <fstream>
#include "utils.hpp"
#include "timer.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "poseidon_g_permutation.hpp"
#include "utils/time_metric.hpp"
#include "zklog.hpp"
#include "ecrecover.hpp"

namespace fork_6
{
void MainExecutorC::execute (ProverRequest &proverRequest)
{
    TimerStart(MAIN_EXEC_C);

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    struct timeval t;
    TimeMetricStorage mainMetrics;
#endif

    // Get a HashDBInterface interface, according to the configuration
    HashDBInterface *pHashDB = HashDBClientFactory::createHashDBClient(mainExecutor.fr, mainExecutor.config);
    if (pHashDB == NULL)
    {
        zklog.error("main_exec_c() failed calling HashDBClientFactory::createHashDBClient() uuid=" + proverRequest.uuid);
        proverRequest.result = ZKR_DB_ERROR;
        return;
    }

    // Create context
    ContextC ctxc(mainExecutor.fr, mainExecutor.config, proverRequest, pHashDB);

    // Copy input database content into context database
    if (proverRequest.input.db.size() > 0)
    {
        pHashDB->loadDB(proverRequest.input.db, true);
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

    // Check that forkID is correct
    if (proverRequest.input.publicInputsExtended.publicInputs.forkID != 6) // fork_6
    {
        zklog.error("main_exec_c() called with invalid forkID=" + to_string(proverRequest.input.publicInputsExtended.publicInputs.forkID));
        proverRequest.result = ZKR_SM_MAIN_INVALID_FORK_ID;
        HashDBClientFactory::freeHashDBClient(pHashDB);
        return;
    }

    // Set initial state root
    scalar2fea(fr, proverRequest.input.publicInputsExtended.publicInputs.oldStateRoot, ctxc.root); // TODO: Check range?
    zklog.info("Old root=" + fea2string(fr, ctxc.root));

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
    zklog.info("sequencer=" + ctxc.globalVars.sequencerAddr.get_str(16));

    // Set timestamp
    ctxc.globalVars.timestamp = proverRequest.input.publicInputsExtended.publicInputs.timestamp;

    // Set batchL2DataLength
    ctxc.globalVars.batchL2DataLength = proverRequest.input.publicInputsExtended.publicInputs.batchL2Data.size();

    // Decode batch L2 data
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    gettimeofday(&t, NULL);
#endif
    zkresult result = BatchDecode(proverRequest.input.publicInputsExtended.publicInputs.batchL2Data, ctxc.batch);
    if (result != ZKR_SUCCESS)
    {
        zklog.error("main_exec_c() failed calling BatchDecode()");
        proverRequest.result = result;
        HashDBClientFactory::freeHashDBClient(pHashDB);
        return;
    }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    mainMetrics.add("Batch L2 data decode", TimeDiff(t));
#endif

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    gettimeofday(&t, NULL);
#endif
    result = ((fork_6::FullTracer *)proverRequest.pFullTracer)->onStartBatch(ctxc);
    if (result != ZKR_SUCCESS)
    {
        zklog.error("main_exec_c() failed calling onStartBatch()");
        proverRequest.result = result;
        HashDBClientFactory::freeHashDBClient(pHashDB);
        return;
    }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    mainMetrics.add("FullTracer::onStartBatch", TimeDiff(t));
#endif

    // Create global exit root manager L2 account
    mpz_class globalExitRootManagerL2Address(ADDRESS_GLOBAL_EXIT_ROOT_MANAGER_L2);
    Account globalExitRootManagerL2Account(fr, poseidon, *pHashDB);
    result = globalExitRootManagerL2Account.Init(globalExitRootManagerL2Address);
    if (result != ZKR_SUCCESS)
    {
        zklog.error("main_exec_c() failed calling globalExitRootManagerL2Account.Init()");
        proverRequest.result = result;
        HashDBClientFactory::freeHashDBClient(pHashDB);
        return;
    }

    // Store global exit root
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    gettimeofday(&t, NULL);
#endif
    // TODO: What TX number should we use?  0?
    result = globalExitRootManagerL2Account.SetGlobalExitRoot(proverRequest.uuid, 0, ctxc.root, proverRequest.input.publicInputsExtended.publicInputs.globalExitRoot, mpz_class(proverRequest.input.publicInputsExtended.publicInputs.timestamp));
    if (result != ZKR_SUCCESS)
    {
        zklog.error("main_exec_c() failed calling globalExitRootManagerL2Account.SetGlobalExitRoot()");
        proverRequest.result = result;
        HashDBClientFactory::freeHashDBClient(pHashDB);
        return;
    }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    mainMetrics.add("SMT set", TimeDiff(t));
#endif

    // Create system SC account
    mpz_class systemAddress(ADDRESS_SYSTEM);
    Account systemAccount(fr, poseidon, *pHashDB);
    result = systemAccount.Init(systemAddress);
    if (result != ZKR_SUCCESS)
    {
        zklog.error("main_exec_c() failed calling systemAccount.Init()");
        proverRequest.result = result;
        HashDBClientFactory::freeHashDBClient(pHashDB);
        return;
    }

    // Create sequencer account
    Account sequencerAccount(fr, poseidon, *pHashDB);
    result = sequencerAccount.Init(proverRequest.input.publicInputsExtended.publicInputs.sequencerAddr);
    if (result != ZKR_SUCCESS)
    {
        zklog.error("main_exec_c() failed calling sequencerAccount.Init()");
        proverRequest.result = result;
        HashDBClientFactory::freeHashDBClient(pHashDB);
        return;
    }

    /*************/
    /* ECRecover */
    /*************/

    // ECRecover all transactions present in parsed batch L2 data, in parallel
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    gettimeofday(&t, NULL);
#endif
#pragma omp parallel for  num_threads(16) //TODO: Make this configurable
    for (uint64_t tx=0; tx<ctxc.batch.tx.size(); tx++)
    {
        // Calculate tx hash
        string signHash = ctxc.batch.tx[tx].signHash();
        //zklog.info("signHash=" + signHash);

        // Verify signature and obtain the from account public key
        mpz_class v_ = ctxc.batch.tx[tx].v;
        mpz_class signature(signHash);
        ctxc.batch.tx[tx].ecRecoverResult = ECRecover(signature, ctxc.batch.tx[tx].r, ctxc.batch.tx[tx].s, v_, false, ctxc.batch.tx[tx].fromPublicKey);
    }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    mainMetrics.add("ECRecover", TimeDiff(t));
#endif

    // Process all transactions present in parsed batch L2 data
    for (ctxc.tx=0; ctxc.tx<ctxc.batch.tx.size(); ctxc.tx++)
    {
        // Log TX info
        //zklog.info("main_exec_c() processing tx=" + to_string(tx));
        //batch.tx[tx].print();

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&t, NULL);
#endif
        result = ((fork_6::FullTracer *)proverRequest.pFullTracer)->onProcessTx(ctxc);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling onProcessTx()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        mainMetrics.add("FullTracer::onProcessTx", TimeDiff(t));
#endif

        if (ctxc.batch.tx[ctxc.tx].ecRecoverResult != ECR_NO_ERROR)
        {
            zklog.error("main_exec_c() failed calling ECRecover()");
            proverRequest.result = ZKR_UNSPECIFIED;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
        //zklog.info("fromPublicKey=" + fromPublicKey.get_str(16));

        // Create from and to accounts
        Account fromAccount(fr, poseidon, *pHashDB);
        result = fromAccount.Init(ctxc.batch.tx[ctxc.tx].fromPublicKey);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling fromAccount.Init()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
        Account toAccount(fr, poseidon, *pHashDB);
        result = toAccount.Init(ctxc.batch.tx[ctxc.tx].to);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling toAccount.Init()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }

        /*******************************/
        /* from.nonce = from.nonce + 1 */
        /*******************************/

        // Get nonce of from account
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&t, NULL);
#endif
        uint64_t fromNonce;
        result = fromAccount.GetNonce(proverRequest.uuid, ctxc.root, fromNonce);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling toAccount.GetNonce()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        mainMetrics.add("SMT get", TimeDiff(t));
#endif

        // Check from account nonce against the one provided by the tx batch L2 data
        if (fromNonce != ctxc.batch.tx[ctxc.tx].nonce)
        {
            zklog.error("main_exec_c() found fromNonce=" + to_string(fromNonce) + " different from batch L2 Datan nonce=" + to_string(ctxc.batch.tx[ctxc.tx].nonce));
            proverRequest.result = ZKR_UNSPECIFIED;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }

        // Increment from nonce
        fromNonce++;

        // Set new from nonce
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&t, NULL);
#endif
        result = fromAccount.SetNonce(proverRequest.uuid, ctxc.tx, ctxc.root, fromNonce);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling toAccount.SetNonce()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        mainMetrics.add("SMT set", TimeDiff(t));
#endif

        // Get balance of from account
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&t, NULL);
#endif
        mpz_class fromBalance;
        result = fromAccount.GetBalance(proverRequest.uuid, ctxc.root, fromBalance);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling fromAccount.GetBalance()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        mainMetrics.add("SMT get", TimeDiff(t));
#endif

        // Calculate gas
        ctxc.batch.tx[ctxc.tx].gas = 21000; // Transfer with no data, no CALLDATA cost, no deployment cost

        // Check that gas is not higher than gas limit
        if (ctxc.batch.tx[ctxc.tx].gas > ctxc.batch.tx[ctxc.tx].gasLimit)
        {
            zklog.error("main_exec_c() failed gas=" + ctxc.batch.tx[ctxc.tx].gas.get_str(10) + " < gasLimit=" + to_string(ctxc.batch.tx[ctxc.tx].gasLimit));
            proverRequest.result = ZKR_UNSPECIFIED; // TODO: Review list of errors
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }

        // Calculate effective gas price: txGasPrice = Floor((gasPrice * (effectivePercentage + 1)) / 256)
        if (ctxc.batch.tx[ctxc.tx].gasPercentage != 255)
        {
            uint64_t txGasPercentage = (uint64_t)ctxc.batch.tx[ctxc.tx].gasPercentage;
            ctxc.batch.tx[ctxc.tx].effectiveGasPrice = ctxc.batch.tx[ctxc.tx].gasPrice * (txGasPercentage + 1) / 256;
        }
        else
        {
            ctxc.batch.tx[ctxc.tx].effectiveGasPrice = ctxc.batch.tx[ctxc.tx].gasPrice;
        }

        // Calculate fee
        ctxc.batch.tx[ctxc.tx].fee = ctxc.batch.tx[ctxc.tx].gas * ctxc.batch.tx[ctxc.tx].effectiveGasPrice;

        // Check that from account has enough balance to complete the transfer
        mpz_class fromAmount = ctxc.batch.tx[ctxc.tx].value + ctxc.batch.tx[ctxc.tx].fee;
        if (fromBalance < fromAmount)
        {
            zklog.error("main_exec_c() failed fromBalance=" + fromBalance.get_str(10) + " < fromAmount=" + fromAmount.get_str(10));
            proverRequest.result = ZKR_UNSPECIFIED;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }

        // Update from account balance = balance - value - fee (gas*gasPrice)
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&t, NULL);
#endif
        fromBalance -= fromAmount;
        result = fromAccount.SetBalance(proverRequest.uuid, ctxc.tx, ctxc.root, fromBalance);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling fromAccount.SetBalance()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        mainMetrics.add("SMT set", TimeDiff(t));
#endif

        /***********************************/
        /* to.balance = to.balance + value */
        /***********************************/

        // Get balance of to account
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&t, NULL);
#endif
        mpz_class toBalance;
        result = toAccount.GetBalance(proverRequest.uuid, ctxc.root, toBalance);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling toAccount.GetBalance()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        mainMetrics.add("SMT get", TimeDiff(t));
#endif

        // Update to account balance = balance + value
        toBalance += ctxc.batch.tx[ctxc.tx].value;

        // Set account balance
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&t, NULL);
#endif
        result = toAccount.SetBalance(proverRequest.uuid, ctxc.tx, ctxc.root, toBalance);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling toAccount.SetBalance()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        mainMetrics.add("SMT set", TimeDiff(t));
#endif

        /***********************************************/
        /* sequencer.balance = sequencer.balance + fee */
        /***********************************************/

        // TODO: if sequencer and from are the same, read and write only once

        // Get balance of sequencer account
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&t, NULL);
#endif
        mpz_class sequencerBalance;
        result = sequencerAccount.GetBalance(proverRequest.uuid, ctxc.root, sequencerBalance);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling sequencerAccount.GetBalance()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        mainMetrics.add("SMT get", TimeDiff(t));
#endif

        // sequencer.balance += fee
        sequencerBalance += ctxc.batch.tx[ctxc.tx].fee;

        // Update sequencer account balance = balance + fee
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&t, NULL);
#endif
        result = sequencerAccount.SetBalance(proverRequest.uuid, ctxc.tx, ctxc.root, sequencerBalance);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling sequencerAccount.SetBalance()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        mainMetrics.add("SMT set", TimeDiff(t));
#endif

        // Increase TX count
        ctxc.globalVars.txCount++;

        // TODO: If we call GetTxCount, check range

        // Set new TX count in system account
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&t, NULL);
#endif
        result = systemAccount.SetTxCount(proverRequest.uuid, ctxc.tx, ctxc.root, ctxc.globalVars.txCount);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling systemAccount.SetTxCount()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        mainMetrics.add("SMT set", TimeDiff(t));
#endif

        // Set new state root in system account
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&t, NULL);
#endif
        mpz_class auxScalar;
        fea2scalar(fr, auxScalar, ctxc.root);
        result = systemAccount.SetStateRoot(proverRequest.uuid, ctxc.tx, ctxc.root, ctxc.globalVars.txCount, auxScalar);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling systemAccount.SetStateRoot()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        mainMetrics.add("SMT set", TimeDiff(t));
#endif

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        gettimeofday(&t, NULL);
#endif
        result = ((fork_6::FullTracer *)proverRequest.pFullTracer)->onFinishTx(ctxc);
        if (result != ZKR_SUCCESS)
        {
            zklog.error("main_exec_c() failed calling onFinishTx()");
            proverRequest.result = result;
            HashDBClientFactory::freeHashDBClient(pHashDB);
            return;
        }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
        mainMetrics.add("FullTracer::onFinishTx", TimeDiff(t));
#endif

        //zklog.info("Processed tx=" + to_string(tx) + " newStateRoot=" + fea2string(fr, ctxc.root));
    }


    zklog.info("new root=" + fea2string(fr, ctxc.root));

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    gettimeofday(&t, NULL);
#endif
    result = ((fork_6::FullTracer *)proverRequest.pFullTracer)->onFinishBatch(ctxc);
    if (result != ZKR_SUCCESS)
    {
        zklog.error("main_exec_c() failed calling onFinishBatch()");
        proverRequest.result = result;
        HashDBClientFactory::freeHashDBClient(pHashDB);
        return;
    }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    mainMetrics.add("FullTracer::onFinishBatch", TimeDiff(t));
#endif

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    gettimeofday(&t, NULL);
#endif
    result = pHashDB->flush(proverRequest.uuid, fea2string(fr, ctxc.root), proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE, proverRequest.flushId, proverRequest.lastSentFlushId);
    if (result != ZKR_SUCCESS)
    {
        proverRequest.result = result;
        zklog.error("Failed calling pHashDB->flush() result=" + zkresult2string(result));
        HashDBClientFactory::freeHashDBClient(pHashDB);
        return;
    }
#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    mainMetrics.add("HashDB::flush", TimeDiff(t));
#endif

    HashDBClientFactory::freeHashDBClient(pHashDB);
    proverRequest.result = ZKR_SUCCESS;

#ifdef LOG_TIME_STATISTICS_MAIN_EXECUTOR
    if (config.executorTimeStatistics)
    {
        mainMetrics.print("Main C Executor calls");
    }
#endif
    TimerStopAndLog(MAIN_EXEC_C);
}

}