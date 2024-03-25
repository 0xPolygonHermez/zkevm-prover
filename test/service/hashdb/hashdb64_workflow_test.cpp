#include <unistd.h>
#include "hashdb64_workflow_test.hpp"
#include "zklog.hpp"
#include "zkresult.hpp"
#include "hashdb_factory.hpp"
#include "utils.hpp"
#include "hashdb_singleton.hpp"
#include "timer.hpp"
#include "check_tree_test.hpp"
#include "time_metric.hpp"

bool batchIsDiscarded (uint64_t batch)
{
    if (((batch+4) % 5) == 0) // 1, 6, 11, 16...
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool batchIsConsolidated (uint64_t batch)
{
    if (((batch+1) % 5) == 0) // 4, 9, 14, 19...
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool blockIsDiscarded (uint64_t block)
{
    if (block == 1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool txIsDiscarded (uint64_t tx)
{
    if (tx == 1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool setIsDiscarded (uint64_t set)
{
    if (set == 1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

uint64_t HashDB64WorkflowTest (const Config& config)
{
    TimerStart(HASHDB64_WORKFLOW_TEST);

    TimeMetricStorage timeMetricStorage;
    struct timeval t;

    zklog.info("HashDB64WorkflowTest() started");
    Goldilocks fr;
    PoseidonGoldilocks poseidon;
    uint64_t block = 0;
    zkresult zkr;
    Persistence persistence = PERSISTENCE_DATABASE;
    HashDBInterface* pHashDB = HashDBClientFactory::createHashDBClient(fr, config);
    zkassertpermanent(pHashDB != NULL);
    uint64_t flushId, storedFlushId;
    

    const uint64_t numberOfBatches = 10; // Batches  4, 9, 14, 19... will be reverted, and then state will be consolidated
    const uint64_t numberOfTxsPerBatch = 3;
    const uint64_t numberOfSetsPerTx = 3; // Set 1 will be reverted
    const uint64_t numberOfProgramsPerTx = 1; // Set 1 will be reverted

    zklog.info("HashDB64WorkflowTest() numberOfBatches=" + to_string(numberOfBatches) + " numberOfTxsPerBatch=" + to_string(numberOfTxsPerBatch) + " numberOfSetsPerTx=" + to_string(numberOfSetsPerTx));

    SmtSetResult setResult;
    SmtGetResult getResult;

    Goldilocks::Element key[4]={0,0,0,0};
    Goldilocks::Element root[4]={0,0,0,0};
    Goldilocks::Element newRoot[4]={0,0,0,0};
    Goldilocks::Element keyfea[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    mpz_class value = 0;
    mpz_class keyScalar = 0;
    vector<KeyValue> persistentKeyValues;
    vector<KeyValue> discardedKeyValues;
    unordered_map<string, vector<uint8_t>> persistentPrograms;
    unordered_map<string, vector<uint8_t>> discardedPrograms;
    vector<uint8_t> accumulatedProgram;
    vector<Goldilocks::Element> accumulatedProgramFe;
    uint64_t programCounter = 0;

    pHashDB->getLatestStateRoot(root);

    /*******************************/
    /* SET KEY-VALUES AND PROGRAMS */
    /*******************************/

    for (uint64_t batch=0; batch<numberOfBatches; batch++)
    {
        zklog.info("STARTING BATCH=" + to_string(batch));

        // Start batch
        string batchUUID = getUUID();
        Goldilocks::Element batchOldStateRoot[4];
        for (uint64_t i=0; i<4; i++) batchOldStateRoot[i] = root[i];
        vector<KeyValue> keyValues;

        // Set TXs
        for (uint64_t tx=0; tx<numberOfTxsPerBatch; tx++)
        {

            Goldilocks::Element txOldRoot[4];
            for (uint64_t i=0; i<4; i++) txOldRoot[i] = root[i];

            for (uint64_t set=0; set<numberOfSetsPerTx; set++)
            {
                keyScalar++;
                keyfea[0] = fr.fromU64(keyScalar.get_ui());
                poseidon.hash(key, keyfea);
                //scalar2key(fr, keyScalar, key);
                if (fea2string(fr, key) == "adb5787a1f8676b554f2216c0b37148d303a082109d64fe07615b40971dc29f2")
                {
                    zklog.info("special key found setIsDiscarded(set)=" + to_string(setIsDiscarded(set)) + " batchIsDiscarded(batch)=" + to_string(batchIsDiscarded(batch)));
                }
                value++;
                
                gettimeofday(&t, NULL);
                zkr = pHashDB->set(batchUUID, block, tx, root, key, value, persistence, newRoot, &setResult, NULL);
                timeMetricStorage.add("set", TimeDiff(t));

                //zklog.info("SET zkr=" + zkresult2string(zkr) + " root=" + fea2string(fr, root) + " key=" + fea2string(fr, key) + " value=" + value.get_str() + " newRoot=" + fea2string(fr, newRoot));
                zkassertpermanent(zkr==ZKR_SUCCESS);
                zkassertpermanent(!fr.isZero(newRoot[0]) || !fr.isZero(newRoot[1]) || !fr.isZero(newRoot[2]) || !fr.isZero(newRoot[3]));

                gettimeofday(&t, NULL);
                zkr = pHashDB->get(batchUUID, newRoot, key, value, &getResult, NULL);
                timeMetricStorage.add("get", TimeDiff(t));

                //zklog.info("GET zkr=" + zkresult2string(zkr) + " root=" + fea2string(fr, root) + " key=" + fea2string(fr, key) + " value=" + value.get_str());
                zkassertpermanent(zkr==ZKR_SUCCESS);
                zkassertpermanent(value==getResult.value);

                // Advance the state root only if this set is not discarded
                if (setIsDiscarded(set))
                {
                    // We "revert" this sub-state, i.e. we dont advance the state root and we discard this key
                }
                else
                {
                    // Advance in the state root chain
                    for (uint64_t i=0; i<4; i++) root[i] = setResult.newRoot[i];
                }

                // Build the KeyValue we just set
                KeyValue keyValue;
                for (uint64_t i=0; i<4; i++) keyValue.key[i] = key[i];
                keyValue.value = value;

                // Store the KeyValue in the prover vector, depending on its expected persistence
                if (setIsDiscarded(set) || txIsDiscarded(tx) || batchIsDiscarded(batch))
                {
                    // Store in discardedKeyValues
                    discardedKeyValues.emplace_back(keyValue);
                }
                else
                {
                    // Store in keyValues
                    keyValues.emplace_back(keyValue);
                }

            } // For every set

            // For every program of this tx
            for (uint64_t program=0; program<numberOfProgramsPerTx; program++)
            {
                programCounter++;
                accumulatedProgram.emplace_back(programCounter);
                accumulatedProgramFe.emplace_back(fr.fromU64(programCounter));
                keyfea[0] = fr.fromU64(1000000000+programCounter);
                poseidon.hash(key, keyfea);
                string keyString = fea2string(fr, key);

                zkr = pHashDB->setProgram(batchUUID, block, tx, key, accumulatedProgram, persistence);
                zkassertpermanent(zkr == ZKR_SUCCESS);

                if (batchIsDiscarded(batch) || txIsDiscarded(tx))
                {
                    discardedPrograms[keyString] = accumulatedProgram;
                }
                else
                {
                    persistentPrograms[keyString] = accumulatedProgram;
                }

            } // For every program

            gettimeofday(&t, NULL);
            pHashDB->finishTx(batchUUID, fea2string(fr, root), persistence);
            pHashDB->finishBlock(batchUUID, fea2string(fr, root), persistence);
            timeMetricStorage.add("semiFlush", TimeDiff(t));

            if (txIsDiscarded(tx))
            {
                for (uint64_t i=0; i<4; i++) root[i] = txOldRoot[i];
            }

        } // For every tx

        // Flush
        gettimeofday(&t, NULL);
        /*uint64_t flushId, storedFlushId;
        zkr = pHashDB->flush(batchUUID, fea2string(fr, root), persistence, flushId, storedFlushId);
        timeMetricStorage.add("flush", TimeDiff(t));
        zkassertpermanent(zkr==ZKR_SUCCESS);
        zklog.info("FLUSH zkr=" + zkresult2string(zkr) + " root=" + fea2string(fr, root));
*/
        // Discard some of the batches, and accumulate the key values in the rest
        if (batchIsDiscarded(batch))
        {
            for (uint64_t i=0; i<4; i++) root[i] = batchOldStateRoot[i];
        }
        else
        {
            persistentKeyValues.insert(persistentKeyValues.end(), keyValues.begin(), keyValues.end());
        }

        // Consolidate state root every 5 batches, at batches 4, 9, 14, 19...
        Goldilocks::Element batchNewStateRoot[4];
        if (batchIsConsolidated(batch))
        {
            Goldilocks::Element consolidatedStateRoot[4];
            gettimeofday(&t, NULL);
            zkr = pHashDB->consolidateState(root, persistence, consolidatedStateRoot, flushId, storedFlushId);
            timeMetricStorage.add("consolidateState", TimeDiff(t));
            zkassertpermanent(zkr==ZKR_SUCCESS);
            zklog.info("CONSOLIDATE zkr=" + zkresult2string(zkr) + " virtualRoot=" + fea2string(fr, root) + " consolidatedRoot=" + fea2string(fr, consolidatedStateRoot) + " flushId=" + to_string(flushId) + " storedFlushId=" + to_string(storedFlushId));

            // New state root
            for (uint64_t i=0; i<4; i++) batchNewStateRoot[i] = consolidatedStateRoot[i];
            for (uint64_t i=0; i<4; i++) root[i] = consolidatedStateRoot[i];

            // Wait for data to be sent
            while (true)
            {
                uint64_t storedFlushId, storingFlushId, lastFlushId, pendingToFlushNodes, pendingToFlushProgram, storingNodes, storingProgram;
                string proverId;
                zkr = pHashDB->getFlushStatus(storedFlushId, storingFlushId, lastFlushId, pendingToFlushNodes, pendingToFlushProgram, storingNodes, storingProgram, proverId);
                zkassertpermanent(zkr==ZKR_SUCCESS);
                zklog.info("GET FLUSH STATUS storedFlushId=" + to_string(storedFlushId));
                if (storedFlushId >= flushId)
                {
                    break;
                }
                sleep(1);
            }
            zklog.info("FLUSHED");

            // Call ReadTree with the old state root to get the hashes of the initial values of all read or written keys
            /*vector<HashValueGL> oldHashValues;
            zkr = pHashDB->readTree(batchOldStateRoot, keyValues, oldHashValues);
            zkassertpermanent(zkr==ZKR_SUCCESS);
            zklog.info("READ TREE batchOldStateRoot=" + fea2string(fr, batchOldStateRoot) + " keyValues.size=" + to_string(keyValues.size()) + " hashValues.size=" + to_string(oldHashValues.size()));*/

            // Call ReadTree with the new state root to get the hashes of the initial values of all read or written keys
            vector<HashValueGL> hashValues;
            vector<KeyValue> auxKeyValues = persistentKeyValues;
            for (uint64_t i=0; i<auxKeyValues.size(); i++)
            {
                auxKeyValues[i].value = 0;
            }

            //CheckTreeTest(config);

            gettimeofday(&t, NULL);
            zkr = pHashDB->readTree(batchNewStateRoot, auxKeyValues, hashValues);
            timeMetricStorage.add("readTree", TimeDiff(t));
            zkassertpermanent(zkr==ZKR_SUCCESS);
            zklog.info("READ TREE batchNewStateRoot=" + fea2string(fr, batchNewStateRoot) + " keyValues.size=" + to_string(auxKeyValues.size()) + " hashValues.size=" + to_string(hashValues.size()));

            zkassertpermanent(auxKeyValues.size() == persistentKeyValues.size());
            for (uint64_t i=0; i<auxKeyValues.size(); i++)
            {
                //zklog.info("auxKeyValues[" + to_string(i) + "].key=" + fea2string(fr, auxKeyValues[i].key) + " .value=" + auxKeyValues[i].value.get_str(10));
                //zklog.info("allKeyValues[i].key=" + fea2string(fr, allKeyValues[i].key) + " .value=" + allKeyValues[i].value.get_str(10));
                if (auxKeyValues[i].value != persistentKeyValues[i].value)
                {
                    zklog.error("HashDB64WorkflowTest() found value=" + auxKeyValues[i].value.get_str() + " != expected value=" + persistentKeyValues[i].value.get_str());
                    return 1;
                }
                else
                {
                    //zklog.error("Found value=" + auxKeyValues[i].value.get_str() + " == expected value=" + allKeyValues[i].value.get_str());
                }
                zkassertpermanent( fr.equal(auxKeyValues[i].key[0], persistentKeyValues[i].key[0]) &&
                                   fr.equal(auxKeyValues[i].key[1], persistentKeyValues[i].key[1]) &&
                                   fr.equal(auxKeyValues[i].key[2], persistentKeyValues[i].key[2]) &&
                                   fr.equal(auxKeyValues[i].key[3], persistentKeyValues[i].key[3]) );
            }
        }
        else
        {
            for (uint64_t i=0; i<4; i++) batchNewStateRoot[i] = root[i];
        }

    } // For every batch

    /*********************/
    /* CHECK PERSISTENCE */
    /*********************/

    string batchUUID = "";

    // Check that all persistent keys have their expected value
    for (uint64_t i=0; i<persistentKeyValues.size(); i++)
    {
        //zklog.info("persistentKeyValues[" + to_string(i) + "].key=" + fea2string(fr, persistentKeyValues[i].key) + " .value=" + persistentKeyValues[i].value.get_str(10));
        mpz_class auxValue;
        gettimeofday(&t, NULL);
        zkr = pHashDB->get(batchUUID, root, persistentKeyValues[i].key, auxValue, &getResult, NULL);
        timeMetricStorage.add("get", TimeDiff(t));
        zkassertpermanent(zkr==ZKR_SUCCESS);
        zkassertpermanent(auxValue==persistentKeyValues[i].value);
    }

    // Check that all discarded keys have value=0
    for (uint64_t i=0; i<discardedKeyValues.size(); i++)
    {
        //zklog.info("discardedKeyValues[" + to_string(i) + "].key=" + fea2string(fr, discardedKeyValues[i].key) + " .value=" + discardedKeyValues[i].value.get_str(10));
        mpz_class auxValue;
        gettimeofday(&t, NULL);
        zkr = pHashDB->get(batchUUID, root, discardedKeyValues[i].key, auxValue, &getResult, NULL);
        timeMetricStorage.add("get", TimeDiff(t));
        zkassertpermanent(zkr==ZKR_SUCCESS);
        zkassertpermanent(auxValue==0);
    }

    // Check that all persistent programs have their expected value
    unordered_map<string, vector<uint8_t>>::const_iterator itProgram;
    for (itProgram = persistentPrograms.begin(); itProgram != persistentPrograms.end(); itProgram++)
    {
        vector<uint8_t> auxValue;
        gettimeofday(&t, NULL);
        Goldilocks::Element key[4];
        string2fea(fr, itProgram->first, key);
        zkr = pHashDB->getProgram(batchUUID, key, auxValue, NULL);
        timeMetricStorage.add("getProgram", TimeDiff(t));
        zkassertpermanent(zkr==ZKR_SUCCESS);
        zkassertpermanent(auxValue.size() == itProgram->second.size());
        for (uint64_t i=0; i<auxValue.size(); i++)
        {
            zkassertpermanent(auxValue[i] == itProgram->second[i]);
        }
    }
    
    // Check that all discarded programs are not present
    for (itProgram = discardedPrograms.begin(); itProgram != discardedPrograms.end(); itProgram++)
    {
        vector<uint8_t> auxValue;
        gettimeofday(&t, NULL);
        Goldilocks::Element key[4];
        string2fea(fr, itProgram->first, key);
        zkr = pHashDB->getProgram(batchUUID, key, auxValue, NULL);
        timeMetricStorage.add("getProgram", TimeDiff(t));
        if (zkr != ZKR_DB_KEY_NOT_FOUND)
        {
            zklog.info("getProgram failed key=" + itProgram->first);
        }
        else
        {
            //zklog.info("getProgram failed gracefully key=" + itProgram->first);
        }
        zkassertpermanent(zkr==ZKR_DB_KEY_NOT_FOUND);
    }

    timeMetricStorage.print("HashDB64 Workflow test metrics");

    CheckTreeTest(config);

    TimerStopAndLog(HASHDB64_WORKFLOW_TEST);

    return 0;
}