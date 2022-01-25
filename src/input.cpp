#include <iostream>
#include "config.hpp"
#include "input.hpp"
#include "scalar.hpp"
#include "rlpvalue/rlpvalue.h"

void Input::load (json &input)
{
    loadGlobals      (input);
    loadTransactions (input);
#ifdef USE_LOCAL_STORAGE
    loadStorage      (input);
#endif
    loadDatabase     (input);
}

/* Load old/new state roots, sequencer address and chain ID */

void Input::loadGlobals (json &input)
{
    // Input JSON file must contain a globalExitRoot key at the root level
    if ( !input.contains("globalExitRoot") ||
         !input["globalExitRoot"].is_string() )
    {
        cerr << "Error: globalExitRoot key not found in input JSON file" << endl;
        exit(-1);
    }
    globalExitRoot = input["globalExitRoot"];
    cout << "loadGobals(): globalExitRoot=" << globalExitRoot << endl;

    // Input JSON file must contain a oldStateRoot key at the root level
    if ( !input.contains("oldStateRoot") ||
         !input["oldStateRoot"].is_string() )
    {
        cerr << "Error: oldStateRoot key not found in input JSON file" << endl;
        exit(-1);
    }
    publicInputs.oldStateRoot = input["oldStateRoot"];
    cout << "loadGobals(): oldStateRoot=" << publicInputs.oldStateRoot << endl;

    // Input JSON file must contain a newStateRoot key at the root level
    if ( !input.contains("newStateRoot") ||
         !input["newStateRoot"].is_string() )
    {
        cerr << "Error: newStateRoot key not found in input JSON file" << endl;
        exit(-1);
    }
    publicInputs.newStateRoot = input["newStateRoot"];
    cout << "loadGobals(): newStateRoot=" << publicInputs.newStateRoot << endl;

    // Input JSON file must contain a oldLocalExitRoot key at the root level
    if ( !input.contains("oldLocalExitRoot") ||
         !input["oldLocalExitRoot"].is_string() )
    {
        cerr << "Error: oldLocalExitRoot key not found in input JSON file" << endl;
        exit(-1);
    }
    publicInputs.oldLocalExitRoot = input["oldLocalExitRoot"];
    cout << "loadGobals(): oldLocalExitRoot=" << publicInputs.oldLocalExitRoot << endl;

    // Input JSON file must contain a newLocalExitRoot key at the root level
    if ( !input.contains("newLocalExitRoot") ||
         !input["newLocalExitRoot"].is_string() )
    {
        cerr << "Error: newLocalExitRoot key not found in input JSON file" << endl;
        exit(-1);
    }
    publicInputs.newLocalExitRoot = input["newLocalExitRoot"];
    cout << "loadGobals(): newLocalExitRoot=" << publicInputs.newLocalExitRoot << endl;

    // Input JSON file must contain a sequencerAddr key at the root level
    if ( !input.contains("sequencerAddr") ||
         !input["sequencerAddr"].is_string() )
    {
        cerr << "Error: sequencerAddr key not found in input JSON file" << endl;
        exit(-1);
    }
    publicInputs.sequencerAddr = input["sequencerAddr"];
    cout << "loadGobals(): sequencerAddr=" << publicInputs.sequencerAddr << endl;

    // Input JSON file must contain a chainId key at the root level
    if ( !input.contains("chainId") ||
         !input["chainId"].is_number_unsigned() )
    {
        cerr << "Error: chainId key not found in input JSON file" << endl;
        exit(-1);
    }
    publicInputs.chainId = input["chainId"];
    cout << "loadGobals(): chainId=" << publicInputs.chainId << endl;

    // Input JSON file could contain a defaultChainId key at the root level (not mandatory)
    if ( !input.contains("defaultChainId") ||
         !input["defaultChainId"].is_number_unsigned() )
    {
        // This is the default value: 1000
        publicInputs.defaultChainId = 1000;
    }
    else
    {
        publicInputs.defaultChainId = input["defaultChainId"];
    }
    cout << "loadGobals(): defaultChainId=" << publicInputs.defaultChainId << endl;

    // Input JSON file must contain a numBatch key at the root level
    if ( !input.contains("numBatch") ||
         !input["numBatch"].is_number_unsigned() )
    {
        cerr << "Error: numBatch key not found in input JSON file" << endl;
        exit(-1);
    }
    publicInputs.batchNum = input["numBatch"];
    cout << "loadGobals(): batchNum=" << publicInputs.batchNum << endl;
}

/* Load transactions and resulting globalHash */

void Input::loadTransactions (json &input)
{
    // Input JSON file must contain a txs string array at the root level
    if ( !input.contains("txs") ||
         !input["txs"].is_array() )
    {
        cerr << "Error: txs key not found in input JSON file" << endl;
        exit(-1);
    }

    // Get the number of transactions present in the input JSON file
    uint64_t numberOfTransactions = input["txs"].size();
    cout << "loadTransactions() found " << numberOfTransactions << " transactions in input JSON file" << endl;

    // Store all transactions into txs
    for (uint64_t i=0; i<numberOfTransactions; i++)
    {
        txs.push_back(input["txs"][i]);
    }

    // Generate derivated data
    preprocessTxs();
}

void Input::preprocessTxs (void)
{
    // Concatenate all transactions into one sigle string
    batchL2Data = "0x";
    for (uint64_t i=0; i<txs.size(); i++)
    {
        batchL2Data += Remove0xIfPresent(txs[i]);
    }
    txsLen = batchL2Data.size() - 2;
    cout << "Input::preprocessTxs() input.txsLen=" << txsLen << endl;

    // Calculate the TX batch hash
    string keccakInput = batchL2Data + NormalizeToNFormat(Remove0xIfPresent(globalExitRoot), 64);
    string keccakOutput = keccak256(keccakInput);
    batchHashData.set_str(Remove0xIfPresent(keccakOutput), 16);
    cout << "Input::preprocessTxs() input.batchHashData=" << keccakOutput << endl;

    // Prepare the string to calculate the new root hash
    keccakInput = "0x";
    keccakInput += NormalizeToNFormat(publicInputs.oldStateRoot, 64);
    keccakInput += NormalizeToNFormat(publicInputs.oldLocalExitRoot, 64);
    keccakInput += NormalizeToNFormat(publicInputs.newStateRoot, 64);
    keccakInput += NormalizeToNFormat(publicInputs.newLocalExitRoot, 64);
    keccakInput += NormalizeToNFormat(publicInputs.sequencerAddr, 40);
    keccakInput += NormalizeToNFormat(keccakOutput, 64); // batchHashData string
    mpz_class aux(publicInputs.chainId);
    keccakInput += NormalizeToNFormat(aux.get_str(16), 8);
    mpz_class aux2(publicInputs.batchNum);
    keccakInput += NormalizeToNFormat(aux2.get_str(16), 8);
    
    // Calculate the new root hash from the concatenated string
    keccakOutput = keccak256(keccakInput);
    globalHash.set_str(Remove0xIfPresent(keccakOutput), 16);
    cout << "Input::preprocessTxs() input.globalHash=" << globalHash.get_str(16) << endl;
}

#ifdef USE_LOCAL_STORAGE

/* Store keys into storage ctx.sto[] */

void Input::loadStorage (json &input)
{
    // Input JSON file must contain a keys structure at the root level
    if ( !input.contains("keys") ||
         !input["keys"].is_structured() )
    {
        cerr << "Error: keys key not found in input JSON file" << endl;
        exit(-1);
    }
    //cout << "keys content:" << endl;
    for (json::iterator it = input["keys"].begin(); it != input["keys"].end(); ++it)
    {
        // Read fe from it.key()
        RawFr::Element fe;
        string2fe(fr, it.key(), fe);

        // Read scalar from it.value()
        mpz_class scalar;
        scalar.set_str(it.value(), 16);

        // Store the key:value pair in context storage
        sto[fe] = scalar;

#ifdef LOG_STORAGE
        cout << "loadStorage() added record with key(fe): " << ctx.fr.toString(fe, 16) << " value(scalar): " << scalar.get_str(16) << endl;
#endif
    }
}

#endif

/* Store db into database ctx.db[] */

void Input::loadDatabase (json &input)
{
    // Input JSON file must contain a db structure at the root level
    if ( !input.contains("db") ||
         !input["db"].is_structured() )
    {
        cout << "Input::loadDatabase warning: db key not found in input JSON file" << endl;
        return;
    }
    cout << "loadDatabase() db content:" << endl;
    for (json::iterator it = input["db"].begin(); it != input["db"].end(); ++it)
    {
        // Every value must be a 16-fe-elements array
        if (!it.value().is_array() ||
            !(it.value().size()==16))
        {
            cerr << "Error: keys value not a 16-elements array in input JSON file: " << it.value() << endl;
            exit(-1);
        }

        // Add the 16 fe elements into the database value
        vector<RawFr::Element> dbValue;
        for (int i=0; i<16; i++)
        {
            RawFr::Element fe;
            string2fe(fr, it.value()[i], fe);
            dbValue.push_back(fe);
        }

        // Get the key fe element
        RawFr::Element key;
        string2fe(fr, it.key(), key);

        // Add the key:value pair to the context database
        db[key] = dbValue;
        cout << "    key: " << it.key() << " value: " << it.value()[0] << " etc." << endl;
    }   
}