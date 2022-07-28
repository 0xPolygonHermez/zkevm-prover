#include <iostream>
#include "config.hpp"
#include "input.hpp"
#include "scalar.hpp"
#include "database.hpp"

void Input::load (json &input)
{
    loadGlobals      (input);
    preprocessTxs(); // Generate derivated data
#ifdef USE_LOCAL_STORAGE
    loadStorage      (input);
#endif
    loadDatabase     (input);
}

void Input::save (json &input) const
{
    saveGlobals      (input);
#ifdef USE_LOCAL_STORAGE
    saveStorage      (input);
#endif
    saveDatabase     (input);
}

void Input::save (json &input, const Database &database) const
{
    saveGlobals      (input);
#ifdef USE_LOCAL_STORAGE
    saveStorage      (input);
#endif
    saveDatabase     (input, database);
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

    // Input JSON file must contain a timestamp key at the root level
    if ( !input.contains("timestamp") ||
         !input["timestamp"].is_number_unsigned() )
    {
        cerr << "Error: timestamp key not found in input JSON file" << endl;
        exit(-1);
    }
    publicInputs.timestamp = input["timestamp"];
    cout << "loadGobals(): timestamp=" << publicInputs.timestamp << endl;

    // Input JSON file must contain a batchL2Data key at the root level
    if ( !input.contains("batchL2Data") ||
         !input["batchL2Data"].is_string() )
    {
        cerr << "Error: batchL2Data key not found in input JSON file" << endl;
        exit(-1);
    }
    batchL2Data = Add0xIfMissing(input["batchL2Data"]);
    cout << "loadGobals(): batchL2Data=" << batchL2Data << endl;
}

void Input::saveGlobals (json &input) const
{
    input["globalExitRoot"] = globalExitRoot;
    input["oldStateRoot"] = publicInputs.oldStateRoot;
    input["newStateRoot"] = publicInputs.newStateRoot;
    input["oldLocalExitRoot"] = publicInputs.oldLocalExitRoot;
    input["newLocalExitRoot"] = publicInputs.newLocalExitRoot;
    input["sequencerAddr"] = publicInputs.sequencerAddr;
    input["defaultChainId"] = publicInputs.defaultChainId;
    input["numBatch"] = publicInputs.batchNum;
    input["timestamp"] = publicInputs.timestamp;
    input["batchL2Data"] = batchL2Data;
}

void Input::preprocessTxs (void)
{
    cout << "Input::preprocessTxs() input.txsLen=" << txsLen << endl;

    // Calculate the TX batch hash
    string keccakInput = batchL2Data;
    keccakInput += NormalizeToNFormat(Remove0xIfPresent(globalExitRoot), 64);
    keccakInput += NormalizeToNFormat(publicInputs.sequencerAddr, 40);

    string keccakOutput = keccak256(keccakInput);

    batchHashData.set_str(Remove0xIfPresent(keccakOutput), 16);
    cout << "Input::preprocessTxs() input.batchHashData=" << keccakOutput << endl;

    // Calculate STARK input

    // Prepare the string to calculate the new root hash
    keccakInput = "0x";
    keccakInput += NormalizeToNFormat(publicInputs.oldStateRoot, 64);
    keccakInput += NormalizeToNFormat(publicInputs.oldLocalExitRoot, 64);
    keccakInput += NormalizeToNFormat(publicInputs.newStateRoot, 64);
    keccakInput += NormalizeToNFormat(publicInputs.newLocalExitRoot, 64);
    keccakInput += NormalizeToNFormat(keccakOutput, 64); // batchHashData string
    mpz_class aux3(publicInputs.batchNum);
    keccakInput += NormalizeToNFormat(aux3.get_str(16), 16);
    mpz_class aux1(publicInputs.timestamp);
    keccakInput += NormalizeToNFormat(aux1.get_str(16), 16);

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
        Goldilocks::Element fe;
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

void Input::saveStorage (json &input) const
{
}

#endif

/* Store db into database ctx.db[] */

void Input::loadDatabase (json &input)
{
    // Input JSON file must contain a db structure at the root level
    if ( !input.contains("db") ||
         !input["db"].is_structured() )
    {
        cout << "Input::loadDatabase() warning: db key not found in input JSON file" << endl;
        return;
    }
    cout << "loadDatabase() db content:" << endl;
    for (json::iterator it = input["db"].begin(); it != input["db"].end(); ++it)
    {
        // Every value must be a 12-fe array if intermediate node, or 8-fe array if value
        if (!it.value().is_array() ||
            !((it.value().size()==12) || (it.value().size()==8)) )
        {
            cerr << "Error: Input::loadDatabase() keys value array with invalid length in input JSON file: " << it.value() << endl;
            exit(-1);
        }

        // Add the 16 fe elements into the database value
        vector<Goldilocks::Element> dbValue;
        for (uint64_t i=0; i<it.value().size(); i++)
        {
            Goldilocks::Element fe;
            string2fe(fr, it.value()[i], fe);
            dbValue.push_back(fe);
        }

        // Get the key fe element
        string key = NormalizeToNFormat(it.key(), 64);

        // Add the key:value pair to the context database
        db[key] = dbValue;
        cout << "    key: " << it.key() << " value: " << it.value()[0] << " etc." << endl;
    }   
}

void Input::db2json (json &input, const std::map<string, vector<Goldilocks::Element>> &db, string name) const

{
    input[name] = json::object();
    for(std::map<string, vector<Goldilocks::Element>>::const_iterator iter = db.begin(); iter != db.end(); iter++)
    {
        string key = NormalizeToNFormat(iter->first, 64);
        vector<Goldilocks::Element> dbValue = iter->second;
        json value;
        for (uint64_t i=0; i<dbValue.size(); i++)
        {
            value[i] = NormalizeToNFormat(fr.toString(dbValue[i], 16), 64);
        }
        input[name][key] = value;
    }
}

void Input::saveDatabase (json &input) const
{
    db2json(input, db, "db");
}

void Input::saveDatabase (json &input, const Database &database) const
{
    db2json(input, db, "db");
}