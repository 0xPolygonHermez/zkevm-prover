#include <iostream>
#include "config.hpp"
#include "input.hpp"
#include "scalar.hpp"
#include "database.hpp"
#include "utils.hpp"

zkresult Input::load (json &input)
{
    loadGlobals(input);

    zkresult zkResult = preprocessTxs(); // Generate derivated data
    if (zkResult != ZKR_SUCCESS)
    {
        cerr << "Error: Input::load() failed calling preprocessTxs()" << endl;
        return zkResult;
    }

    loadDatabase(input);

    return ZKR_SUCCESS;
}

void Input::save (json &input) const
{
    saveGlobals(input);
    saveDatabase(input);
}

void Input::save (json &input, Database &database) const
{
    saveGlobals(input);
    saveDatabase(input, database);
}

/* Load old/new state roots, sequencer address and chain ID */

void Input::loadGlobals (json &input)
{
    // Input JSON file must contain a globalExitRoot key at the root level
    if ( !input.contains("globalExitRoot") ||
         !input["globalExitRoot"].is_string() )
    {
        cerr << "Error: globalExitRoot key not found in input JSON file" << endl;
        exitProcess();
    }
    globalExitRoot = input["globalExitRoot"];
#ifdef LOG_INPUT
    cout << "loadGobals(): globalExitRoot=" << globalExitRoot << endl;
#endif

    // Input JSON file must contain a oldStateRoot key at the root level
    if ( !input.contains("oldStateRoot") ||
         !input["oldStateRoot"].is_string() )
    {
        cerr << "Error: oldStateRoot key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputs.oldStateRoot = input["oldStateRoot"];
#ifdef LOG_INPUT
    cout << "loadGobals(): oldStateRoot=" << publicInputs.oldStateRoot << endl;
#endif

    // Input JSON file must contain a newStateRoot key at the root level
    if ( !input.contains("newStateRoot") ||
         !input["newStateRoot"].is_string() )
    {
        cerr << "Error: newStateRoot key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputs.newStateRoot = input["newStateRoot"];
#ifdef LOG_INPUT
    cout << "loadGobals(): newStateRoot=" << publicInputs.newStateRoot << endl;
#endif

    // Input JSON file must contain a oldLocalExitRoot key at the root level
    if ( !input.contains("oldLocalExitRoot") ||
         !input["oldLocalExitRoot"].is_string() )
    {
        cerr << "Error: oldLocalExitRoot key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputs.oldLocalExitRoot = input["oldLocalExitRoot"];
#ifdef LOG_INPUT
    cout << "loadGobals(): oldLocalExitRoot=" << publicInputs.oldLocalExitRoot << endl;
#endif

    // Input JSON file must contain a newLocalExitRoot key at the root level
    if ( !input.contains("newLocalExitRoot") ||
         !input["newLocalExitRoot"].is_string() )
    {
        cerr << "Error: newLocalExitRoot key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputs.newLocalExitRoot = input["newLocalExitRoot"];
#ifdef LOG_INPUT
    cout << "loadGobals(): newLocalExitRoot=" << publicInputs.newLocalExitRoot << endl;
#endif

    // Input JSON file must contain a sequencerAddr key at the root level
    if ( !input.contains("sequencerAddr") ||
         !input["sequencerAddr"].is_string() )
    {
        cerr << "Error: sequencerAddr key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputs.sequencerAddr = input["sequencerAddr"];
#ifdef LOG_INPUT
    cout << "loadGobals(): sequencerAddr=" << publicInputs.sequencerAddr << endl;
#endif

    // Input JSON file could contain a chainId key at the root level (not mandatory)
    if ( !input.contains("chainId") ||
         !input["chainId"].is_number_unsigned() )
    {
        cerr << "Error: chainId key not found in input JSON file" << endl;
        exitProcess();
    }
    else
    {
        publicInputs.chainId = input["chainId"];
    }
#ifdef LOG_INPUT
    cout << "loadGobals(): chainId=" << publicInputs.chainId << endl;
#endif

    // Input JSON file must contain a numBatch key at the root level
    if ( !input.contains("numBatch") ||
         !input["numBatch"].is_number_unsigned() )
    {
        cerr << "Error: numBatch key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputs.batchNum = input["numBatch"];
#ifdef LOG_INPUT
    cout << "loadGobals(): batchNum=" << publicInputs.batchNum << endl;
#endif

    // Input JSON file must contain a timestamp key at the root level
    if ( !input.contains("timestamp") ||
         !input["timestamp"].is_number_unsigned() )
    {
        cerr << "Error: timestamp key not found in input JSON file" << endl;
        exitProcess();
    }
    publicInputs.timestamp = input["timestamp"];
#ifdef LOG_INPUT
    cout << "loadGobals(): timestamp=" << publicInputs.timestamp << endl;
#endif

    // Input JSON file may contain a aggregatorAddress key at the root level
    if ( input.contains("aggregatorAddress") && 
         input["aggregatorAddress"].is_string() )
    {
        publicInputs.aggregatorAddress = Add0xIfMissing(input["aggregatorAddress"]);
        cout << "loadGobals(): aggregatorAddress=" << publicInputs.aggregatorAddress << endl; 
    }
    else
    {
        publicInputs.aggregatorAddress = "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266"; // Default aggregator address, for testing purposes
    }

    // Input JSON file must contain a batchL2Data key at the root level
    if ( !input.contains("batchL2Data") ||
         !input["batchL2Data"].is_string() )
    {
        cerr << "Error: batchL2Data key not found in input JSON file" << endl;
        exitProcess();
    }
    batchL2Data = Add0xIfMissing(input["batchL2Data"]);
#ifdef LOG_INPUT
    cout << "loadGobals(): batchL2Data=" << batchL2Data << endl;
#endif

    // Input JSON file may contain a from key at the root level
    if ( input.contains("from") && 
         input["from"].is_string() )
    {
        from = Add0xIfMissing(input["from"]);
#ifdef LOG_INPUT
        cout << "loadGobals(): from=" << from << endl;
#endif
    }
}

void Input::saveGlobals (json &input) const
{
    input["globalExitRoot"] = globalExitRoot;
    input["oldStateRoot"] = publicInputs.oldStateRoot;
    input["newStateRoot"] = publicInputs.newStateRoot;
    input["oldLocalExitRoot"] = publicInputs.oldLocalExitRoot;
    input["newLocalExitRoot"] = publicInputs.newLocalExitRoot;
    input["sequencerAddr"] = publicInputs.sequencerAddr;
    input["chainId"] = publicInputs.chainId;
    input["aggregatorAddress"] = publicInputs.aggregatorAddress;    
    input["numBatch"] = publicInputs.batchNum;
    input["timestamp"] = publicInputs.timestamp;
    input["aggregatorAddress"] = publicInputs.aggregatorAddress;  
    input["batchL2Data"] = batchL2Data;
    input["from"] = from;
}

zkresult Input::preprocessTxs (void)
{
#ifdef LOG_INPUT
    cout << "Input::preprocessTxs() input.txsLen=" << txsLen << endl;
#endif

    // Check the batchL2Data length
    if (batchL2Data.size() > (MAX_BATCH_L2_DATA_SIZE*2 + 2))
    {
        cerr << "Error: Input::preprocessTxs() found batchL2Data.size()=" << batchL2Data.size() << " > (MAX_BATCH_L2_DATA_SIZE*2+2)=" << (MAX_BATCH_L2_DATA_SIZE*2+2) << endl;
        return ZKR_SM_MAIN_BATCH_L2_DATA_TOO_BIG;
    }

    // Calculate the TX batch hash
    string keccakInput = batchL2Data;
    keccakInput += NormalizeToNFormat(Remove0xIfPresent(globalExitRoot), 64);
    keccakInput += NormalizeToNFormat(publicInputs.sequencerAddr, 40);

    string keccakOutput = keccak256(keccakInput);

    batchHashData.set_str(Remove0xIfPresent(keccakOutput), 16);
#ifdef LOG_INPUT
    cout << "Input::preprocessTxs() input.batchHashData=" << keccakOutput << endl;
#endif

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
    mpz_class aux2(publicInputs.chainId);
    keccakInput += NormalizeToNFormat(aux2.get_str(16), 16);

    // Calculate the new root hash from the concatenated string
    keccakOutput = keccak256(keccakInput);
    globalHash.set_str(Remove0xIfPresent(keccakOutput), 16);

#ifdef LOG_INPUT
    cout << "Input::preprocessTxs() input.globalHash=" << globalHash.get_str(16) << endl;
#endif
    return ZKR_SUCCESS;
}

/* Store db into database ctx.db[] */

void Input::loadDatabase (json &input)
{
    // Input JSON file must contain a db structure at the root level
    if ( !input.contains("db") ||
         !input["db"].is_structured() )
    {
#ifdef LOG_INPUT
        //cout << "Input::loadDatabase() warning: db key not found in input JSON file" << endl;
#endif
    }
    else
    {
#ifdef LOG_INPUT
        cout << "loadDatabase() db content:" << endl;
#endif
        for (json::iterator it = input["db"].begin(); it != input["db"].end(); ++it)
        {
            // Every value must be a 12-fe array if intermediate node, or 8-fe array if value
            if (!it.value().is_array() ||
                !((it.value().size()==12) || (it.value().size()==8)) )
            {
                cerr << "Error: Input::loadDatabase() keys value array with invalid length in input JSON file: " << it.value() << endl;
                exitProcess();
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
#ifdef LOG_INPUT
            cout << "    key: " << it.key() << " value: " << it.value()[0] << " etc." << endl;
#endif
        }
    }

    // Input JSON file must contain a contractsBytecode structure at the root level
    if ( !input.contains("contractsBytecode") ||
         !input["contractsBytecode"].is_structured() )
    {
#ifdef LOG_INPUT
        //cout << "Input::loadDatabase() warning: contractsBytecode key not found in input JSON file" << endl;
#endif
    }
    else
    {
#ifdef LOG_INPUT
        cout << "loadDatabase() contractsBytecode content:" << endl;
#endif
        for (json::iterator it = input["contractsBytecode"].begin(); it != input["contractsBytecode"].end(); ++it)
        {
            // Add the 16 fe elements into the database value
            vector<uint8_t> dbValue;
            string contractValue = string2ba(it.value());
            for (uint64_t i=0; i<contractValue.size(); i++)
            {
                dbValue.push_back(contractValue.at(i));
            }

            // Get the key fe element
            string key = NormalizeToNFormat(it.key(), 64);

            // Add the key:value pair to the context database
            contractsBytecode[key] = dbValue;
#ifdef LOG_INPUT
            cout << "    key: " << it.key() << " value: " << it.value() << endl;
#endif
        }
    }
}

void Input::db2json (json &input, const std::map<string, vector<Goldilocks::Element>> &db, string name) const

{
    input[name] = json::object();
    for(std::map<string, vector<Goldilocks::Element>>::const_iterator iter = db.begin(); iter != db.end(); iter++)
    {
        string key = NormalizeTo0xNFormat(iter->first, 64);
        vector<Goldilocks::Element> dbValue = iter->second;
        json value;
        for (uint64_t i=0; i<dbValue.size(); i++)
        {
            value[i] = NormalizeToNFormat(fr.toString(dbValue[i], 16), 16);
        }
        input[name][key] = value;
    }
}

void Input::contractsBytecode2json (json &input, const std::map<string, vector<uint8_t>> &contractsBytecode, string name) const
{
    input[name] = json::object();
    for(std::map<string, vector<uint8_t>>::const_iterator iter = contractsBytecode.begin(); iter != contractsBytecode.end(); iter++)
    {
        string key = NormalizeToNFormat(iter->first, 64);
        vector<uint8_t> dbValue = iter->second;
        string value;
        for (uint64_t i=0; i<dbValue.size(); i++)
        {
            value += byte2string(dbValue[i]);
        }
        input[name][key] = value;
    }
}

void Input::saveDatabase (json &input) const 
{
    db2json(input, db, "db");
    contractsBytecode2json(input, contractsBytecode, "contractsBytecode");
}

void Input::saveDatabase (json &input, Database &database) const
{
    database.lock();
    db2json(input, database.dbReadLog, "db");
    database.unlock();
    contractsBytecode2json(input, contractsBytecode, "contractsBytecode");    
}
