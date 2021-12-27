#include <iostream>
#include "config.hpp"
#include "input.hpp"
#include "scalar.hpp"
#include "rlpvalue/rlpvalue.h"

void loadGlobals      (Context &ctx, json &input);
void loadTransactions (Context &ctx, json &input);
void loadStorage      (Context &ctx, json &input);
void loadDatabase     (Context &ctx, json &input);

void loadInput (Context &ctx, json &input)
{
    loadGlobals      (ctx, input);
    loadTransactions (ctx, input);
    loadStorage      (ctx, input);
#ifdef INIT_DATABASE_WITH_INPUT_JSON_DATA
    loadDatabase     (ctx, input);
#endif
}

/* Load old/new state roots, sequencer address and chain ID */

void loadGlobals (Context &ctx, json &input)
{
    // Input JSON file must contain a oldStateRoot key at the root level
    if ( !input.contains("oldStateRoot") ||
         !input["oldStateRoot"].is_string() )
    {
        cerr << "Error: oldStateRoot key not found in input JSON file" << endl;
        exit(-1);
    }
    ctx.oldStateRoot = input["oldStateRoot"];
    cout << "loadGobals(): oldStateRoot=" << ctx.oldStateRoot << endl;

    // Input JSON file must contain a newStateRoot key at the root level
    if ( !input.contains("newStateRoot") ||
         !input["newStateRoot"].is_string() )
    {
        cerr << "Error: newStateRoot key not found in input JSON file" << endl;
        exit(-1);
    }
    ctx.newStateRoot = input["newStateRoot"];
    cout << "loadGobals(): newStateRoot=" << ctx.newStateRoot << endl;

    // Input JSON file must contain a sequencerAddr key at the root level
    if ( !input.contains("sequencerAddr") ||
         !input["sequencerAddr"].is_string() )
    {
        cerr << "Error: sequencerAddr key not found in input JSON file" << endl;
        exit(-1);
    }
    ctx.sequencerAddr = input["sequencerAddr"];
    cout << "loadGobals(): sequencerAddr=" << ctx.sequencerAddr << endl;

    // Input JSON file must contain a chainId key at the root level
    if ( !input.contains("chainId") ||
         !input["chainId"].is_number_unsigned() )
    {
        cerr << "Error: chainId key not found in input JSON file" << endl;
        exit(-1);
    }
    ctx.chainId = input["chainId"];
    cout << "loadGobals(): chainId=" << ctx.chainId << endl;
}

/* Load transactions and resulting globalHash */

void loadTransactions (Context &ctx, json &input)
{
    // Store input data in a vector of strings
    vector<string> d;
    d.push_back(NormalizeTo0xNFormat(ctx.sequencerAddr,64));
    mpz_class aux(ctx.chainId);
    d.push_back(NormalizeTo0xNFormat(aux.get_str(16),4));
    d.push_back(NormalizeTo0xNFormat(ctx.oldStateRoot,64));

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

    // For every transaction, collect the required data
    for (uint64_t i=0; i<numberOfTransactions; i++)
    {
        string tx = input["txs"][i];
        //cout << "loadTransactions(): tx=" << tx << endl;

        // Allocate memory to store the byte array version of the string tx
        uint64_t dataSize = tx.size() + 2;
        uint8_t * pData = (uint8_t *)malloc(dataSize);
        if (pData==NULL)
        {
            cerr << "Error: loadTransactions() failed calling malloc() of " << dataSize << " bytes" << endl;
            exit(-1);
        }

        // Convert the string tx into a byte array
        dataSize = string2ba(tx, pData, dataSize);

        // Parse the transaction byte array into an RLPValue
        RLPValue rtx;
        size_t consumed=0;
        size_t wanted=0;
        rtx.read(pData, dataSize, consumed, wanted );
        free(pData);

        // The root value must be a 9-elements array
        if (!rtx.isArray())
        {
            cerr << "Error: loadTransactions() did not find an array when parsing tx: " << tx << endl;
            exit(-1);
        }
        if (rtx.size()!=9)
        {
            cerr << "Error: loadTransactions() did not find a 9-elements array when parsing tx: " << rtx.size() << " elements in " << tx << endl;
            exit(-1);
        }

        // Check that all children are buffers
        for (uint64_t i=0; i<rtx.size(); i++)
        {
            if (!rtx[i].isBuffer())
            {
            cerr << "Error: loadTransactions() found a non-buffer child when parsing tx: " << i << " element in " << tx << endl;
            exit(-1);
            }
        }

        // Calculate chainID = (rtx[6] - 35) >> 1
        string aux = rtx[6].getValStr(); // Max size will be 64 bits (8B)
        if (aux.size() > 8)
        {
            cerr << "Error: loadTransactions() found invalid rtx[6] size: " << aux.size() << endl;
            exit(-1);
        }
        mpz_class rtx6;
        ba2scalar((const uint8_t *)aux.c_str(), aux.size(), rtx6);
        mpz_class chainID = (rtx6 - 35) >> 1;

        // Calculate sign = 1 - (rtx[6] & 1)
        mpz_class auxScalar = 1 - (rtx6 & 1);
        uint16_t sign = auxScalar.get_ui();

        // Calculate r = rtx[7]
        aux = rtx[7].getValStr();
        if (aux.size() > 32)
        {
            cerr << "Error: loadTransactions() found invalid rtx[7] size: " << aux.size() << endl;
            exit(-1);
        }
        mpz_class r;
        ba2scalar((const uint8_t *)aux.c_str(), aux.size(), r);

        // Calculate s = rtx[8]
        aux = rtx[8].getValStr();
        if (aux.size() > 32)
        {
            cerr << "Error: loadTransactions() found invalid rtx[8] size: " << aux.size() << endl;
            exit(-1);
        }
        mpz_class s;
        ba2scalar((const uint8_t *)aux.c_str(), aux.size(), s);

        // Calculate v = sign + 27;
        uint16_t v = sign + 27;

        // Build a new array of RLP values to calculate the new root state
        RLPValue e;
        e.setArray();

        // Elements 0 to 5 are the same as in the old transaction
        e.push_back(rtx[0]);
        e.push_back(rtx[1]);
        e.push_back(rtx[2]);
        e.push_back(rtx[3]);
        e.push_back(rtx[4]);
        e.push_back(rtx[5]);

        // Element 6 is the encoded chainID
        uint8_t ba[9]; // One extra byte for the final 0
        dataSize = 8; // Max size is 64b = 8B
        scalar2ba(&ba[0], dataSize, chainID);
        aux="";
        for (uint64_t i=0; i<dataSize; i++) aux.push_back(ba[i]);
        RLPValue chainIDValue(aux.c_str());
        
        e.push_back(chainIDValue);

        // Elements 7 and 8 are empty elements
        RLPValue empty("");
        e.push_back(empty);
        e.push_back(empty);

        // Get the RLP-encoded version of the new array of RLP elements
        string auxString;
        auxString = e.write();
        string signData;
        ba2string(signData, (const uint8_t *)auxString.data(), auxString.size());
        signData = "0x" + signData;

        // Add it to the strings vector
        d.push_back(signData);

        // Build a transaction data instance, and add it to the ctx.txs[] vector
        TxData txData;
        txData.originalTx = tx;
        txData.signData = signData;
        txData.r = r;
        txData.s = s;
        txData.v = v;
        ctx.txs.push_back(txData);

    }

    // Finally, add the new root state to the vector of strings
    d.push_back(NormalizeTo0xNFormat(ctx.newStateRoot,64));

    // Concatenate d into one single string concat with the pattern 0xnnn...
    string concat = "0x";
    for (uint64_t i=0; i<d.size(); i++)
    {
        concat += Remove0xIfPresent(d[i]);
    }

    // Calculate the new root hash from the concatenated string
    string hash = keccak256(concat);
    ctx.globalHash.set_str(Remove0xIfPresent(hash), 16);
    cout << "ctx.globalHash=" << ctx.globalHash.get_str(16) << endl;
}

/* Store keys into storage ctx.sto[] */

void loadStorage (Context &ctx, json &input)
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
        string2fe(ctx.fr, it.key(), fe);

        // Read scalar from it.value()
        mpz_class scalar;
        scalar.set_str(it.value(), 16);

        // Store the key:value pair in context storage
        ctx.sto[fe] = scalar;

#ifdef LOG_STORAGE
        cout << "loadStorage() added record with key(fe): " << ctx.fr.toString(fe, 16) << " value(scalar): " << scalar.get_str(16) << endl;
#endif
    }
}

/* Store db into database ctx.db[] */

void loadDatabase (Context &ctx, json &input)
{
    // Input JSON file must contain a db structure at the root level
    if ( !input.contains("db") ||
         !input["db"].is_structured() )
    {
        cerr << "Error: db key not found in input JSON file" << endl;
        exit(-1);
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
            string2fe(ctx.fr, it.value()[i], fe);
            dbValue.push_back(fe);
        }

        // Get the key fe element
        RawFr::Element key;
        string2fe(ctx.fr, it.key(), key);

        // Add the key:value pair to the context database
        ctx.db.create(key, dbValue);
        cout << "    key: " << it.key() << " value: " << it.value()[0] << " etc." << endl;
    }   
}