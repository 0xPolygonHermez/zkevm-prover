#include <iostream>

#include "config.hpp"
#include "input.hpp"
#include "scalar.hpp"
#include "rlpvalue/rlpvalue.h"

void preprocessTxs (Context &ctx, json &input)
{
    // Input JSON file must contain a oldStateRoot key at the root level
    if ( !input.contains("oldStateRoot") ||
         !input["oldStateRoot"].is_string() )
    {
        cerr << "Error: oldStateRoot key not found in input JSON file" << endl;
        exit(-1);
    }
    ctx.oldStateRoot = input["oldStateRoot"];
    cout << "preprocessTxs(): oldStateRoot=" << ctx.oldStateRoot << endl;

    // Input JSON file must contain a oldStateRoot key at the root level
    if ( !input.contains("newStateRoot") ||
         !input["newStateRoot"].is_string() )
    {
        cerr << "Error: newStateRoot key not found in input JSON file" << endl;
        exit(-1);
    }
    ctx.newStateRoot = input["newStateRoot"];
    cout << "preprocessTxs(): newStateRoot=" << ctx.newStateRoot << endl;

    // Input JSON file must contain a sequencerAddr key at the root level
    if ( !input.contains("sequencerAddr") ||
         !input["sequencerAddr"].is_string() )
    {
        cerr << "Error: sequencerAddr key not found in input JSON file" << endl;
        exit(-1);
    }
    ctx.sequencerAddr = input["sequencerAddr"];
    cout << "preprocessTxs(): sequencerAddr=" << ctx.sequencerAddr << endl;

    // Input JSON file must contain a chainId key at the root level
    if ( !input.contains("chainId") ||
         !input["chainId"].is_number_unsigned() )
    {
        cerr << "Error: chainId key not found in input JSON file" << endl;
        exit(-1);
    }
    ctx.chainId = input["chainId"];
    cout << "preprocessTxs(): chainId=" << ctx.chainId << endl;

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
    for (int i=0; i<input["txs"].size(); i++)
    {
        string tx = input["txs"][i];
        cout << "preprocessTxs(): tx=" << tx << endl;

        

        RLPValue rtx;
        size_t consumed=0;
        size_t wanted=0;

        uint8_t data[1024]; // TODO: Should we have a fixed limit, or dynamically allocate memory?
        uint64_t maxSize = sizeof(data);
        uint64_t dataSize = string2ba(tx, data, maxSize);

        // Parse tx, and expect a 9-elements array
        rtx.read(data, dataSize, consumed, wanted );
        if (!rtx.isArray())
        {
            cerr << "Error: preprocessTxs() did not find an array when parsing tx: " << tx << endl;
            exit(-1);
        }
        if (rtx.size() != 9)
        {
            cerr << "Error: preprocessTxs() did not find a 9-elements array when parsing tx: " << rtx.size() << " elements in " << tx << endl;
            exit(-1);
        }

        // Check that all children are buffers
        for (int i=0; i<rtx.size(); i++)
        {
            if (!rtx[i].isBuffer())
            {
            cerr << "Error: preprocessTxs() found a non-buffer child when parsing tx: " << i << " element in " << tx << endl;
            exit(-1);
            }
        }

        // chainID = (rtx[6] - 35) >> 1
        string aux = rtx[6].getValStr();
        if (aux.size() != 2)
        {
            cerr << "Error: preprocessTxs() found invalid rtx[6] size: " << aux.size() << endl;
            exit(-1);
        }
        uint16_t rtx6;
        ba2u16((const uint8_t *)aux.c_str(), rtx6);
        uint16_t chainID = (rtx6 - 35) >> 1; // 400

        // sign = 1 - (rtx[6] & 1)
        uint16_t sign = 1 - (rtx6 & 1); // 0

        // r = rtx[7]
        aux = rtx[7].getValStr();
        if (aux.size() != 32)
        {
            cerr << "Error: preprocessTxs() found invalid rtx[7] size: " << aux.size() << endl;
            exit(-1);
        }
        mpz_class r;
        ba2scalar((const uint8_t *)aux.c_str(), r);

        // s = rtx[8]
        aux = rtx[8].getValStr();
        if (aux.size() != 32)
        {
            cerr << "Error: preprocessTxs() found invalid rtx[8] size: " << aux.size() << endl;
            exit(-1);
        }
        mpz_class s;
        ba2scalar((const uint8_t *)aux.c_str(), s);

        // v = sign + 27;
        uint16_t v = sign + 27;

        RLPValue e;
        e.setArray();
        e.push_back(rtx[0]);
        e.push_back(rtx[1]);
        e.push_back(rtx[2]);
        e.push_back(rtx[3]);
        e.push_back(rtx[4]);
        e.push_back(rtx[5]);

        uint8_t ba[3];
        ba[0] = chainID >> 8;
        ba[1] = chainID & 0xFF;
        ba[2] = 0;
        RLPValue chainIDValue((const char*)&ba[0]);
        e.push_back(chainIDValue);

        RLPValue empty("");
        e.push_back(empty);
        e.push_back(empty);

        string auxString;
        auxString = e.write();
        string signData;
        ba2string(signData, (const uint8_t *)auxString.data(), auxString.size());
        signData = "0x" + signData;
        d.push_back(signData);

        TxData txData;
        txData.originalTx = tx;
        txData.signData = signData;
        txData.r = r;
        txData.s = s;
        txData.v = v;  // TODO: can we avoid these copies by converting directly to these elements?
        ctx.txs.push_back(txData);

    }

    d.push_back(NormalizeTo0xNFormat(ctx.newStateRoot,64));

    // Concatenate d into one single string concat with the pattern 0xnnn...
    string concat = "0x";
    for (int i=0; i<d.size(); i++)
    {
        concat += RemoveOxIfPresent(d[i]);
    }
    cout << "concat: " << concat << endl;

    // globalHash = keccak256(concat)
    string hash = keccak256(concat);
    ctx.globalHash.set_str(RemoveOxIfPresent(hash), 16);
    cout << "ctx.globalHash=" << ctx.globalHash.get_str(16) << endl;

    // Input JSON file must contain a keys structure at the root level
    if ( !input.contains("keys") ||
         !input["keys"].is_structured() )
    {
        cerr << "Error: keys key not found in input JSON file" << endl;
        exit(-1);
    }
    cout << "keys content:" << endl;
    for (json::iterator it = input["keys"].begin(); it != input["keys"].end(); ++it)
    {
        RawFr::Element fe;
        mpz_class scalar;

        // Read fe from it.key()
        string s;
        s = it.key();
        scalar.set_str(s, 16);
        ctx.fr.fromMpz(fe, scalar.get_mpz_t());

        // Read scalar from it.value()
        s = it.value();
        scalar.set_str(s,16);

        // Store the key:value pair in context storage
        ctx.sto[fe] = scalar;

#ifdef LOG_STORAGE
        cout << "Storage added record with key(fe): " << ctx.fr.toString(fe, 16) << " value(scalar): " << scalar.get_str(16) << endl;
#endif
    }

    // Input JSON file must contain a db structure at the root level
    if ( !input.contains("db") ||
         !input["db"].is_structured() )
    {
        cerr << "Error: db key not found in input JSON file" << endl;
        exit(-1);
    }
    cout << "db content:" << endl;
    for (json::iterator it = input["db"].begin(); it != input["db"].end(); ++it)
    {
        if (!it.value().is_array() ||
            !it.value().size()==16)
        {
            cerr << "Error: keys value not a 16-elements array in input JSON file: " << it.value() << endl;
            exit(-1);
        }
        vector<RawFr::Element> dbValue;
        for (int i=0; i<16; i++)
        {
            RawFr::Element auxFe;
            string s = it.value()[i];
            mpz_class scalar;
            scalar.set_str(s, 16);
            ctx.fr.fromMpz(auxFe, scalar.get_mpz_t());
            dbValue.push_back(auxFe);
        }
        RawFr::Element key;
        mpz_class scalar;
        scalar.set_str(it.key(), 16);
        ctx.fr.fromMpz(key, scalar.get_mpz_t());
        ctx.db[key] = dbValue;
        cout << "key: " << it.key() << " value: " << it.value()[0] << " etc." << endl;
    }
}
/* TODO: Migrate this code

function preprocessTxs(ctx) {
    ctx.pTxs = [];
    const d = [];
    d.push(ethers.utils.hexZeroPad(ethers.utils.hexlify(ethers.BigNumber.from(ctx.input.sequencerAddr)), 32));
    d.push(ethers.utils.hexZeroPad(ethers.utils.hexlify(ethers.BigNumber.from(ctx.input.chainId)), 2));
    d.push(ethers.utils.hexZeroPad(ethers.utils.hexlify(ethers.BigNumber.from(ctx.input.oldStateRoot)), 32));
    for (let i=0; i<ctx.input.txs.length; i++) {
        const rtx = ethers.utils.RLP.decode(ctx.input.txs[i]);
        const chainId = (Number(rtx[6]) - 35) >> 1;
        const sign = Number(rtx[6])  & 1;
        const e =[rtx[0], rtx[1], rtx[2], rtx[3], rtx[4], rtx[5], ethers.utils.hexlify(chainId), "0x", "0x"];
        const signData = ethers.utils.RLP.encode( e );
        ctx.pTxs.push({
            signData: signData,
            signature: {
                r: rtx[7],
                s: rtx[8],
                v: sign + 26
            }
        });
        d.push(signData);
    }
    d.push(ethers.utils.hexZeroPad(ethers.utils.hexlify(ethers.BigNumber.from(ctx.input.newStateRoot)), 32));
    ctx.globalHash = ethers.utils.keccak256(ctx.globalHash = ethers.utils.concat(d));
}*/