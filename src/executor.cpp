
#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "ffiasm/fr.hpp"
#include "executor.hpp"

using namespace std;
using json = nlohmann::json;

#define NEVALUATIONS 1000 //1<<23 // 8M
#define NPOLS 100 //512
// TODO: Segmentation fault: out of memory

typedef RawFr::Element tPolynomial[NEVALUATIONS];
typedef tPolynomial tExecutorOutput[NPOLS]; // polsArray
// TODO: Allocate dynamically?

#define INVALID_ID 0xFFFFFFFFFFFFFFFF
uint64_t A0 = INVALID_ID;
uint64_t A1 = INVALID_ID;
uint64_t A2 = INVALID_ID;
uint64_t A3 = INVALID_ID;
uint64_t B0 = INVALID_ID;
uint64_t B1 = INVALID_ID;
uint64_t B2 = INVALID_ID;
uint64_t B3 = INVALID_ID;
uint64_t C0 = INVALID_ID;
uint64_t C1 = INVALID_ID;
uint64_t C2 = INVALID_ID;
uint64_t C3 = INVALID_ID;
uint64_t D0 = INVALID_ID;
uint64_t D1 = INVALID_ID;
uint64_t D2 = INVALID_ID;
uint64_t D3 = INVALID_ID;
uint64_t E0 = INVALID_ID;
uint64_t E1 = INVALID_ID;
uint64_t E2 = INVALID_ID;
uint64_t E3 = INVALID_ID;
uint64_t FREE0 = INVALID_ID;
uint64_t FREE1 = INVALID_ID;
uint64_t FREE2 = INVALID_ID;
uint64_t FREE3 = INVALID_ID;
uint64_t CONST = INVALID_ID;
uint64_t CTX = INVALID_ID;
uint64_t GAS = INVALID_ID;
uint64_t JMP = INVALID_ID;
uint64_t JMPC = INVALID_ID;
uint64_t MAXMEM = INVALID_ID;
uint64_t PC = INVALID_ID;
uint64_t SP = INVALID_ID;
uint64_t SR = INVALID_ID;
uint64_t arith = INVALID_ID;
uint64_t assert = INVALID_ID;
uint64_t bin = INVALID_ID;
uint64_t comparator = INVALID_ID;
uint64_t ecRecover = INVALID_ID;
uint64_t hashE = INVALID_ID;
uint64_t hashRD = INVALID_ID;
uint64_t hashWR = INVALID_ID;
uint64_t inA = INVALID_ID;
uint64_t inB = INVALID_ID;
uint64_t inC = INVALID_ID;
uint64_t inD = INVALID_ID;
uint64_t inE = INVALID_ID;
uint64_t inCTX = INVALID_ID;
uint64_t inFREE = INVALID_ID;
uint64_t inGAS = INVALID_ID;
uint64_t inMAXMEM = INVALID_ID;
uint64_t inPC = INVALID_ID;
uint64_t inSP = INVALID_ID;
uint64_t inSR = INVALID_ID;
uint64_t inSTEP = INVALID_ID;
uint64_t inc = INVALID_ID;
uint64_t ind = INVALID_ID;
uint64_t isCode = INVALID_ID;
uint64_t isMaxMem = INVALID_ID;
uint64_t isMem = INVALID_ID;
uint64_t isNeg = INVALID_ID;
uint64_t isStack = INVALID_ID;
uint64_t mRD = INVALID_ID;
uint64_t mWR = INVALID_ID;
uint64_t neg = INVALID_ID;
uint64_t offset = INVALID_ID;
uint64_t opcodeRomMap = INVALID_ID;
uint64_t sRD = INVALID_ID;
uint64_t sWR = INVALID_ID;
uint64_t setA = INVALID_ID;
uint64_t setB = INVALID_ID;
uint64_t setC = INVALID_ID;
uint64_t setD = INVALID_ID;
uint64_t setE = INVALID_ID;
uint64_t setCTX = INVALID_ID;
uint64_t setGAS = INVALID_ID;
uint64_t setMAXMEM = INVALID_ID;
uint64_t setPC = INVALID_ID;
uint64_t setSP = INVALID_ID;
uint64_t setSR = INVALID_ID;
uint64_t shl = INVALID_ID;
uint64_t shr = INVALID_ID;
uint64_t useCTX = INVALID_ID;
uint64_t zkPC = INVALID_ID;
uint64_t byte4_freeIN = INVALID_ID;
uint64_t byte4_out = INVALID_ID;

// TODO: check if map performance is better
void addPol(const char * pName, uint64_t id)
{
         if (!strcmp(pName,"main.A0")) A0 = id;
    else if (!strcmp(pName,"main.A1")) A1 = id;
    else if (!strcmp(pName,"main.A2")) A2 = id;
    else if (!strcmp(pName,"main.A3")) A3 = id;
    else if (!strcmp(pName,"main.B0")) B0 = id;
    else if (!strcmp(pName,"main.B1")) B1 = id;
    else if (!strcmp(pName,"main.B2")) B2 = id;
    else if (!strcmp(pName,"main.B3")) B3 = id;
    else if (!strcmp(pName,"main.C0")) C0 = id;
    else if (!strcmp(pName,"main.C1")) C1 = id;
    else if (!strcmp(pName,"main.C2")) C2 = id;
    else if (!strcmp(pName,"main.C3")) C3 = id;
    else if (!strcmp(pName,"main.D0")) D0 = id;
    else if (!strcmp(pName,"main.D1")) D1 = id;
    else if (!strcmp(pName,"main.D2")) D2 = id;
    else if (!strcmp(pName,"main.D3")) D3 = id;
    else if (!strcmp(pName,"main.E0")) E0 = id;
    else if (!strcmp(pName,"main.E1")) E1 = id;
    else if (!strcmp(pName,"main.E2")) E2 = id;
    else if (!strcmp(pName,"main.E3")) E3 = id;
    else if (!strcmp(pName,"main.FREE0")) FREE0 = id;
    else if (!strcmp(pName,"main.FREE1")) FREE1 = id;
    else if (!strcmp(pName,"main.FREE2")) FREE2 = id;
    else if (!strcmp(pName,"main.FREE3")) FREE3 = id;
    else if (!strcmp(pName,"main.CONST")) CONST = id;
    else if (!strcmp(pName,"main.CTX")) CTX = id;
    else if (!strcmp(pName,"main.GAS")) GAS = id;
    else if (!strcmp(pName,"main.JMP")) JMP = id;
    else if (!strcmp(pName,"main.JMPC")) JMPC = id;
    else if (!strcmp(pName,"main.MAXMEM")) MAXMEM = id;
    else if (!strcmp(pName,"main.PC")) PC = id;
    else if (!strcmp(pName,"main.SP")) SP = id;
    else if (!strcmp(pName,"main.SR")) SR = id;
    else if (!strcmp(pName,"main.arith")) arith = id;
    else if (!strcmp(pName,"main.assert")) assert = id;
    else if (!strcmp(pName,"main.bin")) bin = id;
    else if (!strcmp(pName,"main.comparator")) comparator = id;
    else if (!strcmp(pName,"main.ecRecover")) ecRecover = id;
    else if (!strcmp(pName,"main.hashE")) hashE = id;
    else if (!strcmp(pName,"main.hashRD")) hashRD = id;
    else if (!strcmp(pName,"main.hashWR")) hashWR = id;
    else if (!strcmp(pName,"main.inA")) inA = id;
    else if (!strcmp(pName,"main.inB")) inB = id;
    else if (!strcmp(pName,"main.inC")) inC = id;
    else if (!strcmp(pName,"main.inD")) inD = id;
    else if (!strcmp(pName,"main.inE")) inE = id;
    else if (!strcmp(pName,"main.inCTX")) inCTX = id;
    else if (!strcmp(pName,"main.inFREE")) inFREE = id;
    else if (!strcmp(pName,"main.inGAS")) inGAS = id;
    else if (!strcmp(pName,"main.inMAXMEM")) inMAXMEM = id;
    else if (!strcmp(pName,"main.inPC")) inPC = id;
    else if (!strcmp(pName,"main.inSP")) inSP = id;
    else if (!strcmp(pName,"main.inSR")) inSR = id;
    else if (!strcmp(pName,"main.inSTEP")) inSTEP = id;
    else if (!strcmp(pName,"main.inc")) inc = id;
    else if (!strcmp(pName,"main.ind")) ind = id;
    else if (!strcmp(pName,"main.isCode")) isCode = id;
    else if (!strcmp(pName,"main.isMaxMem")) isMaxMem = id;
    else if (!strcmp(pName,"main.isMem")) isMem = id;
    else if (!strcmp(pName,"main.isNeg")) isNeg = id;
    else if (!strcmp(pName,"main.isStack")) isStack = id;
    else if (!strcmp(pName,"main.mRD")) mRD = id;
    else if (!strcmp(pName,"main.mWR")) mWR = id;
    else if (!strcmp(pName,"main.neg")) neg = id;
    else if (!strcmp(pName,"main.offset")) offset = id;
    else if (!strcmp(pName,"main.opcodeRomMap")) opcodeRomMap = id;
    else if (!strcmp(pName,"main.sRD")) sRD = id;
    else if (!strcmp(pName,"main.sWR")) sWR = id;
    else if (!strcmp(pName,"main.setA")) setA = id;
    else if (!strcmp(pName,"main.setB")) setB = id;
    else if (!strcmp(pName,"main.setC")) setC = id;
    else if (!strcmp(pName,"main.setD")) setD = id;
    else if (!strcmp(pName,"main.setE")) setE = id;
    else if (!strcmp(pName,"main.setCTX")) setCTX = id;
    else if (!strcmp(pName,"main.setGAS")) setGAS = id;
    else if (!strcmp(pName,"main.setMAXMEM")) setMAXMEM = id;
    else if (!strcmp(pName,"main.setPC")) setPC = id;
    else if (!strcmp(pName,"main.setSP")) setSP = id;
    else if (!strcmp(pName,"main.setSR")) setSR = id;
    else if (!strcmp(pName,"main.shl")) shl = id;
    else if (!strcmp(pName,"main.shr")) shr = id;
    else if (!strcmp(pName,"main.useCTX")) useCTX = id;
    else if (!strcmp(pName,"main.zkPC")) zkPC = id;
    else if (!strcmp(pName,"byte4.freeIN")) byte4_freeIN = id;
    else if (!strcmp(pName,"byte4.out")) byte4_out = id;
    else
    {
        cerr << "Error: pol() could not find a polynomial for name " << pName << endl;
        exit(-1); // TODO: Should we kill the process?
    }
}

typedef struct {
    uint64_t ln; // Program Counter (PC)
    uint64_t step; // Interation, instruction execution loop counter
    string fileName; // From ROM JSON file instruction
    uint64_t line; // From ROM JSON file instruction
} tContext;

void createPols(tExecutorOutput &pols, json &pil);
void initState(RawFr &fr, tExecutorOutput &pols);
void preprocessTxs(tContext &ctx, json &input);

void execute(RawFr &fr, json &input, json &rom, json &pil)
{
    cout << "execute()" << endl;

    tExecutorOutput pols;
    tContext ctx;
    RawFr::Element op3, op2, op1, op0;

    pols[A0][0] = fr.zero(); // Fast version
    //pols[polmap("A0")][0] = fr.zero(); // Slow version

    //vector<vector<RawFr::Element>> polsArray;

    //[pols, polsArray] = createPols(pil);
    createPols(pols, pil); // TODO: rename to addPols()

    initState(fr, pols);

    preprocessTxs(ctx, input);

    assert(rom.is_array());

    for (uint64_t i=0; i<NEVALUATIONS; i++)
    {
        //ctx.ln = Fr_toInt(pols[zkPC][i].v); // TODO: ctx.ln = Fr.toObject(pols.main.zkPC[i]);
        ctx.ln = i;
        ctx.step = i;
        //ctx.A = [pols.main.A0[i], pols.main.A1[i], pols.main.A2[i], pols.main.A3[i]];
        //ctx.B = [pols.main.B0[i], pols.main.B1[i], pols.main.B2[i], pols.main.B3[i]];
        //ctx.C = [pols.main.C0[i], pols.main.C1[i], pols.main.C2[i], pols.main.C3[i]];
        //ctx.D = [pols.main.D0[i], pols.main.D1[i], pols.main.D2[i], pols.main.D3[i]];
        //ctx.E = [pols.main.E0[i], pols.main.E1[i], pols.main.E2[i], pols.main.E3[i]];
        //ctx.SR = pols.main.SR[i];
        //ctx.CTX = pols.main.CTX[i];
        //ctx.SP = pols.main.SP[i];
        //ctx.PC = pols.main.PC[i];
        //ctx.MAXMEM = pols.main.MAXMEM[i];
        //ctx.GAS = pols.main.GAS[i];
        //ctx.zkPC = pols.main.zkPC[i];
    
        // Get the line ctx.ln from the ROM JSON file
        //const l = rom[ fe2n(Fr, ctx.zkPC) ];
        json l = rom[ctx.ln];

        // In case we reached the end of the ROM JSON file, break the for() loop
        if (!l.is_object())
        {
            break;
        }

        ctx.fileName = l["fileName"]; // TODO: check presence and type
        ctx.line = l["line"]; // TODO: check presence and type

        //printRegs(Fr, ctx);

        //if (i==104) {
        //    console.log("pause");
        //}


        /*if (l.cmdBefore) {
            for (let j=0; j< l.cmdBefore.length; j++) {
                evalCommand(ctx, l.cmdBefore[j]);
            }
        }*/
        json cmdBefore = l["cmdBefore"];
        if (cmdBefore.is_object())
        {
            assert(cmdBefore.is_array()); // TODO: Should we assert, exit(-1), or something else?
            for (json::iterator it = cmdBefore.begin(); it != cmdBefore.end(); ++it) {
                std::cout << *it << '\n';
                //evalCommand(ctx,*it);
            }
        }

        op0 = op1 = op2 = op3 = fr.zero(); //[op0, op1, op2, op3] = [Fr.zero, Fr.zero, Fr.zero, Fr.zero];

        if (l.contains("inA") && l["inA"].is_number_unsigned() && l["inA"]==1) // TODO: We are grebbing 3 times the string "inA".  Can we do it in only 1 check?
        {
            fr.add(op0,op0,pols[A0][i]); // TODO: Shouldn't it be a fr.copy()?
            fr.add(op1,op1,pols[A1][i]);
            fr.add(op2,op2,pols[A2][i]);
            fr.add(op3,op3,pols[A3][i]);
            pols[inA][i] = fr.one();
        }
        else {
            pols[inA][i] = fr.zero();
        }
        /*if (l.inA == 1) {
            [op0, op1, op2, op3] = [Fr.add(op0, ctx.A[0]), Fr.add(op1, ctx.A[1]), Fr.add(op2, ctx.A[2]), Fr.add(op3, ctx.A[3])];
            pols.main.inA[i] = Fr.one;
        } else {
            pols.main.inA[i] = Fr.zero;
        }*/

    }
}

void createPols(tExecutorOutput &pols, json &pil)
{
    // PIL JSON file must contain a nCommitments key at the root level
    if ( !pil.contains("nCommitments") ||
         !pil["nCommitments"].is_number_unsigned() )
    {
        cerr << "Error: nCommitments key not found in PIL JSON file" << endl;
        exit(-1);
    }
    uint64_t nCommitments;
    nCommitments = pil["nCommitments"];
    cout << nCommitments << endl;

    // PIL JSON file must contain a references structure at the root level
    if ( !pil.contains("references") ||
         !pil["references"].is_structured() )
    {
        cerr << "Error: references key not found in PIL JSON file" << endl;
        exit(-1);
    }

    // Iterate the PIL JSON references array
    json references = pil["references"];
    //json kk = pil["kk"];
    //cout << "kk.is_object()=" << kk.is_object() << endl;
    //cout << "kk.is_structured()=" << kk.is_structured() << endl;
    //cout << "references.is_object()=" << references.is_object() << endl;
    //cout << "references.is_structured()=" << references.is_structured() << endl;
    uint64_t addedPols = 0;
    for (json::iterator it = references.begin(); it != references.end(); ++it) {
        string key = it.key();
        json value = it.value();
        if ( value.is_object() &&
             value.contains("type") && 
             value["type"].is_string() &&
             value.contains("id") &&
             value["id"].is_number_unsigned() ) 
        {
            string type = it.value()["type"];
            uint64_t id = it.value()["id"];
            //cout << key << " - " << type << " - " << id << endl;
            if (type.compare("cmP")==0) { // TODO: check that there are exactly 2 substrings
                /*string nameSpace;
                string namePol;
                istringstream iss(key);
                getline(iss,nameSpace,'.');
                getline(iss,namePol,'.');
                if (nameSpace.compare("main")!=0) continue;
                uint64_t numPol = polmap(namePol.c_str());
                cout << "    " << nameSpace << " - " << namePol << " - " << type << " - " << id << " - " << numPol << endl;*/
                if (id>=NPOLS)
                {
                    cerr << "Error: polynomial " << key << " id(" << id << ") >= NPOLS(" << NPOLS << ")" << endl;
                    exit(-1);
                }
                addPol(key.c_str(),id);
                addedPols++;
                cout << "Added polynomial " << addedPols << ": " << key << " with ID " << id << endl;
            }
        }

    }
}

/* 
    This function creates an array of polynomials and a mapping that maps the reference name in pil to the polynomial
*/
/*
function createPols(pil) {
    polsArray = [];
    pols = {};
    for (let i=0; i<pil.nCommitments; i++) polsArray.push([]);
    for (refName in pil.references) {
        if (pil.references.hasOwnProperty(refName)) {
            ref = pil.references[refName];
            if (ref.type == "cmP") {
                [nameSpace, namePol] = refName.split(".");
                if (!pols[nameSpace]) pols[nameSpace] = {};
                pols[nameSpace][namePol] = polsArray[ref.id];
            }
        }
    }

    return [pols, polsArray];
}*/

void initState(RawFr &fr, tExecutorOutput &pols)
{
    // Register value initial parameters
    pols[A0][0] = fr.zero();
    pols[A1][0] = fr.zero();
    pols[A2][0] = fr.zero();
    pols[A3][0] = fr.zero();
    pols[B0][0] = fr.zero();
    pols[B1][0] = fr.zero();
    pols[B2][0] = fr.zero();
    pols[B3][0] = fr.zero();
    pols[C0][0] = fr.zero();
    pols[C1][0] = fr.zero();
    pols[C2][0] = fr.zero();
    pols[C3][0] = fr.zero();
    pols[D0][0] = fr.zero();
    pols[D1][0] = fr.zero();
    pols[D2][0] = fr.zero();
    pols[D3][0] = fr.zero();
    pols[E0][0] = fr.zero();
    pols[E1][0] = fr.zero();
    pols[E2][0] = fr.zero();
    pols[E3][0] = fr.zero();
    pols[SR][0] = fr.zero();
    pols[CTX][0] = fr.zero();
    pols[SP][0] = fr.zero();
    pols[PC][0] = fr.zero();
    pols[MAXMEM][0] = fr.zero();
    pols[GAS][0] = fr.zero();
    pols[zkPC][0] = fr.zero();
}

void preprocessTxs(tContext &ctx, json &input)
{
    // Input JSON file must contain a oldStateRoot key at the root level
    if ( !input.contains("oldStateRoot") ||
         !input["oldStateRoot"].is_string() )
    {
        cerr << "Error: oldStateRoot key not found in input JSON file" << endl;
        exit(-1);
    }
    string oldStateRoot = input["oldStateRoot"];
    cout << "preprocessTxs(): oldStateRoot=" << oldStateRoot << endl;

    // Input JSON file must contain a oldStateRoot key at the root level
    if ( !input.contains("newStateRoot") ||
         !input["newStateRoot"].is_string() )
    {
        cerr << "Error: newStateRoot key not found in input JSON file" << endl;
        exit(-1);
    }
    string newStateRoot = input["newStateRoot"];
    cout << "preprocessTxs(): newStateRoot=" << newStateRoot << endl;

    // Input JSON file must contain a sequencerAddr key at the root level
    if ( !input.contains("sequencerAddr") ||
         !input["sequencerAddr"].is_string() )
    {
        cerr << "Error: sequencerAddr key not found in input JSON file" << endl;
        exit(-1);
    }
    string sequencerAddr = input["sequencerAddr"];
    cout << "preprocessTxs(): sequencerAddr=" << sequencerAddr << endl;

    // Input JSON file must contain a chainId key at the root level
    if ( !input.contains("chainId") ||
         !input["chainId"].is_number_unsigned() )
    {
        cerr << "Error: chainId key not found in input JSON file" << endl;
        exit(-1);
    }
    uint64_t chainId = input["chainId"];
    cout << "preprocessTxs(): chainId=" << chainId << endl;

    // Input JSON file must contain a txs structure at the root level
    if ( !input.contains("txs") ||
         !input["txs"].is_array() )
    {
        cerr << "Error: txs key not found in input JSON file" << endl;
        exit(-1);
    }
    vector<string> txs = input["txs"]; // TODO: check that all array values are strings
    for (std::vector<string>::iterator it = txs.begin() ; it != txs.end(); ++it)
    {
        cout << "preprocessTxs(): tx=" << *it << endl;
    }

}
/*
function preprocessTxs(ctx) {
    ctx.pTxs = [];
    const d = [];
    d.push(ethers.utils.hexZeroPad(ethers.utils.hexlify(ethers.BigNumber.from(ctx.input.oldStateRoot)), 32));
    d.push(ethers.utils.hexZeroPad(ethers.utils.hexlify(ethers.BigNumber.from(ctx.input.newStateRoot)), 32));
    for (let i=0; i<ctx.input.txs.length; i++) {
        const rtx = ethers.utils.RLP.decode(ctx.input.txs[i]);
        const chainId = (Number(rtx[6]) - 35) >> 1;
        const sign = Number(rtx[6])  & 1;
        const e =[rtx[0], rtx[1], rtx[2], rtx[3], rtx[4], rtx[5], ethers.utils.hexlify(chainId), "0x", "0x"];
        const signData = ethers.utils.RLP.encode( e );
        ctx.pTxs.push({
            signData: signData,
            signature: {
                r: rtx[7],k
                s: rtx[8],
                v: sign + 26
            }
        });
        d.push(signData);
    }
    ctx.globalHash = ethers.utils.keccak256(ctx.globalHash = ethers.utils.concat(d));
}*/