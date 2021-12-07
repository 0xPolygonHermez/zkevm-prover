#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <gmpxx.h>

#include "config.hpp"
#include "ffiasm/fr.hpp"
#include "executor.hpp"
#include "rom_line.hpp"
#include "rom_command.hpp"
#include "rom.hpp"
#include "context.hpp"
#include "pols.hpp"
#include "input.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "eval_command.hpp"
#include "poseidon_opt/poseidon_opt.hpp"
#include "smt.hpp"
#include "ecrecover/ecrecover.hpp"

using namespace std;
using json = nlohmann::json;

#define MEMORY_SIZE 1000 // TODO: decide maximum size
#define MEM_OFFSET 0x300000000
#define STACK_OFFSET 0x200000000
#define CODE_OFFSET 0x100000000
#define CTX_OFFSET 0x400000000

void initState(RawFr &fr, Context &ctx);
void checkFinalState(RawFr &fr, Context &ctx);

void execute (RawFr &fr, json &input, json &romJson, json &pil, string &outputFile)
{

    std::string a = ecrecover("0x508875071183e0839fc7992b309c3a9d557b7274782d307f33706fddbea193fb3d799f700f43296a8432594ed7de022eab1d43b608949ca098da7824739f57041c","hola"); 
    printf("%s\n",a.c_str());
    // It should return 0x112d1e402c161057ad6dcbb31737f11420d9f209

    //string test = "0x00"; // 0xbc36...
    string test = "0x"; // 0xc5d2...
    // keccak hash: 044852b2a670ade5407e78fb2863c51de9fcb96542a07186fe3aeda6bb8a116d
    cout << "Keccak hash: " << keccak256(test) << endl;

    Context ctx(fr);
    memset(&ctx.pols, 0, sizeof(ctx.pols));

    Poseidon_opt poseidon;
    Smt smt(ARITY);
    
    ctx.outputFile = outputFile;

    GetPrimeNumber(fr, ctx.prime);
    cout << "Prime=0x" << ctx.prime.get_str(16) << endl;

    // opN are local, uncommitted polynomials
    RawFr::Element op0;
    uint64_t op3, op2, op1;

    /* Load ROM JSON file content to memory */
    loadRom(ctx, romJson);

    /* Create polynomials list in memory */
    createPols(ctx, pil);

    /* Create and map pols file to memory */
    mapPols(ctx);

    /* Sets first evaluation of all polynomials to zero */
    initState(fr, ctx);

    preprocessTxs(ctx, input);

    uint64_t zkPC = 0; // execution line, i.e. zkPC

    for (uint64_t i=0; i<NEVALUATIONS; i++)
    {
        zkPC = pols(zkPC)[i]; // This is the read line of code, but using step for debugging purposes, to execute all possible instructions

#ifdef LOG_STEPS
        //cout << "--> Starting step: " << i << " with zkPC: " << zkPC << endl;
#endif
        ctx.ln = zkPC;

        // ctx.step is used inside evaluateCommand() to find the current value of the registers, e.g. (*ctx.pPols)(A0)[ctx.step]
        ctx.step = i;


        /*if (ctx.step==231)
        {
            cout << "pause" << endl;
        }*/

        // Store fileName and line
        ctx.fileName = rom[zkPC].fileName; // TODO: Is this required?  It is only used in printRegs(), and it is an overhead in every loop.
        ctx.line = rom[zkPC].line; // TODO: Is this required? It is only used in printRegs(), and it is an overhead in every loop.

        if (i%1000==0) cout << "Step: " << i << endl;

        // Evaluate the list cmdBefore commands, and any children command, recursively
        for (uint64_t j=0; j<rom[zkPC].cmdBefore.size(); j++)
        {
            CommandResult cr;
            evalCommand(ctx, *rom[zkPC].cmdBefore[j], cr);
        }

        // Initialize the local registers to zero
        op0 = fr.zero();
        op1 = 0;
        op2 = 0;
        op3 = 0;

        // inX adds the corresponding register values to the op local register set
        // In case several inXs are set to 1, those values will be added

        if (rom[zkPC].inA == 1)
        {
            fr.add(op0, op0, pols(A0)[i]);
            op1 = op1 + pols(A1)[i];
            op2 = op2 + pols(A2)[i];
            op3 = op3 + pols(A3)[i];
#ifdef LOG_INX
            cout << "inA op=" << op3 << ":" << op2 << ":" << op1 << ":" << fr.toString(op0) << endl;
#endif
        }
        pols(inA)[i] = rom[zkPC].inA;

        if (rom[zkPC].inB == 1) {
            fr.add(op0, op0, pols(B0)[i]);
            op1 = op1 + pols(B1)[i];
            op2 = op2 + pols(B2)[i];
            op3 = op3 + pols(B3)[i];
#ifdef LOG_INX
            cout << "inB op=" << op3 << ":" << op2 << ":" << op1 << ":" << fr.toString(op0) << endl;
#endif
        }
        pols(inB)[i] = rom[zkPC].inB;

        if (rom[zkPC].inC == 1) {
            fr.add(op0, op0, pols(C0)[i]);
            op1 = op1 + pols(C1)[i];
            op2 = op2 + pols(C2)[i];
            op3 = op3 + pols(C3)[i];
#ifdef LOG_INX
            cout << "inC op=" << op3 << ":" << op2 << ":" << op1 << ":" << fr.toString(op0) << endl;
#endif
        }
        pols(inC)[i] = rom[zkPC].inC;

        if (rom[zkPC].inD == 1) {
            fr.add(op0, op0, pols(D0)[i]);
            op1 = op1 + pols(D1)[i];
            op2 = op2 + pols(D2)[i];
            op3 = op3 + pols(D3)[i];
#ifdef LOG_INX
            cout << "inD op=" << op3 << ":" << op2 << ":" << op1 << ":" << fr.toString(op0) << endl;
#endif
        }
        pols(inD)[i] = rom[zkPC].inD;

        if (rom[zkPC].inE == 1) {
            fr.add(op0, op0, pols(E0)[i]);
            op1 = op1 + pols(E1)[i];
            op2 = op2 + pols(E2)[i];
            op3 = op3 + pols(E3)[i];
#ifdef LOG_INX
            cout << "inE op=" << op3 << ":" << op2 << ":" << op1 << ":" << fr.toString(op0) << endl;
#endif
        }
        pols(inE)[i] = rom[zkPC].inE;

        if (rom[zkPC].inSR == 1) {
            fr.add(op0, op0, pols(SR)[i]);
#ifdef LOG_INX
            cout << "inSR op=" << fr.toString(op0) << endl;
#endif
        }
        pols(inSR)[i] = rom[zkPC].inSR;

        RawFr::Element aux;

        if (rom[zkPC].inCTX == 1) {
            fr.fromUI(aux,pols(CTX)[i]);
            fr.add(op0, op0, aux);
#ifdef LOG_INX
            cout << "inCTX op=" << fr.toString(op0) << endl;
#endif
        }
        pols(inCTX)[i] = rom[zkPC].inCTX;

        if (rom[zkPC].inSP == 1) {
            fr.fromUI(aux,pols(SP)[i]);
            fr.add(op0, op0, aux);
#ifdef LOG_INX
            cout << "inSP op=" << fr.toString(op0) << endl;
#endif
        }
        pols(inSP)[i] = rom[zkPC].inSP;

        if (rom[zkPC].inPC == 1) {
            fr.fromUI(aux,pols(PC)[i]);
            fr.add(op0, op0, aux);
#ifdef LOG_INX
            cout << "inPC op=" << fr.toString(op0) << endl;
#endif
        }
        pols(inPC)[i] = rom[zkPC].inPC;

        if (rom[zkPC].inGAS == 1) {
            fr.fromUI(aux,pols(GAS)[i]);
            fr.add(op0, op0, aux);
#ifdef LOG_INX
            cout << "inGAS op=" << fr.toString(op0) << endl;
#endif
        }
        pols(inGAS)[i] = rom[zkPC].inGAS;
        
        if (rom[zkPC].inMAXMEM == 1) {
            fr.fromUI(aux,pols(MAXMEM)[i]);
            fr.add(op0, op0, aux);
#ifdef LOG_INX
            cout << "inMAXMEM op=" << fr.toString(op0) << endl;
#endif
        }
        pols(inMAXMEM)[i] = rom[zkPC].inMAXMEM;

        if (rom[zkPC].inSTEP == 1) {
            fr.fromUI(aux, i);
            fr.add(op0, op0, aux);
#ifdef LOG_INX
            cout << "inSTEP op=" << fr.toString(op0) << endl;
#endif
        }
        pols(inSTEP)[i] = rom[zkPC].inSTEP;

        if (rom[zkPC].bConstPresent) {
#ifdef LOG_INX
            cout << "inCONST --> op0=" << fr.toString(op0,16) << endl;
#endif
            pols(CONST)[i] = rom[zkPC].CONST; // TODO: Check rom types: U64, U32, etc.  They should match the pols types
            if (pols(CONST)[i]>=0)
            {
                fr.fromUI(aux,pols(CONST)[i]);
            }
            else
            {
                fr.fromUI(aux,-pols(CONST)[i]);
                fr.neg(aux,aux);
            }
            fr.add(op0, op0, aux);
            ctx.byte4[0x80000000 + rom[zkPC].CONST] = true;
#ifdef LOG_INX
            cout << "inCONST <-- op0=" << fr.toString(op0,16) << endl;
#endif
        } else {
            pols(CONST)[i] = 0;
            ctx.byte4[0x80000000] = true;
        }

        uint32_t addrRel = 0; // TODO: Check with Jordi if this is the right type for an address
        uint64_t addr = 0;

        // If address involved, load offset into addr
        if (rom[zkPC].mRD==1 || rom[zkPC].mWR==1 || rom[zkPC].hashRD==1 || rom[zkPC].hashWR==1 || rom[zkPC].hashE==1 || rom[zkPC].JMP==1 || rom[zkPC].JMPC==1) {
            if (rom[zkPC].ind == 1)
            {
                addrRel = fe2n(ctx, pols(E0)[i]);
            }
            if (rom[zkPC].bOffsetPresent && rom[zkPC].offset!=0)
            {
                // If offset is possitive, and the sum is too big, fail
                if (rom[zkPC].offset>0 && (addrRel+rom[zkPC].offset)>=0x100000000)
                {
                    cerr << "Error: addrRel >= 0x100000000 ln: " << ctx.ln << endl;
                    exit(-1); // TODO: Should we kill the process?                    
                }
                // If offset is negative, and its modulo is bigger than addrRel, fail
                if (rom[zkPC].offset<0 && (-rom[zkPC].offset)>addrRel)
                {
                    cerr << "Error: addrRel < 0 ln: " << ctx.ln << endl;
                    exit(-1); // TODO: Should we kill the process?
                }
                addrRel += rom[zkPC].offset;
            }
            addr = addrRel;
#ifdef LOG_ADDR
            cout << "Any addr=" << addr << endl;
#endif
        }

        if (rom[zkPC].useCTX == 1) {
            addr += pols(CTX)[i]*CTX_OFFSET;
#ifdef LOG_ADDR
            cout << "useCTX addr=" << addr << endl;
#endif
        }
        pols(useCTX)[i] = rom[zkPC].useCTX;

        if (rom[zkPC].isCode == 1) {
            addr += CODE_OFFSET;
#ifdef LOG_ADDR
            cout << "isCode addr=" << addr << endl;
#endif
        }
        pols(isCode)[i] = rom[zkPC].isCode;

        if (rom[zkPC].isStack == 1) {
            addr += STACK_OFFSET;
#ifdef LOG_ADDR
            cout << "isStack addr=" << addr << endl;
#endif
        }
        pols(isStack)[i] = rom[zkPC].isStack;

        if (rom[zkPC].isMem == 1) {
            addr += MEM_OFFSET;
#ifdef LOG_ADDR
            cout << "isMem addr=" << addr << endl;
#endif
        }
        pols(isMem)[i] = rom[zkPC].isMem;

        pols(inc)[i] = rom[zkPC].inc;
        pols(dec)[i] = rom[zkPC].dec;
        pols(ind)[i] = rom[zkPC].ind;

        if (rom[zkPC].bOffsetPresent && (rom[zkPC].offset!=0)) {
            pols(offset)[i] = rom[zkPC].offset;
            ctx.byte4[0x80000000 + rom[zkPC].offset] = true;
        } else {
            pols(offset)[i] = 0;
            ctx.byte4[0x80000000] = true;
        }

        if (rom[zkPC].inFREE == 1) {

            if (rom[zkPC].freeInTag.isPresent == false) {
                cerr << "Error: Instruction with freeIn without freeInTag:" << ctx.ln << endl;
                exit(-1);
            }

            RawFr::Element fi0;
            RawFr::Element fi1;
            RawFr::Element fi2;
            RawFr::Element fi3;

            if (rom[zkPC].freeInTag.op == "") {
                uint64_t nHits = 0;
                if (rom[zkPC].mRD == 1) {
                    if (ctx.mem.find(addr) != ctx.mem.end()) {
#ifdef LOG_MEMORY
                        cout << "Memory read mRD: addr:" << addr << " " << printFea(ctx, ctx.mem[addr]) << endl;
#endif
                        fi0 = ctx.mem[addr].fe0;
                        fi1 = ctx.mem[addr].fe1;
                        fi2 = ctx.mem[addr].fe2;
                        fi3 = ctx.mem[addr].fe3;
                    } else {
                        fi0 = fr.zero();
                        fi1 = fr.zero();
                        fi2 = fr.zero();
                        fi3 = fr.zero();
                    }
                    nHits++;
                }
                if (rom[zkPC].sRD == 1) {
                    // Fill a vector of field elements
                    vector<RawFr::Element> keyV;
                    RawFr::Element aux;
                    keyV.push_back(pols(A0)[i]);
                    fr.fromUI(aux, pols(A1)[i]);
                    keyV.push_back(aux);
                    fr.fromUI(aux, pols(A2)[i]);
                    keyV.push_back(aux);
                    keyV.push_back(pols(B0)[i]);
                    keyV.push_back(pols(C0)[i]);
                    fr.fromUI(aux, pols(C1)[i]);
                    keyV.push_back(aux);
                    fr.fromUI(aux, pols(C2)[i]);
                    keyV.push_back(aux);
                    fr.fromUI(aux, pols(C3)[i]);
                    keyV.push_back(aux);

                    // Add tailing fr.zero's to complete 2^ARITY field elements
                    while (keyV.size() < (1<<ARITY)) {
                        keyV.push_back(fr.zero());
                    }
                    
                    //for (int k=0; k<keyV.size(); k++)
                      //  cout << "keyV["<<k<<"]="<< ctx.fr.toString(keyV[k], 16) << endl;
                    // Call poseidon
                    poseidon.hash(keyV, &ctx.lastSWrite.key);
#ifdef LOG_STORAGE
                    cout << "Storage read sRD got poseidon key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << endl;
#endif 
                    
                    // Check that storage entry exists
                    if (ctx.sto.find(ctx.lastSWrite.key) == ctx.sto.end())
                    {
                        cerr << "Error: Storage not initialized, key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << " line: " << ctx.ln << " step: " << ctx.step << endl;
                        exit(-1);
                    }

                    // Read the value from storage, and store it in fin
                    scalar2fea(fr, ctx.sto[ctx.lastSWrite.key], fi0, fi1, fi2, fi3);

                    nHits++;
#ifdef LOG_STORAGE
                    cout << "Storage read sRD read from key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << " value:" << ctx.sto[ctx.lastSWrite.key] << endl;
#endif 
                }
                if (rom[zkPC].sWR == 1) {
                    // reset lastSWrite
                    ctx.lastSWrite.key = fr.zero();
                    ctx.lastSWrite.newRoot = fr.zero();
                    ctx.lastSWrite.step = 0;

                    // Fill a vector of field elements
                    vector<RawFr::Element> keyV;
                    RawFr::Element aux;
                    keyV.push_back(pols(A0)[i]);
                    fr.fromUI(aux, pols(A1)[i]);
                    keyV.push_back(aux);
                    fr.fromUI(aux, pols(A2)[i]);
                    keyV.push_back(aux);
                    keyV.push_back(pols(B0)[i]);
                    keyV.push_back(pols(C0)[i]);
                    fr.fromUI(aux, pols(C1)[i]);
                    keyV.push_back(aux);
                    fr.fromUI(aux, pols(C2)[i]);
                    keyV.push_back(aux);
                    fr.fromUI(aux, pols(C3)[i]);
                    keyV.push_back(aux);

                    // Add tailing fr.zero's to complete 2^ARITY field elements
                    while (keyV.size() < (1<<ARITY)) {
                        keyV.push_back(fr.zero());
                    }
                    
                    // Call poseidon
                    poseidon.hash(keyV, &ctx.lastSWrite.key);
#ifdef LOG_STORAGE
                    cout << "Storage write sWR got poseidon key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << endl;
#endif                    
                    // Check that storage entry exists
                    if (ctx.sto.find(ctx.lastSWrite.key) == ctx.sto.end())
                    {
                        cerr << "Error: Storage write sWR not initialized key: " << fr.toString(ctx.lastSWrite.key, 16) << " line: " << ctx.ln << endl;
                        exit(-1);
                    }

                    SmtSetResult res;
                    mpz_class scalarD;
                    fea2scalar(fr, scalarD, pols(D0)[i], pols(D1)[i], pols(D2)[i], pols(D3)[i]);
                    smt.set(ctx.fr, ctx.db, pols(SR)[i], ctx.lastSWrite.key, scalarD, res);
                    ctx.lastSWrite.newRoot = res.newRoot;
                    ctx.lastSWrite.step = i;

                    fi0 = ctx.lastSWrite.newRoot;
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                    nHits++;
#ifdef LOG_STORAGE
                    cout << "Storage write sWR stored at key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << " newRoot: " << fr.toString(res.newRoot, 16) << endl;
#endif
                }
                if (rom[zkPC].hashRD == 1) {
                    if (ctx.hash.find(addr) == ctx.hash.end()) {
                        cerr << "Error: Hash address not initialized" << endl;
                        exit(-1);
                    }
                    mpz_class auxScalar(ctx.hash[addr].hash);
                    scalar2fea(fr, auxScalar, fi0, fi1, fi2, fi3);
                    nHits++;
#ifdef LOG_HASH
                    cout << "Hash read hashRD: addr:" << addr << " hash:" << auxScalar.get_str(16) << endl;
#endif
                }
                if (rom[zkPC].ecRecover == 1) {
                    mpz_class aux;
                    
                    fea2scalar(fr, aux, pols(A0)[i], pols(A1)[i], pols(A2)[i], pols(A3)[i]);
                    string d = NormalizeTo0xNFormat(aux.get_str(16),64);

                    // Signature = 0x + r(32B) + s(32B) + v(1B) = 0x + 130chars
                    fea2scalar(fr, aux, pols(B0)[i], pols(B1)[i], pols(B2)[i], pols(B3)[i]);
                    string r = NormalizeToNFormat(aux.get_str(16),64);
                    fea2scalar(fr, aux, pols(C0)[i], pols(C1)[i], pols(C2)[i], pols(C3)[i]);
                    string s = NormalizeToNFormat(aux.get_str(16),64);
                    aux = fe2n(ctx, pols(D0)[i]);
                    string v = NormalizeToNFormat(aux.get_str(16),2);
                    string signature = "0x" + r + s + v;

                    /* Return the address associated with the public key signature from elliptic curve signature.
                       Signature parts: r: first 32 bytes of signature; s: second 32 bytes of signature; v: final 1 byte of signature.
                       Hash: d: 32 bytes. */
                    //string ecResult = ecrecover(signature, d);
                    string ecResult = "0x617b3a3528F9cDd6630fd3301B9c8911F7Bf063D"; // TODO: undo this hardcoded value when ecrecover() works as expected
                    mpz_class raddr(ecResult);
                    /*const d = ethers.utils.hexlify(fea2scalar(Fr, ctx.A));
                    const r = ethers.utils.hexlify(fea2scalar(Fr, ctx.B));
                    const s = ethers.utils.hexlify(fea2scalar(Fr, ctx.C));
                    const v = ethers.utils.hexlify(fe2n(Fr, ctx.D[0]));
                    const raddr =  ethers.utils.recoverAddress(d, {
                        r: r,
                        s: s,
                        v: v
                    });*/
                    scalar2fea(fr, raddr, fi0, fi1, fi2, fi3);
                    nHits++;
                }
                if (rom[zkPC].shl == 1) {
                    mpz_class a;
                    fea2scalar(fr, a, pols(A0)[i], pols(A1)[i], pols(A2)[i], pols(A3)[i]);
                    uint64_t s = fe2n(ctx, pols(D0)[i]);
                    if ((s>32) || (s<0)) {
                        cerr << "Error: SHL too big: " << ctx.ln << endl;
                        exit(-1);
                    }
                    mpz_class band("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
                    mpz_class b;
                    b = (a << s*8) & band;
                    scalar2fea(fr, b, fi0, fi1, fi2, fi3);
                    nHits++;
                } 
                if (rom[zkPC].shr == 1) {
                    mpz_class a;
                    fea2scalar(fr, a, pols(A0)[i], pols(A1)[i], pols(A2)[i], pols(A3)[i]);
                    uint64_t s = fe2n(ctx, pols(D0)[i]);
                    if ((s>32) || (s<0)) {
                        cerr << "Error: SHR too big: " << ctx.ln << endl;
                        exit(-1);
                    }
                    mpz_class b = a >> s*8;
                    scalar2fea(fr, b, fi0, fi1, fi2, fi3);
                    nHits++;
                } 
                if (nHits == 0) {
                    cerr << "Error: Empty freeIn without a valid instruction: " << ctx.ln << endl;
                    exit(-1);
                }
                if (nHits > 1) {
                    cerr << "Error: Only one instructuin that requires freeIn is alllowed: " << ctx.ln << endl;
                }
            } else {
                CommandResult cr;
                evalCommand(ctx, rom[zkPC].freeInTag, cr);
                if (cr.type == crt_fea) {
                    fi0 = cr.fea0;
                    fi1 = cr.fea1;
                    fi2 = cr.fea2;
                    fi3 = cr.fea3;
                } else if (cr.type == crt_fe) {
                    fi0 = cr.fe;
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                } else if (cr.type == crt_scalar) {
                    scalar2fea(fr, cr.scalar, fi0, fi1, fi2, fi3);
                } else if (cr.type == crt_u16) {
                    fr.fromUI(fi0, cr.u16);
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                } else if (cr.type == crt_u32) {
                    fr.fromUI(fi0, cr.u32);
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                } else if (cr.type == crt_u64) {
                    fr.fromUI(fi0, cr.u64);
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                } else {
                    cerr << "Error: unexpected command result type: " << cr.type << endl;
                    exit(-1);
                }
            }
            pols(FREE0)[i] = fi0;
            pols(FREE1)[i] = fi1;
            pols(FREE2)[i] = fi2;
            pols(FREE3)[i] = fi3;

            fr.add(op0, op0, fi0);
            op1 += fe2u64(fr, fi1);
            op2 += fe2u64(fr, fi2);
            op3 += fe2u64(fr, fi3);
        } else {
            pols(FREE0)[i] = fr.zero();
            pols(FREE1)[i] = fr.zero();
            pols(FREE2)[i] = fr.zero();
            pols(FREE3)[i] = fr.zero();
        }
        pols(inFREE)[i] = rom[zkPC].inFREE;
        if (rom[zkPC].neg == 1) {
            fr.neg(op0,op0);
#ifdef LOG_NEG
            cout << "neg op0=" << fr.toString(op0, 16) << endl;
#endif
        }
        pols(neg)[i] = rom[zkPC].neg;

        if (rom[zkPC].assert == 1) {
            if ( (!fr.eq(pols(A0)[i],op0)) ||
                 (pols(A1)[i] != op1) ||
                 (pols(A2)[i] != op2) ||
                 (pols(A3)[i] != op3) )
            {
                cerr << "Error: ROM assert failed: AN!=opN ln: " << ctx.ln << endl;
                cout << "A: " << pols(A3)[i] << ":" << pols(A2)[i] << ":" << pols(A1)[i] << ":" << fr.toString(pols(A0)[i],16) << endl;
                cout << "OP:" << op3 << ":" << op2 << ":" << op1 << ":" << fr.toString(op0,16) << endl;
                exit(-1);
            }
#ifdef LOG_ASSERT
            cout << "assert" << endl;
#endif
        }
        pols(assert)[i] = rom[zkPC].assert;

        // The registers of the evaluation 0 will be overwritten with the values from the last evaluation,
        // closing the evaluation circle
        uint64_t nexti = (i+1)%NEVALUATIONS;

        if (rom[zkPC].setA == 1) {
            pols(A0)[nexti] = op0;
            pols(A1)[nexti] = op1;
            pols(A2)[nexti] = op2;
            pols(A3)[nexti] = op3;
#ifdef LOG_SETX
            cout << "setA A[nexti]=" << pols(A3)[nexti] << ":" << pols(A2)[nexti] << ":" << pols(A1)[nexti] << ":" << fr.toString(pols(A0)[nexti],16) << endl;
#endif
        } else {
            pols(A0)[nexti] = pols(A0)[i];
            pols(A1)[nexti] = pols(A1)[i];
            pols(A2)[nexti] = pols(A2)[i];
            pols(A3)[nexti] = pols(A3)[i];
        }
        pols(setA)[i] = rom[zkPC].setA;

        if (rom[zkPC].setB == 1) {
            pols(B0)[nexti] = op0;
            pols(B1)[nexti] = op1;
            pols(B2)[nexti] = op2;
            pols(B3)[nexti] = op3;
#ifdef LOG_SETX
            cout << "setB B[nexti]=" << pols(B3)[nexti] << ":" << pols(B2)[nexti] << ":" << pols(B1)[nexti] << ":" << fr.toString(pols(B0)[nexti], 16) << endl;
#endif
        } else {
            pols(B0)[nexti] = pols(B0)[i];
            pols(B1)[nexti] = pols(B1)[i];
            pols(B2)[nexti] = pols(B2)[i];
            pols(B3)[nexti] = pols(B3)[i];
        }
        pols(setB)[i] = rom[zkPC].setB;

        if (rom[zkPC].setC == 1) {
            pols(C0)[nexti] = op0;
            pols(C1)[nexti] = op1;
            pols(C2)[nexti] = op2;
            pols(C3)[nexti] = op3;
#ifdef LOG_SETX
            cout << "setC C[nexti]=" << pols(C3)[nexti] << ":" << pols(C2)[nexti] << ":" << pols(C1)[nexti] << ":" << fr.toString(pols(C0)[nexti], 16) << endl;
#endif
        } else {
            pols(C0)[nexti] = pols(C0)[i];
            pols(C1)[nexti] = pols(C1)[i];
            pols(C2)[nexti] = pols(C2)[i];
            pols(C3)[nexti] = pols(C3)[i];
        }
        pols(setC)[i] = rom[zkPC].setC;

        if (rom[zkPC].setD == 1) {
            pols(D0)[nexti] = op0;
            pols(D1)[nexti] = op1;
            pols(D2)[nexti] = op2;
            pols(D3)[nexti] = op3;
#ifdef LOG_SETX
            cout << "setD D[nexti]=" << pols(D3)[nexti] << ":" << pols(D2)[nexti] << ":" << pols(D1)[nexti] << ":" << fr.toString(pols(D0)[nexti], 16) << endl;
#endif
        } else {
            pols(D0)[nexti] = pols(D0)[i];
            pols(D1)[nexti] = pols(D1)[i];
            pols(D2)[nexti] = pols(D2)[i];
            pols(D3)[nexti] = pols(D3)[i];
        }
        pols(setD)[i] = rom[zkPC].setD;
        if (rom[zkPC].setE == 1) {
            pols(E0)[nexti] = op0;
            pols(E1)[nexti] = op1;
            pols(E2)[nexti] = op2;
            pols(E3)[nexti] = op3;
#ifdef LOG_SETX
            cout << "setE E[nexti]=" << pols(E3)[nexti] << ":" << pols(E2)[nexti] << ":" << pols(E1)[nexti] << ":" << fr.toString(pols(E0)[nexti] ,16) << endl;
#endif
        } else {
            pols(E0)[nexti] = pols(E0)[i];
            pols(E1)[nexti] = pols(E1)[i];
            pols(E2)[nexti] = pols(E2)[i];
            pols(E3)[nexti] = pols(E3)[i];
        }
        pols(setE)[i] = rom[zkPC].setE;

        if (rom[zkPC].setSR == 1) {
            pols(SR)[nexti] = op0;
#ifdef LOG_SETX
            cout << "setSR SR[nexti]=" << fr.toString(pols(SR)[nexti],16) << endl;
#endif
        } else {
            pols(SR)[nexti] = pols(SR)[i];
        }
        pols(setSR)[i] = rom[zkPC].setSR;

        if (rom[zkPC].setCTX == 1) {
            pols(CTX)[nexti] = fe2n(ctx, op0);
#ifdef LOG_SETX
            cout << "setCTX CTX[nexti]=" << pols(CTX)[nexti] << endl;
#endif
        } else {
            pols(CTX)[nexti] = pols(CTX)[i];
        }
        pols(setCTX)[i] = rom[zkPC].setCTX;

        if (rom[zkPC].setSP == 1) {
            pols(SP)[nexti] = fe2n(ctx, op0);
#ifdef LOG_SETX
            cout << "setSP SP[nexti]=" << pols(SP)[nexti] << endl;
#endif
        } else {
            pols(SP)[nexti] = pols(SP)[i];
            if ((rom[zkPC].inc==1) && (rom[zkPC].isStack==1)){
                pols(SP)[nexti] = pols(SP)[nexti] + 1;
            }
            if ((rom[zkPC].dec==1) && (rom[zkPC].isStack==1)){
                pols(SP)[nexti] = pols(SP)[nexti] - 1;
            }
        }
        pols(setSP)[i] = rom[zkPC].setSP;
        if (rom[zkPC].setPC == 1) {
            pols(PC)[nexti] = fe2n(ctx, op0);
#ifdef LOG_SETX
            cout << "setPC PC[nexti]=" << pols(PC)[nexti] << endl;
#endif
        } else {
            pols(PC)[nexti] = pols(PC)[i];
            if ( (rom[zkPC].inc==1) && (rom[zkPC].isCode==1) ) {
                pols(PC)[nexti] = pols(PC)[nexti] + 1; // PC is part of Ethereum's program
            }
            if ( (rom[zkPC].dec==1) && (rom[zkPC].isCode==1) ) {
                pols(PC)[nexti] = pols(PC)[nexti] - 1; // PC is part of Ethereum's program
            }
        }
        pols(setPC)[i] = rom[zkPC].setPC;

        if (rom[zkPC].JMPC == 1) {
#ifdef LOG_JMP
            cout << "JMPC: op0=" << fr.toString(op0) << endl;
#endif
            int64_t o = fe2n(ctx, op0);
#ifdef LOG_JMP
            cout << "JMPC: o=" << o << endl;
#endif
            if (o<0) {
                pols(isNeg)[i] = 1;
                pols(zkPC)[nexti] = addr;
                ctx.byte4[0x100000000 + o] = true;
#ifdef LOG_JMP
               cout << "Next zkPC(1)=" << pols(zkPC)[nexti] << endl;
#endif
            } else {
                pols(isNeg)[i] = 0;
                pols(zkPC)[nexti] = pols(zkPC)[i] + 1;
#ifdef LOG_JMP
                cout << "Next zkPC(2)=" << pols(zkPC)[nexti] << endl;
#endif
                ctx.byte4[o] = true;
            }
            pols(JMP)[i] = 0;
            pols(JMPC)[i] = 1;
        } else if (rom[zkPC].JMP == 1) {
            pols(isNeg)[i] = 0;
            pols(zkPC)[nexti] = addr;
#ifdef LOG_JMP
            cout << "Next zkPC(3)=" << pols(zkPC)[nexti] << endl;
#endif
            pols(JMP)[i] = 1;
            pols(JMPC)[i] = 0;
        } else {
            pols(isNeg)[i] = 0;
            pols(zkPC)[nexti] = pols(zkPC)[i] + 1;
            pols(JMP)[i] = 0;
            pols(JMPC)[i] = 0;
        }

        uint32_t maxMemCalculated = 0;
        uint32_t mm = pols(MAXMEM)[i];
        if (rom[zkPC].isMem==1 && addrRel>mm) {
            pols(isMaxMem)[i] = 1;
            maxMemCalculated = addrRel;
            ctx.byte4[maxMemCalculated - mm] = true;
        } else {
            pols(isMaxMem)[i] = 0;
            maxMemCalculated = mm;
            ctx.byte4[0] = true;
        }

        if (rom[zkPC].setMAXMEM == 1) {
            pols(MAXMEM)[nexti] = fe2n(ctx, op0);
#ifdef LOG_SETX
            cout << "setMAXMEM MAXMEM[nexti]=" << pols(MAXMEM)[nexti] << endl;
#endif
        } else {
            pols(MAXMEM)[nexti] = maxMemCalculated;
        }
        pols(setMAXMEM)[i] = rom[zkPC].setMAXMEM;

        if (rom[zkPC].setGAS == 1) {
            pols(GAS)[nexti] = fe2n(ctx, op0);
#ifdef LOG_SETX
            cout << "setGAS GAS[nexti]=" << pols(GAS)[nexti] << endl;
#endif
        } else {
            pols(GAS)[nexti] = pols(GAS)[i];
        }
        pols(setGAS)[i] = rom[zkPC].setGAS;

        pols(mRD)[i] = rom[zkPC].mRD;
        if (rom[zkPC].mWR == 1) {
            ctx.mem[addr].fe0 = op0;
            fr.fromUI(ctx.mem[addr].fe1, op1);
            fr.fromUI(ctx.mem[addr].fe2, op2);
            fr.fromUI(ctx.mem[addr].fe3, op3);
#ifdef LOG_MEMORY
            cout << "Memory write mWR: addr:" << addr << " " << printFea(ctx, ctx.mem[addr]) << endl;
#endif
        }
        pols(mWR)[i] = rom[zkPC].mWR;

        pols(sRD)[i] = rom[zkPC].sRD;

        if (rom[zkPC].sWR == 1) {
            if (ctx.lastSWrite.step != i) {
                // Fill a vector of field elements
                vector<RawFr::Element> keyV;
                RawFr::Element aux;
                keyV.push_back(pols(A0)[i]);
                fr.fromUI(aux, pols(A1)[i]);
                keyV.push_back(aux);
                fr.fromUI(aux, pols(A2)[i]);
                keyV.push_back(aux);
                keyV.push_back(pols(B0)[i]);
                keyV.push_back(pols(C0)[i]);
                fr.fromUI(aux, pols(C1)[i]);
                keyV.push_back(aux);
                fr.fromUI(aux, pols(C2)[i]);
                keyV.push_back(aux);
                fr.fromUI(aux, pols(C3)[i]);
                keyV.push_back(aux);

                // Add tailing fr.zero's to complete 2^ARITY field elements
                while (keyV.size() < (1<<ARITY)) {
                    keyV.push_back(fr.zero());
                }
                
                // Call poseidon
                poseidon.hash(keyV, &ctx.lastSWrite.key);
                
                // Check that storage entry exists
                if (ctx.sto.find(ctx.lastSWrite.key) == ctx.sto.end())
                {
                    cerr << "Error: Storage not initialized key: " << fr.toString(ctx.lastSWrite.key, 16) << " line: " << ctx.ln << endl;
                    exit(-1);
                }

                SmtSetResult res;
                mpz_class scalarD;
                fea2scalar(fr, scalarD, pols(D0)[i], pols(D1)[i], pols(D2)[i], pols(D3)[i]);
                smt.set(ctx.fr, ctx.db, pols(SR)[i], ctx.lastSWrite.key, scalarD, res);
                ctx.lastSWrite.newRoot = res.newRoot;
                ctx.lastSWrite.step = i;
            }

            if (!fr.eq(ctx.lastSWrite.newRoot, op0)) {
                cerr << "Error: Storage write does not match: " << ctx.ln << endl;
                exit(-1);
            }
            mpz_class auxScalar;
            fea2scalar(fr, auxScalar, pols(D0)[i], pols(D1)[i], pols(D2)[i], pols(D3)[i]);
            ctx.sto[ctx.lastSWrite.key] = auxScalar;
        }
        pols(sWR)[i] = rom[zkPC].sWR;
        pols(hashRD)[i] = rom[zkPC].hashRD;

        if (rom[zkPC].hashWR == 1) {

            // Get the size of the hash from D0
            uint64_t size = fe2n(ctx, pols(D0)[i]);
            if ((size<0) || (size>32)) {
                cerr << "Error: Invalid size for hash.  Size:" << size << " Line:" << ctx.ln << endl;
                exit(-1);
            }

            // Get contents of opN into a
            mpz_class a;
            fea2scalar(fr, a, op0, op1, op2, op3);

            // If there is no entry in the hash database for this address, then create a new one
            if (ctx.hash.find(addr) == ctx.hash.end())
            {
                HashValue hashValue;
                ctx.hash[addr] = hashValue; // TODO: Could this be just a copy assignment, instead of an allocation?
            }

            for (uint64_t j=0; j<size; j++) {
                mpz_class band(0xFF);
                mpz_class result = (a >> (size-j-1)*8) & band;
                uint64_t uiResult = result.get_ui();
                ctx.hash[addr].data.push_back((uint8_t)uiResult);
            }
            //ctx.hash[addr].dataSize = size;
#ifdef LOG_HASH
            cout << "Hash write  hashWR: addr:" << addr << endl;
#endif
        }
        pols(hashWR)[i] = rom[zkPC].hashWR;
        if (rom[zkPC].hashE == 1) {            
            // Datos: '0xee80843b9aca00830186a0944d5cf5032b2a844602278b01199ed191a86c93ff88016345785d8a0000808201908080'
            // Hash: 0xf18aa4ed1378d34eac01f7bb19d391c581cae25ca96e5e229e4ceec4ddffd137
            //ctx.hash[addr].result = ethers.utils.keccak256(ethers.utils.hexlify(ctx.hash[addr].data));
            uint64_t dataSize = ctx.hash[addr].data.size();
            ctx.hash[addr].hash = keccak256(ctx.hash[addr].data.data(), dataSize);
            if (addr==1) ctx.hash[addr].hash = "0xf18aa4ed1378d34eac01f7bb19d391c581cae25ca96e5e229e4ceec4ddffd137";
            if (addr==0) ctx.hash[addr].hash = "0x90e25a5adfc9791fb824cc7d34703d61dee3b0d178b5fc9c88b8da2f974786b7";
            // TODO: undo this hardcoded value when the right keccak256 is available
#ifdef LOG_HASH
            cout << "Hash write  hashWR+hashE: addr:" << addr << " hash:" << ctx.hash[addr].hash << " size:" << ctx.hash[addr].data.size() << " data:";
            for (int k=0; k<ctx.hash[addr].data.size(); k++) cout << byte2string(ctx.hash[addr].data[k]) << ":";
            cout << endl;
#endif            
        }
        pols(hashE)[i] = rom[zkPC].hashE;

        pols(ecRecover)[i] = rom[zkPC].ecRecover;

        if (rom[zkPC].arith == 1) {
            mpz_class A, B, C, D, op;
            fea2scalar(fr, A, pols(A0)[i], pols(A1)[i], pols(A2)[i], pols(A3)[i]);
            fea2scalar(fr, B, pols(B0)[i], pols(B1)[i], pols(B2)[i], pols(B3)[i]);
            fea2scalar(fr, C, pols(C0)[i], pols(C1)[i], pols(C2)[i], pols(C3)[i]);
            fea2scalar(fr, D, pols(D0)[i], pols(D1)[i], pols(D2)[i], pols(D3)[i]);
            fea2scalar(fr, op, op0, op1, op2, op3);

            if ( (A*B) + C != (D<<256) + op ) {
                cerr << "Error: Arithmetic does not match: " << ctx.ln << endl;
                exit(-1);
            }
        }
        pols(arith)[i] = rom[zkPC].arith;

        pols(shl)[i] = rom[zkPC].shl;
        pols(shr)[i] = rom[zkPC].shr;
        pols(bin)[i] = rom[zkPC].bin;
        pols(comparator)[i] = rom[zkPC].comparator;
        pols(opcodeRomMap)[i] = rom[zkPC].opcodeRomMap;

        // Evaluate the list cmdAfter commands, and any children command, recursively
        for (uint64_t j=0; j<rom[zkPC].cmdAfter.size(); j++)
        {
            CommandResult cr;
            evalCommand(ctx, *rom[zkPC].cmdAfter[j], cr);
        }

#ifdef LOG_STEPS
        cout << "<-- Completed step: " << ctx.step << " zkPC: " << zkPC << " op0: " << fr.toString(op0,16) << " FREE0: " << fr.toString(pols(FREE0)[i],16) << endl;
#endif
        //if(i>240) break;
    }

    //printRegs(ctx);
    //printVars(ctx);
    //printMem(ctx);
    //printStorage(ctx);
    //printDb(ctx);

    checkFinalState(fr, ctx);

    uint64_t p = 0;
    uint64_t last = 0;
    for (map<uint32_t,bool>::iterator it=ctx.byte4.begin(); it!=ctx.byte4.end(); it++)
    {
        uint32_t num = it->first;
        pols(byte4_freeIN)[p] = num >> 16;
        pols(byte4_out)[p] = last;
        p++;
        pols(byte4_freeIN)[p] = num & 0xFFFF;
        pols(byte4_out)[p] = num >> 16;
        p++;
        last = num;
    }
    pols(byte4_freeIN)[p] = 0;
    pols(byte4_out)[p] = last;
    p++;
    pols(byte4_freeIN)[p] = 0;
    pols(byte4_out)[p] = 0;
    p++;

    if (p >= NEVALUATIONS)
    {
        cerr << "Error: Too many byte4 entries" << endl;
        exit(-1);
    }

    while (p < NEVALUATIONS)
    {
        pols(byte4_freeIN)[p] = 0;
        pols(byte4_out)[p] = 0;
        p++;        
    }

    /* Unmap output file from memory */
    unmapPols(ctx);

    /* Unload ROM JSON file data from memory, i.e. free memory */
    unloadRom(ctx);
    
}

/* Sets first evaluation of all polynomials to zero */
void initState(RawFr &fr, Context &ctx)
{
    // Register value initial parameters
    pols(A0)[0] = fr.zero();
    pols(A1)[0] = 0;
    pols(A2)[0] = 0;
    pols(A3)[0] = 0;
    pols(B0)[0] = fr.zero();
    pols(B1)[0] = 0;
    pols(B2)[0] = 0;
    pols(B3)[0] = 0;
    pols(C0)[0] = fr.zero();
    pols(C1)[0] = 0;
    pols(C2)[0] = 0;
    pols(C3)[0] = 0;
    pols(D0)[0] = fr.zero();
    pols(D1)[0] = 0;
    pols(D2)[0] = 0;
    pols(D3)[0] = 0;
    pols(E0)[0] = fr.zero();
    pols(E1)[0] = 0;
    pols(E2)[0] = 0;
    pols(E3)[0] = 0;
    pols(SR)[0] = fr.zero();
    pols(CTX)[0] = 0;
    pols(SP)[0] = 0;
    pols(PC)[0] = 0;
    pols(MAXMEM)[0] = 0;
    pols(GAS)[0] = 0;
    pols(zkPC)[0] = 0;
}

void checkFinalState(RawFr &fr, Context &ctx)
{
    if ( 
        (!fr.isZero(pols(A0)[0])) ||
        (pols(A1)[0]!=0) ||
        (pols(A2)[0]!=0) ||
        (pols(A3)[0]!=0) ||
        (!fr.isZero(pols(B0)[0])) ||
        (pols(B1)[0]!=0) ||
        (pols(B2)[0]!=0) ||
        (pols(B3)[0]!=0) ||
        (!fr.isZero(pols(C0)[0])) ||
        (pols(C1)[0]!=0) ||
        (pols(C2)[0]!=0) ||
        (pols(C3)[0]!=0) ||
        (!fr.isZero(pols(D0)[0])) ||
        (pols(D1)[0]!=0) ||
        (pols(D2)[0]!=0) ||
        (pols(D3)[0]!=0) ||
        (!fr.isZero(pols(E0)[0])) ||
        (pols(E1)[0]!=0) ||
        (pols(E2)[0]!=0) ||
        (pols(E3)[0]!=0) ||
        (!fr.isZero(pols(SR)[0])) ||
        (pols(CTX)[0]!=0) ||
        (pols(SP)[0]!=0) ||
        (pols(PC)[0]!=0) ||
        (pols(MAXMEM)[0]!=0) ||
        (pols(GAS)[0]!=0) ||
        (pols(zkPC)[0]!=0)
    ) {
        cerr << "Error: Program terminated with registers not set to zero" << endl;
        exit(-1);
    }
    else{
        cout << "checkFinalState() succeeded" << endl;
    }
}
