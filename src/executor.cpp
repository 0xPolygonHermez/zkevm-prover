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

#define MEM_OFFSET 0x300000000
#define STACK_OFFSET 0x200000000
#define CODE_OFFSET 0x100000000
#define CTX_OFFSET 0x400000000

void initState(RawFr &fr, Context &ctx);
void checkFinalState(RawFr &fr, Context &ctx);

void execute (RawFr &fr, json &input, json &romJson, json &pil, string &outputFile)
{
    // Create context and store a finite field reference in it
    Context ctx(fr);
    GetPrimeNumber(fr, ctx.prime);
    cout << "Prime=0x" << ctx.prime.get_str(16) << endl;

    // Poseidon hash instance
    Poseidon_opt poseidon;

    // SMT instance
    Smt smt(ARITY);
    
    // Store the name of the file to store all polynomial evaluations as memory-mapped HDD space in mapPols()
    ctx.outputFile = outputFile;

    /* Load ROM JSON file content into memory */
    loadRom(ctx, romJson);

    /* Load PIL JSON file content into memory */
    loadPols(ctx, pil);

    /* Sets first evaluation of all polynomials to zero */
    initState(fr, ctx);

    /* Load input JSON file content into memory */
    loadInput(ctx, input);

    // opN are local, uncommitted polynomials
    RawFr::Element op0;
    uint64_t op3, op2, op1;

    // Zero-knowledge program counter
    uint64_t zkPC = 0;

    for (uint64_t i=0; i<NEVALUATIONS; i++)
    {
        zkPC = pol(zkPC)[i]; // This is the read line of ZK code
        ctx.zkPC = zkPC;

        // ctx.step is used inside evaluateCommand() to find the current value of the registers, e.g. pols(A0)[ctx.step]
        ctx.step = i;

#ifdef LOG_STEPS
        cout << "--> Starting step: " << i << " with zkPC: " << zkPC << endl;
#endif

#ifdef LOG_FILENAME
        // Store fileName and line
        ctx.fileName = rom[zkPC].fileName;
        ctx.line = rom[zkPC].line;
#endif

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
        // In case several inXs are set to 1, those values will be added together to opN
        // e.g. op0 = inX*X0 + inY*Y0 + inZ*Z0 +...

        // If inA, op=op+A
        if (rom[zkPC].inA == 1)
        {
            fr.add(op0, op0, pol(A0)[i]);
            op1 = op1 + pol(A1)[i];
            op2 = op2 + pol(A2)[i];
            op3 = op3 + pol(A3)[i];
#ifdef LOG_INX
            cout << "inA op=" << op3 << ":" << op2 << ":" << op1 << ":" << fr.toString(op0) << endl;
#endif
        }
        pol(inA)[i] = rom[zkPC].inA;

        // If inB, op=op+B
        if (rom[zkPC].inB == 1) {
            fr.add(op0, op0, pol(B0)[i]);
            op1 = op1 + pol(B1)[i];
            op2 = op2 + pol(B2)[i];
            op3 = op3 + pol(B3)[i];
#ifdef LOG_INX
            cout << "inB op=" << op3 << ":" << op2 << ":" << op1 << ":" << fr.toString(op0) << endl;
#endif
        }
        pol(inB)[i] = rom[zkPC].inB;

        // If inC, op=op+C
        if (rom[zkPC].inC == 1) {
            fr.add(op0, op0, pol(C0)[i]);
            op1 = op1 + pol(C1)[i];
            op2 = op2 + pol(C2)[i];
            op3 = op3 + pol(C3)[i];
#ifdef LOG_INX
            cout << "inC op=" << op3 << ":" << op2 << ":" << op1 << ":" << fr.toString(op0) << endl;
#endif
        }
        pol(inC)[i] = rom[zkPC].inC;

        // If inD, op=op+D
        if (rom[zkPC].inD == 1) {
            fr.add(op0, op0, pol(D0)[i]);
            op1 = op1 + pol(D1)[i];
            op2 = op2 + pol(D2)[i];
            op3 = op3 + pol(D3)[i];
#ifdef LOG_INX
            cout << "inD op=" << op3 << ":" << op2 << ":" << op1 << ":" << fr.toString(op0) << endl;
#endif
        }
        pol(inD)[i] = rom[zkPC].inD;

        // If inE, op=op+E
        if (rom[zkPC].inE == 1) {
            fr.add(op0, op0, pol(E0)[i]);
            op1 = op1 + pol(E1)[i];
            op2 = op2 + pol(E2)[i];
            op3 = op3 + pol(E3)[i];
#ifdef LOG_INX
            cout << "inE op=" << op3 << ":" << op2 << ":" << op1 << ":" << fr.toString(op0) << endl;
#endif
        }
        pol(inE)[i] = rom[zkPC].inE;

        // If inSR, op=op+SR
        if (rom[zkPC].inSR == 1) {
            fr.add(op0, op0, pol(SR)[i]);
#ifdef LOG_INX
            cout << "inSR op=" << fr.toString(op0) << endl;
#endif
        }
        pol(inSR)[i] = rom[zkPC].inSR;

        RawFr::Element aux;

        // If inCTX, op=op+CTX
        if (rom[zkPC].inCTX == 1) {
            fr.fromUI(aux,pol(CTX)[i]);
            fr.add(op0, op0, aux);
#ifdef LOG_INX
            cout << "inCTX op=" << fr.toString(op0) << endl;
#endif
        }
        pol(inCTX)[i] = rom[zkPC].inCTX;

        // If inSP, op=op+SP
        if (rom[zkPC].inSP == 1) {
            fr.fromUI(aux,pol(SP)[i]);
            fr.add(op0, op0, aux);
#ifdef LOG_INX
            cout << "inSP op=" << fr.toString(op0) << endl;
#endif
        }
        pol(inSP)[i] = rom[zkPC].inSP;

        // If inPC, op=op+PC
        if (rom[zkPC].inPC == 1) {
            fr.fromUI(aux,pol(PC)[i]);
            fr.add(op0, op0, aux);
#ifdef LOG_INX
            cout << "inPC op=" << fr.toString(op0) << endl;
#endif
        }
        pol(inPC)[i] = rom[zkPC].inPC;

        // If inGAS, op=op+GAS
        if (rom[zkPC].inGAS == 1) {
            fr.fromUI(aux,pol(GAS)[i]);
            fr.add(op0, op0, aux);
#ifdef LOG_INX
            cout << "inGAS op=" << fr.toString(op0) << endl;
#endif
        }
        pol(inGAS)[i] = rom[zkPC].inGAS;
        
        // If inMAXMEM, op=op+MAXMEM
        if (rom[zkPC].inMAXMEM == 1) {
            fr.fromUI(aux,pol(MAXMEM)[i]);
            fr.add(op0, op0, aux);
#ifdef LOG_INX
            cout << "inMAXMEM op=" << fr.toString(op0) << endl;
#endif
        }
        pol(inMAXMEM)[i] = rom[zkPC].inMAXMEM;

        // If inSTEP, op=op+STEP
        if (rom[zkPC].inSTEP == 1) {
            fr.fromUI(aux, i);
            fr.add(op0, op0, aux);
#ifdef LOG_INX
            cout << "inSTEP op=" << fr.toString(op0) << endl;
#endif
        }
        pol(inSTEP)[i] = rom[zkPC].inSTEP;

        // If inCONST, op=op+CONST
        if (rom[zkPC].bConstPresent) {
#ifdef LOG_INX
            cout << "inCONST --> op0=" << fr.toString(op0,16) << endl;
#endif
            pol(CONST)[i] = rom[zkPC].CONST;
            if (pol(CONST)[i]>=0)
            {
                fr.fromUI(aux,pol(CONST)[i]);
            }
            else
            {
                fr.fromUI(aux,-pol(CONST)[i]);
                fr.neg(aux,aux);
            }
            fr.add(op0, op0, aux);
            ctx.byte4[0x80000000 + rom[zkPC].CONST] = true;
#ifdef LOG_INX
            cout << "inCONST <-- op0=" << fr.toString(op0,16) << endl;
#endif
        } else {
            pol(CONST)[i] = 0;
            ctx.byte4[0x80000000] = true;
        }

        uint32_t addrRel = 0;
        uint64_t addr = 0;

        // If address is involved, load offset into addr
        if (rom[zkPC].mRD==1 || rom[zkPC].mWR==1 || rom[zkPC].hashRD==1 || rom[zkPC].hashWR==1 || rom[zkPC].hashE==1 || rom[zkPC].JMP==1 || rom[zkPC].JMPC==1) {
            if (rom[zkPC].ind == 1)
            {
                addrRel = fe2n(ctx, pol(E0)[i]);
            }
            if (rom[zkPC].bOffsetPresent && rom[zkPC].offset!=0)
            {
                // If offset is possitive, and the sum is too big, fail
                if (rom[zkPC].offset>0 && (addrRel+rom[zkPC].offset)>=0x100000000)
                {
                    cerr << "Error: addrRel >= 0x100000000 ln: " << ctx.zkPC << endl;
                    exit(-1);                  
                }
                // If offset is negative, and its modulo is bigger than addrRel, fail
                if (rom[zkPC].offset<0 && (-rom[zkPC].offset)>addrRel)
                {
                    cerr << "Error: addrRel < 0 ln: " << ctx.zkPC << endl;
                    exit(-1);
                }
                addrRel += rom[zkPC].offset;
            }
            addr = addrRel;
#ifdef LOG_ADDR
            cout << "Any addr=" << addr << endl;
#endif
        }

        // If useCTX, addr=addr+CTX*CTX_OFFSET
        if (rom[zkPC].useCTX == 1) {
            addr += pol(CTX)[i]*CTX_OFFSET;
#ifdef LOG_ADDR
            cout << "useCTX addr=" << addr << endl;
#endif
        }
        pol(useCTX)[i] = rom[zkPC].useCTX;

        // If isCode, addr=addr+CODE_OFFSET
        if (rom[zkPC].isCode == 1) {
            addr += CODE_OFFSET;
#ifdef LOG_ADDR
            cout << "isCode addr=" << addr << endl;
#endif
        }
        pol(isCode)[i] = rom[zkPC].isCode;

        // If isStack, addr=addr+STACK_OFFSET
        if (rom[zkPC].isStack == 1) {
            addr += STACK_OFFSET;
#ifdef LOG_ADDR
            cout << "isStack addr=" << addr << endl;
#endif
        }
        pol(isStack)[i] = rom[zkPC].isStack;

        // If isMem, addr=addr+MEM_OFFSET
        if (rom[zkPC].isMem == 1) {
            addr += MEM_OFFSET;
#ifdef LOG_ADDR
            cout << "isMem addr=" << addr << endl;
#endif
        }
        pol(isMem)[i] = rom[zkPC].isMem;

        // Copy ROM flags into the polynomials
        pol(inc)[i] = rom[zkPC].inc;
        pol(dec)[i] = rom[zkPC].dec;
        pol(ind)[i] = rom[zkPC].ind;

        // If offset, record it in byte4
        if (rom[zkPC].bOffsetPresent && (rom[zkPC].offset!=0)) {
            pol(offset)[i] = rom[zkPC].offset;
            ctx.byte4[0x80000000 + rom[zkPC].offset] = true;
        } else {
            pol(offset)[i] = 0;
            ctx.byte4[0x80000000] = true;
        }

        // If inFREE, calculate the free value, and add it to op
        if (rom[zkPC].inFREE == 1)
        {
            // freeInTag must be present
            if (rom[zkPC].freeInTag.isPresent == false) {
                cerr << "Error: Instruction with freeIn without freeInTag:" << ctx.zkPC << endl;
                exit(-1);
            }

            // Store free value here, and add it to op later
            RawFr::Element fi0;
            RawFr::Element fi1;
            RawFr::Element fi2;
            RawFr::Element fi3;

            // If there is no operation specified in freeInTag.op, then get the free value directly from the corresponding source
            if (rom[zkPC].freeInTag.op == "") {
                uint64_t nHits = 0;

                // If mRD (memory read) get fi=mem[addr], if it exsists
                if (rom[zkPC].mRD == 1)
                {
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

                // If sRD (storage read) get a poseidon hash, and read fi=sto[hash]
                if (rom[zkPC].sRD == 1)
                {
                    // Fill a vector of field elements: [A0, A1, A2, B0, C0, C1, C2, C3, 0, 0, 0, 0, 0, 0, 0, 0]
                    vector<RawFr::Element> keyV;
                    RawFr::Element aux;
                    keyV.push_back(pol(A0)[i]);
                    fr.fromUI(aux, pol(A1)[i]);
                    keyV.push_back(aux);
                    fr.fromUI(aux, pol(A2)[i]);
                    keyV.push_back(aux);
                    keyV.push_back(pol(B0)[i]);
                    keyV.push_back(pol(C0)[i]);
                    fr.fromUI(aux, pol(C1)[i]);
                    keyV.push_back(aux);
                    fr.fromUI(aux, pol(C2)[i]);
                    keyV.push_back(aux);
                    fr.fromUI(aux, pol(C3)[i]);
                    keyV.push_back(aux);

                    // Add tailing fr.zero's to complete 2^ARITY field elements
                    while (keyV.size() < (1<<ARITY)) {
                        keyV.push_back(fr.zero());
                    }
                    
                    // Call poseidon and get the hash key
                    poseidon.hash(keyV, &ctx.lastSWrite.key);
#ifdef LOG_STORAGE
                    cout << "Storage read sRD got poseidon key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << endl;
#endif 
                    
                    // Check that storage entry exists
                    if (ctx.sto.find(ctx.lastSWrite.key) == ctx.sto.end())
                    {
                        cerr << "Error: Storage not initialized, key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << " line: " << ctx.zkPC << " step: " << ctx.step << endl;
                        exit(-1);
                    }

                    // Read the value from storage, and store it in fin
                    scalar2fea(fr, ctx.sto[ctx.lastSWrite.key], fi0, fi1, fi2, fi3);

                    nHits++;
#ifdef LOG_STORAGE
                    cout << "Storage read sRD read from key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << " value:" << ctx.sto[ctx.lastSWrite.key] << endl;
#endif 
                }

                // If sWR (storage write) calculate the poseidon hash key, check its entry exists in storage, and update new root hash
                if (rom[zkPC].sWR == 1)
                {
                    // reset lastSWrite
                    ctx.lastSWrite.key = fr.zero();
                    ctx.lastSWrite.newRoot = fr.zero();
                    ctx.lastSWrite.step = 0;

                    // Fill a vector of field elements
                    vector<RawFr::Element> keyV;
                    RawFr::Element aux;
                    keyV.push_back(pol(A0)[i]);
                    fr.fromUI(aux, pol(A1)[i]);
                    keyV.push_back(aux);
                    fr.fromUI(aux, pol(A2)[i]);
                    keyV.push_back(aux);
                    keyV.push_back(pol(B0)[i]);
                    keyV.push_back(pol(C0)[i]);
                    fr.fromUI(aux, pol(C1)[i]);
                    keyV.push_back(aux);
                    fr.fromUI(aux, pol(C2)[i]);
                    keyV.push_back(aux);
                    fr.fromUI(aux, pol(C3)[i]);
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
                        cerr << "Error: Storage write sWR not initialized key: " << fr.toString(ctx.lastSWrite.key, 16) << " line: " << ctx.zkPC << endl;
                        exit(-1);
                    }

                    // Call SMT to get the new Merkel Tree root hash
                    SmtSetResult res;
                    mpz_class scalarD;
                    fea2scalar(fr, scalarD, pol(D0)[i], pol(D1)[i], pol(D2)[i], pol(D3)[i]);
                    smt.set(ctx.fr, ctx.db, pol(SR)[i], ctx.lastSWrite.key, scalarD, res);
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

                // If hashRD (hash read)
                if (rom[zkPC].hashRD == 1)
                {
                    // Check the entry addr exists in hash
                    if (ctx.hash.find(addr) == ctx.hash.end()) {
                        cerr << "Error: Hash address not initialized" << endl;
                        exit(-1);
                    }

                    // Read fi=hash[addr]
                    mpz_class auxScalar(ctx.hash[addr].hash);
                    scalar2fea(fr, auxScalar, fi0, fi1, fi2, fi3);
                    nHits++;
#ifdef LOG_HASH
                    cout << "Hash read hashRD: addr:" << addr << " hash:" << auxScalar.get_str(16) << endl;
#endif
                }

                // If ecRecover, build the transaction signature, recover the address that generated it, and copy fi=recovered address
                if (rom[zkPC].ecRecover == 1) {
                    mpz_class aux;
                    
                    // Get d=A
                    fea2scalar(fr, aux, pol(A0)[i], pol(A1)[i], pol(A2)[i], pol(A3)[i]);
                    string d = NormalizeTo0xNFormat(aux.get_str(16),64);

                    // Signature string = 0x + r(32B) + s(32B) + v(1B) = 0x + 130chars
                    fea2scalar(fr, aux, pol(B0)[i], pol(B1)[i], pol(B2)[i], pol(B3)[i]);
                    string r = NormalizeToNFormat(aux.get_str(16),64);
                    fea2scalar(fr, aux, pol(C0)[i], pol(C1)[i], pol(C2)[i], pol(C3)[i]);
                    string s = NormalizeToNFormat(aux.get_str(16),64);
                    aux = fe2n(ctx, pol(D0)[i]);
                    string v = NormalizeToNFormat(aux.get_str(16),2);
                    string signature = "0x" + r + s + v;

                    /* Return the address associated with the public key signature from elliptic curve signature.
                       Signature parts: r: first 32 bytes of signature; s: second 32 bytes of signature; v: final 1 byte of signature.
                       Hash: d: 32 bytes. */
                    string ecResult = ecrecover(signature, d);
                    mpz_class raddr(ecResult);
                    scalar2fea(fr, raddr, fi0, fi1, fi2, fi3);
                    nHits++;
                }

                // If shl, shift A, D bytes to the left, and discard highest bits
                if (rom[zkPC].shl == 1)
                {
                    // Read a=A
                    mpz_class a;
                    fea2scalar(fr, a, pol(A0)[i], pol(A1)[i], pol(A2)[i], pol(A3)[i]);

                    // Read s=D
                    uint64_t s = fe2n(ctx, pol(D0)[i]);
                    if ((s>32) || (s<0)) {
                        cerr << "Error: SHL too big: " << ctx.zkPC << endl;
                        exit(-1);
                    }

                    // Calculate b = shift a, s bytes to the left
                    mpz_class band("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
                    mpz_class b;
                    b = (a << s*8) & band;

                    // Copy fi=b
                    scalar2fea(fr, b, fi0, fi1, fi2, fi3);
                    nHits++;
                }

                // If shr, shift A, D bytes to the right
                if (rom[zkPC].shr == 1)
                {
                    // Read a=A
                    mpz_class a;
                    fea2scalar(fr, a, pol(A0)[i], pol(A1)[i], pol(A2)[i], pol(A3)[i]);

                    // Read s=D
                    uint64_t s = fe2n(ctx, pol(D0)[i]);
                    if ((s>32) || (s<0)) {
                        cerr << "Error: SHR too big: " << ctx.zkPC << endl;
                        exit(-1);
                    }

                    // Calculate b = shift a, s bytes to the right
                    mpz_class b = a >> s*8;

                    // Copy fi=b
                    scalar2fea(fr, b, fi0, fi1, fi2, fi3);
                    nHits++;
                } 

                // Check that one and only one instruction has been requested
                if (nHits == 0) {
                    cerr << "Error: Empty freeIn without a valid instruction: " << ctx.zkPC << endl;
                    exit(-1);
                }
                if (nHits > 1) {
                    cerr << "Error: Only one instruction that requires freeIn is alllowed: " << ctx.zkPC << endl;
                }
            }
            // If freeInTag.op!="", then evaluate the requested command (recursively)
            else
            {
                // Call evalCommand()
                CommandResult cr;
                evalCommand(ctx, rom[zkPC].freeInTag, cr);

                // Copy fi=command result, depending on its type 
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

            // Store polynomial FREE=fi
            pol(FREE0)[i] = fi0;
            pol(FREE1)[i] = fi1;
            pol(FREE2)[i] = fi2;
            pol(FREE3)[i] = fi3;

            // op = op + fi
            fr.add(op0, op0, fi0);
            op1 += fe2u64(fr, fi1);
            op2 += fe2u64(fr, fi2);
            op3 += fe2u64(fr, fi3);
        } else {
            pol(FREE0)[i] = fr.zero();
            pol(FREE1)[i] = fr.zero();
            pol(FREE2)[i] = fr.zero();
            pol(FREE3)[i] = fr.zero();
        }
        pol(inFREE)[i] = rom[zkPC].inFREE;

        // If neg, op=-op
        if (rom[zkPC].neg == 1) {
            fr.neg(op0,op0);
#ifdef LOG_NEG
            cout << "neg op0=" << fr.toString(op0, 16) << endl;
#endif
        }
        pol(neg)[i] = rom[zkPC].neg;

        // If assert, check that A=op
        if (rom[zkPC].assert == 1) {
            if ( (!fr.eq(pol(A0)[i],op0)) ||
                 (pol(A1)[i] != op1) ||
                 (pol(A2)[i] != op2) ||
                 (pol(A3)[i] != op3) )
            {
                cerr << "Error: ROM assert failed: AN!=opN ln: " << ctx.zkPC << endl;
                cout << "A: " << pol(A3)[i] << ":" << pol(A2)[i] << ":" << pol(A1)[i] << ":" << fr.toString(pol(A0)[i],16) << endl;
                cout << "OP:" << op3 << ":" << op2 << ":" << op1 << ":" << fr.toString(op0,16) << endl;
                exit(-1);
            }
#ifdef LOG_ASSERT
            cout << "assert" << endl;
#endif
        }
        pol(assert)[i] = rom[zkPC].assert;

        // Calculate nexti to write the next evaluation register values according to setX
        // The registers of the evaluation 0 will be overwritten with the values from the last evaluation, closing the evaluation circle
        uint64_t nexti = (i+1)%NEVALUATIONS;

        // If setA, A'=op
        if (rom[zkPC].setA == 1) {
            pol(A0)[nexti] = op0;
            pol(A1)[nexti] = op1;
            pol(A2)[nexti] = op2;
            pol(A3)[nexti] = op3;
#ifdef LOG_SETX
            cout << "setA A[nexti]=" << pols(A3)[nexti] << ":" << pols(A2)[nexti] << ":" << pols(A1)[nexti] << ":" << fr.toString(pols(A0)[nexti],16) << endl;
#endif
        } else {
            pol(A0)[nexti] = pol(A0)[i];
            pol(A1)[nexti] = pol(A1)[i];
            pol(A2)[nexti] = pol(A2)[i];
            pol(A3)[nexti] = pol(A3)[i];
        }
        pol(setA)[i] = rom[zkPC].setA;

        // If setB, B'=op
        if (rom[zkPC].setB == 1) {
            pol(B0)[nexti] = op0;
            pol(B1)[nexti] = op1;
            pol(B2)[nexti] = op2;
            pol(B3)[nexti] = op3;
#ifdef LOG_SETX
            cout << "setB B[nexti]=" << pols(B3)[nexti] << ":" << pols(B2)[nexti] << ":" << pols(B1)[nexti] << ":" << fr.toString(pols(B0)[nexti], 16) << endl;
#endif
        } else {
            pol(B0)[nexti] = pol(B0)[i];
            pol(B1)[nexti] = pol(B1)[i];
            pol(B2)[nexti] = pol(B2)[i];
            pol(B3)[nexti] = pol(B3)[i];
        }
        pol(setB)[i] = rom[zkPC].setB;

        // If setC, C'=op
        if (rom[zkPC].setC == 1) {
            pol(C0)[nexti] = op0;
            pol(C1)[nexti] = op1;
            pol(C2)[nexti] = op2;
            pol(C3)[nexti] = op3;
#ifdef LOG_SETX
            cout << "setC C[nexti]=" << pols(C3)[nexti] << ":" << pols(C2)[nexti] << ":" << pols(C1)[nexti] << ":" << fr.toString(pols(C0)[nexti], 16) << endl;
#endif
        } else {
            pol(C0)[nexti] = pol(C0)[i];
            pol(C1)[nexti] = pol(C1)[i];
            pol(C2)[nexti] = pol(C2)[i];
            pol(C3)[nexti] = pol(C3)[i];
        }
        pol(setC)[i] = rom[zkPC].setC;

        // If setD, D'=op
        if (rom[zkPC].setD == 1) {
            pol(D0)[nexti] = op0;
            pol(D1)[nexti] = op1;
            pol(D2)[nexti] = op2;
            pol(D3)[nexti] = op3;
#ifdef LOG_SETX
            cout << "setD D[nexti]=" << pols(D3)[nexti] << ":" << pols(D2)[nexti] << ":" << pols(D1)[nexti] << ":" << fr.toString(pols(D0)[nexti], 16) << endl;
#endif
        } else {
            pol(D0)[nexti] = pol(D0)[i];
            pol(D1)[nexti] = pol(D1)[i];
            pol(D2)[nexti] = pol(D2)[i];
            pol(D3)[nexti] = pol(D3)[i];
        }
        pol(setD)[i] = rom[zkPC].setD;
        
        // If setE, E'=op
        if (rom[zkPC].setE == 1) {
            pol(E0)[nexti] = op0;
            pol(E1)[nexti] = op1;
            pol(E2)[nexti] = op2;
            pol(E3)[nexti] = op3;
#ifdef LOG_SETX
            cout << "setE E[nexti]=" << pols(E3)[nexti] << ":" << pols(E2)[nexti] << ":" << pols(E1)[nexti] << ":" << fr.toString(pols(E0)[nexti] ,16) << endl;
#endif
        } else {
            pol(E0)[nexti] = pol(E0)[i];
            pol(E1)[nexti] = pol(E1)[i];
            pol(E2)[nexti] = pol(E2)[i];
            pol(E3)[nexti] = pol(E3)[i];
        }
        pol(setE)[i] = rom[zkPC].setE;

        // If setSR, SR'=op
        if (rom[zkPC].setSR == 1) {
            pol(SR)[nexti] = op0;
#ifdef LOG_SETX
            cout << "setSR SR[nexti]=" << fr.toString(pols(SR)[nexti],16) << endl;
#endif
        } else {
            pol(SR)[nexti] = pol(SR)[i];
        }
        pol(setSR)[i] = rom[zkPC].setSR;

        // If setCTX, CTX'=op
        if (rom[zkPC].setCTX == 1) {
            pol(CTX)[nexti] = fe2n(ctx, op0);
#ifdef LOG_SETX
            cout << "setCTX CTX[nexti]=" << pols(CTX)[nexti] << endl;
#endif
        } else {
            pol(CTX)[nexti] = pol(CTX)[i];
        }
        pol(setCTX)[i] = rom[zkPC].setCTX;

        // If setSP, SP'=op
        if (rom[zkPC].setSP == 1) {
            pol(SP)[nexti] = fe2n(ctx, op0);
#ifdef LOG_SETX
            cout << "setSP SP[nexti]=" << pols(SP)[nexti] << endl;
#endif
        } else {
            // SP'=SP
            pol(SP)[nexti] = pol(SP)[i];
            // If inc stack, SP'++
            if ((rom[zkPC].inc==1) && (rom[zkPC].isStack==1)){
                pol(SP)[nexti] = pol(SP)[nexti] + 1;
            }
            // If dec stack, SP'--
            if ((rom[zkPC].dec==1) && (rom[zkPC].isStack==1)){
                pol(SP)[nexti] = pol(SP)[nexti] - 1;
            }
        }
        pol(setSP)[i] = rom[zkPC].setSP;

        // If setPC, PC'=op
        if (rom[zkPC].setPC == 1) {
            pol(PC)[nexti] = fe2n(ctx, op0);
#ifdef LOG_SETX
            cout << "setPC PC[nexti]=" << pols(PC)[nexti] << endl;
#endif
        } else {
            // PC'=PC
            pol(PC)[nexti] = pol(PC)[i];
            // If inc code, PC'++
            if ( (rom[zkPC].inc==1) && (rom[zkPC].isCode==1) ) {
                pol(PC)[nexti] = 
                pol(PC)[nexti] + 1; // PC is part of Ethereum's program
            }
            // If dec code, PC'--
            if ( (rom[zkPC].dec==1) && (rom[zkPC].isCode==1) ) {
                pol(PC)[nexti] = pol(PC)[nexti] - 1; // PC is part of Ethereum's program
            }
        }
        pol(setPC)[i] = rom[zkPC].setPC;

        // If JMPC, jump conditionally based on op value
        if (rom[zkPC].JMPC == 1) {
#ifdef LOG_JMP
            cout << "JMPC: op0=" << fr.toString(op0) << endl;
#endif
            int64_t o = fe2n(ctx, op0);
#ifdef LOG_JMP
            cout << "JMPC: o=" << o << endl;
#endif
            // If op<0, jump to addr: zkPC'=addr
            if (o < 0) {
                pol(isNeg)[i] = 1;
                pol(zkPC)[nexti] = addr;
                ctx.byte4[0x100000000 + o] = true;
#ifdef LOG_JMP
               cout << "Next zkPC(1)=" << pols(zkPC)[nexti] << endl;
#endif
            }
            // If op>=0, simply increase zkPC'=zkPC+1
            else
            {
                pol(isNeg)[i] = 0;
                pol(zkPC)[nexti] = pol(zkPC)[i] + 1;
#ifdef LOG_JMP
                cout << "Next zkPC(2)=" << pols(zkPC)[nexti] << endl;
#endif
                ctx.byte4[o] = true;
            }
            pol(JMP)[i] = 0;
            pol(JMPC)[i] = 1;
        }
        // If JMP, directly jump zkPC'=addr
        else if (rom[zkPC].JMP == 1)
        {
            pol(isNeg)[i] = 0;
            pol(zkPC)[nexti] = addr;
#ifdef LOG_JMP
            cout << "Next zkPC(3)=" << pols(zkPC)[nexti] << endl;
#endif
            pol(JMP)[i] = 1;
            pol(JMPC)[i] = 0;
        }
        // Else, simply increase zkPC'=zkPC+1
        else
        {
            pol(isNeg)[i] = 0;
            pol(zkPC)[nexti] = pol(zkPC)[i] + 1;
            pol(JMP)[i] = 0;
            pol(JMPC)[i] = 0;
        }

        // Calculate the new max mem address, if any
        uint32_t maxMemCalculated = 0;
        uint32_t mm = pol(MAXMEM)[i];
        if (rom[zkPC].isMem==1 && addrRel>mm) {
            pol(isMaxMem)[i] = 1;
            maxMemCalculated = addrRel;
            ctx.byte4[maxMemCalculated - mm] = true;
        } else {
            pol(isMaxMem)[i] = 0;
            maxMemCalculated = mm;
            ctx.byte4[0] = true;
        }

        // If setMAXMEM, MAXMEM'=op
        if (rom[zkPC].setMAXMEM == 1) {
            pol(MAXMEM)[nexti] = fe2n(ctx, op0);
#ifdef LOG_SETX
            cout << "setMAXMEM MAXMEM[nexti]=" << pols(MAXMEM)[nexti] << endl;
#endif
        } else {
            pol(MAXMEM)[nexti] = maxMemCalculated;
        }
        pol(setMAXMEM)[i] = rom[zkPC].setMAXMEM;

        // If setGAS, GAS'=op
        if (rom[zkPC].setGAS == 1) {
            pol(GAS)[nexti] = fe2n(ctx, op0);
#ifdef LOG_SETX
            cout << "setGAS GAS[nexti]=" << pols(GAS)[nexti] << endl;
#endif
        } else {
            pol(GAS)[nexti] = pol(GAS)[i];
        }
        pol(setGAS)[i] = rom[zkPC].setGAS;

        // Copy ROM flags into the polynomials
        pol(mRD)[i] = rom[zkPC].mRD;

        // If mWR, mem[addr]=op
        if (rom[zkPC].mWR == 1) {
            ctx.mem[addr].fe0 = op0;
            fr.fromUI(ctx.mem[addr].fe1, op1);
            fr.fromUI(ctx.mem[addr].fe2, op2);
            fr.fromUI(ctx.mem[addr].fe3, op3);
#ifdef LOG_MEMORY
            cout << "Memory write mWR: addr:" << addr << " " << printFea(ctx, ctx.mem[addr]) << endl;
#endif
        }
        pol(mWR)[i] = rom[zkPC].mWR;

        // Copy ROM flags into the polynomials
        pol(sRD)[i] = rom[zkPC].sRD;

        if (rom[zkPC].sWR == 1)
        {
            if (ctx.lastSWrite.step != i)
            {
                // Fill a vector of field elements
                vector<RawFr::Element> keyV;
                RawFr::Element aux;
                keyV.push_back(pol(A0)[i]);
                fr.fromUI(aux, pol(A1)[i]);
                keyV.push_back(aux);
                fr.fromUI(aux, pol(A2)[i]);
                keyV.push_back(aux);
                keyV.push_back(pol(B0)[i]);
                keyV.push_back(pol(C0)[i]);
                fr.fromUI(aux, pol(C1)[i]);
                keyV.push_back(aux);
                fr.fromUI(aux, pol(C2)[i]);
                keyV.push_back(aux);
                fr.fromUI(aux, pol(C3)[i]);
                keyV.push_back(aux);

                // Add tailing fr.zero's to complete 2^ARITY field elements
                while (keyV.size() < (1<<ARITY)) {
                    keyV.push_back(fr.zero());
                }
                
                // Call poseidon to get the hash
                poseidon.hash(keyV, &ctx.lastSWrite.key);
                
                // Check that storage entry exists
                if (ctx.sto.find(ctx.lastSWrite.key) == ctx.sto.end())
                {
                    cerr << "Error: Storage not initialized key: " << fr.toString(ctx.lastSWrite.key, 16) << " line: " << ctx.zkPC << endl;
                    exit(-1);
                }

                // Call SMT to get the new Merkel Tree root hash
                SmtSetResult res;
                mpz_class scalarD;
                fea2scalar(fr, scalarD, pol(D0)[i], pol(D1)[i], pol(D2)[i], pol(D3)[i]);
                smt.set(ctx.fr, ctx.db, pol(SR)[i], ctx.lastSWrite.key, scalarD, res);

                // Store it in lastSWrite
                ctx.lastSWrite.newRoot = res.newRoot;
                ctx.lastSWrite.step = i;
            }

            // Check that the new root hash equals op0
            if (!fr.eq(ctx.lastSWrite.newRoot, op0)) {
                cerr << "Error: Storage write does not match: " << ctx.zkPC << endl;
                exit(-1);
            }

            // Store sto[poseidon_hash]=D
            mpz_class auxScalar;
            fea2scalar(fr, auxScalar, pol(D0)[i], pol(D1)[i], pol(D2)[i], pol(D3)[i]);
            ctx.sto[ctx.lastSWrite.key] = auxScalar;
        }
        pol(sWR)[i] = rom[zkPC].sWR;

        // Copy ROM flags into the polynomials
        pol(hashRD)[i] = rom[zkPC].hashRD;

        if (rom[zkPC].hashWR == 1) {

            // Get the size of the hash from D0
            uint64_t size = fe2n(ctx, pol(D0)[i]);
            if ((size<0) || (size>32)) {
                cerr << "Error: Invalid size for hash.  Size:" << size << " Line:" << ctx.zkPC << endl;
                exit(-1);
            }

            // Get contents of opN into a
            mpz_class a;
            fea2scalar(fr, a, op0, op1, op2, op3);

            // If there is no entry in the hash database for this address, then create a new one
            if (ctx.hash.find(addr) == ctx.hash.end())
            {
                HashValue hashValue;
                ctx.hash[addr] = hashValue;
            }

            // Fill the hash data vector with chunks of the scalar value
            for (uint64_t j=0; j<size; j++) {
                mpz_class band(0xFF);
                mpz_class result = (a >> (size-j-1)*8) & band;
                uint64_t uiResult = result.get_ui();
                ctx.hash[addr].data.push_back((uint8_t)uiResult);
            }

#ifdef LOG_HASH
            cout << "Hash write  hashWR: addr:" << addr << endl;
#endif
        }
        pol(hashWR)[i] = rom[zkPC].hashWR;

        // If hashE, calculate hash[addr] using keccak256
        if (rom[zkPC].hashE == 1) {
            ctx.hash[addr].hash = keccak256(ctx.hash[addr].data.data(), ctx.hash[addr].data.size());
#ifdef LOG_HASH
            cout << "Hash write  hashWR+hashE: addr:" << addr << " hash:" << ctx.hash[addr].hash << " size:" << ctx.hash[addr].data.size() << " data:";
            for (int k=0; k<ctx.hash[addr].data.size(); k++) cout << byte2string(ctx.hash[addr].data[k]) << ":";
            cout << endl;
#endif            
        }
        pol(hashE)[i] = rom[zkPC].hashE;

        // Copy ROM flags into the polynomials
        pol(ecRecover)[i] = rom[zkPC].ecRecover;

        // If arith, check that A*B + C = D<<256 + op, using scalars (result can be a big number)
        if (rom[zkPC].arith == 1)
        {
            // Convert to scalar
            mpz_class A, B, C, D, op;
            fea2scalar(fr, A, pol(A0)[i], pol(A1)[i], pol(A2)[i], pol(A3)[i]);
            fea2scalar(fr, B, pol(B0)[i], pol(B1)[i], pol(B2)[i], pol(B3)[i]);
            fea2scalar(fr, C, pol(C0)[i], pol(C1)[i], pol(C2)[i], pol(C3)[i]);
            fea2scalar(fr, D, pol(D0)[i], pol(D1)[i], pol(D2)[i], pol(D3)[i]);
            fea2scalar(fr, op, op0, op1, op2, op3);

            // Check the condition
            if ( (A*B) + C != (D<<256) + op ) {
                cerr << "Error: Arithmetic does not match: " << ctx.zkPC << endl;
                exit(-1);
            }
        }
        pol(arith)[i] = rom[zkPC].arith;

        // Copy ROM flags into the polynomials
        pol(shl)[i] = rom[zkPC].shl;
        pol(shr)[i] = rom[zkPC].shr;
        pol(bin)[i] = rom[zkPC].bin;
        pol(comparator)[i] = rom[zkPC].comparator;
        pol(opcodeRomMap)[i] = rom[zkPC].opcodeRomMap;

        // Evaluate the list cmdAfter commands, and any children command, recursively
        for (uint64_t j=0; j<rom[zkPC].cmdAfter.size(); j++)
        {
            CommandResult cr;
            evalCommand(ctx, *rom[zkPC].cmdAfter[j], cr);
        }

#ifdef LOG_STEPS
        cout << "<-- Completed step: " << ctx.step << " zkPC: " << zkPC << " op0: " << fr.toString(op0,16) << " FREE0: " << fr.toString(pols(FREE0)[i],16) << endl;
#endif
    }

    //printRegs(ctx);
    //printVars(ctx);
    //printMem(ctx);
    //printStorage(ctx);
    //printDb(ctx);

    // Check that all registers are set to 0
    checkFinalState(fr, ctx);

    // Based on the content of byte4[], fill the byte4_freeIn and byte4_out polynomials
    uint64_t p = 0;
    uint64_t last = 0;

    // Check that we have enough room in polynomials
    if (ctx.byte4.size()*2 > NEVALUATIONS)
    {
        cerr << "Error: Too many byte4 entries" << endl;
        exit(-1);
    }    
    
    // Generate polynomials content out of byte4 content
    for (map<uint32_t,bool>::iterator it=ctx.byte4.begin(); it!=ctx.byte4.end(); it++)
    {
        uint32_t num = it->first;
        pol(byte4_freeIN)[p] = num >> 16;
        pol(byte4_out)[p] = last;
        p++;
        pol(byte4_freeIN)[p] = num & 0xFFFF;
        pol(byte4_out)[p] = num >> 16;
        p++;
        last = num;
    }
    pol(byte4_freeIN)[p] = 0;
    pol(byte4_out)[p] = last;
    p++;
    pol(byte4_freeIN)[p] = 0;
    pol(byte4_out)[p] = 0;
    p++;

    // Check that we are not too far away   TODO: Delete after discussing it with Jordi
    /*if (p >= NEVALUATIONS)
    {
        cerr << "Error: Too many byte4 entries" << endl;
        exit(-1);
    }

    // Complete the rest of evaluations with zeros
    while (p < NEVALUATIONS)
    {
        pol(byte4_freeIN)[p] = 0;
        pol(byte4_out)[p] = 0;
        p++;        
    }*/

    /* Unmap output file from memory and cleanup */
    unloadPols(ctx);

    /* Unload ROM JSON file data from memory, i.e. free memory */
    unloadRom(ctx);
    
}

/* Sets first evaluation of all polynomials to zero */
void initState(RawFr &fr, Context &ctx)
{
    // Register value initial parameters
    pol(A0)[0] = fr.zero();
    pol(A1)[0] = 0;
    pol(A2)[0] = 0;
    pol(A3)[0] = 0;
    pol(B0)[0] = fr.zero();
    pol(B1)[0] = 0;
    pol(B2)[0] = 0;
    pol(B3)[0] = 0;
    pol(C0)[0] = fr.zero();
    pol(C1)[0] = 0;
    pol(C2)[0] = 0;
    pol(C3)[0] = 0;
    pol(D0)[0] = fr.zero();
    pol(D1)[0] = 0;
    pol(D2)[0] = 0;
    pol(D3)[0] = 0;
    pol(E0)[0] = fr.zero();
    pol(E1)[0] = 0;
    pol(E2)[0] = 0;
    pol(E3)[0] = 0;
    pol(SR)[0] = fr.zero();
    pol(CTX)[0] = 0;
    pol(SP)[0] = 0;
    pol(PC)[0] = 0;
    pol(MAXMEM)[0] = 0;
    pol(GAS)[0] = 0;
    pol(zkPC)[0] = 0;
}

// Check that last evaluation (which is in fact the first one) is zero
void checkFinalState(RawFr &fr, Context &ctx)
{
    if ( 
        (!fr.isZero(pol(A0)[0])) ||
        (pol(A1)[0]!=0) ||
        (pol(A2)[0]!=0) ||
        (pol(A3)[0]!=0) ||
        (!fr.isZero(pol(B0)[0])) ||
        (pol(B1)[0]!=0) ||
        (pol(B2)[0]!=0) ||
        (pol(B3)[0]!=0) ||
        (!fr.isZero(pol(C0)[0])) ||
        (pol(C1)[0]!=0) ||
        (pol(C2)[0]!=0) ||
        (pol(C3)[0]!=0) ||
        (!fr.isZero(pol(D0)[0])) ||
        (pol(D1)[0]!=0) ||
        (pol(D2)[0]!=0) ||
        (pol(D3)[0]!=0) ||
        (!fr.isZero(pol(E0)[0])) ||
        (pol(E1)[0]!=0) ||
        (pol(E2)[0]!=0) ||
        (pol(E3)[0]!=0) ||
        (!fr.isZero(pol(SR)[0])) ||
        (pol(CTX)[0]!=0) ||
        (pol(SP)[0]!=0) ||
        (pol(PC)[0]!=0) ||
        (pol(MAXMEM)[0]!=0) ||
        (pol(GAS)[0]!=0) ||
        (pol(zkPC)[0]!=0)
    ) {
        cerr << "Error: Program terminated with registers not set to zero" << endl;
        exit(-1);
    }
    else{
        cout << "checkFinalState() succeeded" << endl;
    }
}
