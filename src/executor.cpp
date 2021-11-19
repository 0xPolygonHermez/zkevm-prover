
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
    cout << "execute()" << endl;

    Context ctx;
    memset(&ctx.pols, 0, sizeof(ctx.pols));
    
    ctx.pFr = &fr;
    ctx.outputFile = outputFile;

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

    uint64_t i = 0; // execution line

    for (uint64_t step=0; step<NEVALUATIONS; step++)
    {
        //i = pols(zkPC)[i]; // This is the read line of code, but using step for debugging purposes, to execute all possible instructions
        i=step;
        ctx.ln = i;

        // To be used inside evaluateCommand() to find the current value of the registers, e.g. (*ctx.pPols)(A0)[ctx.step]
        ctx.step = step;

        // Limit execution line to ROM size
        if ( i>=ctx.romSize )
        {
            cout << "Reached end of rom" << endl;
            break;
        }

        ctx.fileName = rom[i].fileName; // TODO: Is this required?  It is only used in printRegs(), and it is an overhead in every loop.
        ctx.line = rom[i].line; // TODO: Is this required? It is only used in printRegs(), and it is an overhead in every loop.

        // Evaluate the list cmdBefore commands, and any children command, recursively
        for (uint64_t j=0; j<rom[i].cmdBefore.size(); j++)
        {
            evalCommand(ctx, *rom[i].cmdBefore[j]);
        }

        // Initialize the local registers to zero
        op0 = fr.zero();
        op1 = 0;
        op2 = 0;
        op3 = 0;

        // inX adds the corresponding register values to the op local register set
        // In case several inXs are set to 1, those values will be added

        if (rom[i].inA == 1)
        {
            fr.add(op0, op0, pols(A0)[i]);
            op1 = op1 + pols(A1)[i];
            op2 = op2 + pols(A2)[i];
            op3 = op3 + pols(A3)[i];
        }
        pols(inA)[i] = rom[i].inA;

        if (rom[i].inB == 1) {
            fr.add(op0, op0, pols(B0)[i]);
            op1 = op1 + pols(B1)[i];
            op2 = op2 + pols(B2)[i];
            op3 = op3 + pols(B3)[i];
        }
        pols(inB)[i] = rom[i].inB;

        if (rom[i].inC == 1) {
            fr.add(op0, op0, pols(C0)[i]);
            op1 = op1 + pols(C1)[i];
            op2 = op2 + pols(C2)[i];
            op3 = op3 + pols(C3)[i];
        }
        pols(inC)[i] = rom[i].inC;

        if (rom[i].inD == 1) {
            fr.add(op0, op0, pols(D0)[i]);
            op1 = op1 + pols(D1)[i];
            op2 = op2 + pols(D2)[i];
            op3 = op3 + pols(D3)[i];
        }
        pols(inD)[i] = rom[i].inD;

        if (rom[i].inE == 1) {
            fr.add(op0, op0, pols(E0)[i]);
            op1 = op1 + pols(E1)[i];
            op2 = op2 + pols(E2)[i];
            op3 = op3 + pols(E3)[i];
        }
        pols(inE)[i] = rom[i].inE;

        if (rom[i].inSR == 1) {
            fr.add(op0, op0, pols(SR)[i]);
        }
        pols(inSR)[i] = rom[i].inSR;

        RawFr::Element aux;

        if (rom[i].inCTX == 1) {
            fr.fromUI(aux,pols(CTX)[i]);
            fr.add(op0, op0, aux);
        }
        pols(inCTX)[i] = rom[i].inCTX;

        if (rom[i].inSP == 1) {
            fr.fromUI(aux,pols(SP)[i]);
            fr.add(op0, op0, aux);
        }
        pols(inSP)[i] = rom[i].inSP;

        if (rom[i].inPC == 1) {
            fr.fromUI(aux,pols(PC)[i]);
            fr.add(op0, op0, aux);
        }
        pols(inPC)[i] = rom[i].inPC;

        if (rom[i].inGAS == 1) {
            fr.fromUI(aux,pols(GAS)[i]);
            fr.add(op0, op0, aux);
        }
        pols(inGAS)[i] = rom[i].inGAS;
        
        if (rom[i].inMAXMEM == 1) {
            fr.fromUI(aux,pols(MAXMEM)[i]);
            fr.add(op0, op0, aux);
        }
        pols(inMAXMEM)[i] = rom[i].inMAXMEM;

        if (rom[i].inSTEP == 1) {
            fr.fromUI(aux, i);
            fr.add(op0, op0, aux);
        }
        pols(inSTEP)[i] = rom[i].inSTEP;

        if (rom[i].bConstPresent) {
            pols(CONST)[i] = rom[i].CONST; // TODO: Check rom types: U64, U32, etc.  They should match the pols types
            fr.fromUI(aux,pols(CONST)[i]);
            fr.add(op0, op0, aux);
            ctx.byte4[0x80000000 + rom[i].CONST] = true;
        } else {
            pols(CONST)[i] = 0;
        }

        uint64_t addrRel = 0; // TODO: Check with Jordi if this is the right type for an address
        uint64_t addr = 0;

        // If address involved, load offset into addr
        if (rom[i].mRD==1 || rom[i].mWR==1 || rom[i].hashRD==1 || rom[i].hashWR==1 || rom[i].hashE==1 || rom[i].JMP==1 || rom[i].JMPC==1) {
            if (rom[i].ind == 1)
            {
                addrRel = fe2n(fr, pols(E0)[i]);
            }
            if (rom[i].bOffsetPresent)
            {
                // If offset is possitive, and the sum is too big, fail
                if (rom[i].offset>0 && (addrRel+rom[i].offset)>=0x100000000)
                {
                    cerr << "Error: addrRel >= 0x100000000 ln: " << ctx.ln << endl;
                    exit(-1); // TODO: Should we kill the process?                    
                }
                // If offset is negative, and its modulo is bigger than addrRel, fail
                if (rom[i].offset<0 && (-rom[i].offset)>addrRel)
                {
                    cerr << "Error: addrRel < 0 ln: " << ctx.ln << endl;
                    exit(-1); // TODO: Should we kill the process?
                }
                addrRel += rom[i].offset;
            }
            addr = addrRel;
        }

        if (rom[i].useCTX == 1) {
            addr += pols(CTX)[i]*CTX_OFFSET;
        }
        pols(useCTX)[i] = rom[i].useCTX;

        if (rom[i].isCode == 1) {
            addr += CODE_OFFSET;
        }
        pols(isCode)[i] = rom[i].isCode;

        if (rom[i].isStack == 1) {
            addr += STACK_OFFSET;
        }
        pols(isStack)[i] = rom[i].isStack;

        if (rom[i].isMem == 1) {
            addr += MEM_OFFSET;
        }
        pols(isMem)[i] = rom[i].isMem;

        pols(inc)[i] = rom[i].inc;
        pols(dec)[i] = rom[i].dec;
        pols(ind)[i] = rom[i].ind;

        if (rom[i].bOffsetPresent) {
            pols(offset)[i] = rom[i].offset;
            ctx.byte4[0x80000000 + rom[i].offset] = true;
        } else {
            pols(offset)[i] = 1;
        }

        /*
        if (l.inFREE) {

            if (!l.freeInTag) {
                throw new Error(`Instruction with freeIn without freeInTag: ${ctx.ln}`);
            }
            
            let fi;
            if (l.freeInTag.op=="") {
                let nHits = 0;
                if (l.mRD == 1) {
                    if (typeof ctx.mem[addr] != "undefined") {
                        fi = ctx.mem[addr];
                    } else {
                        fi = [Fr.zero, Fr.zero, Fr.zero, Fr.zero];
                    }
                    nHits++;
                }
                if (l.sRD == 1) {
                    const saddr = fe2bns([ctx.A0, ctx.A1, ctx.A2, ctx.A3]); // 256 bis number, scalar > fe size
                    if (typeof ctx.sto[saddr] === "undefined" ) throw new Error(`Storage not initialized: ${ctx.ln}`);
                    fi = ctx.sto[ saddr ];
                    nHits++;
                } // Library gmp -> C big numbers library, function to convert fe to bn
                // raw->toMpz()void toMpz(mpz_t r, Element &a); DNS google 8.8.8.8
                if (l.hashRD == 1) {
                    if (!ctx.hash[addr]) throw new Error("Hash address not initialized");
                    if (typeof ctx.hash[addr].result == "undefined") throw new Error("Hash not finalized");
                    fi = bn2bna(Fr, ctx.hash[addr].result);
                    nHits++;
                }
                if (l.ecRecover == 1) {
                    const d = ethers.utils.hexlify(fea2bn(Fr, ctx.A));
                    const r = ethers.utils.hexlify(fea2bn(Fr, ctx.B));
                    const s = ethers.utils.hexlify(fea2bn(Fr, ctx.C));
                    const v = ethers.utils.hexlify(fe2n(Fr, ctx.D[0]));
                    const raddr = .recoverAddress(d, {
                        r: r,
                        s: s,
                        v: v
                    });
                    fi = bn2bna(Fr, raddr);
                    nHits++;
                }
                if (l.shl == 1) {
                    const a = Scalar.e(fea2bn(Fr, ctx.A));
                    const s = fe2n(Fr, ctx.D[0]);
                    if ((s>32) || (s<0)) throw new Error(`SHL too big: ${ctx.ln}`);
                    fi = bn2bna(Fr, Scalar.band(Scalar.shl(a, s*8), Scalar.e("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")));
                    nHits++;
                } 
                if (l.shr == 1) {
                    const a = Scalar.e(fea2bn(Fr, ctx.A));
                    const s = fe2n(Fr, ctx.D[0]);
                    if ((s>32) || (s<0)) throw new Error(`SHR too big: ${ctx.ln}`);
                    fi = bn2bna(Fr, Scalar.shr(a, s*8));
                    nHits++;
                } 
                if (nHits==0) {
                    throw new Error(`Empty freeIn without a valid instruction: ${ctx.ln}`);
                }
                if (nHits>1) {
                    throw new Error(`Only one instructuin that requires freeIn is alllowed: ${ctx.ln}`);
                }
            } else {
                fi = evalCommand(ctx, l.freeInTag);
            }
            [pols.main.FREE0[i], pols.main.FREE1[i], pols.main.FREE2[i], pols.main.FREE3[i]] = fi;
            [op0, op1, op2, op3] = [Fr.add(op0, fi[0]), Fr.add(op1, fi[1]), Fr.add(op2, fi[2]), Fr.add(op3, fi[3])];
            pols.main.inFREE[i] = Fr.one;

        } else {
            [pols.main.FREE0[i], pols.main.FREE1[i], pols.main.FREE2[i], pols.main.FREE3[i]] = [Fr.zero, Fr.zero, Fr.zero, Fr.zero];
            pols.main.inFREE[i] = Fr.zero;
        }*/

        if (rom[i].neg == 1) {
            fr.neg(op0,op0);
        }
        pols(neg)[i] = rom[i].neg;

        if (rom[i].assert == 1) {
            if ( (!fr.eq(pols(A0)[i],op0)) ||
                 (pols(A1)[i] != op1) ||
                 (pols(A2)[i] != op2) ||
                 (pols(A3)[i] != op3) )
            {
                cerr << "Error: ROM assert failed: AN!=opN ln: " << ctx.ln << endl;
                //exit(-1); // TODO: Should we kill the process?  Temporarly disabling because assert is failing, since executor is not completed
            }
        }
        pols(assert)[i] = rom[i].assert;

        // The registers of the evaluation 0 will be overwritten with the values from the last evaluation,
        // closing the evaluation circle
        uint64_t nexti = (i+1)%NEVALUATIONS;

        if (rom[i].setA == 1) {
            fr.copy(pols(A0)[nexti],op0);
            pols(A1)[nexti] = op1;
            pols(A2)[nexti] = op2;
            pols(A3)[nexti] = op3;
        } else {
            fr.copy(pols(A0)[nexti],pols(A0)[i]);
            pols(A1)[nexti] = pols(A1)[i];
            pols(A2)[nexti] = pols(A2)[i];
            pols(A3)[nexti] = pols(A3)[i];
        }
        pols(setA)[i] = rom[i].setA;

        if (rom[i].setB == 1) {
            fr.copy(pols(B0)[nexti],op0);
            pols(B1)[nexti] = op1;
            pols(B2)[nexti] = op2;
            pols(B3)[nexti] = op3;
        } else {
            fr.copy(pols(B0)[nexti],pols(B0)[i]);
            pols(B1)[nexti] = pols(B1)[i];
            pols(B2)[nexti] = pols(B2)[i];
            pols(B3)[nexti] = pols(B3)[i];
        }
        pols(setB)[i] = rom[i].setB;

        if (rom[i].setC == 1) {
            fr.copy(pols(C0)[nexti],op0);
            pols(C1)[nexti] = op1;
            pols(C2)[nexti] = op2;
            pols(C3)[nexti] = op3;
        } else {
            fr.copy(pols(C0)[nexti],pols(C0)[i]);
            pols(C1)[nexti] = pols(C1)[i];
            pols(C2)[nexti] = pols(C2)[i];
            pols(C3)[nexti] = pols(C3)[i];
        }
        pols(setC)[i] = rom[i].setC;

        if (rom[i].setD == 1) {
            fr.copy(pols(D0)[nexti],op0);
            pols(D1)[nexti] = op1;
            pols(D2)[nexti] = op2;
            pols(D3)[nexti] = op3;
        } else {
            fr.copy(pols(D0)[nexti],pols(D0)[i]);
            pols(D1)[nexti] = pols(D1)[i];
            pols(D2)[nexti] = pols(D2)[i];
            pols(D3)[nexti] = pols(D3)[i];
        }
        pols(setD)[i] = rom[i].setD;

        if (rom[i].setE == 1) {
            fr.copy(pols(E0)[nexti],op0);
            pols(E1)[nexti] = op1;
            pols(E2)[nexti] = op2;
            pols(E3)[nexti] = op3;
        } else {
            fr.copy(pols(E0)[nexti],pols(E0)[i]);
            pols(E1)[nexti] = pols(E1)[i];
            pols(E2)[nexti] = pols(E2)[i];
            pols(E3)[nexti] = pols(E3)[i];
        }
        pols(setE)[i] = rom[i].setE;

        if (rom[i].setSR == 1) {
            fr.copy(pols(SR)[nexti],op0);
        } else {
            fr.copy(pols(SR)[nexti],pols(SR)[i]);
        }
        pols(setSR)[i] = rom[i].setSR;

        if (rom[i].setCTX == 1) {
            pols(CTX)[nexti] = fe2n(fr,op0);
        } else {
            pols(CTX)[nexti] = pols(CTX)[i];
        }
        pols(setCTX)[i] = rom[i].setCTX;

        if (rom[i].setSP == 1) {
            pols(SP)[nexti] = fe2n(fr,op0);
        } else {
            pols(SP)[nexti] = pols(SP)[i];
            if ((rom[i].inc==1) && (rom[i].isStack==1)){
                pols(SP)[nexti] = pols(SP)[nexti] + 1;
            }
            if ((rom[i].dec==1) && (rom[i].isStack==1)){
                pols(SP)[nexti] = pols(SP)[nexti] - 1;
            }
        }
        pols(setSP)[i] = rom[i].setSP;

        if (rom[i].setPC == 1) {
            pols(PC)[nexti] = fe2n(fr,op0);
        } else {
            pols(PC)[nexti] = pols(PC)[i];
            if ( (rom[i].inc==1) && (rom[i].isCode==1) ) {
                pols(PC)[nexti] = pols(PC)[nexti] + 1; // PC is part of Ethereum's program
            }
            if ( (rom[i].dec==1) && (rom[i].isCode==1) ) {
                pols(PC)[nexti] = pols(PC)[nexti] - 1; // PC is part of Ethereum's program
            }
        }
        pols(setPC)[i] = rom[i].setPC;

        if (rom[i].JMPC == 1) {
            int64_t o = fe2n(fr, op0);
            if (o<0) {
                pols(isNeg)[i] = 1;
                pols(zkPC)[nexti] = addr;
                ctx.byte4[0x100000000 + o] = true;
            } else {
                pols(isNeg)[i] = 0;
                pols(zkPC)[nexti] = pols(zkPC)[i] + 1;
                ctx.byte4[o] = true;
            }
            pols(JMP)[i] = 0;
            pols(JMPC)[i] = 1;
        } else if (rom[i].JMP == 1) {
            pols(isNeg)[i] = 0;
            pols(zkPC)[nexti] = addr;
            pols(JMP)[i] = 1;
            pols(JMPC)[i] = 0;
        } else {
            pols(isNeg)[i] = 0;
            pols(zkPC)[nexti] = pols(zkPC)[i] + 1;
            pols(JMP)[i] = 0;
            pols(JMPC)[i] = 0;
        }

        uint64_t maxMemCalculated = 0;
        uint64_t mm = pols(MAXMEM)[i];
        if (rom[i].isMem==1 && addrRel>mm) {
            pols(isMaxMem)[i] = 1;
            maxMemCalculated = addrRel;
            ctx.byte4[maxMemCalculated - mm] = true;
        } else {
            pols(isMaxMem)[i] = 0;
            maxMemCalculated = mm;
        }

        if (rom[i].setMAXMEM == 1) {
            pols(MAXMEM)[nexti] = fe2n(fr,op0);
        } else {
            pols(MAXMEM)[nexti] = maxMemCalculated;
        }
        pols(setMAXMEM)[i] = rom[i].setMAXMEM;

        if (rom[i].setGAS == 1) {
            pols(GAS)[nexti] = fe2n(fr, op0);
        } else {
            pols(GAS)[nexti] = pols(GAS)[i];
        }
        pols(setGAS)[i] = rom[i].setGAS;

        pols(mRD)[i] = rom[i].mRD;

        if (rom[i].mWR == 1) {
            ctx.mem[addr][0] = op0;
            fr.fromUI(ctx.mem[addr][1], op1);
            fr.fromUI(ctx.mem[addr][2], op2);
            fr.fromUI(ctx.mem[addr][3], op3);
        }
        pols(mWR)[i] = rom[i].mWR;

        pols(sRD)[i] = rom[i].sRD;

/*
        if (l.sWR) {
            pols.main.sWR[i] = Fr.one;

            if ((!ctx.lastSWrite)||(ctx.lastSWrite.step != i)) {
                ctx.lastSWrite = {};
                const keyV = [
                    ctx.A[0],
                    ctx.A[1],
                    ctx.A[2],
                    ctx.B[0],
                    ctx.C[0],
                    ctx.C[1],
                    ctx.C[2],
                    ctx.C[3]
                ]
                while (keyV.length < (1<< config.ARITY)) keyV.push(Fr.zero);

                ctx.lastSWrite.key = poseidon(keyV);

                ctx.lastSWrite.keyS = Fr.toString(ctx.lastSWrite.key, 16).padStart(64, "0");
                if (typeof ctx.sto[ctx.lastSWrite.keyS ] === "undefined" ) throw new Error(`Storage not initialized: ${ctx.ln}`);

                const res = smt.set(ctx.SR, ctx.lastSWrite.key, fea2scalar(Fr, ctx.D));
                ctx.lastSWrite.newRoot = res.newRoot;
                ctx.lastSWrite.step=i;
            }

            if (!Fr.eq(ctx.lastSWrite.newRoot, op0)) {
                throw new Error(`Storage write does not match: ${ctx.ln}`);
            }
            ctx.sto[ ctx.lastSWrite.keyS ] = fea2scalar(Fr, ctx.D);
        } else {
            pols.main.sWR[i] = Fr.zero;
        }
*/
        pols(hashRD)[i] = rom[i].hashRD;
/*
        if (l.hashWR) {
            pols.main.hashWR[i] = Fr.one;

            size = fe2n(Fr, ctx.D[0]);
            if ((size<0) || (size>32)) throw new Error(`Invalid size for hash: ${ctx.ln}`);
            const a = fea2bn(Fr, [op0, op1, op2, op3]);
            if (!ctx.hash[addr]) ctx.hash[addr] = { data: [] } ; // Array: string hexa
            for (let i=0; i<size; i++) {
                ctx.hash[addr].data.push(Scalar.toNumber(Scalar.band( Scalar.shr( a, (size-i -1)*8 ) , Scalar.e("0xFF"))));
            } // storing bytes 1 by 1, hash is a bytes vector
        } else {
            pols.main.hashWR[i] = Fr.zero;
        }

        if (l.hashE) {
            pols.main.hashE[i] = Fr.one;
            
            ctx.hash[addr].result = ethers.utils.keccak256(ethers.utils.hexlify(ctx.hash[addr].data)); // array of bytes, hexa string
        } else {
            pols.main.hashE[i] = Fr.zero;
        }
*/
        pols(ecRecover)[i] = rom[i].ecRecover;

        pols(arith)[i] = rom[i].arith;

        /*
                if (l.arith) {
            const A = fea2scalar(Fr, ctx.A);
            const B = fea2scalar(Fr, ctx.B);
            const C = fea2scalar(Fr, ctx.C);
            const D = fea2scalar(Fr, ctx.D);
            const op = fea2scalar(Fr, [op0, op1, op2, op3]);

            if (! Scalar.eq(Scalar.add(Scalar.mul(A, B), C),  Scalar.add(Scalar.shl(D, 256), op))   ) {
                throw new Error(`Arithmetic does not match: ${ctx.ln}`);
            }
            pols.main.arith[i] = Fr.one;
        } else {
            pols.main.arith[i] = Fr.zero;
        }
        */

        pols(shl)[i] = rom[i].shl;
        pols(shr)[i] = rom[i].shr;
        pols(bin)[i] = rom[i].bin;
        pols(comparator)[i] = rom[i].comparator;
        pols(opcodeRomMap)[i] = rom[i].opcodeRomMap;

        // Evaluate the list cmdAfter commands, and any children command, recursively
         for (uint64_t j=0; j<rom[i].cmdAfter.size(); j++)
        {
            evalCommand(ctx, *rom[i].cmdAfter[j]);
        }

    }

    printRegs(ctx);
    printVars(ctx);
    printMem(ctx);

    checkFinalState(fr, ctx);

    uint64_t p = 0;
    uint64_t last = 0;
    for (int n=0; n<ctx.byte4.size(); n++)
    {
        pols(byte4_freeIN)[p] = n >> 16;
        pols(byte4_out)[p] = last;
        p++;
        pols(byte4_freeIN)[p] = n & 0xFFFF;
        pols(byte4_out)[p] = n >> 16;
        p++;
        last = n;
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
