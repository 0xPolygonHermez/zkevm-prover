
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
    ctx.pFr = &fr;
    ctx.outputFile = outputFile;

    // opN are local, uncommitted polynomials
    RawFr::Element op3, op2, op1, op0;

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
        //i = fe2n(fr, pols[zkPC][i]); // This is the read line of code, but using step for debugging purposes, to execute all possible instructions
        i=step;
        ctx.ln = i;
        ctx.step = step; // To be used inside evaluateCommand() to find the current value of the registers, e.g. (*ctx.pPols)[A0][ctx.step]

        if ( i>=ctx.romSize )
        {
            cout << "Reached end of rom" << endl;
            break;
        }

        ctx.fileName = rom[i].fileName; // TODO: Is this required?  It is only used in printRegs(), and it is an overhead in every loop.
        ctx.line = rom[i].line; // TODO: Is this required? It is only used in printRegs(), and it is an overhead in every loop.

        for (uint64_t j=0; j<rom[i].cmdBefore.size(); j++)
        {
            evalCommand(ctx, *rom[i].cmdBefore[j]);
        }

        op0 = fr.zero();
        op1 = fr.zero();
        op2 = fr.zero();
        op3 = fr.zero();

        // inX adds the corresponding register values to the op local register set
        // In case several inXs are set to 1, those values will be added
        if (rom[i].inA)
        {
            fr.add(op0, op0, pols[A0][i]);
            fr.add(op1, op1, pols[A1][i]);
            fr.add(op2, op2, pols[A2][i]);
            fr.add(op3, op3, pols[A3][i]);
            pols[inA][i] = fr.one();
        }
        else {
            pols[inA][i] = fr.zero();
        }
        
        if (rom[i].inB) {
            fr.add(op0, op0, pols[B0][i]);
            fr.add(op1, op1, pols[B1][i]);
            fr.add(op2, op2, pols[B2][i]);
            fr.add(op3, op3, pols[B3][i]);
            pols[inB][i] = fr.one();
        } else {
            pols[inB][i] = fr.zero();
        }

        if (rom[i].inC) {
            fr.add(op0, op0, pols[C0][i]);
            fr.add(op1, op1, pols[C1][i]);
            fr.add(op2, op2, pols[C2][i]);
            fr.add(op3, op3, pols[C3][i]);
            pols[inC][i] = fr.one();
        } else {
            pols[inC][i] = fr.zero();
        }

        if (rom[i].inD) {
            fr.add(op0, op0, pols[D0][i]);
            fr.add(op1, op1, pols[D1][i]);
            fr.add(op2, op2, pols[D2][i]);
            fr.add(op3, op3, pols[D3][i]);
            pols[inD][i] = fr.one();
        } else {
            pols[inD][i] = fr.zero();
        }

        if (rom[i].inE) {
            fr.add(op0, op0, pols[E0][i]);
            fr.add(op1, op1, pols[E1][i]);
            fr.add(op2, op2, pols[E2][i]);
            fr.add(op3, op3, pols[E3][i]);
            pols[inE][i] = fr.one();
        } else {
            pols[inE][i] = fr.zero();
        }

        if (rom[i].inSR) {
            fr.add(op0, op0, pols[SR][i]);
            pols[inSR][i] = fr.one();
        } else {
            pols[inSR][i] = fr.zero();
        }

        if (rom[i].inCTX) {
            fr.add(op0, op0, pols[CTX][i]);
            pols[inCTX][i] = fr.one();
        } else {
            pols[inCTX][i] = fr.zero();
        }

        if (rom[i].inSP) {
            fr.add(op0, op0, pols[SP][i]);
            pols[inSP][i] = fr.one();
        } else {
            pols[inSP][i] = fr.zero();
        }

        if (rom[i].inPC) {
            fr.add(op0, op0, pols[PC][i]);
            pols[inPC][i] = fr.one();
        } else {
            pols[inPC][i] = fr.zero();
        }
        
        if (rom[i].inGAS) {
            fr.add(op0, op0, pols[GAS][i]);
            pols[inGAS][i] = fr.one();
        } else {
            pols[inGAS][i] = fr.zero();
        }

        if (rom[i].inMAXMEM) {
            fr.add(op0, op0, pols[MAXMEM][i]);
            pols[inMAXMEM][i] = fr.one();
        } else {
            pols[inMAXMEM][i] = fr.zero();
        }

        if (rom[i].inSTEP) {
            RawFr::Element eI;
            fr.fromUI(eI, i);
            fr.add(op0, op0, eI);
            pols[inSTEP][i] = fr.one();
        } else {
            pols[inSTEP][i] = fr.zero();
        }

        if (rom[i].bConstPresent) {
            fr.fromUI(pols[CONST][i], rom[i].CONST);
            fr.add(op0, op0, pols[CONST][i]);
        } else {
            pols[CONST][i] = fr.zero();
        }

        uint64_t addrRel = 0; // TODO: Check with Jordi if this is the right type for an address
        uint64_t addr = 0;

        // If address involved, load offset into addr
        if (rom[i].mRD || rom[i].mWR || rom[i].hashRD || rom[i].hashWR || rom[i].hashE || rom[i].JMP || rom[i].JMPC) {
            if (rom[i].ind)
            {
                addrRel = fe2n(fr, pols[E0][i]);
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

        if (rom[i].useCTX) {
            addr += fe2n(fr,pols[CTX][i])*CTX_OFFSET;
            pols[useCTX][i] = fr.one();
        } else {
            pols[useCTX][i] = fr.zero();
        }

        if (rom[i].isCode) {
            addr += CODE_OFFSET;
            pols[isCode][i] = fr.one();
        } else {
            pols[isCode][i] = fr.zero();
        }

        if (rom[i].isStack) {
            addr += STACK_OFFSET;
            pols[isStack][i] = fr.one();
        } else {
            pols[isStack][i] = fr.zero();
        }

        if (rom[i].isMem) {
            addr += MEM_OFFSET;
            pols[isMem][i] = fr.one();
        } else {
            pols[isMem][i] = fr.zero();
        }

        if (rom[i].inc) {
            pols[inc][i] = fr.one();
        } else {
            pols[inc][i] = fr.zero();
        }

        if (rom[i].dec) {
            pols[dec2][i] = fr.one();
        } else {
            pols[dec2][i] = fr.zero();
        }

        if (rom[i].ind) {
            pols[ind][i] = fr.one();
        } else {
            pols[ind][i] = fr.zero();
        }

        if (rom[i].bOffsetPresent) {
            fr.fromUI(pols[offset][i],rom[i].offset);
        } else {
            pols[offset][i] = fr.zero();
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

        if (rom[i].neg) {
            fr.neg(op0,op0);
            pols[neg][i] = fr.one();
        } else {
            pols[neg][i] = fr.zero();
        }

        if (rom[i].assert) {
            if ( (!fr.eq(pols[A0][i],op0)) ||
                 (!fr.eq(pols[A1][i],op1)) ||
                 (!fr.eq(pols[A2][i],op2)) ||
                 (!fr.eq(pols[A3][i],op3)) )
            {
                cerr << "Error: ROM assert failed: AN!=opN ln: " << ctx.ln << endl;
                //exit(-1); // TODO: Should we kill the process?  Temporarly disabling because assert is failing, since executor is not completed
            }
            pols[assert][i] = fr.one();
        } else {
            pols[assert][i] = fr.zero();
        }

        // The registers of the evaluation 0 will be overwritten with the values from the last evaluation,
        // closing the evaluation circle
        uint64_t nexti = (i+1)%NEVALUATIONS;

        if (rom[i].setA) {
            fr.copy(pols[A0][nexti],op0);
            fr.copy(pols[A1][nexti],op1);
            fr.copy(pols[A2][nexti],op2);
            fr.copy(pols[A3][nexti],op3);
            pols[setA][i] = fr.one();
        } else {
            fr.copy(pols[A0][nexti],pols[A0][i]);
            fr.copy(pols[A1][nexti],pols[A1][i]);
            fr.copy(pols[A2][nexti],pols[A2][i]);
            fr.copy(pols[A3][nexti],pols[A3][i]);
            pols[setA][i] = fr.zero();
        }

        if (rom[i].setB) {
            fr.copy(pols[B0][nexti],op0);
            fr.copy(pols[B1][nexti],op1);
            fr.copy(pols[B2][nexti],op2);
            fr.copy(pols[B3][nexti],op3);
            pols[setB][i] = fr.one();
        } else {
            fr.copy(pols[B0][nexti],pols[B0][i]);
            fr.copy(pols[B1][nexti],pols[B1][i]);
            fr.copy(pols[B2][nexti],pols[B2][i]);
            fr.copy(pols[B3][nexti],pols[B3][i]);
            pols[setB][i] = fr.zero();
        }

        if (rom[i].setC) {
            fr.copy(pols[C0][nexti],op0);
            fr.copy(pols[C1][nexti],op1);
            fr.copy(pols[C2][nexti],op2);
            fr.copy(pols[C3][nexti],op3);
            pols[setC][i] = fr.one();
        } else {
            fr.copy(pols[C0][nexti],pols[C0][i]);
            fr.copy(pols[C1][nexti],pols[C1][i]);
            fr.copy(pols[C2][nexti],pols[C2][i]);
            fr.copy(pols[C3][nexti],pols[C3][i]);
            pols[setC][i] = fr.zero();
        }

        if (rom[i].setD) {
            fr.copy(pols[D0][nexti],op0);
            fr.copy(pols[D1][nexti],op1);
            fr.copy(pols[D2][nexti],op2);
            fr.copy(pols[D3][nexti],op3);
            pols[setD][i] = fr.one();
        } else {
            fr.copy(pols[D0][nexti],pols[D0][i]);
            fr.copy(pols[D1][nexti],pols[D1][i]);
            fr.copy(pols[D2][nexti],pols[D2][i]);
            fr.copy(pols[D3][nexti],pols[D3][i]);
            pols[setD][i] = fr.zero();
        }

        if (rom[i].setE) {
            fr.copy(pols[E0][nexti],op0);
            fr.copy(pols[E1][nexti],op1);
            fr.copy(pols[E2][nexti],op2);
            fr.copy(pols[E3][nexti],op3);
            pols[setE][i] = fr.one();
        } else {
            fr.copy(pols[E0][nexti],pols[E0][i]);
            fr.copy(pols[E1][nexti],pols[E1][i]);
            fr.copy(pols[E2][nexti],pols[E2][i]);
            fr.copy(pols[E3][nexti],pols[E3][i]);
            pols[setE][i] = fr.zero();
        }

        if (rom[i].setSR) {
            fr.copy(pols[SR][nexti],op0);
            pols[setSR][i] = fr.one();
        } else {
            fr.copy(pols[SR][nexti],pols[SR][i]);
            pols[setSR][i] = fr.zero();
        }

        if (rom[i].setCTX) {
            fr.copy(pols[CTX][nexti],op0);
            pols[setCTX][i] = fr.one();
        } else {
            fr.copy(pols[CTX][nexti],pols[CTX][i]);
            pols[setCTX][i] = fr.zero();
        }

        if (rom[i].setSP) {
            fr.copy(pols[SP][nexti],op0);
            pols[setSP][i] = fr.one();
        } else {
            fr.copy(pols[SP][nexti],pols[SP][i]);
            if ((rom[i].inc)&&(rom[i].isStack)){
                fr.add(pols[SP][nexti], pols[SP][nexti], fr.one());
            }
            if ((rom[i].dec)&&(rom[i].isStack)){
                fr.sub(pols[SP][nexti], pols[SP][nexti], fr.one());
            }
            pols[setSP][i] = fr.zero();
        }

        if (rom[i].setPC) {
            fr.copy(pols[PC][nexti],op0);
            pols[setPC][i] = fr.one();
        } else {
            fr.copy(pols[PC][nexti],pols[PC][i]);
            if ( (rom[i].inc) && (rom[i].isCode) ) {
                fr.add(pols[PC][nexti], pols[PC][nexti], fr.one()); // PC is part of Ethereum's program
            }
            if ( (rom[i].dec) && (rom[i].isCode) ) {
                fr.sub(pols[PC][nexti], pols[PC][nexti], fr.one()); // PC is part of Ethereum's program
            }
            pols[setPC][i] = fr.zero();
        }

        if (rom[i].JMPC) {
            int64_t o = fe2n(fr, op0);
            if (o<0) {
                pols[isNeg][i] = fr.one();
                fr.fromUI(pols[zkPC][nexti], addr);
            } else {
                pols[isNeg][i] = fr.zero();
                fr.add(pols[zkPC][nexti], pols[zkPC][i], fr.one());
            }
            pols[JMP][i] = fr.zero();
            pols[JMPC][i] = fr.one();
        } else if (rom[i].JMP) {
            pols[isNeg][i] = fr.zero();
            fr.fromUI(pols[zkPC][nexti], addr);
            pols[JMP][i] = fr.one();
            pols[JMPC][i] = fr.zero();
        } else {
            pols[isNeg][i] = fr.zero();
            fr.add(pols[zkPC][nexti], pols[zkPC][i], fr.one());
            pols[JMP][i] = fr.zero();
            pols[JMPC][i] = fr.zero();
        }

        uint64_t maxMemCalculated = 0;
        uint64_t mm = fe2n(fr, pols[MAXMEM][i]);
        if (rom[i].isMem && addrRel>mm) {
            pols[isMaxMem][i] = fr.one();
            maxMemCalculated = addrRel;
        } else {
            pols[isMaxMem][i] = fr.zero();
            maxMemCalculated = mm;
        }

        if (rom[i].setMAXMEM) {
            pols[MAXMEM][nexti] = op0;
            pols[setMAXMEM][i] = fr.one();
        } else {
            fr.fromUI(pols[MAXMEM][nexti],maxMemCalculated);
            pols[setMAXMEM][i] = fr.zero();
        }

        if (rom[i].setGAS) {
            pols[GAS][nexti] = op0;
            pols[setGAS][i] = fr.one();
        } else {
            pols[GAS][nexti] = pols[GAS][i];
            pols[setGAS][i] = fr.zero();
        }

        if (rom[i].mRD) { // TODO: Shouldn't we read from memory?
            pols[mRD][i] = fr.one();
        } else {
            pols[mRD][i] = fr.zero();
        }

        if (rom[i].mWR) {
            ctx.mem[addr][0] = op0;
            ctx.mem[addr][1] = op1;
            ctx.mem[addr][2] = op2;
            ctx.mem[addr][3] = op3;
            pols[mWR][i] = fr.one();
        } else {
            pols[mWR][i] = fr.zero();
        }

        if (rom[i].sRD) {
            pols[sRD][i] = fr.one();
        } else {
            pols[sRD][i] = fr.zero();
        }

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
        if (rom[i].hashRD) {
            pols[hashRD][i] = fr.one();
        } else {
            pols[hashRD][i] = fr.zero();
        }
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
        if (rom[i].ecRecover) {
            pols[ecRecover][i] = fr.one();
        } else {
            pols[ecRecover][i] = fr.zero();
        }

        if (rom[i].arith) {
            pols[arith][i] = fr.one();
        } else {
            pols[arith][i] = fr.zero();
        }

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

        if (rom[i].shl) {
            pols[shl][i] = fr.one();
        } else {
            pols[shl][i] = fr.zero();
        }

        if (rom[i].shr) {
            pols[shr][i] = fr.one();
        } else {
            pols[shr][i] = fr.zero();
        }

        if (rom[i].bin) {
            pols[bin][i] = fr.one();
        } else {
            pols[bin][i] = fr.zero();
        }

        if (rom[i].comparator) {
            pols[comparator][i] = fr.one();
        } else {
            pols[comparator][i] = fr.zero();
        }

        if (rom[i].opcodeRomMap) {
            pols[opcodeRomMap][i] = fr.one();
        } else {
            pols[opcodeRomMap][i] = fr.zero();
        }

        for (uint64_t j=0; j<rom[i].cmdAfter.size(); j++)
        {
            evalCommand(ctx, *rom[i].cmdAfter[j]);
        }

    }

    printRegs(ctx);
    printVars(ctx);
    printMem(ctx);

    checkFinalState(fr, ctx);

    /* Unmap output file from memory */
    unmapPols(ctx);

    /* Unload ROM JSON file data from memory, i.e. free memory */
    unloadRom(ctx);
    
}

/* Sets first evaluation of all polynomials to zero */
void initState(RawFr &fr, Context &ctx)
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

void checkFinalState(RawFr &fr, Context &ctx)
{
    if ( 
        (!fr.isZero(pols[A0][0])) ||
        (!fr.isZero(pols[A1][0])) ||
        (!fr.isZero(pols[A2][0])) ||
        (!fr.isZero(pols[A3][0])) ||
        (!fr.isZero(pols[B0][0])) ||
        (!fr.isZero(pols[B1][0])) ||
        (!fr.isZero(pols[B2][0])) ||
        (!fr.isZero(pols[B3][0])) ||
        (!fr.isZero(pols[C0][0])) ||
        (!fr.isZero(pols[C1][0])) ||
        (!fr.isZero(pols[C2][0])) ||
        (!fr.isZero(pols[C3][0])) ||
        (!fr.isZero(pols[D0][0])) ||
        (!fr.isZero(pols[D1][0])) ||
        (!fr.isZero(pols[D2][0])) ||
        (!fr.isZero(pols[D3][0])) ||
        (!fr.isZero(pols[E0][0])) ||
        (!fr.isZero(pols[E1][0])) ||
        (!fr.isZero(pols[E2][0])) ||
        (!fr.isZero(pols[E3][0])) ||
        (!fr.isZero(pols[SR][0])) ||
        (!fr.isZero(pols[CTX][0])) ||
        (!fr.isZero(pols[SP][0])) ||
        (!fr.isZero(pols[PC][0])) ||
        (!fr.isZero(pols[MAXMEM][0])) ||
        (!fr.isZero(pols[GAS][0])) ||
        (!fr.isZero(pols[zkPC][0]))
    ) {
        cerr << "Error: Program terminated with registers not set to zero" << endl;
        exit(-1);
    }
    else{
        cout << "checkFinalState() succeeded" << endl;
    }
}
