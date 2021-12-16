#include "stark_gen.hpp"
#include "stark_struct.hpp"
#include "poseidon_opt/poseidon_opt.hpp"
#include "scalar.hpp"

using namespace std;
using json = nlohmann::json;

void StarkGen::generate (Pols &cmPols, Pols &constPols, Pols &constTree)
{
    Poseidon_opt poseidon;
    
/*  const E = new ExpressionOps();

    const M = new Merkle(16, poseidon, poseidon.F);
    */

    uint64_t groupSize = 1 << (NBITS + EXTENDED_BITS - starkStruct[0].nBits);
    uint64_t nGroups = 1 << starkStruct[0].nBits;

    /*const MGPC = new MerkleGroupMultipol(M, nGroups, groupSize, pil.nConstants);

    const fri = new FRI(F, poseidon, 16, Nbits+extendBits, Nbits, starkStruct );

    const transcript = new Transcript(poseidon, poseidon.F);*/

    if (pil.nCommitments!=NPOLS)
    {
        cerr << "Error: unexpected number of committed polynomials: " << pil.nCommitments << endl;
        exit(-1);
    }

    // Build ZHInv

    //const zhInv = buildZhInv(F, Nbits, extendBits);
    buildZhInv(fr, NBITS, EXTENDED_BITS);

    /*for (let i=0; i<pil.nCommitments; i++) pols.cm[i] = {};
    for (let i=0; i<pil.expressions.length; i++) pols.exps[i] = {};
    for (let i=0; i<pil.nQ; i++) pols.q[i] = {};
    for (let i=0; i<pil.nConstants; i++) pols.const[i] = {};*/

    // Init pols instance vectors
    for (uint64_t i=0; i<pil.nCommitments;       i++) pols.cm.push_back(StarkCm());
    for (uint64_t i=0; i<pil.expressions.size(); i++) pols.exps.push_back(StarkExpression());
    for (uint64_t i=0; i<pil.nQ;                 i++) pols.q.push_back(StarkQ());
    for (uint64_t i=0; i<pil.nConstants;         i++) pols.constants.push_back(StarkConstants());

// 1.- Prepare commited polynomials. 

    // Convert all committed polynomials from their original types to field elements, and store them in pols.cm[i].v_n[j]
    RawFr::Element fe;
    for (uint64_t i=0; i<cmPols.size; i++)
    {
        cout << "Preparing committed polynomial " << i << endl;

        switch(cmPols.orderedPols[i]->elementType)
        {
            case et_bool:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    u82fe(fr, fe, ((PolBool *)cmPols.orderedPols[i])->pData[j]);
                    pols.cm[i].v_n.push_back(fe);
                }
                break;
            case et_s8:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    s82fe(fr, fe, ((PolS8 *)cmPols.orderedPols[i])->pData[j]);
                    pols.cm[i].v_n.push_back(fe);
                }
                break;
            case et_u8:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    u82fe(fr, fe, ((PolU8 *)cmPols.orderedPols[i])->pData[j]);
                    pols.cm[i].v_n.push_back(fe);
                }
                break;
            case et_s16:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    s162fe(fr, fe, ((PolS16 *)cmPols.orderedPols[i])->pData[j]);
                    pols.cm[i].v_n.push_back(fe);
                }
                break;
            case et_u16:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    u162fe(fr, fe, ((PolU16 *)cmPols.orderedPols[i])->pData[j]);
                    pols.cm[i].v_n.push_back(fe);
                }
                break;
            case et_s32:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    s322fe(fr, fe, ((PolS32 *)cmPols.orderedPols[i])->pData[j]);
                    pols.cm[i].v_n.push_back(fe);
                }
                break;
            case et_u32:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    u322fe(fr, fe, ((PolU32 *)cmPols.orderedPols[i])->pData[j]);
                    pols.cm[i].v_n.push_back(fe);
                }
                break;
            case et_s64:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    s642fe(fr, fe, ((PolS64 *)cmPols.orderedPols[i])->pData[j]);
                    pols.cm[i].v_n.push_back(fe);
                }
                break;
            case et_u64:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    u642fe(fr, fe, ((PolU64 *)cmPols.orderedPols[i])->pData[j]);
                    pols.cm[i].v_n.push_back(fe);
                }
                break;
            case et_field:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    pols.cm[i].v_n.push_back(((PolFieldElement *)cmPols.orderedPols[i])->pData[j]);
                }
                break;
            default:
                cerr << "Error: StarkGen::generate() found committed polynomial of unknown type: " << cmPols.orderedPols[i]->elementType << endl;
                exit(-1);
        }
        
    }

    /*for (let i=0; i<constPols.length; i++) {
        console.log(`Preparing constant polynomial ${i}`);
        if (constPols[i].length!= N) {
            throw new Error(`Constant Polynomial ${i} does not have the right size: ${constPols[i].length} and should be ${N}`);
        }
        pols.const[i].v_2ns = [];
        for (let j=0; j<(N<<extendBits); j++) pols.const[i].v_2ns[j] = MGPC.getElement(constTree, i, j);
        pols.const[i].v_n = [];
        for (let j=0; j<N; j++) pols.const[i].v_n[j] = F.e(constPols[i][j]);
    }*/

    for (uint64_t i=0; i<constPols.size; i++)
    {
        cout << "Preparing constant polynomial " << i << endl;
        for (uint64_t j=0; j<(NEVALUATIONS<<EXTENDED_BITS); j++)
        {
            //fe = MGPC.getElement(constTree, i, j);  TODO: Migrate this code
            pols.constants[i].v_2ns.push_back(fe);
        }
        switch(constPols.orderedPols[i]->elementType)
        {
            case et_bool:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    u82fe(fr, fe, ((PolBool *)constPols.orderedPols[i])->pData[j]);
                    pols.constants[i].v_n.push_back(fe);
                }
                break;
            case et_s8:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    s82fe(fr, fe, ((PolS8 *)constPols.orderedPols[i])->pData[j]);
                    pols.constants[i].v_n.push_back(fe);
                }
                break;
            case et_u8:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    u82fe(fr, fe, ((PolU8 *)constPols.orderedPols[i])->pData[j]);
                    pols.constants[i].v_n.push_back(fe);
                }
                break;
            case et_s16:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    s162fe(fr, fe, ((PolS16 *)constPols.orderedPols[i])->pData[j]);
                    pols.constants[i].v_n.push_back(fe);
                }
                break;
            case et_u16:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    u162fe(fr, fe, ((PolU16 *)constPols.orderedPols[i])->pData[j]);
                    pols.constants[i].v_n.push_back(fe);
                }
                break;
            case et_s32:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    s322fe(fr, fe, ((PolS32 *)constPols.orderedPols[i])->pData[j]);
                    pols.constants[i].v_n.push_back(fe);
                }
                break;
            case et_u32:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    u322fe(fr, fe, ((PolU32 *)constPols.orderedPols[i])->pData[j]);
                    pols.constants[i].v_n.push_back(fe);
                }
                break;
            case et_s64:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    s642fe(fr, fe, ((PolS64 *)constPols.orderedPols[i])->pData[j]);
                    pols.constants[i].v_n.push_back(fe);
                }
                break;
            case et_u64:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    u642fe(fr, fe, ((PolU64 *)constPols.orderedPols[i])->pData[j]);
                    pols.constants[i].v_n.push_back(fe);
                }
                break;
            case et_field:
                for (uint64_t j=0; j<NEVALUATIONS; j++)
                {
                    pols.constants[i].v_n.push_back(((PolFieldElement *)constPols.orderedPols[i])->pData[j]);
                }
                break;
            default:
                cerr << "Error: StarkGen::generate() found const polynomial of unknown type: " << constPols.orderedPols[i]->elementType << endl;
                exit(-1);
        }
    }

// This will calculate all the Q polynomials and extend commits
/*
    await prepareCommitsAndQs(F, pil, pols, extendBits);

    for (let i=0; i<pil.nCommitments; i++) {
            if (!pols.cm[i].v_2ns) {
                console.log(`Extending polynomial ${i}`);
                pols.cm[i].v_2ns = await extendPol(F, pols.cm[i].v_n, extendBits);
            }
        }*/

        /*for (let i=0; i<pil.expressions.length; i++) {
            if (typeof pil.expressions[i].idQ != "undefined") {
                await calculateExpression(i, "v_2ns");
                console.log(`Calculating q ${i}`);
            }
        }
        */
    for (uint64_t i=0; i<pil.expressions.size(); i++)
    {
        if (pil.expressions[i].bIdQPresent)
        {
            calculateExpression(i, "v_2ns");
        }
    }

}



void StarkGen::buildZhInv (RawFr &fr, uint64_t Nbits, uint64_t extendBits)
{
    RawFr::Element w = fr.one();
    RawFr::Element sn;
    fr.fromUI(sn, 25); // F.shift = 25 = 0x16 TODO: Move to constant, or something else.  Confirm with Jordi that this is a constant.
    for (uint64_t i=0; i<Nbits; i++) fr.square(sn, sn);
    for (uint64_t i=0; i<(1<<extendBits); i++)
    {
        RawFr::Element aux1, aux2;
        fr.mul(aux1, sn, w);
        fr.sub(aux2, aux1, fr.one());
        fr.inv(aux1, aux2);
        ZHInv.push_back(aux1);
        //fr.mul(w, w, fr.w[extendBits]); TODO: Port this code.  What is fr.w?
    }
}

void StarkGen::getZhInv (RawFr::Element &fe, uint64_t i)
{
    fe = ZHInv[i%ZHInv.size()];
}

void StarkGen::eval(Expression &exp, const string &subPol, vector<RawFr::Element> &r)
{

}

/*
    function eval(exp, subPol) {
        if (exp.op == "add") {
            const a = eval(exp.values[0], subPol);
            const b = eval(exp.values[1], subPol);
            const r = new Array(a.length);
            for (let i=0; i<a.length; i++) r[i] = F.add(a[i], b[i]);
            debug("add", r, a, b);
            return r;
        } else if (exp.op == "sub") {
            const a = eval(exp.values[0], subPol);
            const b = eval(exp.values[1], subPol);
            const r = new Array(a.length);
            for (let i=0; i<a.length; i++) r[i] = F.sub(a[i], b[i]);
            debug("sub", r, a, b);
            return r;
        } else if (exp.op == "mul") {
            const a = eval(exp.values[0], subPol);
            const b = eval(exp.values[1], subPol);
            const r = new Array(a.length);
            for (let i=0; i<a.length; i++) r[i] = F.mul(a[i], b[i]);
            debug("mul", r, a, b);
            return r;
        } else if (exp.op == "addc") {
            const a = eval(exp.values[0], subPol);
            const c = F.e(exp.const);
            const r = new Array(a.length);
            for (let i=0; i<a.length; i++) r[i] = F.add(a[i], c);
            debug("addc", r, a, null, c);
            return r;
        } else if (exp.op == "mulc") {
            const a = eval(exp.values[0], subPol);
            const c = F.e(exp.const);
            const r = new Array(a.length);
            for (let i=0; i<a.length; i++) r[i] = F.mul(a[i], c);
            debug("mulc", r, a, null, c);
            return r;
        } else if (exp.op == "neg") {
            const a = eval(exp.values[0], subPol);
            const r = new Array(a.length);
            for (let i=0; i<a.length; i++) r[i] = F.neg(a[i]);
            debug("neg", r, a);
            return r;
        } else if (exp.op == "cm") {
            let r = pols.cm[exp.id][subPol];
            if (exp.next) r = getPrime(r, subPol);
            return r;
        } else if (exp.op == "const") {
            let r = pols.const[exp.id][subPol];
            if (exp.next) r = getPrime(r, subPol);
            return r;
        } else if (exp.op == "exp") {
            let r = pols.exps[exp.id][subPol];
            if (exp.next) r = getPrime(r, subPol);
            return r;
        } else if (exp.op == "q") {
            let r = pols.q[exp.id][subPol];
            if (exp.next) r = getPrime(r, subPol);
            return r;
        } else if (exp.op == "number") {
            const N = pols.const[0].length;
            const v = F.e(exp.value);
            const r = new Array(N);
            for (let i=0; iN; i++) r[i] = v;
            return r;
        } else {
            throw new Error(`Invalid op: ${exp.op}`);
        }
    }
    */


/*
    function getPrime(p, subPol) {
        if (subPol == "v_n") {
            const r = p.slice(1);
            r[p.length-1] = p[0];
            return r;
        } else if (subPol == "v_2ns") {
            const r = p.slice(1<<extendBits);
            for (let i=0; i<(1<<extendBits); i++) {
                r[p.length - (1<<extendBits) + i] = p[i];
            }
            return r;
        } else {
            throw new Error(`Invalid subpol: ${subPol}`);
        }
    }
*/
void StarkGen::calculateExpression (uint64_t expId, const string &subPol)
{
    cout << "calculateExpression: " << expId << endl;

    if (subPol=="v_2ns")
    {
        if (pols.exps[expId].v_2ns.size() > 0)
            return;
        if (pil.expressions[expId].bIdQPresent)
            calculateExpression(expId, "v_n");
    }
    else if (subPol=="v_n")
    {
        if (pols.exps[expId].v_n.size() > 0) return;
    }
    else
    {
        cerr << "Error: StarkGen::calculateExpression() called with invalid subPol: " << subPol << endl;
        exit(-1);
    }

    calculateDependencies(pil.expressions[expId], subPol);

    //eval(pil.expressions[expId], subPol);
    /*
            if ((subPol == "v_2ns")&&(expId === debugExpId)) debugging = true;
        const p = eval(pil.expressions[expId], subPol);
        debugging = false;

        if (subPol == "v_2ns") {
            if (typeof pil.expressions[expId].idQ !== "undefined") {
                const r = await extendPol(F, pols.exps[expId].v_n, extendBits);
                const q = new Array(p.length);
                for (let i=0; i<p.length; i++) {
                    q[i] = F.mul(F.sub(p[i], r[i]), zhInv(i))
                }
                pols.exps[expId].v_2ns = r;
                pols.q[pil.expressions[expId].idQ].v_2ns = q;
            } else {
                pols.exps[expId].v_2ns = p;
            }
            return pols.exps[expId].v_2ns;
        } else if (subPol == "v_n")  {
            pols.exps[expId].v_n = p;
            return pols.exps[expId].v_n;
        }
    */
}

void StarkGen:: calculateDependencies(const Expression &exp, const string &subPol)
{
    if (exp.op == "exp")
    {
        if (!exp.bIdPresent)
        {
            cerr << "Error: id expected but not present" << endl;
            exit(-1);
        }
        calculateExpression(exp.id, subPol);
    }
    for (uint64_t i=0; i<exp.values.size(); i++)
    {
        calculateDependencies(exp.values[i], subPol);
    }
}