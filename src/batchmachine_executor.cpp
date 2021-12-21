#include "batchmachine_executor.hpp"
#include "poseidon_opt/poseidon_opt.hpp"
#include "scalar.hpp"
#include "compare_fe.hpp"
#include "utils.hpp"

void batchMachineExecutor (RawFr &fr, Mem &mem, Script &script)
{
    Poseidon_opt poseidon;

    for (uint64_t i=0; i<script.program.size(); i++)
    {
        if (i==213)
            break;
        Program program = script.program[i];
        cout << "Program line: " << i << " operation: " << op2string(program.op) << " result: " << program.result << endl;

        switch (program.op)
        {
        case op_field_set:
        {
            zkassert(mem[program.result].type == rt_field);
            zkassert(program.value.size() > 0);

            fr.fromString(mem[program.result].fe, program.value);

            printReference(fr, mem[program.result]);
            break;
        }
        case op_field_add:
        {
            zkassert(mem[program.result].type == rt_field);
            zkassert(program.values.size() == 2);
            zkassert(mem[program.values[0]].type == rt_field);
            zkassert(mem[program.values[1]].type == rt_field);

            fr.add(mem[program.result].fe, mem[program.values[0]].fe, mem[program.values[1]].fe);

            printReference(fr, mem[program.result]);
            break;
        }
        case op_field_sub:
        {
            zkassert(mem[program.result].type == rt_field);
            zkassert(program.values.size() == 2);
            zkassert(mem[program.values[0]].type == rt_field);
            zkassert(mem[program.values[1]].type == rt_field);

            fr.sub(mem[program.result].fe, mem[program.values[0]].fe, mem[program.values[1]].fe);

            printReference(fr, mem[program.result]);
            break;
        }
        case op_field_neg:
        {
            zkassert(mem[program.result].type == rt_field);
            zkassert(program.values.size() == 1);
            zkassert(mem[program.values[0]].type == rt_field);

            fr.neg(mem[program.result].fe, mem[program.values[0]].fe);

            printReference(fr, mem[program.result]);
            break;
        }
        case op_field_mul:
        {
            zkassert(mem[program.result].type == rt_field);
            zkassert(program.values.size() == 2);
            zkassert(mem[program.values[0]].type == rt_field);
            zkassert(mem[program.values[1]].type == rt_field);

            fr.mul(mem[program.result].fe, mem[program.values[0]].fe, mem[program.values[1]].fe);

            printReference(fr, mem[program.result]);
            break;
        }
        case op_pol_add:
        {
            zkassert(mem[program.result].type == rt_pol);
            zkassert(program.values.size() == 2);
            zkassert(mem[program.values[0]].type == rt_pol);
            zkassert(mem[program.values[1]].type == rt_pol);
            zkassert(mem[program.result].N == mem[program.values[0]].N);
            zkassert(mem[program.result].N == mem[program.values[1]].N);

            for (uint64_t j=0; j<program.N; j++)
            {
                fr.add(mem[program.result].pPol[j], mem[program.values[0]].pPol[j], mem[program.values[1]].pPol[j]);
            }

            printReference(fr, mem[program.result]);
            break;
        }
        case op_pol_sub:
        {
            zkassert(mem[program.result].type == rt_pol);
            zkassert(program.values.size() == 2);
            zkassert(mem[program.values[0]].type == rt_pol);
            zkassert(mem[program.values[1]].type == rt_pol);
            zkassert(mem[program.result].N == mem[program.values[0]].N);
            zkassert(mem[program.result].N == mem[program.values[1]].N);

            for (uint64_t j=0; j<program.N; j++)
            {
                fr.sub(mem[program.result].pPol[j], mem[program.values[0]].pPol[j], mem[program.values[1]].pPol[j]);
            }

            printReference(fr, mem[program.result]);
            break;
        }
        case op_pol_neg:
        {
            zkassert(mem[program.result].type == rt_pol);
            zkassert(program.values.size() == 1);
            zkassert(mem[program.values[0]].type == rt_pol);
            zkassert(mem[program.result].N == mem[program.values[0]].N);

            for (uint64_t j=0; j<program.N; j++)
            {
                fr.neg(mem[program.result].pPol[j], mem[program.values[0]].pPol[j]);
            }

            printReference(fr, mem[program.result]);
            break;
        }
        case op_pol_mul:
        {
            zkassert(mem[program.result].type == rt_pol);
            zkassert(program.values.size() == 2);
            zkassert(mem[program.values[0]].type == rt_pol);
            zkassert(mem[program.values[1]].type == rt_pol);
            zkassert(mem[program.result].N == mem[program.values[0]].N);
            zkassert(mem[program.result].N == mem[program.values[1]].N);

            for (uint64_t j=0; j<program.N; j++)
            {
                fr.mul(mem[program.result].pPol[j], mem[program.values[0]].pPol[j], mem[program.values[1]].pPol[j]);
            }

            printReference(fr, mem[program.result]);
            break;
        }
        case op_pol_addc:
        {
            zkassert(mem[program.result].type == rt_pol);
            zkassert(program.values.size() == 1);
            zkassert(mem[program.values[0]].type == rt_pol);
            zkassert(mem[program.result].N == mem[program.values[0]].N);

            RawFr::Element fe;
            fr.fromUI(fe, program.constant);
            for (uint64_t j=0; j<program.N; j++)
            {
                fr.add(mem[program.result].pPol[j], mem[program.values[0]].pPol[j], fe);
            }

            printReference(fr, mem[program.result]);
            break;
        }
        case op_pol_mulc:
        {
            zkassert(mem[program.result].type == rt_pol);
            zkassert(program.values.size() == 1);
            zkassert(mem[program.values[0]].type == rt_pol);
            zkassert(mem[program.result].N == mem[program.values[0]].N);

            RawFr::Element fe;
            fr.fromUI(fe, program.constant);
            for (uint64_t j=0; j<program.N; j++)
            {
                fr.mul(mem[program.result].pPol[j], mem[program.values[0]].pPol[j], fe);
            }

            printReference(fr, mem[program.result]);
            break;
        }
        case op_pol_grandProduct:
        {
            zkassert(mem[program.result].type == rt_pol);
            zkassert(program.values.size() == 1);
            zkassert(mem[program.values[0]].type == rt_pol);
            zkassert(mem[program.result].N == mem[program.values[0]].N);

            mem[program.result].pPol[0] = fr.one();
            for (uint64_t j=1; j<program.N; j++)
            {
                fr.mul(mem[program.result].pPol[j], mem[program.values[0]].pPol[j-1], mem[program.result].pPol[j-1]);
            }

            printReference(fr, mem[program.result]);
            break;
        }
        case op_pol_batchInverse:
        {
            zkassert(mem[program.result].type == rt_pol);
            zkassert(program.values.size() == 1);
            zkassert(mem[program.values[0]].type == rt_pol);
            zkassert(mem[program.result].N == mem[program.values[0]].N);

            batchInverse(fr, mem[program.values[0]], mem[program.result]);

            printReference(fr, mem[program.result]);
            break;
        }
        case op_pol_rotate:
        {
            zkassert(mem[program.result].type == rt_pol);
            zkassert(program.values.size() == 1);
            zkassert(mem[program.values[0]].type == rt_pol);
            zkassert(mem[program.result].N == mem[program.values[0]].N);

            for (uint64_t j=0; j<program.N; j++)
            {
                mem[program.result].pPol[j] = mem[program.values[0]].pPol[(j+program.shift)%program.N];
            }

            printReference(fr, mem[program.result]);
            break;
        }
        case op_pol_extend:
        {
            /*mem[l.result] = await extendPol(F, mem[l.values[0]], l.extendBits);*/
            break;
        }
        case op_pol_getEvaluation:
        {
            zkassert(mem[program.result].type == rt_field);
            zkassert(mem[program.p].type == rt_pol);
            zkassert(program.idx < mem[program.p].N);

            mem[program.result].fe = mem[program.p].pPol[program.idx];

            printReference(fr, mem[program.result]);
        }
        case op_treeGroupMultipol_extractPol:
        {
            /*const MGP = new MerkleGroupMultipol(M, l.nGroups, l.groupSize, l.nPols);
                    const N = l.nGroups*l.groupSize;
                    const p = new Array(N);
                    for (let j=0; j<N; j++) {
                        p[j] = MGP.getElement(mem[l.tree], l.polIdx, j); 
                    }
                    mem[l.result] = p;*/
            break;
        }
        case op_treeGroupMultipol_merkelize:
        {
            /*const MGP = new MerkleGroupMultipol(M, l.nGroups, l.groupSize, l.pols.length);
                    const pols = [];
                    for (let j=0; j<l.pols.length; j++) {
                        pols.push(mem[l.pols[j]]);
                    }
                    mem[l.result] = MGP.merkelize(pols);*/
            break;
        }
        case op_treeGroupMultipol_root:
        {
            /*const MGP = new MerkleGroupMultipol(M, l.nGroups, l.groupSize, l.nPols);
                    mem[l.result] = MGP.root(mem[l.tree]);*/
            break;
        }
        case op_treeGroupMultipol_getGroupProof:
        {
            /*const MGP = new MerkleGroupMultipol(M, l.nGroups, l.groupSize, l.nPols);
                    mem[l.result] = MGP.getGroupProof(mem[l.tree], mem[l.idx]);*/
            break;
        }
        case op_treeGroup_merkelize:
        {
            /*const MG = new MerkleGroup(M, l.nGroups, l.groupSize);
                    mem[l.result] = MG.merkelize(mem[l.pol]);*/
            break;
        }
        case op_treeGroup_root:
        {
            /*const MG = new MerkleGroup(M, l.nGroups, l.groupSize);
                    mem[l.result] = MG.root(mem[l.tree]);*/
            break;
        }
        case op_treeGroup_getElementProof:
        {
            /*const MG = new MerkleGroup(M, l.nGroups, l.groupSize);
                    mem[l.result] = MG.getElementProof(mem[l.tree], mem[l.idx]);*/
            break;
        }
        case op_treeGroup_getGroupProof:
        {
            /*const MG = new MerkleGroup(M, l.nGroups, l.groupSize);
                    mem[l.result] = MG.getGroupProof(mem[l.tree], mem[l.idx]);*/
            break;
        }
        case op_idxArrayFromFields:
        {
            zkassert(program.fields.size() > 0);
            zkassert(mem[program.result].type == rt_idxArray);

            vector<vector<uint8_t>> fields;
            for (uint64_t j=0; j<program.fields.size(); j++)
            {
                zkassert(mem[program.fields[j]].type == rt_field);

                mpz_class s;
                fe2scalar(fr, s, mem[program.fields[j]].fe);
                vector<uint8_t> bits;
                scalar2bits(s, bits);
                fields.push_back(bits);
            }

            uint64_t curField = 0;
            uint64_t curBit = 0;
            for (uint64_t i=0; i<program.n; i++)
            {
                uint32_t a = 0;
                for (uint64_t j=0; j<program.nBits; j++)
                {
                    if (fields[curField][curBit]) a = a + (1<<j);
                    curBit++;
                    if (curBit == 253)
                    {
                        curBit = 0;
                        curField++;
                    }
                }
                mem[program.result].pIdxArray[i] = a;
            }

            printReference(fr, mem[program.result]);
            break;
        }
        case op_idxArray_get:
        {
            zkassert(mem[program.result].type == rt_int);
            zkassert(mem[program.idxArray].type == rt_idxArray);
            zkassert(program.pos <= mem[program.idxArray].N);

            mem[program.result].integer = mem[program.idxArray].pIdxArray[program.pos];

            printReference(fr, mem[program.result]);
            break;
        }
        case op_idx_addMod:
        {
            zkassert(mem[program.result].type == rt_int);
            zkassert(mem[program.idx].type == rt_int);

            mem[program.result].integer = (uint32_t)((uint64_t(mem[program.idx].integer) + program.add) % program.mod);

            printReference(fr, mem[program.result]);
            break;
        }
        case op_calculateH1H2:
        {
            //calculateH1H2(fr, mem[program.f], mem[program.t], mem[program.resultH1], mem[program.resultH2]); TODO: Review implementation with Jordi
            break;
        }
        case op_friReduce:
        {
            /*let acc = F.e(l.shiftInv);
                    let w = F.e(l.w);
                    let nX = 1 << l.reduceBits;
                    let pol2N = l.N/nX;
                    const pol2_e = new Array(pol2N);
                    for (let g = 0; g<pol2N; g++) {
                        const ppar = new Array(nX);
                        for (let i=0; i<nX; i++) {
                            ppar[i] = mem[l.pol][(i*pol2N)+g];
                        }
                        const ppar_c = await F.ifft(ppar);

                        polMulAxi(F, ppar_c, F.one, acc);    // Multiplies coefs by 1, shiftInv, shiftInv^2, shiftInv^3, ......
        
                        pol2_e[g] = evalPol(F, ppar_c, mem[l.specialX]);
                        acc = F.mul(acc, w);
                    }
                    mem[l.result] = pol2_e;*/
            break;
        }
        case op_hash:
        {
            zkassert(program.values.size() > 0)
            zkassert(mem[program.result].type == rt_field);

            vector<RawFr::Element> keyV;
            for (uint64_t j=0; j<program.values.size(); j++)
            {
                zkassert(mem[program.values[j]].type == rt_field);
                keyV.push_back(mem[program.values[j]].fe);
            }
            poseidon.hash(keyV, &mem[program.result].fe);

            printReference(fr, mem[program.result]);
            break;
        }
        case op_log:
        {
            zkassert(program.msg.size() > 0);
            cout << "BME log: " << program.msg << endl;
            /*      if (typeof(l.refId)!= "undefined") {  TODO: Ask Jordi is we need to support this (no refID occurrences in script)
                    const o = refToObject(F, mem, l.ref);
                    console.log(JSON.stringify(o, null, 1));
                }*/
            break;
        }
        default:
        {
            cerr << "Error: batchMachineExecutor() found unsupported operation: " << program.op << " at program line: " << i << endl;
            exit(-1);
        }        
    }
    }

    //return dereference(F, mem, script.output); TODO: when output format is final
    json j = dereference(fr, mem, script.output);

    cout << "batchMachineExecutor() build proof:" << endl;
    cout << j.dump() << endl;
}





/*
function dereference(F, mem, o) {
    if (Array.isArray(o)) {
        const res = [];
        for (let i=0; i<o.length; i++) {
            res[i] = dereference(F, mem, o[i]);
        }
        return res;
    } else if (typeof o === "object") {
        if (o.$Ref) {
            return refToObject(F, mem, o);
        } else {
            const res = {};
            const keys = Object.keys(o);
            keys.forEach( (k) => {
                res[k] = dereference(F, mem, o[k]);
            });
            return res;
        }
    } else {
        return o;
    }
}
*/

json dereference(RawFr &fr, Mem &mem, Output &output)
{
    if (output.isArray())
    {
        json j;
        for (uint64_t i=0; i<output.array.size(); i++)
        {
            j[i] = dereference(fr, mem, output.array[i]);
        }
        return j;
    }
    else
    {
        return refToObject(fr, mem, output.ref);
    }
}

/*
function refToObject(F, mem, ref) {
    if (ref.type == "int") {
        return mem[ref.id];
    } else if (ref.type == "field") {
        return  F.toString(mem[ref.id]);
    } else if (ref.type == "pol") {
        return  stringifyFElements(F, mem[ref.id]);
    } else if (ref.type == "treeGroup_groupProof") {
        return  stringifyFElements(F, mem[ref.id]);
    } else if (ref.type == "treeGroup_elementProof") {
        return  stringifyFElements(F, mem[ref.id]);
    } else if (ref.type == "treeGroupMultipol_groupProof") {
        return  stringifyFElements(F, mem[ref.id]);
    } else {
        throw new Error('Cannot stringify ${ref.type}');
    }
}
*/

json refToObject(RawFr &fr, Mem &mem, Reference &ref)
{
    switch (ref.type)
    {
        case rt_int:
        {
            return mem[ref.id].integer;
        }
        case rt_field:
        {
            return fr.toString(mem[ref.id].fe, 16);
        }
        case rt_pol:
        {
            json j;
            for (uint64_t i=0; i<ref.N; i++)
            {
                j.push_back(fr.toString(mem[ref.id].pPol[i], 16));
            }
        }
        case rt_treeGroup_groupProof:
        case rt_treeGroup_elementProof:
        case rt_treeGroupMultipol_groupProof:
            return "TODO";
        default:
            cerr << "Error: refToObject cannot return JSON object of ref.type: " << ref.type << endl;
            exit(-1);
    }
}

void calculateH1H2 (RawFr &fr, Reference &f, Reference &t, Reference &h1, Reference &h2)
{
    zkassert(t.type == rt_pol);
    zkassert(f.type == rt_pol);
    zkassert(h1.type == rt_pol);
    zkassert(h2.type == rt_pol);
    zkassert(h1.N == f.N);
    zkassert(h2.N == f.N);

    map<RawFr::Element, uint64_t, CompareFe> idx_t;
    multimap<RawFr::Element, uint64_t, CompareFe> s;

    for (uint64_t i=0; i<(uint32_t)t.N; i++)
    {
        idx_t[t.pPol[i]] = i;
        s.insert(pair<RawFr::Element,uint64_t>(t.pPol[i],i));
    }

    for (uint64_t i=0; i<f.N; i++)
    {
        if (idx_t.find(f.pPol[i]) == idx_t.end())
        {
            cerr << "Error: calculateH1H2() Number not included: " << fr.toString(f.pPol[i], 16) << endl;
            exit(-1);
        }
        uint64_t idx = idx_t[f.pPol[i]];
        s.insert(pair<RawFr::Element,uint64_t>(f.pPol[i],idx));
    }

    multimap<RawFr::Element, uint64_t>::iterator it;
    uint64_t i=0;

    for (it=s.begin(); it!=s.end(); it++, i++)
    {
        if ((i&1) == 0)
        {
            h1.pPol[i/2] = it->first;
        }
        else
        {
            h2.pPol[i/2] = it->first;
        }
    }

    /*
        const idx_t = {};
    const s = [];
    for (i=0; i<t.length; i++) {
        idx_t[t[i]]=i;
        s.push([t[i], i]);
    }
    for (i=0; i<f.length; i++) {
        const idx = idx_t[f[i]];
        if (isNaN(idx)) {
            throw new Error(`Number not included: ${F.toString(f[i])}`);
        }
        s.push([f[i], idx]);
    }

    s.sort( (a, b) => a[1] - b[1] );

    const h1 = new Array(f.length);
    const h2 = new Array(f.length);
    for (let i=0; i<f.length; i++) {
        h1[i] = s[2*i][0];
        h2[i] = s[2*i+1][0];
    }

    return [h1, h2];
    */
}

void batchInverse (RawFr &fr, Reference &source, Reference &result)
{
    zkassert(source.type == rt_pol);
    zkassert(result.type == rt_pol);
    zkassert(source.N == result.N);
    zkassert(source.N >= 2);

    uint64_t N = source.N;

    // Calculate the products: [a, ab, abc, ... abc..xyz]
    RawFr::Element * pProduct;
    pProduct = (RawFr::Element *)malloc(N*sizeof(RawFr::Element));
    if ( pProduct == NULL)
    {
        cerr << "Error: batchInverse() failed calling malloc of bytes: " << N*sizeof(RawFr::Element) << endl;
        exit(-1);
    }
    pProduct[0] = source.pPol[0]; // a
    for (uint64_t i=1; i<N; i++)
    {
        fr.mul(pProduct[i], pProduct[i-1], source.pPol[i]);
    }

    // Calculate the inversions: [1/a, 1/ab, 1/abc, ... 1/abc..xyz]
    RawFr::Element * pInvert;
    pInvert = (RawFr::Element *)malloc(N*sizeof(RawFr::Element));
    if ( pInvert == NULL)
    {
        cerr << "Error: batchInverse() failed calling malloc of bytes: " << N*sizeof(RawFr::Element) << endl;
        exit(-1);
    }    
    fr.inv(pInvert[N-1], pProduct[N-1]);
    for (uint64_t i = N-1; i>0; i--)
    {
        fr.mul(pInvert[i-1], pInvert[i], source.pPol[i]);
    }

    // Generate the output
    result.pPol[0] = pInvert[0];
    for (uint64_t i=1; i<N; i++)
    {
        fr.mul(result.pPol[i], pInvert[i], pProduct[i-1]);
    }

    // Free memory
    free(pProduct);
    free(pInvert);
}
