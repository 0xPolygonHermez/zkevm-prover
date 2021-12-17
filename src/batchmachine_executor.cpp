#include "batchmachine_executor.hpp"
#include "poseidon_opt/poseidon_opt.hpp"

void batchMachineExecutor (RawFr &fr, Mem &mem, Script &script)
{
    Poseidon_opt poseidon;

    for (uint64_t i=0; i<script.program.size(); i++)
    {
        Program program = script.program[i];
        cout << "Program line: " << i << " operation: " << program.op << endl;

        if (program.op == "field_set")
        {
            zkassert(mem[program.result].type == rt_field);
            zkassert(program.value.size() > 0);

            fr.fromString(mem[program.result].fe, program.value);
        }
        else if (program.op == "field_add")
        {
            zkassert(mem[program.result].type == rt_field);
            zkassert(program.values.size() == 2);
            zkassert(mem[program.values[0]].type == rt_field);
            zkassert(mem[program.values[1]].type == rt_field);

            fr.add(mem[program.result].fe, mem[program.values[0]].fe, mem[program.values[1]].fe);
        }
        else if (program.op == "field_sub")
        {
            zkassert(mem[program.result].type == rt_field);
            zkassert(program.values.size() == 2);
            zkassert(mem[program.values[0]].type == rt_field);
            zkassert(mem[program.values[1]].type == rt_field);

            fr.sub(mem[program.result].fe, mem[program.values[0]].fe, mem[program.values[1]].fe);
        }
        else if (program.op == "field_neg")
        {
            zkassert(mem[program.result].type == rt_field);
            zkassert(program.values.size() == 1);
            zkassert(mem[program.values[0]].type == rt_field);

            fr.neg(mem[program.result].fe, mem[program.values[0]].fe);
        }
        else if (program.op == "field_mul")
        {
            zkassert(mem[program.result].type == rt_field);
            zkassert(program.values.size() == 2);
            zkassert(mem[program.values[0]].type == rt_field);
            zkassert(mem[program.values[1]].type == rt_field);

            fr.mul(mem[program.result].fe, mem[program.values[0]].fe, mem[program.values[1]].fe);
        }
        else if (program.op == "pol_add")
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
        }
        else if (program.op == "pol_sub")
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
        }
        else if (program.op == "pol_neg")
        {

            zkassert(mem[program.result].type == rt_pol);
            zkassert(program.values.size() == 1);
            zkassert(mem[program.values[0]].type == rt_pol);
            zkassert(mem[program.result].N == mem[program.values[0]].N);

            for (uint64_t j=0; j<program.N; j++)
            {
                fr.neg(mem[program.result].pPol[j], mem[program.values[0]].pPol[j]);
            }
        }
        else if (program.op == "pol_mul")
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
        }
        else if (program.op == "pol_addc")
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
        }
        else if (program.op == "pol_mulc")
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
        }
        else if (program.op == "pol_grandProduct")
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
        }
        else if (program.op == "pol_batchInverse")
        {
            /*mem[l.result] = await F.batchInverse(mem[l.values[0]]);*/
        }
        else if (program.op == "pol_rotate")
        {

            zkassert(mem[program.result].type == rt_pol);
            zkassert(program.values.size() == 1);
            zkassert(mem[program.values[0]].type == rt_pol);
            zkassert(mem[program.result].N == mem[program.values[0]].N);

            for (uint64_t j=0; j<program.N; j++)
            {
                mem[program.result].pPol[j] = mem[program.values[0]].pPol[(j+program.shift)%program.N];
            }
        }
        else if (program.op == "pol_extend")
        {
            /*mem[l.result] = await extendPol(F, mem[l.values[0]], l.extendBits);*/
        }
        else if (program.op == "treeGroupMultipol_extractPol")
        {
            /*const MGP = new MerkleGroupMultipol(M, l.nGroups, l.groupSize, l.nPols);
                    const N = l.nGroups*l.groupSize;
                    const p = new Array(N);
                    for (let j=0; j<N; j++) {
                        p[j] = MGP.getElement(mem[l.tree], l.polIdx, j); 
                    }
                    mem[l.result] = p;*/
        }
        else if (program.op == "treeGroupMultipol_merkelize")
        {
            /*const MGP = new MerkleGroupMultipol(M, l.nGroups, l.groupSize, l.pols.length);
                    const pols = [];
                    for (let j=0; j<l.pols.length; j++) {
                        pols.push(mem[l.pols[j]]);
                    }
                    mem[l.result] = MGP.merkelize(pols);*/
        }
        else if (program.op == "treeGroupMultipol_root")
        {
            /*const MGP = new MerkleGroupMultipol(M, l.nGroups, l.groupSize, l.nPols);
                    mem[l.result] = MGP.root(mem[l.tree]);*/
        }
        else if (program.op == "treeGroupMultipol_getGroupProof")
        {
            /*const MGP = new MerkleGroupMultipol(M, l.nGroups, l.groupSize, l.nPols);
                    mem[l.result] = MGP.getGroupProof(mem[l.tree], mem[l.idx]);*/
        }
        else if (program.op == "treeGroup_merkelize")
        {
            /*const MG = new MerkleGroup(M, l.nGroups, l.groupSize);
                    mem[l.result] = MG.merkelize(mem[l.pol]);*/
        }
        else if (program.op == "treeGroup_root")
        {
            /*const MG = new MerkleGroup(M, l.nGroups, l.groupSize);
                    mem[l.result] = MG.root(mem[l.tree]);*/
        }
        else if (program.op == "treeGroup_getElementProof")
        {
            /*const MG = new MerkleGroup(M, l.nGroups, l.groupSize);
                    mem[l.result] = MG.getElementProof(mem[l.tree], mem[l.idx]);*/
        }
        else if (program.op == "treeGroup_getGroupProof")
        {
            /*const MG = new MerkleGroup(M, l.nGroups, l.groupSize);
                    mem[l.result] = MG.getGroupProof(mem[l.tree], mem[l.idx]);*/
        }
        else if (program.op == "idxArrayFromFields")
        {
            /*const fields = [];
                    for (let j=0; j<l.fields.length; j++) {
                        fields.push(  Scalar.bits(Scalar.e(F.toObject(mem[l.fields[j]]))));
                    }
        
                    const res = [];
                    let curField =0;
                    let curBit =0;
                    for (let i=0; i<l.n; i++) {
                        let a = 0;
                        for (let j=0; j<l.nBits; j++) {
                            if (fields[curField][curBit]) a = a + (1<<j);
                            curBit ++;
                            if (curBit == 253) {
                                curBit = 0;
                                curField ++;
                            }
                        }
                        res.push(a);
                    }
                    mem[l.result] = res;*/
        }
        else if (program.op == "idxArray_get")
        {
            /*mem[l.result] = mem[l.idxArray][l.pos];*/
        }
        else if (program.op == "idx_addMod")
        {
            /*mem[l.result] = (mem[l.idx] + l.add) % l.mod;*/
        }
        else if (program.op == "calculateH1H2")
        {
            /*const [h1, h2] = calculateH1H2(F, mem[l.f], mem[l.t]);
                    mem[l.resultH1] = h1;
                    mem[l.resultH2] = h2;*/
        }
        else if (program.op == "friReduce")
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
        }
        else if (program.op == "hash")
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
        }
        else if (program.op == "log")
        {
            zkassert(program.msg.size() > 0);
            cout << "BME log: " << program.msg << endl;
            /*      if (typeof(l.refId)!= "undefined") {
                    const o = refToObject(F, mem, l.ref);
                    console.log(JSON.stringify(o, null, 1));
                }*/
        }
        else
        {
            cerr << "Error: batchMachineExecutor() found unsupported operation: " << program.op << " at program line: " << i << endl;
            exit(-1);
        }        
    }

    //return dereference(F, mem, script.output); TODO: when output format is final
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