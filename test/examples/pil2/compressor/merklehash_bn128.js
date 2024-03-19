const LinearHash = require("../linearhash/linearhash.bn128");
const assert = require("assert");
const { log2 } = require("pilcom/src/utils.js");

module.exports = class MerkleHash {

    constructor(poseidon, arity, custom) {
        this.poseidon = poseidon;
        this.F = poseidon.F;
        this.lh = new LinearHash(poseidon, arity, custom);
        this.arity = arity;
    }

    async merkelize(vals, elementSize, elementsInLinear, nLinears, interleaved) {
        const lh = this.lh;

        let sizeTree = 1;
        let nTree = nLinears;
        while (nTree>1) {
            sizeTree += nTree;
            nTree = Math.floor((nTree-1)/this.arity)+1;
        }

        const buff = new BigUint64Array(3 + elementsInLinear*nLinears*elementSize + 4*sizeTree);
        let p=0;
        buff[p++] = BigInt(elementSize);
        buff[p++] = BigInt(elementsInLinear);
        buff[p++] = BigInt(nLinears);
        if (vals.length == elementsInLinear) {
            for (let j=0; j<elementsInLinear; j++) {
                if (vals[j].length != nLinears) throw new Error("Invalid Element size");
            }

            for (let i=0; i<nLinears; i++) {
                for (let j=0; j<elementsInLinear; j++) {
                    if (Array.isArray(vals[j][i])) {
                        if (vals[j][i].length != elementSize) throw new Error("Invalid Element size");
                        for (let k=0; k<elementSize; k++) {
                            buff[p++] = BigInt(vals[j][i][k]);
                        }
                    } else {
                        if (elementSize != 1) throw new Error("Invalid Element size");
                        buff[p++] = BigInt(vals[j][i]);
                    }
                }
            }
        } else if (vals.length == nLinears*elementsInLinear) {
            for (let i=0; i<nLinears; i++) {
                for (let j=0; j<elementsInLinear; j++) {
                    const s = interleaved ? i*elementsInLinear + j : j*nLinears + i;
                    if (Array.isArray(vals[s])) {
                        if (vals[s].length != elementSize) throw new Error("Invalid Element size");
                        for (let k=0; k<elementSize; k++) {
                            buff[p++] = BigInt(vals[s][k]);
                        }
                    } else {
                        if (elementSize != 1) throw new Error("Invalid Element size");
                        buff[p++] = BigInt(vals[s]);
                    }
                }
            }
        } else {
            throw new Error("Invalid vals format");
        }



        const buff8 = new Uint8Array(buff.buffer);
        let pp = p*8;

        let o=3;
        let nextO = pp;

        for (let i=0; i<nLinears; i++) {
            if (i%10000 == 0) console.log(`Hashing... ${i+1}/${nLinears}`);
            const a = buff.slice(o, o+elementsInLinear*elementSize);
            const h = lh.hash(a);
            buff8.set(h, pp);
            pp += 32;
            o += elementsInLinear*elementSize;
        }


        o=nextO;

        let n = nLinears;
        let nextN = Math.floor((n-1)/this.arity)+1;
        const auxBuff = new Uint8Array(this.arity*32);
        while (n>1) {
            nextO = pp;
            for (let i=0; i<nextN; i++) {
                let ih;
                if ((i+1)*this.arity <= n) {
                    ih = buff8.slice(o+i*this.arity*32, o+(i+1)*this.arity*32);
                } else {
                    auxBuff.set(buff8.slice(o+i*this.arity*32, o+n*32));
                    auxBuff.fill(0, (n-i)*32);
                    ih = auxBuff;
                }
                const h = this.poseidon(ih);
                buff8.set(h, pp);
                pp += 32;
            }
            n = nextN;
            nextN = Math.floor((n-1)/this.arity)+1;
            o = nextO;
        }

        assert(buff8.length == pp);

        return buff;
    }

    // idx is the root of unity
    getElement(tree, idx, subIdx) {
        const elementSize = Number(tree[0]);
        const elementsInLinear = Number(tree[1]);
        const nLinears = Number(tree[2]);

        if ((idx<0)||(idx>=nLinears)) throw new Error("Out of range");
        if (elementSize == 1) {
            return tree[3 + idx*elementsInLinear + subIdx];
        } else {
            const res = [];
            for (let k=0; k<elementSize; k++) {
                res.push(tree[3 + (idx*elementsInLinear + subIdx)*elementSize + k]);
            }
            return res;
        }
    }


    getGroupProof(tree, idx) {
        const self = this;
        const buff8 = new Uint8Array(tree.buffer);
        const elementSize = Number(tree[0]);
        const elementsInLinear = Number(tree[1]);
        const nLinears = Number(tree[2]);

        if ((idx<0)||(idx>=nLinears)) throw new Error("Out of range");

        const v = new Array(elementsInLinear);
        for (let i=0; i<elementsInLinear; i++) {
            v[i] = this.getElement(tree, idx, i);
        }

        const mp = merkle_genMerkleProof(idx, (3 + elementsInLinear*nLinears*elementSize)*8, nLinears);

        return [v, mp];

        function merkle_genMerkleProof(idx, offset, n) {
            if (n<=1) return [];
            const nBitsArity = Math.ceil(Math.log2(self.arity));
            const nextIdx = idx >> nBitsArity;
            
            const si =  idx ^ (idx & (self.arity - 1));

            const sibs = [];

            for (let i=0; i<self.arity; i++) {
                if (i<n) {
                    sibs.push(self.F.toObject(buff8.slice(offset + (si+i)*32, offset + (si+i+1)*32)));
                } else {
                    sibs.push( 0n );
                }
            }

            const nextN = Math.floor((n-1)/self.arity)+1;

            return [sibs, ...merkle_genMerkleProof(nextIdx, offset+ n*32, nextN)];
        }
    }

    calculateRootFromGroupProof(mp, idx, vals) {

        const self = this;
        const lh = this.lh;

        const a = [];
        for (let i=0; i<vals.length; i++) {
            if (Array.isArray(a[i])) {
                for (j=0; j<vals[i].length; j++) {
                    a.push(vals[i][j]);
                }
            } else {
                a.push(vals[i]);
            }
        }

        const h = lh.hash(a);

        return this.F.toObject(merkle_calculateRootFromProof(mp, idx, h));

        function merkle_calculateRootFromProof(mp, idx, value, offset) {
            offset = offset || 0;
            if (mp.length == offset) {
                return value;
            }

            const nBitsArity = Math.ceil(Math.log2(self.arity));

            const curIdx = idx & (self.arity - 1);
            const nextIdx = idx >> nBitsArity;

            const buff = new Uint8Array(32*self.arity);
            for (let i=0; i<self.arity; i++) {
                buff.set(self.F.e(mp[offset][i]), i*32);
            }
            buff.set(value, curIdx*32);

            const nextValue = self.poseidon(buff);

            return merkle_calculateRootFromProof(mp, nextIdx, nextValue, offset+1);
        }

    }

    eqRoot(r1, r2) {
        return r1 === r2;
    }

    verifyGroupProof(root, mp, idx, groupElements) {
        const cRoot = this.calculateRootFromGroupProof(mp, idx, groupElements);
        return this.eqRoot(cRoot, root);
    }

    root(tree) {
        const buff8 = new Uint8Array(tree.buffer);
        return this.F.toObject(buff8.slice(-32));
    }

}

