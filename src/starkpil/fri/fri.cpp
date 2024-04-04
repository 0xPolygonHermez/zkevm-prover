

#include "fri.hpp"
#include "timer.hpp"
#include "zklog.hpp"

template <typename ElementType>
void FRI<ElementType>::fold(uint64_t step, FRIProof<ElementType> &proof, Polinomial &friPol, Polinomial& challenge, StarkInfo starkInfo, MerkleTreeType** treesFRI) {

    uint64_t polBits = log2(friPol.degree());

    Goldilocks::Element polShiftInv = Goldilocks::inv(Goldilocks::shift());
    
    if(step > 0) {
        for (uint64_t j = 0; j < starkInfo.starkStruct.steps[0].nBits - starkInfo.starkStruct.steps[step - 1].nBits; j++)
        {
            polShiftInv = polShiftInv * polShiftInv;
        }
    }

    uint64_t reductionBits = polBits - starkInfo.starkStruct.steps[step].nBits;

    uint64_t pol2N = 1 << (polBits - reductionBits);
    uint64_t nX = (1 << polBits) / pol2N;

    Polinomial pol2_e(pol2N, FIELD_EXTENSION);

    Goldilocks::Element wi = Goldilocks::inv(Goldilocks::w(polBits));

    uint64_t nn = ((1 << polBits) / nX);
    u_int64_t maxth = omp_get_max_threads();
    if (maxth > nn) maxth = nn;
#pragma omp parallel num_threads(maxth)
    {
        u_int64_t nth = omp_get_num_threads();
        u_int64_t thid = omp_get_thread_num();
        u_int64_t chunk = nn / nth;
        u_int64_t res = nn - nth * chunk;

        // Evaluate bounds of the loop for the thread
        uint64_t init = chunk * thid;
        uint64_t end;
        if (thid < res) {
            init += thid;
            end = init + chunk + 1;
        } else {
            init += res;
            end = init + chunk;
        }
        //  Evaluate the starting point for the sinv
        Goldilocks::Element aux = wi;
        Goldilocks::Element sinv_ = polShiftInv;
        for (uint64_t i = 0; i < chunk - 1; ++i) aux = aux * wi;
        for (u_int64_t i = 0; i < thid; ++i) sinv_ = sinv_ * aux;   
        u_int64_t ncor = res;
        if (thid < res) ncor = thid;
        for (u_int64_t j = 0; j < ncor; ++j) sinv_ = sinv_ * wi;
        for (uint64_t g = init; g < end; g++)
        {
            if (step == 0)
            {
                Goldilocks3::copy((Goldilocks3::Element &)(*pol2_e[g]), (Goldilocks3::Element &)(*friPol[g]));
            }
            else
            {
                Polinomial ppar(nX, FIELD_EXTENSION);
                Polinomial ppar_c(nX, FIELD_EXTENSION);

                for (uint64_t i = 0; i < nX; i++)
                {
                    Goldilocks3::copy((Goldilocks3::Element &)(*ppar[i]), (Goldilocks3::Element &)(*friPol[(i * pol2N) + g]));
                }
                NTT_Goldilocks ntt(nX, 1);

                ntt.INTT(ppar_c.address(), ppar.address(), nX, FIELD_EXTENSION);
                polMulAxi(ppar_c, Goldilocks::one(), sinv_); // Multiplies coefs by 1, shiftInv, shiftInv^2, shiftInv^3, ......
                evalPol(pol2_e, g, ppar_c, challenge);
                sinv_ = sinv_ * wi;
            }
        }
    }

    if (step != starkInfo.starkStruct.steps.size() - 1) {
        // Re-org in groups
        Polinomial aux(pol2N, FIELD_EXTENSION);
        getTransposed(aux, pol2_e, starkInfo.starkStruct.steps[step + 1].nBits);
        treesFRI[step]->copySource(aux.address());
        treesFRI[step]->merkelize();
        treesFRI[step]->getRoot(&proof.proofs.fri.trees[step + 1].root[0]);
    }

    friPol.potConstruct(friPol.address(), pol2_e.degree(), friPol.dim(), friPol.offset());
    Polinomial::copy(friPol, pol2_e);

    if(step == starkInfo.starkStruct.steps.size() - 1) {
        proof.proofs.fri.setPol(friPol.address());
    }
}

template <typename ElementType>
void FRI<ElementType>::proveQueries(uint64_t* friQueries, FRIProof<ElementType> &fproof, MerkleTreeType **trees, MerkleTreeType **treesFRI, StarkInfo starkInfo) {

    for (uint64_t step = 0; step < starkInfo.starkStruct.steps.size(); step++)
    {
        for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++)
        {
            if (step == 0) {
                queryPol(fproof, trees, starkInfo.nStages + 2, friQueries[i], step);
            } else {
                queryPol(fproof, treesFRI[step - 1], friQueries[i], step);
            }
        }
        if (step < starkInfo.starkStruct.steps.size() - 1)
        {
            for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++)
            {
                friQueries[i] = friQueries[i] % (1 << starkInfo.starkStruct.steps[step + 1].nBits);
            }
        }
    }

    return;
}

template <typename ElementType>
void FRI<ElementType>::queryPol(FRIProof<ElementType> &fproof, MerkleTreeType *trees[], uint64_t nTrees, uint64_t idx, uint64_t treeIdx)
{
    vector<MerkleProof<ElementType>> vMkProof;
    for (uint i = 0; i < nTrees; i++)
    {
        ElementType buff[trees[i]->getMerkleTreeWidth() + trees[i]->getMerkleProofSize()];

        trees[i]->getGroupProof(&buff[0], idx);

        MerkleProof<ElementType> mkProof(trees[i]->getMerkleTreeWidth(), trees[i]->getMerkleProofLength(), trees[i]->getNumSiblings(), &buff[0]);
        vMkProof.push_back(mkProof);
    }
    fproof.proofs.fri.trees[treeIdx].polQueries.push_back(vMkProof);

    return;
}


template <typename ElementType>
void FRI<ElementType>::queryPol(FRIProof<ElementType> &fproof, MerkleTreeType *tree, uint64_t idx, uint64_t treeIdx)
{
    vector<MerkleProof<ElementType>> vMkProof;

    ElementType buff[tree->getMerkleTreeWidth() + tree->getMerkleProofSize()];
    tree->getGroupProof(&buff[0], idx);

    MerkleProof<ElementType> mkProof(tree->getMerkleTreeWidth(), tree->getMerkleProofLength(), tree->getNumSiblings(), &buff[0]);
    vMkProof.push_back(mkProof);

    fproof.proofs.fri.trees[treeIdx].polQueries.push_back(vMkProof);

    return;
}

template <typename ElementType>
void FRI<ElementType>::polMulAxi(Polinomial &pol, Goldilocks::Element init, Goldilocks::Element acc)
{
    Goldilocks::Element r = init;
    for (uint64_t i = 0; i < pol.degree(); i++)
    {
        pol[i][0] = pol[i][0] * r;
        pol[i][1] = pol[i][1] * r;
        pol[i][2] = pol[i][2] * r;
        r = r * acc;
    }
}

template <typename ElementType>
void FRI<ElementType>::evalPol(Polinomial &res, uint64_t res_idx, Polinomial &p, Polinomial &x)
{
    if (p.degree() == 0)
    {
        res[res_idx][0] = Goldilocks::zero();
        res[res_idx][1] = Goldilocks::zero();
        res[res_idx][2] = Goldilocks::zero();
        return;
    }
    Goldilocks3::copy((Goldilocks3::Element &)(*res[res_idx]), (Goldilocks3::Element &)(*p[p.degree() - 1]));
    for (int64_t i = p.degree() - 2; i >= 0; i--)
    {
        Goldilocks3::Element aux;
        Goldilocks3::mul(aux, (Goldilocks3::Element &)(*res[res_idx]), (Goldilocks3::Element &)(*x[0]));

        res[res_idx][0] = aux[0] + p[i][0];
        res[res_idx][1] = aux[1] + p[i][1];
        res[res_idx][2] = aux[2] + p[i][2];
    }
}

template <typename ElementType>
void FRI<ElementType>::getTransposed(Polinomial &aux, Polinomial &pol, uint64_t trasposeBits)
{
    uint64_t w = (1 << trasposeBits);
    uint64_t h = pol.degree() / w;

#pragma omp parallel for
    for (uint64_t i = 0; i < w; i++)
    {
        for (uint64_t j = 0; j < h; j++)
        {

            uint64_t fi = j * w + i;
            uint64_t di = i * h + j;
            assert(di < aux.degree());
            assert(fi < pol.degree());

            Goldilocks3::copy((Goldilocks3::Element &)(*aux[di]), (Goldilocks3::Element &)(*pol[fi]));
        }
    }
}