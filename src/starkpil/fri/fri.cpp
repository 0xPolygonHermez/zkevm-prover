

#include "fri.hpp"
#include "timer.hpp"
#include "zklog.hpp"

template <typename ElementType>
void FRI<ElementType>::fold(uint64_t step, FRIProof<ElementType> &proof, Goldilocks::Element* pol, Goldilocks::Element *challenge, StarkInfo starkInfo, MerkleTreeType** treesFRI) {

    uint64_t polBits = step == 0 ? starkInfo.starkStruct.steps[0].nBits : starkInfo.starkStruct.steps[step - 1].nBits;

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
            if (step != 0)
            {
                Goldilocks::Element ppar[nX * FIELD_EXTENSION];
                Goldilocks::Element ppar_c[nX * FIELD_EXTENSION];

                #pragma omp parallel for
                for (uint64_t i = 0; i < nX; i++)
                {
                    std::memcpy(&ppar[i * FIELD_EXTENSION], &pol[((i * pol2N) + g) * FIELD_EXTENSION], FIELD_EXTENSION * sizeof(Goldilocks::Element));
                }
                NTT_Goldilocks ntt(nX, 1);

                ntt.INTT(ppar_c, ppar, nX, FIELD_EXTENSION);
                polMulAxi(ppar_c, nX, sinv_); // Multiplies coefs by 1, shiftInv, shiftInv^2, shiftInv^3, ......
                evalPol(pol, g, nX, ppar_c, challenge);
                sinv_ = sinv_ * wi;
            }
        }
    }
    if (step != starkInfo.starkStruct.steps.size() - 1) {
        // Re-org in groups
        Goldilocks::Element *aux = new Goldilocks::Element[pol2N * FIELD_EXTENSION];
        getTransposed(aux, pol, pol2N, starkInfo.starkStruct.steps[step + 1].nBits);

        treesFRI[step]->copySource(aux);
        treesFRI[step]->merkelize();
        treesFRI[step]->getRoot(&proof.proof.fri.trees[step + 1].root[0]);

        delete aux;
    }
    
    if(step == starkInfo.starkStruct.steps.size() - 1) {
        proof.proof.fri.setPol(pol, pol2N);
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
    fproof.proof.fri.trees[treeIdx].polQueries.push_back(vMkProof);

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

    fproof.proof.fri.trees[treeIdx].polQueries.push_back(vMkProof);

    return;
}

template <typename ElementType>
void FRI<ElementType>::polMulAxi(Goldilocks::Element *pol, uint64_t degree, Goldilocks::Element acc)
{
    Goldilocks::Element r = Goldilocks::one();
    for (uint64_t i = 0; i < degree; i++)
    {   
        Goldilocks3::mul((Goldilocks3::Element &)(pol[i * FIELD_EXTENSION]), (Goldilocks3::Element &)(pol[i * FIELD_EXTENSION]), r);
        r = r * acc;
    }
}

template <typename ElementType>
void FRI<ElementType>::evalPol(Goldilocks::Element* res, uint64_t res_idx, uint64_t degree, Goldilocks::Element* p, Goldilocks::Element *x)
{
    if (degree == 0)
    {
        res[res_idx * FIELD_EXTENSION] = Goldilocks::zero();
        res[res_idx * FIELD_EXTENSION + 1] = Goldilocks::zero();
        res[res_idx * FIELD_EXTENSION + 2] = Goldilocks::zero();
        return;
    }

    std::memcpy(&res[res_idx * FIELD_EXTENSION], &p[(degree - 1) * FIELD_EXTENSION], FIELD_EXTENSION * sizeof(Goldilocks::Element));
    for (int64_t i = degree - 2; i >= 0; i--)
    {
        Goldilocks3::Element aux;
        Goldilocks3::mul(aux, (Goldilocks3::Element &)(res[res_idx * FIELD_EXTENSION]), (Goldilocks3::Element &)x[0]);
        Goldilocks3::add((Goldilocks3::Element &)(res[res_idx * FIELD_EXTENSION]), aux, (Goldilocks3::Element &)p[i * FIELD_EXTENSION]);
    }
}

template <typename ElementType>
void FRI<ElementType>::getTransposed(Goldilocks::Element *aux, Goldilocks::Element* pol, uint64_t degree, uint64_t trasposeBits)
{
    uint64_t w = (1 << trasposeBits);
    uint64_t h = degree / w;

#pragma omp parallel for
    for (uint64_t i = 0; i < w; i++)
    {
        for (uint64_t j = 0; j < h; j++)
        {

            uint64_t fi = j * w + i;
            uint64_t di = i * h + j;

            std::memcpy(&aux[di * FIELD_EXTENSION], &pol[fi * FIELD_EXTENSION], FIELD_EXTENSION * sizeof(Goldilocks::Element));
        }
    }
}