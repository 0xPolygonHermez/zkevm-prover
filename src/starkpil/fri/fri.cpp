

#include "fri.hpp"
#include "timer.hpp"
#include "zklog.hpp"

template <typename ElementType, typename FieldType, typename MerkleTree>
void FRI<ElementType, FieldType, MerkleTree>::fold(uint64_t step, FRIProof<ElementType, FieldType> &proof, Polinomial &friPol, Polinomial& challenge, StarkInfo starkInfo, MerkleTree** treesFRI) {

    uint64_t polBits = log2(friPol.degree());

    Polinomial polShift(1, 1);
    Polinomial polShiftInv(1, 1);

    *polShift[0] = Goldilocks::shift();
    *polShiftInv[0] = Goldilocks::inv(Goldilocks::shift());
    
    if(step > 0) {
        for (uint64_t j = 0; j < starkInfo.starkStruct.steps[0].nBits - starkInfo.starkStruct.steps[step - 1].nBits; j++)
        {
            Goldilocks::mul(*polShiftInv[0], *polShiftInv[0], *polShiftInv[0]);
        }
    }

    uint64_t reductionBits = polBits - starkInfo.starkStruct.steps[step].nBits;

    uint64_t pol2N = 1 << (polBits - reductionBits);
    uint64_t nX = (1 << polBits) / pol2N;

    Polinomial pol2_e(pol2N, FIELD_EXTENSION);

    Polinomial sinv(1, 1);
    Polinomial wi(1, 1);

    *sinv[0] = *polShiftInv[0];
    *wi[0] = Goldilocks::inv(Goldilocks::w(polBits));

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
        Goldilocks::Element aux = *wi[0];
        Goldilocks::Element sinv_ = *sinv[0];
        for (uint64_t i = 0; i < chunk - 1; ++i) aux = aux * (*wi[0]);
        for (u_int64_t i = 0; i < thid; ++i) sinv_ = sinv_ * aux;   
        u_int64_t ncor = res;
        if (thid < res) ncor = thid;
        for (u_int64_t j = 0; j < ncor; ++j) sinv_ = sinv_ * (*wi[0]);
        for (uint64_t g = init; g < end; g++)
        {
            if (step == 0)
            {
                Polinomial::copyElement(pol2_e, g, friPol, g);
            }
            else
            {
                Polinomial ppar(nX, FIELD_EXTENSION);
                Polinomial ppar_c(nX, FIELD_EXTENSION);

                for (uint64_t i = 0; i < nX; i++)
                {
                    Polinomial::copyElement(ppar, i, friPol, (i * pol2N) + g);
                }
                NTT_Goldilocks ntt(nX, 1);

                ntt.INTT(ppar_c.address(), ppar.address(), nX, FIELD_EXTENSION);
                polMulAxi(ppar_c, Goldilocks::one(), sinv_); // Multiplies coefs by 1, shiftInv, shiftInv^2, shiftInv^3, ......
                evalPol(pol2_e, g, ppar_c, challenge);
                sinv_ = sinv_ * (*wi[0]);
            }
        }
    }

    if (step != starkInfo.starkStruct.steps.size() - 1) {
        uint64_t nGroups = 1 << starkInfo.starkStruct.steps[step + 1].nBits;
        uint64_t groupSize = (1 << starkInfo.starkStruct.steps[step].nBits) / nGroups;
    
        // Re-org in groups
        Polinomial aux(pol2N, FIELD_EXTENSION);
        getTransposed(aux, pol2_e, starkInfo.starkStruct.steps[step + 1].nBits);
        treesFRI[step] = new MerkleTree(nGroups, groupSize * FIELD_EXTENSION, NULL);
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

template <typename ElementType, typename FieldType, typename MerkleTree>
void FRI<ElementType, FieldType, MerkleTree>::proveQueries(uint64_t* friQueries, FRIProof<ElementType, FieldType> &fproof, MerkleTree **trees, MerkleTree **treesFRI, StarkInfo starkInfo) {

    for (uint64_t step = 0; step < starkInfo.starkStruct.steps.size(); step++)
    {
        for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++)
        {
            if (step == 0) {
                queryPol(fproof, trees, 5, friQueries[i], step);
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

template <typename ElementType, typename FieldType, typename MerkleTree>
void FRI<ElementType, FieldType, MerkleTree>::queryPol(FRIProof<ElementType, FieldType> &fproof, MerkleTree *trees[], uint64_t nTrees, uint64_t idx, uint64_t treeIdx)
{
    vector<MerkleProof<ElementType, FieldType>> vMkProof;
    for (uint i = 0; i < nTrees; i++)
    {
        ElementType buff[trees[i]->getMerkleTreeWidth() + trees[i]->getMerkleProofSize()] = {FieldType::zero()};

        trees[i]->getGroupProof(&buff[0], idx);

        MerkleProof<ElementType, FieldType> mkProof(trees[i]->getMerkleTreeWidth(), trees[i]->getMerkleProofLength(), trees[i]->getElementSize(), &buff[0]);
        vMkProof.push_back(mkProof);
    }
    fproof.proofs.fri.trees[treeIdx].polQueries.push_back(vMkProof);

    return;
}


template <typename ElementType, typename FieldType, typename MerkleTree>
void FRI<ElementType, FieldType, MerkleTree>::queryPol(FRIProof<ElementType, FieldType> &fproof, MerkleTree *tree, uint64_t idx, uint64_t treeIdx)
{
    vector<MerkleProof<ElementType, FieldType>> vMkProof;

    ElementType buff[tree->getMerkleTreeWidth() + tree->getMerkleProofSize()] = {FieldType::zero()};
    tree->getGroupProof(&buff[0], idx);

    MerkleProof<ElementType, FieldType> mkProof(tree->getMerkleTreeWidth(), tree->getMerkleProofLength(), tree->getElementSize(), &buff[0]);
    vMkProof.push_back(mkProof);

    fproof.proofs.fri.trees[treeIdx].polQueries.push_back(vMkProof);

    return;
}

template <typename ElementType, typename FieldType, typename MerkleTree>
void FRI<ElementType, FieldType, MerkleTree>::polMulAxi(Polinomial &pol, Goldilocks::Element init, Goldilocks::Element acc)
{
    Goldilocks::Element r = init;
    for (uint64_t i = 0; i < pol.degree(); i++)
    {
        Polinomial::mulElement(pol, i, pol, i, r);
        r = r * acc;
    }
}

template <typename ElementType, typename FieldType, typename MerkleTree>
void FRI<ElementType, FieldType, MerkleTree>::evalPol(Polinomial &res, uint64_t res_idx, Polinomial &p, Polinomial &x)
{
    if (p.degree() == 0)
    {
        res[res_idx][0] = Goldilocks::zero();
        res[res_idx][1] = Goldilocks::zero();
        res[res_idx][2] = Goldilocks::zero();
        return;
    }
    Polinomial::copyElement(res, res_idx, p, p.degree() - 1);
    for (int64_t i = p.degree() - 2; i >= 0; i--)
    {
        Polinomial aux(1, 3);
        Polinomial::mulElement(aux, 0, res, res_idx, x, 0);
        Polinomial::addElement(res, res_idx, aux, 0, p, i);
    }
}

template <typename ElementType, typename FieldType, typename MerkleTree>
void FRI<ElementType, FieldType, MerkleTree>::getTransposed(Polinomial &aux, Polinomial &pol, uint64_t trasposeBits)
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

            Polinomial::copyElement(aux, di, pol, fi);
        }
    }
}