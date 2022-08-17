#include "friProve.hpp"
#include "timer.hpp"

void FRIProve::prove(FRIProof &fproof, Goldilocks::Element **trees, Transcript transcript, Polinomial &friPol, uint64_t polBits, StarkInfo starkInfo)
{
    TimerStart(STARK_FRI_PROVE);

    Polinomial polShift(1, 1);
    Polinomial polShiftInv(1, 1);

    *polShift[0] = Goldilocks::shift();
    *polShiftInv[0] = Goldilocks::inv(Goldilocks::shift());

    uint64_t pol2N = 0;

    std::vector<std::vector<Goldilocks::Element>> treesFRI(starkInfo.starkStruct.steps.size());

    TimerStart(STARK_FRI_PROVE_STEPS);
    for (uint64_t si = 0; si < starkInfo.starkStruct.steps.size(); si++)
    {
        uint64_t reductionBits = polBits - starkInfo.starkStruct.steps[si].nBits;

        pol2N = 1 << (polBits - reductionBits);
        uint64_t nX = (1 << polBits) / pol2N;

        Polinomial pol2_e(pol2N, FIELD_EXTENSION);

        Polinomial special_x(1, FIELD_EXTENSION);
        transcript.getField(special_x.address());

        Polinomial sinv(1, 1);
        Polinomial wi(1, 1);

        *sinv[0] = *polShiftInv[0];
        *wi[0] = Goldilocks::inv(Goldilocks::w(polBits));

        for (uint64_t g = 0; g < (1 << polBits) / nX; g++)
        {
            if (si == 0)
            {
                *pol2_e[g] = *friPol[g];
            }
            else
            {
                Polinomial ppar(nX, FIELD_EXTENSION);
                Polinomial ppar_c(nX, FIELD_EXTENSION);

                for (uint64_t i = 0; i < nX; i++)
                {
                    Polinomial::copyElement(ppar, i, friPol, (i * pol2N) + g);
                }
                NTT_Goldilocks ntt(nX);

                ntt.INTT(ppar_c.address(), ppar.address(), nX, FIELD_EXTENSION);

                polMulAxi(ppar_c, Goldilocks::one(), *sinv[0]); // Multiplies coefs by 1, shiftInv, shiftInv^2, shiftInv^3, ......

                evalPol(pol2_e, g, ppar_c, special_x);
                *sinv[0] = *sinv[0] * *wi[0];
            }
        }

        if (si < starkInfo.starkStruct.steps.size() - 1)
        {
            uint64_t nGroups = 1 << starkInfo.starkStruct.steps[si + 1].nBits;
            uint64_t groupSize = (1 << starkInfo.starkStruct.steps[si].nBits) / nGroups;

            // Re-org in groups
            Polinomial aux(pol2N, FIELD_EXTENSION);
#pragma omp parallel for
            for (uint64_t i = 0; i < nGroups; i++)
            {
                for (uint64_t j = 0; j < groupSize; j++)
                {
                    Polinomial::copyElement(aux, i * groupSize + j, pol2_e, j * nGroups + i);
                }
            }

            uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(groupSize * FIELD_EXTENSION, nGroups);
            std::vector<Goldilocks::Element> tree(numElementsTree);

            Polinomial root(HASH_SIZE, 1);

            PoseidonGoldilocks::merkletree(&tree[0], aux.address(), groupSize, nGroups, FIELD_EXTENSION);
            treesFRI[si + 1] = tree;
            MerklehashGoldilocks::root(root.address(), &tree[0], numElementsTree);

            std::cout << "root[" << si + 1 << "]: " << root.toString(4) << std::endl;
            transcript.put(root.address(), HASH_SIZE);

            fproof.proofs.fri.trees[si + 1].setRoot(root.address());
        }
        else
        {
            for (uint64_t i = 0; i < pol2N; i++)
            {
                transcript.put(pol2_e[i], FIELD_EXTENSION);
            }
        }

#pragma omp parallel for
        for (uint64_t i = 0; i < pol2_e.degree(); i++)
        {
            Polinomial::copyElement(friPol, i, pol2_e, i);
        }

        polBits = polBits - reductionBits;

        for (uint64_t j = 0; j < reductionBits; j++)
        {
            Goldilocks::mul(*polShiftInv[0], *polShiftInv[0], *polShiftInv[0]);
            Goldilocks::mul(*polShift[0], *polShift[0], *polShift[0]);
        }
    }
    TimerStopAndLog(STARK_FRI_PROVE_STEPS);
    fproof.proofs.fri.setPol(friPol.address());

    TimerStart(STARK_FRI_QUERIES);

    uint64_t ys[starkInfo.starkStruct.nQueries];
    transcript.getPermutations(ys, starkInfo.starkStruct.nQueries, starkInfo.starkStruct.steps[0].nBits);

    for (uint64_t si = 0; si < starkInfo.starkStruct.steps.size(); si++)
    {
        for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++)
        {
            if (si == 0)
            {
                queryPol(fproof, trees, ys[i], si);
            }
            else
            {
                queryPol(fproof, &treesFRI[si][0], ys[i], si);
            }
        }

        for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++)
        {
            ys[i] = ys[i] % (1 << starkInfo.starkStruct.steps[si + 1].nBits);
        }
    }
    TimerStopAndLog(STARK_FRI_QUERIES);
    TimerStopAndLog(STARK_FRI_PROVE);
    return;
}

void FRIProve::polMulAxi(Polinomial &pol, Goldilocks::Element init, Goldilocks::Element acc)
{
    Goldilocks::Element r = init;
    for (uint64_t i = 0; i < pol.degree(); i++)
    {
        Polinomial::mulElement(pol, i, pol, i, r);
        r = r * acc;
    }
}
void FRIProve::evalPol(Polinomial &res, uint64_t res_idx, Polinomial &p, Polinomial &x)
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

void FRIProve::queryPol(FRIProof &fproof, Goldilocks::Element *trees[5], uint64_t idx, uint64_t treeIdx)
{
    vector<MerkleProof> vMkProof;
    for (uint i = 0; i < 5; i++)
    {
        uint64_t elementsInLinear = Goldilocks::toU64(trees[i][0]);
        uint64_t elementsTree = MerklehashGoldilocks::MerkleProofSize(Goldilocks::toU64(trees[i][1])) * HASH_SIZE;
        Goldilocks::Element buff[(elementsInLinear + elementsTree)] = {Goldilocks::zero()};

        MerklehashGoldilocks::getGroupProof(&buff[0], trees[i], idx);
        MerkleProof mkProof(elementsInLinear, elementsTree / HASH_SIZE, &buff[0]);
        vMkProof.push_back(mkProof);
    }
    fproof.proofs.fri.trees[treeIdx].polQueries.push_back(vMkProof);

    return;
}

void FRIProve::queryPol(FRIProof &fproof, Goldilocks::Element *tree, uint64_t idx, uint64_t treeIdx)
{
    vector<MerkleProof> vMkProof;

    uint64_t elementsInLinear = Goldilocks::toU64(tree[0]);
    uint64_t elementsTree = MerklehashGoldilocks::MerkleProofSize(Goldilocks::toU64(tree[1])) * HASH_SIZE;
    Goldilocks::Element buff[(elementsInLinear * Goldilocks::toU64(tree[0]) + elementsTree)] = {Goldilocks::zero()};

    MerklehashGoldilocks::getGroupProof(&buff[0], tree, idx);
    MerkleProof mkProof(elementsInLinear, elementsTree / HASH_SIZE, &buff[0]);
    vMkProof.push_back(mkProof);

    fproof.proofs.fri.trees[treeIdx].polQueries.push_back(vMkProof);

    return;
}