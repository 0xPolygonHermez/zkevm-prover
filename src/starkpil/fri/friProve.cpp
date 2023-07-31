#include "friProve.hpp"
#include "timer.hpp"
#include "zklog.hpp"

void FRIProve::prove(FRIProof &fproof, MerkleTreeGL **treesGL, Transcript transcript, Polinomial &friPol, uint64_t polBits, StarkInfo starkInfo)
{
    //TimerStart(STARK_FRI_PROVE);

    Polinomial polShift(1, 1);
    Polinomial polShiftInv(1, 1);

    *polShift[0] = Goldilocks::shift();
    *polShiftInv[0] = Goldilocks::inv(Goldilocks::shift());

    uint64_t pol2N = 0;

    std::vector<MerkleTreeGL *> treesFRIGL(starkInfo.starkStruct.steps.size());

    //TimerStart(STARK_FRI_PROVE_STEPS);
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

        uint64_t nn = ((1 << polBits) / nX);
        u_int64_t maxth = omp_get_max_threads();
        if (maxth > nn)
        {
            maxth = nn;
        }
#pragma omp parallel num_threads(maxth)
        {
            u_int64_t nth = omp_get_num_threads();
            u_int64_t thid = omp_get_thread_num();
            u_int64_t chunk = nn / nth;
            u_int64_t res = nn - nth * chunk;

            // Evaluate bounds of the loop for the thread
            uint64_t init = chunk * thid;
            uint64_t end;
            if (thid < res)
            {
                init += thid;
                end = init + chunk + 1;
            }
            else
            {
                init += res;
                end = init + chunk;
            }
            //  Evaluate the starting point for the sinv
            Goldilocks::Element aux = *wi[0];
            Goldilocks::Element sinv_ = *sinv[0];
            for (uint64_t i = 0; i < chunk - 1; ++i)
            {
                aux = aux * (*wi[0]);
            }
            for (u_int64_t i = 0; i < thid; ++i)
            {
                sinv_ = sinv_ * aux;
            }
            u_int64_t ncor = res;
            if (thid < res)
            {
                ncor = thid;
            }
            for (u_int64_t j = 0; j < ncor; ++j)
            {
                sinv_ = sinv_ * (*wi[0]);
            }

            for (uint64_t g = init; g < end; g++)
            {
                if (si == 0)
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
                    evalPol(pol2_e, g, ppar_c, special_x);
                    sinv_ = sinv_ * (*wi[0]);
                }
            }
        }

        if (si < starkInfo.starkStruct.steps.size() - 1)
        {
            uint64_t nGroups = 1 << starkInfo.starkStruct.steps[si + 1].nBits;
            uint64_t groupSize = (1 << starkInfo.starkStruct.steps[si].nBits) / nGroups;

            // Re-org in groups
            Polinomial aux(pol2N, FIELD_EXTENSION);
            getTransposed(aux, pol2_e, starkInfo.starkStruct.steps[si + 1].nBits);

            Polinomial rootGL(HASH_SIZE, 1);
            treesFRIGL[si + 1] = new MerkleTreeGL(nGroups, groupSize * FIELD_EXTENSION, NULL);
            treesFRIGL[si + 1]->copySource(aux.address());
            treesFRIGL[si + 1]->merkelize();
            treesFRIGL[si + 1]->getRoot(rootGL.address());
            zklog.info("rootGL[" + to_string(si + 1) + "]: " + rootGL.toString(4));
            transcript.put(rootGL.address(), HASH_SIZE);
            fproof.proofs.fri.trees[si + 1].setRoot(rootGL.address());
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
    //TimerStopAndLog(STARK_FRI_PROVE_STEPS);
    fproof.proofs.fri.setPol(friPol.address());

    //TimerStart(STARK_FRI_QUERIES);

    uint64_t ys[starkInfo.starkStruct.nQueries];
    transcript.getPermutations(ys, starkInfo.starkStruct.nQueries, starkInfo.starkStruct.steps[0].nBits);

    for (uint64_t si = 0; si < starkInfo.starkStruct.steps.size(); si++)
    {
        for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++)
        {
            if (si == 0)
            {
                queryPol(fproof, treesGL, ys[i], si);
            }
            else
            {
                queryPol(fproof, treesFRIGL[si], ys[i], si);
            }
        }
        if (si < starkInfo.starkStruct.steps.size() - 1)
        {
            for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++)
            {
                ys[i] = ys[i] % (1 << starkInfo.starkStruct.steps[si + 1].nBits);
            }
        }
    }

    while (!treesFRIGL.empty())
    {
        MerkleTreeGL *mt = treesFRIGL.back();
        treesFRIGL.pop_back();
        delete mt;
    }

    //TimerStopAndLog(STARK_FRI_QUERIES);
    //TimerStopAndLog(STARK_FRI_PROVE);
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

void FRIProve::queryPol(FRIProof &fproof, MerkleTreeGL *treesGL[5], uint64_t idx, uint64_t treeIdx)
{
    vector<MerkleProof> vMkProof;
    for (uint i = 0; i < 5; i++)
    {
        MerkleTreeGL *treesGLTmp = treesGL[i];
        Goldilocks::Element buff[treesGLTmp->width + treesGLTmp->MerkleProofSize() * HASH_SIZE] = {Goldilocks::zero()};

        treesGLTmp->getGroupProof(&buff[0], idx);

        MerkleProof mkProof(treesGLTmp->width, treesGLTmp->MerkleProofSize(), &buff[0]);
        vMkProof.push_back(mkProof);
    }
    fproof.proofs.fri.trees[treeIdx].polQueries.push_back(vMkProof);

    return;
}

void FRIProve::queryPol(FRIProof &fproof, MerkleTreeGL *treeGL, uint64_t idx, uint64_t treeIdx)
{
    vector<MerkleProof> vMkProof;

    Goldilocks::Element buff[treeGL->width * treeGL->width + treeGL->MerkleProofSize() * HASH_SIZE] = {Goldilocks::zero()};
    treeGL->getGroupProof(&buff[0], idx);

    MerkleProof mkProof(treeGL->width, treeGL->MerkleProofSize(), &buff[0]);
    vMkProof.push_back(mkProof);

    fproof.proofs.fri.trees[treeIdx].polQueries.push_back(vMkProof);

    return;
}

void FRIProve::getTransposed(Polinomial &aux, Polinomial &pol, uint64_t trasposeBits)
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