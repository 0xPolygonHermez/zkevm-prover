#include "config.hpp"
#include "mem.hpp"
#include "scalar.hpp"
#if 0
#include "merkle_group.hpp"
#include "merkle_group_multipol.hpp"

// Reference indexes: Mem = constantPols + constTree + committedPols + rest
uint64_t constPolsReference = 0;
uint64_t constTreeReference = NCONSTPOLS;
uint64_t cmPolsReference    = NCONSTPOLS + 1;

#define isConstPols(i) ((i >= constPolsReference) && (i < constPolsReference + NCONSTPOLS))
#define isCmPols(i) ((i >= cmPolsReference) && (i < cmPolsReference + NPOLS))

void CopyPol2Reference(Goldilocks &fr, Reference &ref, const Pol *pPol);

void MemAlloc(Mem &mem, Goldilocks &fr, const Script &script, const Pols &cmPols, const Reference *constRefs, const string &constTreePolsInputFile)
{
    // Local variable
    Merkle M(MERKLE_ARITY);

    for (uint64_t i = 0; i < script.refs.size(); i++)
    {
        Reference ref;
        ref = script.refs[i];
        zkassert(ref.id == i);

        switch (ref.type)
        {
        case rt_pol:
        {
            zkassert(ref.pPol == NULL);
            zkassert(ref.N > 0);
            ref.memSize = sizeof(Goldilocks::Element) * ref.N;
            if ( isConstPols(i) || ( isCmPols(i) && cmPols.orderedPols[i - cmPolsReference]->elementType == et_field ) )
            {
                // No need to allocate memory, since we will reuse the mapped memory address
                // cout << "Skipping mem allocation i: " << i << endl;
            }
            else
            {
                ref.pPol = (Goldilocks::Element *)malloc(ref.memSize);
                if (ref.pPol == NULL)
                {
                    cerr << "Error MemAlloc() failed calling malloc() of size: " << ref.memSize << endl;
                    exit(-1);
                }
            }
            if (isConstPols(i))
            {
                ref.pPol = constRefs[i-constPolsReference].pPol;
            }
            else if (isCmPols(i))
            {
                CopyPol2Reference(fr, ref, cmPols.orderedPols[i - cmPolsReference]);
            }
            break;
        }
        case rt_field:
        {
            break;
        }
        case rt_treeGroup:
        {
            zkassert(ref.pTreeGroup == NULL);
            zkassert(ref.nGroups > 0);
            zkassert(ref.groupSize > 0);
            zkassert(ref.nPols == 0);

            ref.memSize = MerkleGroup::getTreeMemSize(&M, ref.nGroups, ref.groupSize);
            ref.pTreeGroup = (Goldilocks::Element *)malloc(ref.memSize);

            if (ref.pTreeGroup == NULL)
            {
                cerr << "Error MemAlloc() failed calling malloc() of size: " << ref.memSize << endl;
                exit(-1);
            }
            break;
        }
        case rt_treeGroup_elementProof:
        {
            zkassert(ref.pTreeGroup_elementProof == NULL);
            zkassert(ref.nGroups > 0);
            zkassert(ref.groupSize > 0);
            zkassert(ref.nPols == 0);

            MerkleGroup::getElementProofSize(&M, ref.nGroups, ref.groupSize, ref.memSize, ref.sizeValue, ref.sizeMpL, ref.sizeMpH);
            ref.pTreeGroup_elementProof = (Goldilocks::Element *)malloc(ref.memSize);

            zkassert(ref.sizeValue != 0);
            zkassert(ref.sizeMp == 0);
            //zkassert(ref.sizeMpL != 0); sizeMpl can be 0 in s1 Merkle tree proofs
            zkassert(ref.sizeMpH != 0);

            if (ref.pTreeGroup_elementProof == NULL)
            {
                cerr << "Error MemAlloc() failed calling malloc() of size: " << ref.memSize << endl;
                exit(-1);
            }

            break;
        }
        case rt_treeGroup_groupProof:
        {
            zkassert(ref.pTreeGroup_groupProof == NULL);
            zkassert(ref.nGroups > 0);
            zkassert(ref.groupSize > 0);
            zkassert(ref.nPols == 0);

            MerkleGroup::getGroupProofSize(&M, ref.nGroups, ref.groupSize, ref.memSize, ref.sizeValue, ref.sizeMp);
            ref.pTreeGroup_groupProof = (Goldilocks::Element *)malloc(ref.memSize);

            zkassert(ref.sizeValue != 0);
            zkassert(ref.sizeMp != 0);
            zkassert(ref.sizeMpL == 0);
            zkassert(ref.sizeMpH == 0);

            if (ref.pTreeGroup_groupProof == NULL)
            {
                cerr << "Error MemAlloc() failed calling malloc() of size: " << ref.memSize << endl;
                exit(-1);
            }

            break;
        }
        case rt_treeGroupMultipol:
        {
            zkassert(ref.pTreeGroupMultipol == NULL);
            zkassert(ref.nGroups > 0);
            zkassert(ref.groupSize > 0);
            zkassert(ref.nPols > 0);

            if (i == constTreeReference)
            {
                // TODO: Merge together with BME, to new golden prime
                //ref.pTreeGroupMultipol = MerkleGroupMultiPol::fileToMap(constTreePolsInputFile, /*mem[treeReference].pTreeGroupMultipol*/NULL, &M, ref.nGroups, ref.groupSize, ref.nPols); // TODO: Remove this unused attribute, and consider moving out of main loop
            }
            else
            {
                ref.memSize = MerkleGroupMultiPol::getTreeMemSize(&M, ref.nGroups, ref.groupSize, ref.nPols);
                ref.pTreeGroupMultipol = (Goldilocks::Element *)malloc(ref.memSize);
                if (ref.pTreeGroupMultipol == NULL)
                {
                    cerr << "Error MemAlloc() failed calling malloc() of size: " << ref.memSize << endl;
                    exit(-1);
                }
            }
            break;
        }
        case rt_treeGroupMultipol_groupProof:
        {
            zkassert(ref.pTreeGroupMultipol_groupProof == NULL);
            zkassert(ref.nGroups > 0);
            zkassert(ref.groupSize > 0);
            zkassert(ref.nPols > 0);

            MerkleGroupMultiPol::getGroupProofSize(&M, ref.nGroups, ref.groupSize, ref.nPols, ref.memSize, ref.sizeValue, ref.sizeMp);
            ref.pTreeGroupMultipol_groupProof = (Goldilocks::Element *)malloc(ref.memSize);

            zkassert(ref.sizeValue != 0);
            zkassert(ref.sizeMp != 0);
            zkassert(ref.sizeMpL == 0);
            zkassert(ref.sizeMpH == 0);

            if (ref.pTreeGroupMultipol_groupProof == NULL)
            {
                cerr << "Error MemAlloc() failed calling malloc() of size: " << ref.memSize << endl;
                exit(-1);
            }
            break;
        }
        case rt_idxArray:
        {
            zkassert(ref.pIdxArray == NULL);
            zkassert(ref.N > 0);
            ref.memSize = sizeof(uint32_t) * ref.N;
            ref.pIdxArray = (uint32_t *)malloc(ref.memSize);
            if (ref.pIdxArray == NULL)
            {
                cerr << "Error MemAlloc() failed calling malloc() of size: " << ref.memSize << endl;
                exit(-1);
            }
            break;
        }
        case rt_int:
        {
            break;
        }
        default:
        {
            cerr << "Error: MemAlloc() found an unrecognized reference type: " << ref.type << " at id=" << ref.id << endl;
            exit(-1);
        }
        }

        // Store the reference instance
        mem.push_back(ref);
    }
}

void MemFree(Mem &mem)
{
    zkassert(mem[constTreeReference].pTreeGroupMultipol != NULL);
    munmap(mem[constTreeReference].pTreeGroupMultipol, mem[constTreeReference].memSize);
    mem[constTreeReference].pTreeGroupMultipol = NULL;

    for (uint64_t i = NCONSTPOLS + 1 + NEVALUATIONS; i < mem.size(); i++)
    {
        zkassert(mem[i].id == i);

        switch (mem[i].type)
        {
        case rt_pol:
        {
            zkassert(mem[i].pPol != NULL);
            free(mem[i].pPol);
            break;
        }
        case rt_field:
        {
            break;
        }
        case rt_treeGroup:
        {
            zkassert(mem[i].pTreeGroup != NULL);
            free(mem[i].pTreeGroup);
            break;
        }
        case rt_treeGroup_elementProof:
        {
            zkassert(mem[i].pTreeGroup_elementProof != NULL);
            free(mem[i].pTreeGroup_elementProof);
            break;
        }
        case rt_treeGroup_groupProof:
        {
            zkassert(mem[i].pTreeGroup_groupProof != NULL);
            free(mem[i].pTreeGroup_groupProof);
            break;
        }
        case rt_treeGroupMultipol:
        {
            zkassert(mem[i].pTreeGroupMultipol != NULL);
            free(mem[i].pTreeGroupMultipol);
            break;
        }
        case rt_treeGroupMultipol_groupProof:
        {
            zkassert(mem[i].pTreeGroupMultipol_groupProof != NULL);
            free(mem[i].pTreeGroupMultipol_groupProof);
            break;
        }
        case rt_idxArray:
            zkassert(mem[i].pIdxArray != NULL);
            free(mem[i].pIdxArray);
            break;
        case rt_int:
            break;
        default:
        {
            cerr << "Error: MemFree() found an unrecognized reference type: " << mem[i].type << " at id=" << mem[i].id << endl;
            exit(-1);
        }
        }
    }

    // Clear the vector, destroying all its reference instances
    mem.clear();
}
#endif

void CopyPol2Reference(Goldilocks &fr, Reference &ref, const Pol *pPol)
{
    switch (pPol->elementType)
    {
    case et_bool:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            ref.pPol[j] = fr.fromU64(((PolBool *)pPol)->pData[j]);
        }
        break;
    case et_s8:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            ref.pPol[j] = fr.fromS32(((PolS8 *)pPol)->pData[j]);
        }
        break;
    case et_u8:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            ref.pPol[j] = fr.fromU64(((PolU8 *)pPol)->pData[j]);
        }
        break;
    case et_s16:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            ref.pPol[j] = fr.fromS32(((PolS16 *)pPol)->pData[j]);
        }
        break;
    case et_u16:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            ref.pPol[j] = fr.fromU64(((PolU16 *)pPol)->pData[j]);
        }
        break;
    case et_s32:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            ref.pPol[j] = fr.fromS32(((PolS32 *)pPol)->pData[j]);
        }
        break;
    case et_u32:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            ref.pPol[j] = fr.fromU64(((PolU32 *)pPol)->pData[j]);
        }
        break;
    case et_s64:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            ref.pPol[j] = fr.fromS32(((PolS64 *)pPol)->pData[j]);
        }
        break;
    case et_u64:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            ref.pPol[j] = fr.fromU64(((PolU64 *)pPol)->pData[j]);
        }
        break;
    case et_field:
        //memcpy(ref.pPol, ((PolFieldElement *)pPol)->pData, sizeof(FieldElement) * NEVALUATIONS);
        ref.pPol = ((PolFieldElement *)pPol)->pData;
        break;
    default:
        cerr << "Error: CopyPol2Reference() found invalid elementType pol" << endl;
        exit(-1);
    }
}

void Pols2Refs(Goldilocks &fr, const Pols &pol, Reference *ref)
{
    for (uint64_t i=0; i<pol.size; i++)
    {
        ref[i].elementType = et_field;
        ref[i].N = NEVALUATIONS;
        ref[i].memSize = sizeof(Goldilocks::Element) * ref[i].N;
        if (pol.orderedPols[i]->elementType == et_field)
        {
            ref[i].pPol = NULL;
        }
        else
        {
            ref[i].pPol = (Goldilocks::Element *)malloc(ref[i].memSize);
            if (ref[i].pPol == NULL)
            {
                cerr << "Error Pols2Refs() failed calling malloc() of size: " << ref[i].memSize << endl;
                exit(-1);
            }
        }
        CopyPol2Reference(fr, ref[i], pol.orderedPols[i]);
    }
}