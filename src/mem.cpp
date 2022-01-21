#include "config.hpp"
#include "mem.hpp"
#include "scalar.hpp"
#include "merkle_group.hpp"
#include "merkle_group_multipol.hpp"

void CopyPol2Reference(RawFr &fr, Reference &ref, const Pol *pPol);

void MemAlloc(Mem &mem, RawFr &fr, const Script &script, const Pols &cmPols, const Pols &constPols, const string &constTreePolsInputFile)
{
    // Local variable
    Merkle M(MERKLE_ARITY);

    // Load ConstantTree
    uint64_t constPolsReference = 0;
    uint64_t constTreeReference = constPols.size;
    uint64_t cmPolsReference    = constPols.size + 1;

#define isConstPols(i) ((i >= constPolsReference) && (i < constPolsReference + constPols.size))
#define isCmPols(i) ((i >= cmPolsReference) && (i < cmPolsReference + cmPols.size))

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
            ref.memSize = sizeof(RawFr::Element) * ref.N;
            if ( ( isConstPols(i) && constPols.orderedPols[i - constPolsReference]->elementType == et_field ) ||
                 ( isCmPols(i) && cmPols.orderedPols[i - cmPolsReference]->elementType == et_field ) )
            {
                // No need to allocate memory, since we will reuse the mapped memory address
                //cout << "Skipping mem allocation i: " << i << endl;
                // TODO: Convert constPols and cmPols to FE arrays at initialization, to avoid copies at every iteration
            }
            else
            {
                ref.pPol = (RawFr::Element *)malloc(ref.memSize);
                if (ref.pPol == NULL)
                {
                    cerr << "Error MemAlloc() failed calling malloc() of size: " << ref.memSize << endl;
                    exit(-1);
                }
            }
            if (isConstPols(i))
            {
                CopyPol2Reference(fr, ref, constPols.orderedPols[i - constPolsReference]);
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
            ref.pTreeGroup = (RawFr::Element *)malloc(ref.memSize);

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
            ref.pTreeGroup_elementProof = (RawFr::Element *)malloc(ref.memSize);

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
            ref.pTreeGroup_groupProof = (RawFr::Element *)malloc(ref.memSize);

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
                ref.pTreeGroupMultipol = MerkleGroupMultiPol::fileToMap(constTreePolsInputFile, /*mem[treeReference].pTreeGroupMultipol*/NULL, &M, ref.nGroups, ref.groupSize, ref.nPols); // TODO: Remove this unused attribute
            }
            else
            {
                ref.memSize = MerkleGroupMultiPol::getTreeMemSize(&M, ref.nGroups, ref.groupSize, ref.nPols);
                ref.pTreeGroupMultipol = (RawFr::Element *)malloc(ref.memSize);
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
            ref.pTreeGroupMultipol_groupProof = (RawFr::Element *)malloc(ref.memSize);

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

void MemFree(Mem &mem, const Pols &cmPols, const Pols &constPols, const string &constTreePolsInputFile)
{
    uint32_t treeReference = constPols.size;
    zkassert(mem[treeReference].pTreeGroupMultipol != NULL);
    munmap(mem[treeReference].pTreeGroupMultipol, mem[treeReference].memSize);
    mem[treeReference].pTreeGroupMultipol = NULL;

    for (uint64_t i = constPols.size + 1 + cmPols.size; i < mem.size(); i++)
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

void CopyPol2Reference(RawFr &fr, Reference &ref, const Pol *pPol)
{
    switch (pPol->elementType)
    {
    case et_bool:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            u82fe(fr, ref.pPol[j], ((PolBool *)pPol)->pData[j]);
        }
        break;
    case et_s8:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            s82fe(fr, ref.pPol[j], ((PolS8 *)pPol)->pData[j]);
        }
        break;
    case et_u8:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            u82fe(fr, ref.pPol[j], ((PolU8 *)pPol)->pData[j]);
        }
        break;
    case et_s16:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            s162fe(fr, ref.pPol[j], ((PolS16 *)pPol)->pData[j]);
        }
        break;
    case et_u16:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            u162fe(fr, ref.pPol[j], ((PolU16 *)pPol)->pData[j]);
        }
        break;
    case et_s32:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            s322fe(fr, ref.pPol[j], ((PolS32 *)pPol)->pData[j]);
        }
        break;
    case et_u32:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            u322fe(fr, ref.pPol[j], ((PolU32 *)pPol)->pData[j]);
        }
        break;
    case et_s64:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            s642fe(fr, ref.pPol[j], ((PolS64 *)pPol)->pData[j]);
        }
        break;
    case et_u64:
        for (int j = 0; j < NEVALUATIONS; j++)
        {
            u642fe(fr, ref.pPol[j], ((PolU64 *)pPol)->pData[j]);
        }
        break;
    case et_field:
        ref.pPol = ((PolFieldElement *)pPol)->pData;
        break;
    default:
        cerr << "Error: CopyPol2Reference() found invalid elementType pol" << endl;
        exit(-1);
    }
}