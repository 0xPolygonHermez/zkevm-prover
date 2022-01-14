#include "config.hpp"
#include "mem.hpp"
#include "scalar.hpp"
#include "merkle_group.hpp"
#include "merkle_group_multipol.hpp"

void MemAlloc(Mem &mem, const Script &script)
{
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
            ref.memSize = sizeof(RawFr::Element) * ref.N;
            ref.pPol = (RawFr::Element *)malloc(ref.memSize);
            if (ref.pPol == NULL)
            {
                cerr << "Error MemAlloc() failed calling malloc() of size: " << ref.memSize << endl;
                exit(-1);
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
            zkassert(ref.sizeMpL != 0);
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

            ref.memSize = MerkleGroupMultiPol::getTreeMemSize(&M, ref.nGroups, ref.groupSize, ref.nPols);
            ref.pTreeGroupMultipol = (RawFr::Element *)malloc(ref.memSize);

            if (ref.pTreeGroupMultipol == NULL)
            {
                cerr << "Error MemAlloc() failed calling malloc() of size: " << ref.memSize << endl;
                exit(-1);
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

void MemFree(Mem &mem)
{
    for (uint64_t i = 0; i < mem.size(); i++)
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
            if (mem[i].pTreeGroupMultipol != NULL)
            {
                free(mem[i].pTreeGroupMultipol);
            }
            break;
        }
        case rt_treeGroupMultipol_groupProof:
        {
            zkassert(mem[i].pTreeGroupMultipol_groupProof != NULL);
            free(mem[i].pTreeGroupMultipol_groupProof);
            break;
        }
        case rt_idxArray: // TODO
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
        memcpy(ref.pPol, ((PolFieldElement *)pPol)->pData, sizeof(RawFr::Element) * NEVALUATIONS);
        break;
    default:
        cerr << "Error: CopyPol2Reference() found invalid elementType pol" << endl;
        exit(-1);
    }
}

void MemCopyPols(RawFr &fr, Mem &mem, const Pols &cmPols, const Pols &constPols, const string &constTreePolsInputFile)
{
    // Load constant polynomials
    for (uint64_t i = 0; i < constPols.size; i++)
    {
        zkassert(i < mem.size());
        zkassert(mem[i].type == rt_pol);
        zkassert(mem[i].N == NEVALUATIONS);
        CopyPol2Reference(fr, mem[i], constPols.orderedPols[i]);
    }

    // Load ConstantTree
    uint32_t treeReference = constPols.size;

    zkassert(treeReference < mem.size());
    zkassert(mem[treeReference].type == rt_treeGroupMultipol);
    zkassert(mem[treeReference].nGroups != 0);
    zkassert(mem[treeReference].groupSize != 0);
    zkassert(mem[treeReference].nPols != 0);

    free(mem[treeReference].pTreeGroupMultipol);

    Merkle M(MERKLE_ARITY);
    mem[treeReference].pTreeGroupMultipol = MerkleGroupMultiPol::fileToMap(constTreePolsInputFile, mem[treeReference].pTreeGroupMultipol, &M, mem[treeReference].nGroups, mem[treeReference].groupSize, mem[treeReference].nPols);

    // Load committed polynomials
    for (uint64_t i = 0; i < cmPols.size; i++)
    {
        uint64_t ref = i + constPols.size + 1;
        zkassert(ref < mem.size());
        zkassert(mem[ref].type == rt_pol);
        zkassert(mem[ref].N == NEVALUATIONS);
        CopyPol2Reference(fr, mem[ref], cmPols.orderedPols[i]);
    }
}

void MemUncopyPols(RawFr &fr, Mem &mem, const Pols &cmPols, const Pols &constPols, const string &constTreePolsInputFile)
{
    // TODO: Undo this:
    // mem[treeReference].pTreeGroupMultipol = MerkleGroupMultiPol::fileToMap(constTreePolsInputFile, mem[treeReference].pTreeGroupMultipol, &M, mem[treeReference].nGroups, mem[treeReference].groupSize, mem[treeReference].nPols);
    uint32_t treeReference = constPols.size;
    mem[treeReference].pTreeGroupMultipol = NULL;
}