#include "config.hpp"
#include "mem.hpp"
#include "scalar.hpp"

void MemAlloc (Mem &mem, const Script &script)
{
    for (uint64_t i=0; i<script.refs.size(); i++)
    {
        Reference ref;
        ref = script.refs[i];
        zkassert(ref.id==i);

        switch (ref.type)
        {
            case rt_pol:
            {
                zkassert(ref.pPol==NULL);
                zkassert(ref.N>0);
                ref.memSize = sizeof(RawFr::Element)*ref.N;
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
            case rt_treeGroup: // TODO
            case rt_treeGroup_elementProof: // TODO
            case rt_treeGroup_groupProof: // TODO
            case rt_treeGroupMultipol: // TODO
            case rt_treeGroupMultipol_groupProof: // TODO
            case rt_idxArray: // TODO
                break;
            case rt_int:
                break;
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

void MemFree (Mem &mem)
{
    for (uint64_t i=0; i<mem.size(); i++)
    {
        zkassert(mem[i].id==i);

        switch (mem[i].type)
        {
            case rt_pol:
            {
                zkassert(mem[i].pPol!=NULL);
                free(mem[i].pPol);
                break;
            }
            case rt_field:
            {
                break;
            }
            case rt_treeGroup: // TODO
            case rt_treeGroup_elementProof: // TODO
            case rt_treeGroup_groupProof: // TODO
            case rt_treeGroupMultipol: // TODO
            case rt_treeGroupMultipol_groupProof: // TODO
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
    switch(pPol->elementType)
    {
        case et_bool:
            for (int j=0; j<NEVALUATIONS; j++)
            {
                u82fe(fr, ref.pPol[j], ((PolBool *)pPol)->pData[j]);
            }
            break;
        case et_s8:
            for (int j=0; j<NEVALUATIONS; j++)
            {
                s82fe(fr, ref.pPol[j], ((PolS8 *)pPol)->pData[j]);
            }
            break;
        case et_u8:
            for (int j=0; j<NEVALUATIONS; j++)
            {
                u82fe(fr, ref.pPol[j], ((PolU8 *)pPol)->pData[j]);
            }
            break;
        case et_s16:
            for (int j=0; j<NEVALUATIONS; j++)
            {
                s162fe(fr, ref.pPol[j], ((PolS16 *)pPol)->pData[j]);
            }
            break;
        case et_u16:
            for (int j=0; j<NEVALUATIONS; j++)
            {
                u162fe(fr, ref.pPol[j], ((PolU16 *)pPol)->pData[j]);
            }
            break;
        case et_s32:
            for (int j=0; j<NEVALUATIONS; j++)
            {
                s322fe(fr, ref.pPol[j], ((PolS32 *)pPol)->pData[j]);
            }
            break;
        case et_u32:
            for (int j=0; j<NEVALUATIONS; j++)
            {
                u322fe(fr, ref.pPol[j], ((PolU32 *)pPol)->pData[j]);
            }
            break;
        case et_s64:
            for (int j=0; j<NEVALUATIONS; j++)
            {
                s642fe(fr, ref.pPol[j], ((PolS64 *)pPol)->pData[j]);
            }
            break;
        case et_u64:
            for (int j=0; j<NEVALUATIONS; j++)
            {
                u642fe(fr, ref.pPol[j], ((PolU64 *)pPol)->pData[j]);
            }
            break;
        case et_field:
            memcpy(ref.pPol, ((PolFieldElement *)pPol)->pData, sizeof(RawFr::Element)*NEVALUATIONS);
            break;
        default:
            cerr << "Error: CopyPol2Reference() found invalid elementType pol" << endl;
            exit(-1);
    }
}

void MemCopyPols (RawFr &fr, Mem &mem, const Pols &cmPols, const Pols &constPols)
{
    uint64_t i=0;
    for (; i<cmPols.size; i++)
    {
        zkassert(i < mem.size());
        zkassert(mem[i].type == rt_pol);
        zkassert(mem[i].N == NEVALUATIONS);
        CopyPol2Reference(fr, mem[i], cmPols.orderedPols[i]);
    }
    for (; i<constPols.size; i++)
    {
        zkassert(i < mem.size());
        zkassert(mem[i].type == rt_pol);
        zkassert(mem[i].N == NEVALUATIONS);
        CopyPol2Reference(fr, mem[i], constPols.orderedPols[i]);
    }
}