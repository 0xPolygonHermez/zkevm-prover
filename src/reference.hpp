#ifndef REFERENCE_HPP
#define REFERENCE_HPP

#include <pol_types.hpp>
#include "ff/ff.hpp"

enum eReferenceType
{
    rt_unknown = 0,
    rt_pol = 1,
    rt_field = 2,
    rt_treeGroup = 3,
    rt_treeGroup_elementProof = 4,
    rt_treeGroup_groupProof = 5,
    rt_treeGroupMultipol = 6,
    rt_treeGroupMultipol_groupProof = 7,
    rt_idxArray = 8,
    rt_int = 9
};

class Reference
{
public:
    uint64_t id;         // Mandatory
    eReferenceType type; // Mandatory
    uint64_t N;
    eElementType elementType;
    uint64_t nGroups;
    uint64_t groupSize;
    uint64_t nPols;
    uint64_t memSize;   // Size of the element in bytes
    uint64_t sizeValue; // Number of elements of the value
    uint64_t sizeMp;    // Number of elements of the Merkleproof
    uint64_t sizeMpL;   // Number of elements of the Merkleproof Low, only for merkle_group_elementProof and needs SizeMpH != 0
    uint64_t sizeMpH;   // Number of elements of the Merkleproof High, only for merkle_group_elementProof and needs SizeMpL != 0
    FieldElement *pPol;
    FieldElement fe;
    FieldElement *pTreeGroup;
    FieldElement *pTreeGroup_groupProof;
    FieldElement *pTreeGroup_elementProof;
    FieldElement *pTreeGroupMultipol;
    FieldElement *pTreeGroupMultipol_groupProof;
    uint32_t *pIdxArray;
    uint32_t integer;

    Reference() : id(0xFFFFFFFFFFFFFFFF),
                  type(rt_unknown),
                  N(0),
                  elementType(et_unknown),
                  nGroups(0),
                  groupSize(0),
                  nPols(0),
                  memSize(0),
                  sizeValue(0),
                  sizeMp(0),
                  sizeMpL(0),
                  sizeMpH(0),
                  pPol(NULL),
                  pTreeGroup(NULL),
                  pTreeGroup_groupProof(NULL),
                  pTreeGroup_elementProof(NULL),
                  pTreeGroupMultipol(NULL),
                  pTreeGroupMultipol_groupProof(NULL),
                  pIdxArray(NULL),
                  integer(0){};
};

#endif