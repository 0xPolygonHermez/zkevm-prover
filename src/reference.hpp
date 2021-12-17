#ifndef REFERENCE_HPP
#define REFERENCE_HPP

#include <pol_types.hpp>
#include "ffiasm/fr.hpp"

enum eReferenceType {
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
    uint64_t id; // Mandatory
    eReferenceType type; // Mandatory
    uint64_t N;
    eElementType elementType;
    uint64_t nGroups;
    uint64_t groupSize;
    uint64_t nPols;

    uint64_t memSize;
    RawFr::Element * pPol;
    RawFr::Element fe;
    // * pTreeGroup;
    // * pTreeGroup_elementProof;
    // * pTreeGroup_groupProof;
    // * pTreeGroupMultipol;
    // * pTreeGroupMultipol_groupProof;
    uint32_t * pIdxArray;
    uint32_t integer;

    Reference() : id(0xFFFFFFFFFFFFFFFF), 
                  type(rt_unknown),
                  N(0),
                  elementType(et_unknown),
                  nGroups(0),
                  groupSize(0),
                  nPols(0),
                  memSize(0),
                  pPol(NULL),
                  pIdxArray(NULL),
                  integer(0) {};
};

#endif