#include <cstdint>

#ifndef GATE_CONFIG_HPP
#define GATE_CONFIG_HPP

class GateConfig
{
public:
    uint64_t zeroRef;
    uint64_t slotSize;
    uint64_t maxRefs;
    uint64_t firstNextRef;
    uint64_t sinRef0;
    uint64_t sinRefNumber;
    uint64_t sinRefDistance;
    uint64_t soutRef0;
    uint64_t soutRefNumber;
    uint64_t soutRefDistance;
    uint64_t polLength;

    GateConfig() :
        zeroRef(0),
        slotSize(0),
        maxRefs(0),
        firstNextRef(0),
        sinRef0(0),
        sinRefNumber(0),
        sinRefDistance(0),
        soutRef0(0),
        soutRefNumber(0),
        soutRefDistance(0),
        polLength(0) {};

    GateConfig(uint64_t zeroRef, uint64_t slotSize, uint64_t maxRefs, uint64_t firstNextRef, uint64_t sinRef0, uint64_t sinRefNumber, uint64_t sinRefDistance, uint64_t soutRef0, uint64_t soutRefNumber, uint64_t soutRefDistance, uint64_t polLength) :
        zeroRef(zeroRef),
        slotSize(slotSize),
        maxRefs(maxRefs),
        firstNextRef(firstNextRef),
        sinRef0(sinRef0),
        sinRefNumber(sinRefNumber),
        sinRefDistance(sinRefDistance),
        soutRef0(soutRef0),
        soutRefNumber(soutRefNumber),
        soutRefDistance(soutRefDistance),
        polLength(polLength) {};

    // Converts relative references to absolute references, based on the slot
    inline uint64_t relRef2AbsRef (uint64_t ref, uint64_t slot)
    {

        // ZeroRef is the same for all the slots, and it is at reference 0
        if (ref==zeroRef) return zeroRef;

        // Next references have an offset of one slot size per slot
        return slot*slotSize + ref;
    };
};

#endif