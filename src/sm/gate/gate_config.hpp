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
        soutRefDistance(0) {};
};

#endif