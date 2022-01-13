#ifndef PAR_MULTIEXP2
#define PAR_MULTIEXP2

#define PME2_PACK_FACTOR 2
#define PME2_MAX_CHUNK_SIZE_BITS 16
#define PME2_MIN_CHUNK_SIZE_BITS 2

template <typename Curve>
class ParallelMultiexp {

    struct PaddedPoint {
        typename Curve::Point p;
//        uint8_t padding[32];
    };

    typename Curve::PointAffine *bases;
    uint8_t* scalars;
    uint32_t scalarSize;
    uint32_t n;
    uint32_t nThreads;
    uint32_t bitsPerChunk;
    uint64_t accsPerChunk;
    uint32_t nChunks;
    Curve &g;
    PaddedPoint *accs;

    void initAccs();

    uint32_t getChunk(uint32_t scalarIdx, uint32_t chunkIdx);
    void processChunk(uint32_t idxChunk);
    void packThreads();
    void reduce(typename Curve::Point &res, uint32_t nBits);

public:
    ParallelMultiexp(Curve &_g): g(_g) {}
    void multiexp(typename Curve::Point &r, typename Curve::PointAffine *_bases, uint8_t* _scalars, uint32_t _scalarSize, uint32_t _n, uint32_t _nThreads=0);

};

#include "multiexp.c.hpp"

#endif // PAR_MULTIEXP2
