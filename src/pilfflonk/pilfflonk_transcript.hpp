#ifndef PILFFLONK_TRANSCRIPT_HPP
#define PILFFLONK_TRANSCRIPT_HPP

#include <any>
#include <vector>
#include <sstream>
#include <alt_bn128.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

class PilFflonkTranscript {
    using FrElement = typename AltBn128::Engine::FrElement;
    using G1Point = typename AltBn128::Engine::G1Point;

    enum ElementTypeEnum {
        FrType, G1Type
    };
    struct ElementTypeStruct {
        ElementTypeEnum type;
        std::any element;
    };

    AltBn128::Engine &E;

    int fieldElements;
    int groupElements;

    std::vector<ElementTypeStruct> elements;

    u_int64_t toRprBE(G1Point &point, uint8_t *data, int64_t seek, int64_t size);

    void hashToFr(FrElement &element, u_int8_t *data, int64_t size);

public:
    PilFflonkTranscript(AltBn128::Engine &E);

    void addScalar(FrElement value);

    void addPolCommitment(G1Point value);

    void reset();

    int nElements() { return elements.size(); }

    typename AltBn128::Engine::FrElement getChallenge();

    int64_t keccak (void *data, int64_t dataSize, void *hash, int64_t HashSize);
};

#endif
