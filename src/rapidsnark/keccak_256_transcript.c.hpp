#include "keccak_256_transcript.hpp"
#include <cstring>
#include "keccak_wrapper.hpp"

template<typename Engine>
Keccak256Transcript<Engine>::Keccak256Transcript(Engine &_E) : E(_E) {
    reset();
}

template<typename Engine>
void Keccak256Transcript<Engine>::addScalar(FrElement value) {
    ElementTypeStruct e = { .type = FrType, .element = value};
    elements.push_back(e);
    fieldElements++;
}

template<typename Engine>
void Keccak256Transcript<Engine>::addPolCommitment(G1Point value) {
    ElementTypeStruct e = { .type = G1Type, .element = value};
    elements.push_back(e);
    groupElements++;
}

template<typename Engine>
void Keccak256Transcript<Engine>::reset() {
    fieldElements = 0;
    groupElements = 0;

    this->elements.clear();
}

template<typename Engine>
typename Engine::FrElement Keccak256Transcript<Engine>::getChallenge() {
    const u_int32_t length = E.fr.bytes() * fieldElements + E.g1.F.bytes() * groupElements * 2 * 3;
    u_int8_t data[length];
    u_int64_t bytes = 0;

    memset(data, 0, length);
    for (unsigned long i = 0; i < this->elements.size(); i++) {
        if(FrType == elements[i].type) {
            FrElement element = std::any_cast<FrElement>(elements[i].element);
            bytes += E.fr.toRprBE(element, data + bytes, E.fr.bytes());
        } else {
            G1Point element = std::any_cast<G1Point>(elements[i].element);
            bytes += toRprBE(element, data, bytes, E.g1.F.bytes());
        }
    }

    FrElement res;
    hashToFr(res, data, bytes);

    return res;
}

template <typename Engine>
u_int64_t Keccak256Transcript<Engine>::toRprBE(G1Point &point, uint8_t *data, int64_t seek, int64_t size)
{
    int64_t bytes = E.g1.F.bytes() * 2;
    typename Engine::G1PointAffine p;

    E.g1.copy(p, point);

    if (E.g1.isZero(p)) {
        memset(data, 0, bytes);
        return 0;
    }
    bytes = E.g1.F.toRprBE(p.x, data + seek, size);
    bytes += E.g1.F.toRprBE(p.y, data + seek + bytes, size);
    return bytes;
}

template <typename Engine>
void Keccak256Transcript<Engine>::hashToFr(FrElement &element, u_int8_t *data, int64_t size)
{
    u_int8_t hash[32];
    keccak(data, size, hash, sizeof(hash));
    E.fr.fromRprBE(element, hash, sizeof(hash));
}
