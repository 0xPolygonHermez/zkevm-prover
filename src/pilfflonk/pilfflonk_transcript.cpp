#include "pilfflonk_transcript.hpp"
#include <cstring>
#include "Keccak-more-compact.hpp"

PilFflonkTranscript::PilFflonkTranscript(AltBn128::Engine &_E) : E(_E)
{
    reset();
}

int64_t PilFflonkTranscript::keccak(void *data, int64_t dataSize, void *hash, int64_t hashSize)
{
    if (hashSize < 32) {
        return -1;
    }
    Keccak(1088, 512, (unsigned char *) data, dataSize, 0x01, (unsigned char *) hash, 32);
    return 32;
}

void PilFflonkTranscript::addScalar(FrElement value)
{
    ElementTypeStruct e = {.type = FrType, .element = value};
    elements.push_back(e);
    fieldElements++;
}

void PilFflonkTranscript::addPolCommitment(G1Point value)
{
    ElementTypeStruct e = {.type = G1Type, .element = value};
    elements.push_back(e);
    groupElements++;
}

void PilFflonkTranscript::reset()
{
    fieldElements = 0;
    groupElements = 0;

    this->elements.clear();
}

typename AltBn128::Engine::FrElement PilFflonkTranscript::getChallenge()
{
    const u_int32_t length = E.fr.bytes() * fieldElements + E.g1.F.bytes() * groupElements * 2 * 3;
    u_int8_t data[length];
    u_int64_t bytes = 0;

    memset(data, 0, length);
    for (unsigned long i = 0; i < this->elements.size(); i++)
    {
        if (FrType == elements[i].type)
        {
            FrElement element = std::any_cast<FrElement>(elements[i].element);
            bytes += E.fr.toRprBE(element, data + bytes, E.fr.bytes());
        }
        else
        {
            G1Point element = std::any_cast<G1Point>(elements[i].element);
            bytes += toRprBE(element, data, bytes, E.g1.F.bytes());
        }
    }

    FrElement res;
    hashToFr(res, data, bytes);

    return res;
}

u_int64_t PilFflonkTranscript::toRprBE(G1Point &point, uint8_t *data, int64_t seek, int64_t size)
{
    int64_t bytes = E.g1.F.bytes() * 2;
    typename AltBn128::Engine::G1PointAffine p;

    E.g1.copy(p, point);

    bytes = E.g1.F.toRprBE(p.x, data + seek, size);
    bytes += E.g1.F.toRprBE(p.y, data + seek + bytes, size);
    return bytes;
}

void PilFflonkTranscript::hashToFr(FrElement &element, u_int8_t *data, int64_t size)
{
    u_int8_t hash[32];
    keccak(data, size, hash, sizeof(hash));
    E.fr.fromRprBE(element, hash, sizeof(hash));
}
