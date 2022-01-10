#ifndef MERKLE
#define MERKLE

#include "poseidon_opt.hpp"
#include "ffiasm/fr.hpp"
#include <vector>

#define N8 32

using namespace std;

class Merkle
{
    typedef RawFr::Element FrElement;

private:
    Poseidon_opt poseidon;
    RawFr field;

    uint32_t _nElements(uint32_t n);

    uint32_t merkelizeLevel(vector<FrElement> &elements, uint32_t offset);
    uint32_t merkelizeLevel(Merkle::FrElement *elements, uint32_t size, uint32_t offset);

public:
    Merkle(uint8_t _arity);
    uint8_t arity;

    uint32_t numHashes(uint32_t n);
    void merkelize(vector<Merkle::FrElement> &elements);
    void merkelize(Merkle::FrElement *elements, uint32_t size);

    RawFr::Element getElement(vector<FrElement> &elements, uint32_t idx);
    FrElement getElement(Merkle::FrElement *elements, uint32_t idx);

    vector<FrElement> genMerkleProof(vector<FrElement> &tree, uint32_t idx, uint32_t offset);
    void genMerkleProof(Merkle::FrElement *elements, uint32_t size, uint32_t idx, uint32_t offset, Merkle::FrElement *proof, uint32_t proofSize);

    uint32_t MerkleProofSize(uint32_t size);
    uint32_t MerkleProofSizeBytes(uint32_t size);

    FrElement calculateRootFromProof(vector<FrElement> &mp, uint32_t idx, FrElement value, uint32_t offset);
    FrElement calculateRootFromProof(Merkle::FrElement *mp, uint32_t size, uint32_t idx, FrElement value, uint32_t offset);

    bool verifyMerkleProof(FrElement root, vector<FrElement> &mp, uint32_t idx, FrElement value, uint32_t offset);
    bool verifyMerkleProof(FrElement root, Merkle::FrElement *mp, uint32_t mp_size, uint32_t idx, FrElement value, uint32_t offset);

    RawFr::Element root(vector<FrElement> &tree);
    RawFr::Element root(Merkle::FrElement *tree, uint32_t size);
};
#endif // MERKLE