#ifndef MERKLE
#define MERKLE

#include "poseidon_opt/poseidon_opt.hpp"
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
    uint8_t arity;

    uint32_t merkelizeLevel(vector<FrElement> &elements, uint32_t offset);
    uint32_t _nElements(uint32_t n);

public:
    Merkle(uint8_t _arity);
    void merkelize(vector<Merkle::FrElement> &elements);
    FrElement getElement(vector<FrElement> &elements, uint32_t idx);
    vector<FrElement> genMerkleProof(vector<FrElement> &tree, uint32_t idx, uint32_t offset);
    FrElement calculateRootFromProof(vector<FrElement> &mp, uint32_t idx, FrElement value, uint32_t offset);
    bool verifyMerkleProof(FrElement root, vector<FrElement> &mp, uint32_t idx, FrElement value, uint32_t offset);
    Merkle::FrElement root(vector<FrElement> &tree);
};
#endif // MERKLE