#ifndef MERKLE_GROUP
#define MERKLE_GROUP

#include "poseidon_opt.hpp"
#include "ffiasm/fr.hpp"
#include "merkle.hpp"
#include <vector>
#include <cassert>
#include <math.h> /* floor */
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace std;

class MerkleGroup
{

private:
    Poseidon_opt poseidon;
    RawFr field;
    Merkle *M;
    uint32_t nGroups, groupSize;
    string fileName;

public:
    uint64_t merkleGroupTreeSize;
    uint32_t mainTreeSize, mainTreeProofSize, groupProofSize, groupTreesSize, elementProofSize;

    MerkleGroup(Merkle *_M, uint32_t _nGroups, uint32_t _groupSize);
    void merkelize(RawFr::Element *tree, vector<RawFr::Element> pols);
    RawFr::Element root(RawFr::Element *tree);
    void getGroupProof(RawFr::Element *tree, uint32_t idx, RawFr::Element *v, uint32_t v_size, RawFr::Element *mp, uint32_t mp_size);
    RawFr::Element calculateRootFromGroupProof(RawFr::Element *mp, uint32_t mp_size, uint32_t groupIdx, RawFr::Element *groupElements, uint32_t groupElements_size);
    bool verifyGroupProof(RawFr::Element root, RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *groupElements, uint32_t groupElements_size);
    void getElementsProof(RawFr::Element *tree, uint32_t idx, RawFr::Element *val, uint32_t val_size, RawFr::Element *mp, uint32_t mp_size);
    RawFr::Element calculateRootFromElementProof(RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *val, uint32_t val_size);
    bool verifyElementProof(RawFr::Element root, RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *val, uint32_t val_size);

    static uint32_t getTreeSize(Merkle *M, uint32_t nGroups, uint32_t groupSize);
    static uint32_t getElementProofSize(Merkle *M, uint32_t nGroups, uint32_t groupSize);
    static uint32_t getGroupProofSize(Merkle *M, uint32_t nGroups, uint32_t groupSize);
};
#endif // MERKLE_GROUP