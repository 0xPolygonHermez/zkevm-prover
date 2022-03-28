#if 0
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
    uint32_t mainTreeSize, mainTreeProofSize, groupProofSize, groupTreesSize, elementProofSize, mp_group_size, mp_main_size;

    MerkleGroup(Merkle *_M, uint32_t _nGroups, uint32_t _groupSize);
    void merkelize(RawFr::Element *tree, vector<RawFr::Element> pols);
    void merkelize(RawFr::Element *tree, RawFr::Element *pols);

    RawFr::Element root(RawFr::Element *tree);
    void getGroupProof(RawFr::Element *tree, uint32_t idx, RawFr::Element *v);
    void getGroupProof(RawFr::Element *tree, uint32_t idx, RawFr::Element *v, uint32_t v_size, RawFr::Element *mp, uint32_t mp_size);
    RawFr::Element calculateRootFromGroupProof(RawFr::Element *mp, uint32_t mp_size, uint32_t groupIdx, RawFr::Element *groupElements, uint32_t groupElements_size);
    bool verifyGroupProof(RawFr::Element root, RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *groupElements, uint32_t groupElements_size);
    void getElementsProof(RawFr::Element *tree, uint32_t idx, RawFr::Element *elementsProof);

    RawFr::Element calculateRootFromElementProof(RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *val, uint32_t val_size);
    bool verifyElementProof(RawFr::Element root, RawFr::Element *mp, uint32_t mp_size, uint32_t idx, RawFr::Element *val, uint32_t val_size);

    static uint32_t getTreeMemSize(Merkle *M, uint32_t nGroups, uint32_t groupSize);
    static void getElementProofSize(Merkle *M, uint32_t nGroups, uint32_t groupSize, uint64_t &memSize, uint64_t &memSizeValue, uint64_t &memSizeMpL, uint64_t &memSizeMpH);
    static void getGroupProofSize(Merkle *M, uint32_t nGroups, uint32_t groupSize, uint64_t &memSize, uint64_t &memSizeValue, uint64_t &memSizeMp);
};
#endif // MERKLE_GROUP
#endif