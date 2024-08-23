
#include "merkleTreeBN128.hpp"
#include <algorithm> // std::max
#include <cassert>

MerkleTreeBN128::MerkleTreeBN128(uint64_t _arity, bool _custom, uint64_t _height, uint64_t _width, Goldilocks::Element *_source, bool allocate) : height(_height), width(_width), source(_source)
{

    if (source == NULL && allocate)
    {
        source = (Goldilocks::Element *)calloc(height * width, sizeof(Goldilocks::Element));
        isSourceAllocated = true;
    }

    arity = _arity;
    custom = _custom;
    numNodes = getNumNodes(height);
    nodes = (RawFr::Element *)calloc(numNodes, sizeof(RawFr::Element));
    isNodesAllocated = true;
}

MerkleTreeBN128::MerkleTreeBN128(uint64_t _arity, bool _custom, Goldilocks::Element *tree)
{
    width = Goldilocks::toU64(tree[0]);
    height = Goldilocks::toU64(tree[1]);
    source = &tree[2];
    arity = _arity;
    custom = _custom;
    numNodes = getNumNodes(height);
    
    nodes = (RawFr::Element *)&source[width * height];
}

MerkleTreeBN128::~MerkleTreeBN128()
{
    if (isNodesAllocated)
    {
        free(nodes);
    }
    if (isSourceAllocated)
    {
        free(source);
    }
}

uint64_t MerkleTreeBN128::getNumSiblings() 
{
    return arity * nFieldElements;
}

uint64_t MerkleTreeBN128::getMerkleTreeWidth()
{
    return width;
}

uint64_t MerkleTreeBN128::getMerkleProofLength()
{
    return ceil((double)log(height) / log(arity));
}


uint64_t MerkleTreeBN128::getMerkleProofSize()
{
    return getMerkleProofLength() * arity * sizeof(RawFr::Element);
}

uint64_t MerkleTreeBN128::getNumNodes(uint64_t n)
{   
    uint n_tmp = n;
    uint64_t nextN = floor(((double)(n_tmp - 1) / arity) + 1);
    uint64_t acc = nextN * arity;
    while (n_tmp > 1)
    {
        // FIll with zeros if n nodes in the leve is not even
        n_tmp = nextN;
        nextN = floor((n_tmp - 1) / arity) + 1;
        if (n_tmp > 1)
        {
            acc += nextN * arity;
        }
        else
        {
            acc += 1;
        }
    }

    return acc;
}


void MerkleTreeBN128::getRoot(RawFr::Element *root)
{
    std::memcpy(root, &nodes[numNodes - 1], sizeof(RawFr::Element));
    zklog.info("MerkleTree root: " + RawFr::field.toString(root[0], 10));

}

void MerkleTreeBN128::copySource(Goldilocks::Element *_source)
{
    std::memcpy(source, _source, height * width * sizeof(Goldilocks::Element));
}

void MerkleTreeBN128::setSource(Goldilocks::Element *_source)
{
    source = _source;
}

Goldilocks::Element MerkleTreeBN128::getElement(uint64_t idx, uint64_t subIdx)
{
    assert((idx > 0) || (idx < width));
    return source[width * idx + subIdx];
}

void MerkleTreeBN128::getGroupProof(RawFr::Element *proof, uint64_t idx)
{
    assert(idx < height);

    Goldilocks::Element v[width];
    for (uint64_t i = 0; i < width; i++)
    {
        v[i] = getElement(idx, i);
    }
    std::memcpy(proof, &v[0], width * sizeof(Goldilocks::Element));
    void *proofCursor = (uint8_t *)proof + width * sizeof(Goldilocks::Element);

    RawFr::Element *mp = (RawFr::Element *)calloc(getMerkleProofSize(), 1);
    genMerkleProof(mp, idx, 0, height);

    std::memcpy(proofCursor, &mp[0], getMerkleProofSize());
    free(mp);
}

void MerkleTreeBN128::genMerkleProof(RawFr::Element *proof, uint64_t idx, uint64_t offset, uint64_t n)
{
    if (n <= 1) return;

    uint64_t nBitsArity = std::ceil(std::log2(arity));

    uint64_t nextIdx = idx >> nBitsArity;
    uint64_t si = idx ^ (idx & (arity - 1));

    std::memcpy(proof, &nodes[offset + si], arity * sizeof(RawFr::Element));
    uint64_t nextN = (std::floor((n - 1) / arity) + 1);
    genMerkleProof(&proof[arity], nextIdx, offset + nextN * arity, nextN);
}

/*
 * LinearHash BN128
 */
void MerkleTreeBN128::linearHash()
{
    if (width > 4)
    {
        uint64_t widthRawFrElements = ceil((double)width / FIELD_EXTENSION);
        RawFr::Element *buff = (RawFr::Element *)calloc(height * widthRawFrElements, sizeof(RawFr::Element));

    uint64_t nElementsGL = (width > FIELD_EXTENSION + 1) ? ceil((double)width / FIELD_EXTENSION) : 0;
#pragma omp parallel for
        for (uint64_t i = 0; i < height; i++)
        {
            for (uint64_t j = 0; j < nElementsGL; j++)
            {
                uint64_t pending = width - j * FIELD_EXTENSION;
                uint64_t batch;
                (pending >= FIELD_EXTENSION) ? batch = FIELD_EXTENSION : batch = pending;
                for (uint64_t k = 0; k < batch; k++)
                {
                    buff[i * nElementsGL + j].v[k] = Goldilocks::toU64(source[i * width + j * FIELD_EXTENSION + k]);
                }
                RawFr::field.toMontgomery(buff[i * nElementsGL + j], buff[i * nElementsGL + j]);
            }
        }

#pragma omp parallel for
        for (uint64_t i = 0; i < height; i++)
        {
            uint pending = nElementsGL;
            Poseidon_opt p;
            std::vector<RawFr::Element> elements(arity + 1);
            while (pending > 0)
            {
                std::memset(&elements[0], 0, (arity + 1) * sizeof(RawFr::Element));
                if (pending >= arity)
                {
                    std::memcpy(&elements[1], &buff[i * nElementsGL + nElementsGL - pending], arity * sizeof(RawFr::Element));
                    std::memcpy(&elements[0], &nodes[i], sizeof(RawFr::Element));
                    p.hash(elements, &nodes[i]);
                    pending = pending - arity;
                }
                else if(custom) 
                {
                    std::memcpy(&elements[1], &buff[i * nElementsGL + nElementsGL - pending], pending * sizeof(RawFr::Element));
                    std::memcpy(&elements[0], &nodes[i], sizeof(RawFr::Element));
                    p.hash(elements, &nodes[i]);
                    pending = 0;
                }
                else
                {
                    std::vector<RawFr::Element> elements_last(pending + 1);
                    std::memcpy(&elements_last[1], &buff[i * nElementsGL + nElementsGL - pending], pending * sizeof(RawFr::Element));
                    std::memcpy(&elements_last[0], &nodes[i], sizeof(RawFr::Element));
                    p.hash(elements_last, &nodes[i]);
                    pending = 0;
                }
            }
        }
        free(buff);
    }
    else
    {
#pragma omp parallel for
        for (uint64_t i = 0; i < height; i++)
        {
            for (uint64_t k = 0; k < width; k++)
            {
                nodes[i].v[k] = Goldilocks::toU64(source[i * width + k]);
            }
            RawFr::field.toMontgomery(nodes[i], nodes[i]);
        }
    }
}

void MerkleTreeBN128::merkelize()
{

    linearHash();

    RawFr::Element *cursor = &nodes[0];
    uint64_t n256 = height;
    uint64_t nextN256 = floor((double)(n256 - 1) / arity) + 1;
    RawFr::Element *cursorNext = &nodes[nextN256 * arity];
    while (n256 > 1)
    {
        uint64_t batches = ceil((double)n256 / arity);
#pragma omp parallel for
        for (uint64_t i = 0; i < batches; i++)
        {
            Poseidon_opt p;
            vector<RawFr::Element> elements(arity + 1);
            std::memset(&elements[0], 0, (arity + 1) * sizeof(RawFr::Element));
            uint numHashes = (i == batches - 1) ? n256 - i*arity : arity;
            std::memcpy(&elements[1], &cursor[i * arity], numHashes * sizeof(RawFr::Element));
            p.hash(elements, &cursorNext[i]);
        }

        n256 = nextN256;
        nextN256 = floor((double)(n256 - 1) / arity) + 1;
        cursor = cursorNext;
        cursorNext = &cursor[nextN256 * arity];
    }
}
