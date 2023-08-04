#include "build_const_tree.hpp"
#include <string>
#include <nlohmann/json.hpp>
#include "utils.hpp"
#include <algorithm>
#include "goldilocks_base_field.hpp"
#include "merklehash_goldilocks.hpp"
#include "poseidon_goldilocks.hpp"
#include <fstream>
#include "timer.hpp"
#include "merkleTreeBN128.hpp"
#include <filesystem>
#include <cstdint>

using namespace std;
using json = nlohmann::json;

#define blocksPerThread 8
#define maxBlockBits 16
#define minBlockBits 12
#define maxNperThread 1 << 18

#define SIZE_GL 8

Goldilocks fr;

string time()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return DateAndTime(t);
}

void writeToTextFile(std::string const &filename, uint64_t *data, uint64_t length)
{
    ofstream fw(filename, std::ofstream::out);
    for (uint64_t i = 0; i < length; i++)
        fw << i << ": " << data[i] << endl;
    fw.close();
}

void _fft_block(Goldilocks::Element *buff, uint64_t rel_pos, uint64_t start_pos, uint64_t nPols, uint64_t nBits, uint64_t s, uint64_t blockBits, uint64_t layers)
{
    uint64_t n = 1 << nBits;
    uint64_t m = 1 << blockBits;
    uint64_t md2 = m >> 1;

    if (layers < blockBits)
    {
        _fft_block(buff, rel_pos, start_pos, nPols, nBits, s, blockBits - 1, layers);
        _fft_block(buff, rel_pos, start_pos + md2, nPols, nBits, s, blockBits - 1, layers);
        return;
    }
    if (layers > 1)
    {
        _fft_block(buff, rel_pos, start_pos, nPols, nBits, s - 1, blockBits - 1, layers - 1);
        _fft_block(buff, rel_pos, start_pos + md2, nPols, nBits, s - 1, blockBits - 1, layers - 1);
    }

    Goldilocks::Element w;
    if (s > blockBits)
    {
        uint64_t width = 1 << (s - layers);
        uint64_t heigth = n / width;
        uint64_t y = floor(start_pos / heigth);
        uint64_t x = start_pos % heigth;
        uint64_t p = x * width + y;
        w = fr.exp(fr.w(s), p);
    }
    else
    {
        w = fr.one();
    }

    for (uint64_t i = 0; i < md2; i++)
    {
        for (uint64_t j = 0; j < nPols; j++)
        {
            Goldilocks::Element t = fr.mul(w, buff[(start_pos - rel_pos + md2 + i) * nPols + j]);
            Goldilocks::Element u = buff[(start_pos - rel_pos + i) * nPols + j];
            buff[(start_pos - rel_pos + i) * nPols + j] = fr.add(u, t);
            buff[(start_pos - rel_pos + md2 + i) * nPols + j] = fr.sub(u, t);
        }
        w = fr.mul(w, fr.w(layers));
    }
}

Goldilocks::Element *fft_block(Goldilocks::Element *buff, uint64_t start_pos, uint64_t nPols, uint64_t nBits, uint64_t s, uint64_t blockBits, uint64_t layers)
{
    // cout << "start block " << s << " " << start_pos << endl;
    _fft_block(buff, start_pos, start_pos, nPols, nBits, s, blockBits, layers);
    // cout << "end block " << s << " " << start_pos << endl;
    return buff;
}

static inline u_int64_t BR(u_int64_t x, u_int64_t domainPow)
{
    x = (x >> 16) | (x << 16);
    x = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8);
    x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4);
    x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2);
    return (((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1)) >> (32 - domainPow);
}

void bitReverse(Goldilocks::Element *buffDst, Goldilocks::Element *buffSrc, uint64_t nPols, uint64_t nBits)
{
    uint64_t n = 1 << nBits;
    for (uint64_t i = 0; i < n; i++)
    {
        uint64_t ri = BR(i, nBits);
        memcpy(buffDst + (i * nPols), buffSrc + (ri * nPols), nPols * SIZE_GL);
    }
}

void interpolateBitReverse(Goldilocks::Element *buffDst, Goldilocks::Element *buffSrc, uint64_t nPols, uint64_t nBits)
{
    uint64_t n = 1 << nBits;
    for (uint64_t i = 0; i < n; i++)
    {
        uint64_t ri = BR(i, nBits);
        uint64_t rii = (n - ri) % n;
        memcpy(buffDst + (i * nPols), buffSrc + (rii * nPols), nPols * SIZE_GL);
    }
}

void traspose(Goldilocks::Element *buffDst, Goldilocks::Element *buffSrc, uint64_t nPols, uint64_t nBits, uint64_t trasposeBits)
{
    uint64_t n = 1 << nBits;
    uint64_t w = 1 << trasposeBits;
    uint64_t h = n / w;
    for (uint64_t i = 0; i < w; i++)
    {
        for (uint64_t j = 0; j < h; j++)
        {
            uint64_t fi = j * w + i;
            uint64_t di = i * h + j;
            memcpy(buffDst + (di * nPols), buffSrc + (fi * nPols), nPols * SIZE_GL);
        }
    }
}

Goldilocks::Element *interpolatePrepareBlock(Goldilocks::Element *buff, uint64_t bufflen, uint64_t width, Goldilocks::Element start, Goldilocks::Element inc, uint64_t st_i, uint64_t st_n)
{
    // cout << time() << " linear interpolatePrepare start.... " << st_i << "/" << st_n << endl;

    uint64_t heigth = bufflen / width;
    Goldilocks::Element w = start;
    for (uint64_t i = 0; i < heigth; i++)
    {
        for (uint64_t j = 0; j < width; j++)
        {
            buff[i * width + j] = fr.mul(buff[i * width + j], w);
        }
        w = fr.mul(w, inc);
    }
    // cout << time() << " linear interpolatePrepare end.... " << st_i << "/" << st_n << endl;
    return buff;
}

void interpolatePrepare(Goldilocks::Element *buff, uint64_t nPols, uint64_t nBits, uint64_t nBitsExt)
{
    uint64_t n = 1 << nBits;
    Goldilocks::Element invN = fr.inv(fr.fromU64(n));

    uint64_t maxNPerThread = 1 << 18;
    uint64_t minNPerThread = 1 << 12;

    int numThreads = omp_get_max_threads() / 2;
    uint64_t nPerThreadF = floor((n - 1) / numThreads) + 1;

    uint64_t maxCorrected = floor(maxNPerThread / nPols);
    uint64_t minCorrected = floor(minNPerThread / nPols);

    if (nPerThreadF > maxCorrected)
        nPerThreadF = maxCorrected;
    if (nPerThreadF < minCorrected)
        nPerThreadF = minCorrected;

#pragma omp parallel for
    for (uint64_t i = 0; i < n; i += nPerThreadF)
    {
        uint64_t curN = min(nPerThreadF, n - i);

        Goldilocks::Element *bb = (Goldilocks::Element *)malloc(curN * nPols * SIZE_GL);
        memcpy(bb, buff + (i * nPols), curN * nPols * SIZE_GL);

        Goldilocks::Element start = fr.mul(invN, fr.exp(fr.shift(), i));
        Goldilocks::Element inc = fr.shift();
        Goldilocks::Element *res = interpolatePrepareBlock(bb, curN * nPols, nPols, start, inc, i / nPerThreadF, floor(n / nPerThreadF));
        memcpy(buff + (i * nPols), res, curN * nPols * SIZE_GL);

        free(bb);
    }

    // writeToTextFile ("interpolatePrepare.new.txt", buff, (1 << nBitsExt)*nPols);
}

void interpolate(Goldilocks::Element *buffSrc, uint64_t nPols, uint64_t nBits, Goldilocks::Element *buffDst, uint64_t nBitsExt)
{
    uint64_t n = 1 << nBits;
    uint64_t nExt = 1 << nBitsExt;
    Goldilocks::Element *tmpBuff = (Goldilocks::Element *)malloc(nExt * nPols * SIZE_GL);
    Goldilocks::Element *outBuff = buffDst;

    Goldilocks::Element *bIn;
    Goldilocks::Element *bOut;

    int numThreads = omp_get_max_threads() / 2;
    uint64_t idealNBlocks = numThreads * blocksPerThread;
    uint64_t nTrasposes = 0;

    uint64_t blockBits = (uint64_t)log2(n * nPols / idealNBlocks);
    if (blockBits < minBlockBits)
        blockBits = minBlockBits;
    if (blockBits > maxBlockBits)
        blockBits = maxBlockBits;
    blockBits = min(nBits, blockBits);
    uint64_t blockSize = 1 << blockBits;
    uint64_t nBlocks = n / blockSize;

    if (blockBits < nBits)
    {
        nTrasposes += floor((nBits - 1) / blockBits) + 1;
    }

    nTrasposes += 1; // The middle convertion

    uint64_t blockBitsExt = (uint64_t)log2(nExt * nPols / idealNBlocks);
    if (blockBitsExt < minBlockBits)
        blockBitsExt = minBlockBits;
    if (blockBitsExt > maxBlockBits)
        blockBitsExt = maxBlockBits;
    blockBitsExt = min(nBitsExt, blockBitsExt);
    uint64_t blockSizeExt = 1 << blockBitsExt;
    uint64_t nBlocksExt = nExt / blockSizeExt;

    if (blockBitsExt < nBitsExt)
    {
        nTrasposes += floor((nBitsExt - 1) / blockBitsExt) + 1;
    }

    if (nTrasposes & 1)
    {
        bOut = tmpBuff;
        bIn = outBuff;
    }
    else
    {
        bOut = outBuff;
        bIn = tmpBuff;
    }

    cout << time() << " Interpolating bit reverse" << endl;
    interpolateBitReverse(bOut, buffSrc, nPols, nBits);

    //    writeToTextFile ("interpolateBitReverse.new.txt", bOut, (1 << nBitsExt)*nPols);

    Goldilocks::Element *bTmp;
    bTmp = bIn;
    bIn = bOut;
    bOut = bTmp;

    for (uint64_t i = 0; i < nBits; i += blockBits)
    {
        cout << time() << " Layer ifft " << i << endl;
        uint64_t sInc = min(blockBits, nBits - i);

#pragma omp parallel for
        for (uint64_t j = 0; j < nBlocks; j++)
        {
            Goldilocks::Element *bb = (Goldilocks::Element *)malloc(blockSize * nPols * SIZE_GL);
            memcpy(bb, bIn + (j * blockSize * nPols), blockSize * nPols * SIZE_GL);

            Goldilocks::Element *res = fft_block(bb, j * blockSize, nPols, nBits, i + sInc, blockBits, sInc);
            memcpy(bIn + (j * blockSize * nPols), res, blockSize * nPols * SIZE_GL);

            free(bb);
        }

        if (sInc < nBits)
        { // Do not transpose if it's the same
            traspose(bOut, bIn, nPols, nBits, sInc);
            bTmp = bIn;
            bIn = bOut;
            bOut = bTmp;
        }
    }

    // writeToTextFile("bIn.new.txt", bIn, (1 << nBitsExt));
    // writeToTextFile("bOut.new.txt", bOut, (1 << nBitsExt));

    cout << time() << " Interpolating prepare" << endl;
    interpolatePrepare(bIn, nPols, nBits, nBitsExt);
    cout << time() << " Bit reverse" << endl;
    bitReverse(bOut, bIn, nPols, nBitsExt);

    // writeToTextFile("bitReverse.bIn.new.txt", bIn, (1 << nBitsExt));
    // writeToTextFile("bitReverse.bOut.new.txt", bOut, (1 << nBitsExt));

    bTmp = bIn;
    bIn = bOut;
    bOut = bTmp;

    for (uint64_t i = 0; i < nBitsExt; i += blockBitsExt)
    {
        cout << time() << " Layer fft " << i << endl;
        uint64_t sInc = min(blockBitsExt, nBitsExt - i);

#pragma omp parallel for
        for (uint64_t j = 0; j < nBlocksExt; j++)
        {
            Goldilocks::Element *bb = (Goldilocks::Element *)malloc(blockSizeExt * nPols * SIZE_GL);
            memcpy(bb, bIn + (j * blockSizeExt * nPols), blockSizeExt * nPols * SIZE_GL);

            Goldilocks::Element *res = fft_block(bb, j * blockSizeExt, nPols, nBitsExt, i + sInc, blockBitsExt, sInc);
            memcpy(bIn + (j * blockSizeExt * nPols), res, blockSizeExt * nPols * SIZE_GL);

            free(bb);
        }

        if (sInc < nBitsExt)
        { // Do not transpose if it's the same
            traspose(bOut, bIn, nPols, nBitsExt, sInc);
            bTmp = bIn;
            bIn = bOut;
            bOut = bTmp;
        }
    }

    free(tmpBuff);
}

void buildConstTree(const string constFile, const string starkStructFile, const string constTreeFile, const string verKeyFile)
{
    TimerStart(BUILD_CONST_TREE);

    json starkStruct;
    file2json(starkStructFile, starkStruct);

    uint64_t nBits = starkStruct["nBits"];
    uint64_t nBitsExt = starkStruct["nBitsExt"];
    uint64_t n = 1 << nBits;
    uint64_t nExt = 1 << nBitsExt;

    uintmax_t constFileSize = filesystem::file_size(constFile);
    uint64_t nPols = constFileSize / (n * SIZE_GL);

    cout << time() << " Pols=" << nPols << endl;
    cout << time() << " nBits=" << nBits << endl;
    cout << time() << " nBitsExt=" << nBitsExt << endl;

    cout << time() << " Loading const file " << constFile << endl;
    Goldilocks::Element *pConstPols = (Goldilocks::Element *)copyFile(constFile, constFileSize);
    Goldilocks::Element *constPolsArrayE = (Goldilocks::Element *)malloc(nExt * nPols * SIZE_GL);

    TimerStart(Interpolate);
    interpolate(pConstPols, nPols, nBits, constPolsArrayE, nBitsExt);
    TimerStopAndLog(Interpolate);

    // writeToTextFile ("../build/tmp/constPolsArrayE.new.txt", constPolsArrayE, nExt*nPols);

    if (starkStruct["verificationHashType"] == "GL")
    {

        TimerStart(MerkleTree_GL);
        uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(nExt);
        uint64_t header = 2;
        uint64_t numElementsCopy = header + nPols * nExt;
        uint64_t numElements = numElementsCopy + numElementsTree;

        uint64_t sizeConstTree = numElements * sizeof(Goldilocks::Element);
        uint64_t batchSize = std::max((uint64_t)8, (nPols + 3) / 4);
        Goldilocks::Element *constTree = (Goldilocks::Element *)malloc(sizeConstTree);
        constTree[0] = Goldilocks::fromU64(nPols);
        constTree[1] = Goldilocks::fromU64(nExt);
        int numThreads = omp_get_max_threads() / 2;
        if (numThreads == 0)
        {
            numThreads = 1;
        }
        Goldilocks::parcpy(&constTree[header], constPolsArrayE, nPols * nExt, numThreads);
        PoseidonGoldilocks::merkletree(&constTree[numElementsCopy], constPolsArrayE, nPols, nExt);
        TimerStopAndLog(MerkleTree_GL);

        cout << time() << " Generating files..." << endl;

        // VerKey
        if (verKeyFile != "")
        {
            json jsonVerKey;
            json value;
            value[0] = fr.toU64(constTree[numElements - 4]);
            value[1] = fr.toU64(constTree[numElements - 3]);
            value[2] = fr.toU64(constTree[numElements - 2]);
            value[3] = fr.toU64(constTree[numElements - 1]);
            jsonVerKey["constRoot"] = value;
            json2file(jsonVerKey, verKeyFile);
        }

        // ConstTree
        ofstream fw(constTreeFile.c_str(), std::fstream::out | std::fstream::binary);
        fw.write((char const *)constTree, sizeConstTree);
        fw.close();

        cout << time() << " Files Generated Correctly" << endl;

        free(constTree);
    }
    else if (starkStruct["verificationHashType"] == "BN128")
    {

        TimerStart(MerkleTree_BN128);
        MerkleTreeBN128 mt(nExt, nPols, constPolsArrayE);
        mt.merkelize();
        TimerStopAndLog(MerkleTree_BN128);

        cout << time() << " Generating files..." << endl;

        // VerKey
        if (verKeyFile != "")
        {
            RawFr::Element constRoot;
            mt.getRoot(&constRoot);
            RawFr rawfr;
            json jsonVerKey;
            json value;
            value = rawfr.toString(constRoot);
            jsonVerKey["constRoot"] = value;
            json2file(jsonVerKey, verKeyFile);
        }

        // ConstTree
        std::ofstream fw(constTreeFile.c_str(), std::fstream::out | std::fstream::binary);
        fw.write((const char *)&(mt.source_width), sizeof(mt.source_width));
        fw.write((const char *)&(mt.height), sizeof(mt.height));
        fw.write((const char *)mt.source, nPols * nExt * SIZE_GL);
        fw.write((const char *)mt.nodes, mt.numNodes * sizeof(RawFr::Element));
        fw.close();

        cout << time() << " Files Generated Correctly" << endl;
    }
    else
    {
        cerr << "Invalid Hash Type: " << starkStruct["verificationHashType"] << endl;
        exit(-1);
    }

    free(constPolsArrayE);
    TimerStopAndLog(BUILD_CONST_TREE);
}