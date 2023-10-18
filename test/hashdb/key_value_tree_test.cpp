#include "key_value_tree_test.hpp"
#include "level_tree_key_value.hpp"
#include "level_tree.hpp"
#include "timer.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <vector>
#include <cstring>
#include <random>
#include <gmpxx.h>
#include <string>
#include "zkglobals.hpp"

void test_LevelTree_insertCounters()
{
    int level;
    bool bfound;
    //
    // test LevelTree InsertCounters
    //
    // std::cout << "test LevelTree: useInsertCounters=true" << std::endl;
    LevelTree tree(4, true);
    std::string binaryStr0 = "1101001110101011010000111011100101010111101011010101101101010101";
    std::string binaryStr1 = "1011101010110011101100101011011000100100010010000010101101000111";
    std::string binaryStr2 = "1101111010100011110001101110000010101010101011101001001111111011";
    std::string binaryStr3 = "1001001110101110011000010010001010101011111100011111001110000110";

    uint64_t key1[4];
    key1[0] = std::bitset<64>(binaryStr0).to_ullong();
    key1[1] = std::bitset<64>(binaryStr1).to_ullong();
    key1[2] = std::bitset<64>(binaryStr2).to_ullong();
    key1[3] = std::bitset<64>(binaryStr3).to_ullong();

    level = tree.insert(key1);
    assert(level == 0);

    // key2: 1 bits in common with key1 (bits of components are interleaved)
    std::string binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    std::string binaryStr1_ = "0011101010110011101100101011011000100100010010000010101101000111";
    std::string binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    std::string binaryStr3_ = "1001001110101110011000010010001010101011111100011111001110000110";
    uint64_t key2[4];
    key2[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key2[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key2[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key2[3] = std::bitset<64>(binaryStr3_).to_ullong();

    level = tree.insert(key2);
    assert(level == 1);
    bfound = tree.extract(key2);
    assert(bfound == true);
    level = tree.level(key1);
    assert(level == 0);
    level = tree.insert(key2);
    assert(level == 1);
    level = tree.insert(key2);
    assert(level == 1);
    level = tree.insert(key2);
    assert(level == 1);
    bfound = tree.extract(key2);
    assert(bfound == true);
    bfound = tree.extract(key2);
    assert(bfound == true);
    bfound = tree.extract(key2);
    assert(bfound == true);
    bfound = tree.extract(key2);
    assert(bfound == false);

    // key3: 201 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1011101010110011101100101011011000100100010010000000101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011000010010001010101011111100011111001110000110";
    uint64_t key3[4];
    key3[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key3[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key3[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key3[3] = std::bitset<64>(binaryStr3_).to_ullong();

    level = tree.insert(key3);
    assert(level == 201);
    bfound = tree.extract(key3);
    assert(bfound == true);
    level = tree.level(key1);
    assert(level == 0);
    level = tree.insert(key3);
    assert(level == 201);
    level = tree.insert(key3);
    assert(level == 201);
    level = tree.insert(key3);
    assert(level == 201);
    bfound = tree.extract(key3);
    assert(bfound == true);
    bfound = tree.extract(key3);
    assert(bfound == true);
    bfound = tree.extract(key3);
    assert(bfound == true);
    bfound = tree.extract(key3);
    assert(bfound == false);
}

void test_LevelTree(uint64_t nBitsStep)
{
    int level;
    bool bfound;
    int64_t pileIdx;
    LevelTree tree(nBitsStep);
    //
    // test LevelTree.insert & LevelTree.extract
    //
    // std::cout << "test LevelTree: level, insert & extract with nBitsStep: " << nBitsStep << std::endl;
    std::string binaryStr0 = "1101001110101011010000111011100101010111101011010101101101010101";
    std::string binaryStr1 = "1011101010110011101100101011011000100100010010000010101101000111";
    std::string binaryStr2 = "1101111010100011110001101110000010101010101011101001001111111011";
    std::string binaryStr3 = "1001001110101110011000010010001010101011111100011111001110000110";

    uint64_t key1[4];
    key1[0] = std::bitset<64>(binaryStr0).to_ullong();
    key1[1] = std::bitset<64>(binaryStr1).to_ullong();
    key1[2] = std::bitset<64>(binaryStr2).to_ullong();
    key1[3] = std::bitset<64>(binaryStr3).to_ullong();

    bfound = tree.extract(key1, &pileIdx);
    assert(pileIdx == -1);
    assert(bfound == false);
    pileIdx = 0;
    level = tree.level(key1, &pileIdx);
    assert(pileIdx == -1);
    assert(level == 0);
    level = tree.insert(key1, &pileIdx);
    assert(level == 0);
    assert(pileIdx == 1);
    pileIdx = 0;
    level = tree.level(key1, &pileIdx);
    assert(level == 0);
    assert(pileIdx == 1);

    // key2: 0 bits in common with key1 (bits of components are interleaved)
    std::string binaryStr0_ = "0101001110101011010000111011100101010111101011010101101101010101";
    std::string binaryStr1_ = "1011101010110011101100101011011000100100010010000010101101000111";
    std::string binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    std::string binaryStr3_ = "1001001110101110011000010010001010101011111100011111001110000110";
    uint64_t key2[4];
    key2[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key2[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key2[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key2[3] = std::bitset<64>(binaryStr3_).to_ullong();

    level = tree.level(key2, &pileIdx);
    assert(level == 0);
    assert(pileIdx == -1);
    pileIdx = 0;
    bfound = tree.extract(key2, &pileIdx);
    assert(bfound == false);
    assert(pileIdx == -1);
    level = tree.insert(key2, &pileIdx);
    assert(level == 0);
    assert(pileIdx == 2);
    level = tree.insert(key1, &pileIdx);
    assert(level == 0);
    assert(pileIdx == 1);

    // key3: 1 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "0011101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011000010010001010101011111100011111001110000110";
    uint64_t key3[4];
    key3[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key3[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key3[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key3[3] = std::bitset<64>(binaryStr3_).to_ullong();

    bfound = tree.extract(key3);
    assert(bfound == false);
    level = tree.level(key3);
    assert(level == 1);
    level = tree.insert(key3);
    assert(level == 1);
    level = tree.level(key1);
    assert(level == 1);

    bfound = tree.extract(key3);
    assert(bfound == true);
    level = tree.insert(key1);
    assert(level == 0);

    level = tree.insert(key3);
    assert(level == 1);
    level = tree.level(key1);
    assert(level == 1);

    // key4: 9 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1001101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011000010010001010101011111100011111001110000110";
    uint64_t key4[4];
    key4[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key4[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key4[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key4[3] = std::bitset<64>(binaryStr3_).to_ullong();

    bfound = tree.extract(key4);
    assert(bfound == false);
    level = tree.level(key4);
    assert(level == 9);
    level = tree.insert(key4);
    assert(level == 9);
    level = tree.level(key1);
    assert(level == 9);

    bfound = tree.extract(key4);
    assert(bfound == true);
    level = tree.insert(key1);
    assert(level == 1);
    level = tree.insert(key4);
    assert(level == 9);
    level = tree.level(key1);
    assert(level == 9);

    // key5: 63 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1011101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101111011000010010001010101011111100011111001110000110";
    uint64_t key5[4];
    key5[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key5[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key5[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key5[3] = std::bitset<64>(binaryStr3_).to_ullong();

    bfound = tree.extract(key5);
    assert(bfound == false);
    level = tree.level(key5);
    assert(level == 63);
    level = tree.insert(key5);
    assert(level == 63);
    level = tree.insert(key1);
    assert(level == 63);

    bfound = tree.extract(key5);
    assert(bfound == true);
    level = tree.insert(key1);
    assert(level == 9);

    bfound = tree.extract(key4);
    assert(bfound == true);
    level = tree.level(key1);
    assert(level == 1);

    // key6: 79 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1011101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011100010010001010101011111100011111001110000110";
    uint64_t key6[4];
    key6[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key6[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key6[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key6[3] = std::bitset<64>(binaryStr3_).to_ullong();

    level = tree.level(key6);
    assert(level == 79);
    bfound = tree.extract(key6);
    assert(bfound == false);
    level = tree.insert(key6);
    assert(level == 79);
    level = tree.level(key1);
    assert(level == 79);

    // key7: 187 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1011101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011000010010001010101011111100111111001110000110";
    uint64_t key7[4];
    key7[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key7[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key7[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key7[3] = std::bitset<64>(binaryStr3_).to_ullong();

    bfound = tree.extract(key7);
    assert(bfound == false);
    level = tree.level(key7);
    assert(level == 187);
    level = tree.insert(key7);
    assert(level == 187);
    level = tree.level(key1);
    assert(level == 187);

    // key8: 201 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1011101010110011101100101011011000100100010010000000101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011000010010001010101011111100011111001110000110";
    uint64_t key8[4];
    key8[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key8[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key8[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key8[3] = std::bitset<64>(binaryStr3_).to_ullong();

    bfound = tree.extract(key8);
    assert(bfound == false);
    level = tree.level(key8);
    assert(level == 201);
    level = tree.insert(key8);
    assert(level == 201);
    level = tree.level(key1);
    assert(level == 201);

    bfound = tree.extract(key8);
    assert(bfound == true);
    level = tree.insert(key1);
    assert(level == 187);

    bfound = tree.extract(key7);
    assert(bfound == true);
    level = tree.insert(key1);
    assert(level == 79);

    bfound = tree.extract(key6);
    assert(bfound == true);
    level = tree.level(key1);
    assert(level == 1);
}

void test_LevelTree_resize()
{
    int level;
    uint64_t nBitsStep = 4;
    LevelTree tree(nBitsStep);
    //
    // test LevelTree.insert & LevelTree.extract
    //
    // std::cout << "test LevelTree: resize " << std::endl;
    std::string binaryStr0 = "1101001110101011010000111011100101010111101011010101101101010101";
    std::string binaryStr1 = "1011101010110011101100101011011000100100010010000010101101000111";
    std::string binaryStr2 = "1101111010100011110001101110000010101010101011101001001111111011";
    std::string binaryStr3 = "1001001110101110011000010010001010101011111100011111001110000110";

    uint64_t key1[4];
    key1[0] = std::bitset<64>(binaryStr0).to_ullong();
    key1[1] = std::bitset<64>(binaryStr1).to_ullong();
    key1[2] = std::bitset<64>(binaryStr2).to_ullong();
    key1[3] = std::bitset<64>(binaryStr3).to_ullong();

    level = tree.level(key1);
    assert(level == 0);
    level = tree.insert(key1);
    assert(level == 0);
    level = tree.level(key1);
    assert(level == 0);

    // key2: 0 bits in common with key1 (bits of components are interleaved)
    std::string binaryStr0_ = "0101001110101011010000111011100101010111101011010101101101010101";
    std::string binaryStr1_ = "1011101010110011101100101011011000100100010010000010101101000111";
    std::string binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    std::string binaryStr3_ = "1001001110101110011000010010001010101011111100011111001110000110";
    uint64_t key2[4];
    key2[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key2[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key2[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key2[3] = std::bitset<64>(binaryStr3_).to_ullong();

    level = tree.level(key2);
    assert(level == 0);
    level = tree.insert(key2);
    assert(level == 0);
    level = tree.insert(key1);
    assert(level == 0);

    // key3: 1 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "0011101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011000010010001010101011111100011111001110000110";
    uint64_t key3[4];
    key3[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key3[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key3[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key3[3] = std::bitset<64>(binaryStr3_).to_ullong();

    level = tree.level(key3);
    assert(level == 1);
    level = tree.insert(key3);
    assert(level == 1);
    level = tree.level(key1);
    assert(level == 1);

    tree.extract(key3);
    level = tree.insert(key1);
    assert(level == 0);

    level = tree.insert(key3);
    assert(level == 1);
    level = tree.level(key1);
    assert(level == 1);

    // key4: 9 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1001101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011000010010001010101011111100011111001110000110";
    uint64_t key4[4];
    key4[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key4[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key4[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key4[3] = std::bitset<64>(binaryStr3_).to_ullong();

    level = tree.level(key4);
    assert(level == 9);
    level = tree.insert(key4);
    assert(level == 9);
    level = tree.level(key1);
    assert(level == 9);

    tree.extract(key4);
    level = tree.insert(key1);
    assert(level == 1);
    level = tree.insert(key4);
    assert(level == 9);
    level = tree.level(key1);
    assert(level == 9);

    // key5: 63 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1011101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101111011000010010001010101011111100011111001110000110";
    uint64_t key5[4];
    key5[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key5[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key5[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key5[3] = std::bitset<64>(binaryStr3_).to_ullong();

    level = tree.level(key5);
    assert(level == 63);
    level = tree.insert(key5);
    assert(level == 63);
    level = tree.insert(key1);
    assert(level == 63);

    tree.extract(key5);
    level = tree.insert(key1);
    assert(level == 9);

    tree.extract(key4);
    level = tree.level(key1);
    assert(level == 1);

    // key6: 79 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1011101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011100010010001010101011111100011111001110000110";
    uint64_t key6[4];
    key6[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key6[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key6[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key6[3] = std::bitset<64>(binaryStr3_).to_ullong();

    level = tree.level(key6);
    assert(level == 79);
    level = tree.insert(key6);
    assert(level == 79);
    level = tree.level(key1);
    assert(level == 79);

    // key7: 187 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1011101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011000010010001010101011111100111111001110000110";
    uint64_t key7[4];
    key7[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key7[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key7[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key7[3] = std::bitset<64>(binaryStr3_).to_ullong();

    level = tree.level(key7);
    assert(level == 187);
    level = tree.insert(key7);
    assert(level == 187);
    level = tree.level(key1);
    assert(level == 187);

    // key8: 201 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1011101010110011101100101011011000100100010010000000101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011000010010001010101011111100011111001110000110";
    uint64_t key8[4];
    key8[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key8[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key8[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key8[3] = std::bitset<64>(binaryStr3_).to_ullong();

    level = tree.level(key8);
    assert(level == 201);
    level = tree.insert(key8);
    assert(level == 201);
    level = tree.level(key1);
    assert(level == 201);

    tree.extract(key8);
    level = tree.insert(key1);
    assert(level == 187);

    tree.extract(key7);
    level = tree.insert(key1);
    assert(level == 79);

    tree.extract(key6);
    level = tree.level(key1);
    assert(level == 1);
}

void test_KVTree_asLevelTree()
{
    int level;
    bool bfound;
    KVTree tree(4);
    //
    // test LevelTree.insert & LevelTree.extract
    //
    // std::cout << "test KVTree: as LevelTree with nBitsStep: 4 " << std::endl;
    std::string binaryStr0 = "1101001110101011010000111011100101010111101011010101101101010101";
    std::string binaryStr1 = "1011101010110011101100101011011000100100010010000010101101000111";
    std::string binaryStr2 = "1101111010100011110001101110000010101010101011101001001111111011";
    std::string binaryStr3 = "1001001110101110011000010010001010101011111100011111001110000110";

    uint64_t key1[4];
    key1[0] = std::bitset<64>(binaryStr0).to_ullong();
    key1[1] = std::bitset<64>(binaryStr1).to_ullong();
    key1[2] = std::bitset<64>(binaryStr2).to_ullong();
    key1[3] = std::bitset<64>(binaryStr3).to_ullong();

    bfound = tree.LevelTree::extract(key1);
    assert(bfound == false);
    level = static_cast<LevelTree&>(tree).level(key1);
    assert(level == 0);
    level = tree.insert(key1);
    assert(level == 0);
    level = static_cast<LevelTree&>(tree).level(key1);
    assert(level == 0);

    // key2: 0 bits in common with key1 (bits of components are interleaved)
    std::string binaryStr0_ = "0101001110101011010000111011100101010111101011010101101101010101";
    std::string binaryStr1_ = "1011101010110011101100101011011000100100010010000010101101000111";
    std::string binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    std::string binaryStr3_ = "1001001110101110011000010010001010101011111100011111001110000110";
    uint64_t key2[4];
    key2[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key2[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key2[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key2[3] = std::bitset<64>(binaryStr3_).to_ullong();

    level =  static_cast<LevelTree&>(tree).level(key2);
    assert(level == 0);
    bfound = tree.LevelTree::extract(key2);
    assert(bfound == false);
    level = tree.insert(key2);
    assert(level == 0);
    level = tree.insert(key1);
    assert(level == 0);

    // key3: 1 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "0011101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011000010010001010101011111100011111001110000110";
    uint64_t key3[4];
    key3[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key3[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key3[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key3[3] = std::bitset<64>(binaryStr3_).to_ullong();

    bfound = tree.LevelTree::extract(key3);
    assert(bfound == false);
    level =  static_cast<LevelTree&>(tree).level(key3);
    assert(level == 1);
    level = tree.insert(key3);
    assert(level == 1);
    level =  static_cast<LevelTree&>(tree).level(key1);
    assert(level == 1);

    bfound = tree.LevelTree::extract(key3);
    assert(bfound == true);
    level = tree.insert(key1);
    assert(level == 0);

    level = tree.insert(key3);
    assert(level == 1);
    level =  static_cast<LevelTree&>(tree).level(key1);
    assert(level == 1);

    // key4: 9 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1001101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011000010010001010101011111100011111001110000110";
    uint64_t key4[4];
    key4[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key4[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key4[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key4[3] = std::bitset<64>(binaryStr3_).to_ullong();

    bfound = tree.LevelTree::extract(key4);
    assert(bfound == false);
    level =  static_cast<LevelTree&>(tree).level(key4);
    assert(level == 9);
    level = tree.insert(key4);
    assert(level == 9);
    level =  static_cast<LevelTree&>(tree).level(key1);
    assert(level == 9);

    bfound = tree.LevelTree::extract(key4);
    assert(bfound == true);
    level = tree.insert(key1);
    assert(level == 1);
    level = tree.insert(key4);
    assert(level == 9);
    level =  static_cast<LevelTree&>(tree).level(key1);
    assert(level == 9);

    // key5: 63 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1011101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101111011000010010001010101011111100011111001110000110";
    uint64_t key5[4];
    key5[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key5[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key5[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key5[3] = std::bitset<64>(binaryStr3_).to_ullong();

    bfound = tree.LevelTree::extract(key5);
    assert(bfound == false);
    level =  static_cast<LevelTree&>(tree).level(key5);
    assert(level == 63);
    level = tree.insert(key5);
    assert(level == 63);
    level = tree.insert(key1);
    assert(level == 63);

    bfound = tree.LevelTree::extract(key5);
    assert(bfound == true);
    level = tree.insert(key1);
    assert(level == 9);

    bfound = tree.LevelTree::extract(key4);
    assert(bfound == true);
    level =  static_cast<LevelTree&>(tree).level(key1);
    assert(level == 1);

    // key6: 79 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1011101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011100010010001010101011111100011111001110000110";
    uint64_t key6[4];
    key6[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key6[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key6[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key6[3] = std::bitset<64>(binaryStr3_).to_ullong();

    level =  static_cast<LevelTree&>(tree).level(key6);
    assert(level == 79);
    bfound = tree.LevelTree::extract(key6);
    assert(bfound == false);
    level = tree.insert(key6);
    assert(level == 79);
    level =  static_cast<LevelTree&>(tree).level(key1);
    assert(level == 79);

    // key7: 187 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1011101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011000010010001010101011111100111111001110000110";
    uint64_t key7[4];
    key7[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key7[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key7[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key7[3] = std::bitset<64>(binaryStr3_).to_ullong();

    bfound = tree.LevelTree::extract(key7);
    assert(bfound == false);
    level =  static_cast<LevelTree&>(tree).level(key7);
    assert(level == 187);
    level = tree.insert(key7);
    assert(level == 187);
    level =  static_cast<LevelTree&>(tree).level(key1);
    assert(level == 187);

    // key8: 201 bits in common with key1 (bits of components are interleaved)
    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1011101010110011101100101011011000100100010010000000101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011000010010001010101011111100011111001110000110";
    uint64_t key8[4];
    key8[0] = std::bitset<64>(binaryStr0_).to_ullong();
    key8[1] = std::bitset<64>(binaryStr1_).to_ullong();
    key8[2] = std::bitset<64>(binaryStr2_).to_ullong();
    key8[3] = std::bitset<64>(binaryStr3_).to_ullong();

    bfound = tree.LevelTree::extract(key8);
    assert(bfound == false);
    level =  static_cast<LevelTree&>(tree).level(key8);
    assert(level == 201);
    level = tree.insert(key8);
    assert(level == 201);
    level =  static_cast<LevelTree&>(tree).level(key1);
    assert(level == 201);

    bfound = tree.LevelTree::extract(key8);
    assert(bfound == true);
    level = tree.insert(key1);
    assert(level == 187);

    bfound = tree.LevelTree::extract(key7);
    assert(bfound == true);
    level = tree.insert(key1);
    assert(level == 79);

    bfound = tree.LevelTree::extract(key6);
    assert(bfound == true);
    level =  static_cast<LevelTree&>(tree).level(key1);
    assert(level == 1);
}

void test_KVTree(uint64_t nBitsStep)
{
    uint64_t level;
    zkresult bfound;
    KVTree kvtree(nBitsStep);
    mpz_class value;
    //
    // test LevelTree.insert & LevelTree.extract
    //
    // std::cout << "test KVTree: level, insert & extract with nBitsStep: " << nBitsStep << std::endl;

    std::string binaryStr0 = "1101001110101011010000111011100101010111101011010101101101010101";
    std::string binaryStr1 = "1011101010110011101100101011011000100100010010000010101101000111";
    std::string binaryStr2 = "1101111010100011110001101110000010101010101011101001001111111011";
    std::string binaryStr3 = "1001001110101110011000010010001010101011111100011111001110000110";

    Goldilocks::Element key1[4];
    key1[0].fe = std::bitset<64>(binaryStr0).to_ullong();
    key1[1].fe = std::bitset<64>(binaryStr1).to_ullong();
    key1[2].fe = std::bitset<64>(binaryStr2).to_ullong();
    key1[3].fe = std::bitset<64>(binaryStr3).to_ullong();

    bfound = kvtree.read(key1, value, level);
    assert(bfound == ZKR_DB_KEY_NOT_FOUND);
    bfound = kvtree.extract(key1, value);
    assert(bfound == ZKR_DB_KEY_NOT_FOUND);
    level = kvtree.level(key1);
    assert(level == 0);
    kvtree.write(key1, 1, level);
    bfound = kvtree.read(key1, value, level);
    assert(bfound == ZKR_SUCCESS && value == 1 && level == 0);

    std::string binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    std::string binaryStr1_ = "1011101010110011101100101011011000100100010010000000101101000111";
    std::string binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    std::string binaryStr3_ = "1001001110101110011000010010001010101011111100011111001110000110";
    
    Goldilocks::Element key2[4];
    key2[0].fe = std::bitset<64>(binaryStr0_).to_ullong();
    key2[1].fe = std::bitset<64>(binaryStr1_).to_ullong();
    key2[2].fe = std::bitset<64>(binaryStr2_).to_ullong();
    key2[3].fe = std::bitset<64>(binaryStr3_).to_ullong();

    kvtree.write(key2, 2, level);
    assert(level == 201);
    bfound = kvtree.read(key1, value, level);
    assert(bfound == ZKR_SUCCESS && value == 1 && level == 201);
    bfound = kvtree.read(key2, value, level);
    assert(bfound == ZKR_SUCCESS && value == 2 && level == 201);
    value = 0;
    bfound = kvtree.extract(key2, value);
    assert(bfound == ZKR_SUCCESS && value == 2);
    bfound = kvtree.read(key1, value, level);
    assert(bfound == ZKR_SUCCESS && value == 1 && level == 0);

    binaryStr0_ = "1101001110101011010000111011100101010111101011010101101101010101";
    binaryStr1_ = "1011101010110011101100101011011000100100010010000010101101000111";
    binaryStr2_ = "1101111010100011110001101110000010101010101011101001001111111011";
    binaryStr3_ = "1001001110101110011000010010001010101011111100111111001110000110";
    
    Goldilocks::Element key3[4];
    key3[0].fe = std::bitset<64>(binaryStr0_).to_ullong();
    key3[1].fe = std::bitset<64>(binaryStr1_).to_ullong();
    key3[2].fe = std::bitset<64>(binaryStr2_).to_ullong();
    key3[3].fe = std::bitset<64>(binaryStr3_).to_ullong();

    kvtree.write(key3, 2, level);
    assert(level == 187);
    bfound = kvtree.read(key1, value, level);
    assert(bfound == ZKR_SUCCESS && value == 1 && level == 187);
    kvtree.write(key1, 0, level);
    assert(level == 187);
    bfound = kvtree.read(key1, value, level);
    assert(bfound == ZKR_SUCCESS && value == 0 && level == 187);
    kvtree.write(key1, 10, level);
    assert(level == 187);
    bfound = kvtree.read(key1, value, level);
    assert(bfound == ZKR_SUCCESS && value == 10 && level == 187);
    kvtree.write(key1, 20, level);
    assert(level == 187);
    bfound = kvtree.read(key1, value, level);
    assert(bfound == ZKR_SUCCESS && value == 20 && level == 187);
    bfound = kvtree.extract(key1, value);
    assert(bfound == ZKR_SUCCESS && value == 20);
    bfound = kvtree.read(key1, value, level);
    assert(bfound == ZKR_SUCCESS && value == 10 && level == 187);
    kvtree.write(key1, 20, level);
    assert(level == 187);
    bfound = kvtree.read(key1, value, level);
    assert(bfound == ZKR_SUCCESS && value == 20 && level == 187);
    bfound = kvtree.extract(key1, value);
    assert(bfound == ZKR_SUCCESS && value == 20);
    bfound = kvtree.extract(key1, value);
    assert(bfound == ZKR_SUCCESS && value == 10);
    bfound = kvtree.extract(key1, value);
    assert(bfound == ZKR_SUCCESS && value == 0);
    bfound = kvtree.extract(key1, value);
    assert(bfound == ZKR_SUCCESS && value == 1);
    bfound = kvtree.extract(key1, value);
    assert(bfound == ZKR_DB_KEY_NOT_FOUND);
}

uint64_t KeyValueTreeTest(void)
{
    TimerStart(KEY_VALUE_TREE_TEST);

    test_LevelTree(2);
    test_LevelTree(4);
    test_LevelTree(8);
    test_LevelTree_resize();
    test_LevelTree_insertCounters();
    test_KVTree_asLevelTree();
    test_KVTree(2);
    test_KVTree(4);
    test_KVTree(8);

    TimerStopAndLog(KEY_VALUE_TREE_TEST);
    return 0;
}