#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <bitset>
#include <gmpxx.h>
#include "key_utils.hpp"
#include "scalar.hpp"

uint64_t splitKey9Test(void){

    std::string binaryStr = "1101001110101011010000111011100101010111101011010101101101010101101110101011001110110010101101100010010001001000001010110100011111011110101000111100011011100000101010101010111010010011111110111001001110101110011000010010001010101011111100011111001110000110";

    mpz_class keyMPZ(binaryStr, 2);
    string keyBA = scalar2ba(keyMPZ);

    vector<uint64_t> key9;
    splitKey9(keyBA, key9);

    assert(key9[0]  == std::bitset<32>("110100111").to_ulong() );
    assert(key9[1]  == std::bitset<32>("010101101").to_ulong() );
    assert(key9[2]  == std::bitset<32>("000011101").to_ulong() );
    assert(key9[3]  == std::bitset<32>("110010101").to_ulong() );
    assert(key9[4]  == std::bitset<32>("011110101").to_ulong() );
    assert(key9[5]  == std::bitset<32>("101010110").to_ulong() );
    assert(key9[6]  == std::bitset<32>("110101010").to_ulong() );
    assert(key9[7]  == std::bitset<32>("110111010").to_ulong() );
    assert(key9[8]  == std::bitset<32>("101100111").to_ulong() );
    assert(key9[9]  == std::bitset<32>("011001010").to_ulong() );
    assert(key9[10] == std::bitset<32>("110110001").to_ulong() );
    assert(key9[11] == std::bitset<32>("001000100").to_ulong() );
    assert(key9[12] == std::bitset<32>("100000101").to_ulong() );
    assert(key9[13] == std::bitset<32>("011010001").to_ulong() );
    assert(key9[14] == std::bitset<32>("111101111").to_ulong() );
    assert(key9[15] == std::bitset<32>("010100011").to_ulong() );
    assert(key9[16] == std::bitset<32>("110001101").to_ulong() );
    assert(key9[17] == std::bitset<32>("110000010").to_ulong() );
    assert(key9[18] == std::bitset<32>("101010101").to_ulong() );
    assert(key9[19] == std::bitset<32>("011101001").to_ulong() );
    assert(key9[20] == std::bitset<32>("001111111").to_ulong() );
    assert(key9[21] == std::bitset<32>("011100100").to_ulong() );
    assert(key9[22] == std::bitset<32>("111010111").to_ulong() );
    assert(key9[23] == std::bitset<32>("001100001").to_ulong() );
    assert(key9[24] == std::bitset<32>("001000101").to_ulong() );
    assert(key9[25] == std::bitset<32>("010101111").to_ulong() );
    assert(key9[26] == std::bitset<32>("110001111").to_ulong() );
    assert(key9[27] == std::bitset<32>("100111000").to_ulong() );
    assert(key9[28] == std::bitset<32>("011000000").to_ulong() );
      
    return 0;
}