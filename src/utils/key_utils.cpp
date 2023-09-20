#include <bitset>
#include "scalar.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

using namespace std;

// Split the fe key into 4-bits chuncks, e.g. 0x123456EF -> { 1, 2, 3, 4, 5, 6, E, F }
void splitKey (Goldilocks &fr, const Goldilocks::Element (&key)[4], bool (&result)[256])
{
    bitset<64> auxb0(fr.toU64(key[0]));
    bitset<64> auxb1(fr.toU64(key[1]));
    bitset<64> auxb2(fr.toU64(key[2]));
    bitset<64> auxb3(fr.toU64(key[3]));
    
    // Split the key in bits, taking one bit from a different scalar every time
    int cont = 0;
    for (uint64_t i=0; i<64; i++)
    {
        result[cont] = auxb0[i];
        result[cont+1] = auxb1[i];
        result[cont+2] = auxb2[i];
        result[cont+3] = auxb3[i];
        cont+=4;
    }
}

// Joins full key from remaining key and path already used
// bits = key path used
// rkey = remaining key
// key = full key (returned)
void joinKey (Goldilocks &fr, const vector<uint64_t> &bits, const Goldilocks::Element (&rkey)[4], Goldilocks::Element (&key)[4])
{
    uint64_t n[4] = {0, 0, 0, 0};
    uint64_t accs[4] = {0, 0, 0, 0};
    for (uint64_t i=0; i<bits.size(); i++)
    {
        if (bits[i])
        {
            accs[i%4] = (accs[i%4] | (uint64_t(1)<<n[i%4]));
        }
        n[i%4] += 1;
    }
    Goldilocks::Element auxk[4];
    for (uint64_t i=0; i<4; i++) auxk[i] = rkey[i];
    for (uint64_t i=0; i<4; i++)
    {
        uint64_t aux = fr.toU64(auxk[i]);
        aux = ((aux<<n[i]) | accs[i]);
        auxk[i] = fr.fromU64(aux);
    }
    for (uint64_t i=0; i<4; i++) key[i] = auxk[i];
}

/**
 * Removes bits from the key depending on the smt level
 * key -key
 * nBits - bits to remove
 * returns rkey - remaining key bits to store
 */
void removeKeyBits (Goldilocks &fr, const Goldilocks::Element (&key)[4], uint64_t nBits, Goldilocks::Element (&rkey)[4])
{
    uint64_t fullLevels = nBits / 4;
    uint64_t auxk[4];

    for (uint64_t i=0; i<4; i++)
    {
        auxk[i] = fr.toU64(key[i]);
    }

    for (uint64_t i = 0; i < 4; i++)
    {
        uint64_t n = fullLevels;
        if (fullLevels * 4 + i < nBits) n += 1;
        auxk[i] = auxk[i] >> n;
    }

    for (uint64_t i=0; i<4; i++)
    {
        rkey[i] = fr.fromU64(auxk[i]);
    }
}

uint64_t getKeyChildren64Position (const bool (&keys)[256], uint64_t level)
{
    if (level > 250)
    {
        zklog.error("getKeyChildren64Position() got invalid level=" + to_string(level));
        exitProcess();
    }
    uint64_t result = 0;
    if (keys[level + 0]) result += 32;
    if (keys[level + 1]) result += 16;
    if (keys[level + 2]) result += 8;
    if (keys[level + 3]) result += 4;
    if (keys[level + 4]) result += 2;
    if (keys[level + 5]) result += 1;
    return result;
}