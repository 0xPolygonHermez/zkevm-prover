#include "sha256.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "timer.hpp"
#include "sha256.hpp"
#include "bcon_sha256.hpp"

// Initialize hash values:
// (first 32 bits of the fractional parts of the square roots of the first 8 primes 2..19)
const uint32_t hIn[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};

// Initialize array of round constants:
// (first 32 bits of the fractional parts of the cube roots of the first 64 primes 2..311):
const uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };

void SHA256String (const string &s, string &hash)
{
    string ba;
    string2ba(s, ba);
    SHA256((uint8_t *)ba.c_str(), ba.size(), hash);
}

void SHA256 (const uint8_t * pData, uint64_t dataSize, string &hash)
{
    mpz_class hashScalar;
    SHA256(pData, dataSize, hashScalar);
    hash = "0x" + hashScalar.get_str(16);
}

void SHA256 (const uint8_t * pData, uint64_t dataSize, mpz_class &hashScalar)
{
	BYTE buf[SHA256_BLOCK_SIZE];
	SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, pData, dataSize);
	sha256_final(&ctx, buf);
    ba2scalar(buf, SHA256_BLOCK_SIZE, hashScalar);
}

void SHA256_old (const uint8_t * pData, uint64_t dataSize, mpz_class &hashScalar)
{
    // Padding:
    // original message of length L bits
    // padded message: <original message of length L> 1 bit <K zero bits> <L as 64 bit integer>
    // (the number of bits will be a multiple of 512)

    // padded data = pData[dataSize] + 0x80 + paddedZeros*0x00 + dataSize[8]
    uint64_t paddedSizeInBitsMin = dataSize*8 + 1 + 64;
    uint64_t paddedSizeInBits = ((paddedSizeInBitsMin / 512) + 1)*512;
    uint64_t paddedSize = paddedSizeInBits / 8;
    uint64_t paddedZeros = (paddedSizeInBits - paddedSizeInBitsMin)/8;

    // Create the padding data buffer
    uint8_t padding[64] = {0};
    u642bytes(dataSize*8, &padding[56], true);
    uint64_t onePosition = 64 - 8 - paddedZeros - 1;
    padding[onePosition] = 0x80;
    uint64_t dataPosition = (dataSize/64)*64;
    for (uint64_t i=0; i<dataSize%64; i++)
    {
        padding[i] = pData[dataPosition+i];
    }
    uint32_t stIn[8];
    memcpy(stIn, hIn, sizeof(hIn));

    // Process the message in successive 512-bit chunks: break message into 512-bit chunks
    for (uint64_t chunk=0; chunk<paddedSize/64; chunk++)
    {
        // determinte the buffer to copy data from
        const uint8_t *pChunkBytes;
        if (chunk == ((paddedSize/64) - 1))
        {
            pChunkBytes = padding;
        }
        else
        {
            pChunkBytes = pData + chunk*64;
        }
        SHA256F(pChunkBytes, stIn, stIn);
    }

    hashScalar = stIn[0];
    hashScalar = hashScalar << 32;
    hashScalar += stIn[1];
    hashScalar = hashScalar << 32;
    hashScalar += stIn[2];
    hashScalar = hashScalar << 32;
    hashScalar += stIn[3];
    hashScalar = hashScalar << 32;
    hashScalar += stIn[4];
    hashScalar = hashScalar << 32;
    hashScalar += stIn[5];
    hashScalar = hashScalar << 32;
    hashScalar += stIn[6];
    hashScalar = hashScalar << 32;
    hashScalar += stIn[7];
}

void SHA256F (const uint8_t inR[64], const uint32_t stIn[8], uint32_t stOut[8]){

    // create a 64-entry message schedule array w[0..63] of 32-bit words
    uint32_t w[64] = {0};

    // copy the 64 bytes (512 bits) into w[0..15]
    for (uint64_t i=0; i<16; i++)
    {
        bytes2u32(inR + 4*i, w[i], true);
    }

    // Extend the first 16 words into the remaining 48 words w[16..63] of the message schedule array:
    for (uint64_t i=16; i<64; i++)
    {
        uint32_t s0 = rotateRight32(w[i-15], 7) ^ rotateRight32(w[i-15], 18) ^ ( w[i-15] >> 3 );
        mpz_class aux = s0;
        uint32_t s1 = rotateRight32(w[i-2], 17) ^ rotateRight32(w[i-2], 19) ^ ( w[i-2] >> 10 );
        aux = s1;
        w[i] = w[i-16] + s0 + w[i-7] + s1;
        aux = w[i];
    }

    // Initialize working variables to current hash value
    uint32_t a = stIn[0];
    uint32_t b = stIn[1];
    uint32_t c = stIn[2];
    uint32_t d = stIn[3];
    uint32_t e = stIn[4];
    uint32_t f = stIn[5];
    uint32_t g = stIn[6];
    uint32_t h = stIn[7];

    // Compression function main loop
    for (uint64_t i=0; i<64; i++)
    {
        uint32_t S1 = rotateRight32(e, 6) ^ rotateRight32(e, 11) ^ rotateRight32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + k[i] + w[i];
        mpz_class aux = temp1;
        uint32_t S0 = rotateRight32(a, 2) ^ rotateRight32(a, 13) ^ rotateRight32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        aux = temp2;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    // Add the compressed chunk to the current hash value
    stOut[0] = stIn[0] + a;
    stOut[1] = stIn[1] + b;
    stOut[2] = stIn[2] + c;
    stOut[3] = stIn[3] + d;
    stOut[4] = stIn[4] + e;
    stOut[5] = stIn[5] + f;
    stOut[6] = stIn[6] + g;
    stOut[7] = stIn[7] + h;

}
