#include "blake_config.hpp"
#include "blake_gate.hpp"
#include "gate_state.hpp"
#include "gate_u64.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "timer.hpp"
#include "zkassert.hpp"

// BLAKE2b uses an initialization vector obtained by taking the first 64 bits of the fractional parts
// of the positive square roots of the first eight prime numbers
const uint64_t IV0 = 0x6a09e667f3bcc908;   // Frac(sqrt(2))
const uint64_t IV1 = 0xbb67ae8584caa73b;   // Frac(sqrt(3))
const uint64_t IV2 = 0x3c6ef372fe94f82b;   // Frac(sqrt(5))
const uint64_t IV3 = 0xa54ff53a5f1d36f1;   // Frac(sqrt(7))
const uint64_t IV4 = 0x510e527fade682d1;   // Frac(sqrt(11))
const uint64_t IV5 = 0x9b05688c2b3e6c1f;   // Frac(sqrt(13))
const uint64_t IV6 = 0x1f83d9abfb41bd6b;   // Frac(sqrt(17))
const uint64_t IV7 = 0x5be0cd19137e2179;   // Frac(sqrt(19))

const uint32_t SIGMA[12][16] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 } };

// Forward declarations
void Blake2b256Gate_Compress ( uint64_t (&h)[8], uint8_t (&chunk)[128], uint64_t cBytesCompressed, bool bLastChunk);
//void Blake2b256Gate_Mix ( uint64_t &Va, uint64_t &Vb, uint64_t &Vc, uint64_t &Vd, uint64_t x, uint64_t y);
void Blake2b256Gate_Mix (GateState &S, GateU64 &Va, GateU64 &Vb, GateU64 &Vc, GateU64 &Vd, GateU64 x, GateU64 y);

// Hex string input, hex string output, e.g.: "0x3030" -> "0xcbc63dc2acb86bd8967453ef98fd4f2be2f26d7337a0937958211c128a18b442"
void Blake2b256Gate_String (const string &s, string &hash)
{
    string ba;
    string2ba(s, ba);
    Blake2b256Gate((uint8_t *)ba.c_str(), ba.size(), hash);
}

// Main function
void Blake2b256Gate (const uint8_t * pData, uint64_t dataSize, string &hash)
{
    uint8_t Key[64]; // Optional 0..64 byte key
    uint64_t cbKeyLen = 0; // Number, (0..64) Length of optional key in bytes
    uint64_t cbHashLen = 32; // Number, (1..64) Desired hash length in bytes
    zkassert(cbKeyLen >= 0);
    zkassert(cbKeyLen <= 64);
    zkassert(cbHashLen >= 1);
    zkassert(cbHashLen <= 64);

    // The BLAKE2b algorithm uses 8-byte (UInt64) words, and 128-byte chunks

    // Initialize State vector h with IV
    uint64_t h[8] = { IV0, IV1, IV2, IV3, IV4, IV5, IV6, IV7 };

    // Mix key size (cbKeyLen) and desired hash length (cbHashLen) into h0
    // h0 ← h0 xor 0x0101kknn
    //     where kk is Key Length (in bytes)
    //           nn is Desired Hash Length (in bytes)
    h[0] = h[0] ^ (0x01010000 | (uint8_t)cbKeyLen<<8 | (uint8_t)cbHashLen);

    // Each time we Compress we record how many bytes have been compressed
    uint64_t cBytesCompressed = 0;
    uint64_t cBytesRemaining = dataSize;

    // If there was a key supplied (i.e. cbKeyLen > 0) 
    // then pad with trailing zeros to make it 128-bytes (i.e. 16 words) 
    // and prepend it to the message M
    uint8_t keyPadding[128] = {0};
    if (cbKeyLen > 0)
    {
        zkassert(cbKeyLen <= 64);
        memcpy(keyPadding, Key, cbKeyLen); // M ← Pad(Key, 128) || M
        cBytesRemaining += 128;
    }

    // Compress whole 128-byte chunks of the message, except the last chunk
    uint8_t chunk[128];

    while (cBytesRemaining > 128)
    {
        memcpy(chunk, pData + cBytesCompressed, 128); // get next 128 bytes of message M
        cBytesCompressed += 128; // increase count of bytes that have been compressed
        cBytesRemaining  -= 128; // decrease count of bytes in M remaining to be processed
        Blake2b256Gate_Compress(h, chunk, cBytesCompressed, false); // false, this is not the last chunk
    }

    // Compress the final bytes from M (i.e. 0..128 bytes)
    zkassert(cBytesRemaining <= 128);
    memcpy(chunk, pData + cBytesCompressed, cBytesRemaining);
    memset(chunk + cBytesRemaining, 0, 128 - cBytesRemaining); // Pad with zeros
    cBytesCompressed += cBytesRemaining;
    zkassert(cBytesCompressed == (dataSize + (cbKeyLen?128:0)));
    Blake2b256Gate_Compress(h, chunk, cBytesCompressed, true); // true, this is the last chunk

    // Result ← first cbHashLen bytes of little endian state vector h
    mpz_class result;
    for (uint64_t i=0; i<cbHashLen/8; i++)
    {
        if (i != 0) result = result << 64;
        result += swapBytes64(h[i]);
    }
    hash = "0x" + result.get_str(16);
}

void Blake2b256Gate_Compress (uint64_t (&h)[8], uint8_t (&chunk)[128], uint64_t cBytesCompressed, bool bLastChunk)
{
    // Create a gate configuration instance
    GateConfig gateConfig;
    gateConfig.zeroRef         = Blake_ZeroRef;
    gateConfig.slotSize        = Blake_SlotSize;
    gateConfig.maxRefs         = Blake_MaxRefs;
    gateConfig.firstNextRef    = Blake_FirstNextRef;
    gateConfig.sinRef0         = Blake_SinRef0;
    gateConfig.sinRefNumber    = Blake_SinRefNumber;
    gateConfig.sinRefDistance  = Blake_SinRefDistance;
    gateConfig.soutRef0        = Blake_SoutRef0;
    gateConfig.soutRefNumber   = Blake_SoutRefNumber;
    gateConfig.soutRefDistance = Blake_SoutRefDistance;
    
    // Create a new gate state per loop
    GateState S(gateConfig);

    // Input State: Treat each 128-byte message chunk as sixteen 8-byte (64-bit) words m
    GateU64 m[16] = {
            GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S),
            GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S) };

    // Input state: the 16x64 data bytes (1024 bits) into Sin[0..1023] and into m[0..15]
    for (uint64_t i=0; i<16; i++)
    {
        // bytes2u64(chunk + 8*i, m[i], false);
        uint64_t aux;
        bytes2u64(chunk + 8*i, aux, false);
        vector<uint8_t> bits;
        u642bits(aux, bits);
        zkassert(bits.size() == 64);
        for (uint64_t j=0; j<64; j++)
        {
            uint64_t ref = Blake_SinRef0 + (i*64 + j)*44;
            S.gate[ref].pin[pin_a].bit = bits[j];
            S.gate[ref].pin[pin_a].source = external;
            m[i].bit[j].ref = ref;
            m[i].bit[j].pin = pin_a;
        }
        //cout << "m[" << i << "]=" << m[i].toString(S) << endl;
    }

    // create a set of 8 variables of 64-bit words = 512 bits
    GateU64 h64[8] = {GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S)};

    // copy the h[8] state (512 bits) into Sin[1024..1535] and into h32[0..7]
    for (uint64_t i=0; i<8; i++)
    {
        vector<uint8_t> bits;
        u642bits(h[i], bits);
        zkassert(bits.size() == 64);
        for (uint64_t j=0; j<64; j++)
        {
            uint64_t ref = Blake_SinRef0 + (1024 + i*64 + j)*44;
            S.gate[ref].pin[pin_a].bit = bits[j];
            S.gate[ref].pin[pin_a].source = external;
            h64[i].bit[j].ref = ref;
            h64[i].bit[j].pin = pin_a;
        }
        //cout << "h64[" << i << "]=" << h64[i].toString(S) << endl;
    }


    // Input State: Set cBytesCompressed into a 64 bits input
    GateU64 bytesCompressed(S);
    {
        vector<uint8_t> bits;
        u642bits(cBytesCompressed, bits);
        zkassert(bits.size() == 64);
        for (uint64_t j=0; j<64; j++)
        {
            uint64_t ref = Blake_SinRef0 + (1536 + j)*44;
            S.gate[ref].pin[pin_a].bit = bits[j];
            S.gate[ref].pin[pin_a].source = external;
            bytesCompressed.bit[j].ref = ref;
            bytesCompressed.bit[j].pin = pin_a;
        }
    }

    // Input State: set bLastChunk into 1088
    GateBit lastChunk;
    {
        uint64_t ref = Blake_SinRef0 + 1600;
        S.gate[ref].pin[pin_a].bit = bLastChunk ? 1 : 0;
        S.gate[ref].pin[pin_a].source = external;
        lastChunk.ref = ref;
        lastChunk.pin = pin_a;
    }

    // Setup local work vector V
    GateU64 V[16] = {
            GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S),
            GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S), GateU64(S) };

    // Input State: First eight items are copied from persistent state vector h
    // Copy the h[8] state (512 bits) into Sin[512..1023] and into V[0..7]
    for (uint64_t i=0; i<8; i++)
    {
        V[i] = h64[i];
        //cout << "V[" << i << "]=" << V[i].toString(S) << endl;
    }

    // Constants: Remaining eight items are initialized from the IV
    V[8] = IV0;
    V[9] = IV1;
    V[10] = IV2;
    V[11] = IV3;
    V[12] = IV4;
    V[13] = IV5;
    V[14] = IV6;
    V[15] = IV7;

    // Input State: Mix the 128-bit counter t into V12:V13
    //V[12] = V[12] ^ cBytesCompressed; // Low 64-bits of UInt128 t
    GateU64 aux(S);
    aux = V[12];
    GateU64_xor(S, aux, bytesCompressed, V[12]);
    //V[13] = V[13] ^ 0; // High 64-bits of UInt128 t
  
    // State: If this is the last block then invert all the bits in V14
    // if (bLastChunk) V[14] = V[14] ^ 0xFFFFFFFFFFFFFFFF (i.e. not V[14])
    aux = V[14];
    GateU64_xor(S, aux, lastChunk, V[14]);

    // Twelve rounds of cryptographic message mixing
    for (uint64_t i=0; i<12; i++)
    {
        Blake2b256Gate_Mix(S, V[0], V[4], V[8],  V[12], m[SIGMA[i][0]], m[SIGMA[i][1]]);
        Blake2b256Gate_Mix(S, V[1], V[5], V[9],  V[13], m[SIGMA[i][2]], m[SIGMA[i][3]]);
        Blake2b256Gate_Mix(S, V[2], V[6], V[10], V[14], m[SIGMA[i][4]], m[SIGMA[i][5]]);
        Blake2b256Gate_Mix(S, V[3], V[7], V[11], V[15], m[SIGMA[i][6]], m[SIGMA[i][7]]);

        Blake2b256Gate_Mix(S, V[0], V[5], V[10], V[15], m[SIGMA[i][8]],  m[SIGMA[i][9]]);
        Blake2b256Gate_Mix(S, V[1], V[6], V[11], V[12], m[SIGMA[i][10]], m[SIGMA[i][11]]);
        Blake2b256Gate_Mix(S, V[2], V[7], V[8],  V[13], m[SIGMA[i][12]], m[SIGMA[i][13]]);
        Blake2b256Gate_Mix(S, V[3], V[4], V[9],  V[14], m[SIGMA[i][14]], m[SIGMA[i][15]]);
    }

    // Mix the upper and lower halves of V into ongoing state vector h
    for (uint64_t i=0; i<8; i++)
    {
        // h[i] = h[i] ^ V[i] ^ V[8 + i];
        GateU64 aux1(S);
        GateU64_xor(S, h64[i], V[i], aux1);
        GateU64_xor(S, aux1, V[8+i], h64[i]);
        h[i] = h64[i].toU64();
    }

    // Make sure that Sout is located in the expected gates, both in pin a and r
    for (uint64_t i=0; i<8; i++)
    {
        for (uint64_t j=0; j<64; j++)
        {
            uint64_t ref = gateConfig.soutRef0 + gateConfig.soutRefDistance*(64*i + j);
            S.XOR( h64[i].bit[j].ref, h64[i].bit[j].pin, gateConfig.zeroRef, pin_a, ref );
            S.SoutRefs[64*i + j] = ref;
            //cout << "SHA256() i=" << i << " aux=" << aux << " pin_a=" << (uint64_t)S.gate[S.SoutRefs[i]].pin[pin_a].bit << " pin_r=" << (uint64_t)S.gate[S.SoutRefs[i]].pin[pin_r].bit << endl;
        }
    }

    S.printCounters();
}

void Blake2b256Gate_Mix (GateState &S, GateU64 &Va, GateU64 &Vb, GateU64 &Vc, GateU64 &Vd, GateU64 x, GateU64 y)
{
    GateU64 aux(S);

    //Va = Va + Vb + x;
    GateU64_add(S, Va, Vb, aux);
    GateU64_add(S, aux, x, Va);

    //Vd = rotateRight64( Vd ^ Va, 32 );
    GateU64_xor(S, Vd, Va, aux);
    Vd = aux;
    Vd.rotateRight(32);

    //Vc = Vc + Vd;
    GateU64_add(S, Vc, Vd, aux);
    Vc = aux;

    //Vb = rotateRight64( Vb ^ Vc, 24);
    GateU64_xor(S, Vb, Vc, aux);
    Vb = aux;
    Vb.rotateRight(24);

    //Va = Va + Vb + y;
    GateU64_add(S, Va, Vb, aux);
    GateU64_add(S, aux, y, Va);

    //Vd = rotateRight64( Vd ^ Va, 16);
    GateU64_xor(S, Vd, Va, aux);
    Vd = aux;
    Vd.rotateRight(16);

    //Vc = Vc + Vd;
    GateU64_add(S, Vc, Vd, aux);
    Vc = aux;

    //Vb = rotateRight64( Vb ^ Vc, 63);
    GateU64_xor(S, Vb, Vc, aux);
    Vb = aux;
    Vb.rotateRight(63);
}
