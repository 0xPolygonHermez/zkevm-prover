#include "sha256_config.hpp"
#include "sha256_gate.hpp"
#include "gate_u32.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "timer.hpp"
#include "zkassert.hpp"

GateConfig SHA256GateConfig = GateConfig(
	uint64_t(0),
	uint64_t(160480),
	uint64_t(170000),
	uint64_t(1),
	uint64_t(45),
	uint64_t(768),
	uint64_t(44),
	uint64_t((45+(768*44))),
	uint64_t(256),
	uint64_t(44),
    uint64_t(1<<23) // TODO
);

void SHA256GateString (const string &s, string &hash)
{
    string ba;
    string2ba(s, ba);
    SHA256Gate((uint8_t *)ba.c_str(), ba.size(), hash);
}

void SHA256Gate (
    const uint8_t * pData, uint64_t dataSize, string &hash,
    string scriptFile, string polsFile, string connectionsFile)
{
    // Initialize hash values:
    // (first 32 bits of the fractional parts of the square roots of the first 8 primes 2..19):
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };

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

    // Process the message in successive 512-bit chunks: break message into 512-bit chunks
    for (uint64_t chunk=0; chunk<paddedSize/64; chunk++)
    {
        // Initialize array of round constants:
        // (first 32 bits of the fractional parts of the cube roots of the first 64 primes 2..311):

        // Create a new gate state per loop
        GateState S(SHA256GateConfig);

        // Create the k constants
        GateU32 k[64] = {
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S),
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S),
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S),
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S),
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S),
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S),
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S),
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S) };
        uint64_t kValue[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };
        for (uint64_t i=0; i<64; i++)
        {
            k[i] = kValue[i];
        }

        // copy chunk into first 16 words w[0..15] of the message schedule array
        
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

        // create a 64-entry message schedule array w[0..63] of 32-bit words
        GateU32 w[64] = {
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S),
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S),
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S),
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S),
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S),
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S),
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S),
            GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S) };

        // copy the 64 data bytes (512 bits) into Sin[0..511] and into w[0..15]
        for (uint64_t i=0; i<16; i++)
        {
            // bytes2u32(pChunkBytes + 4*i, w[i], true);
            uint32_t aux;
            bytes2u32(pChunkBytes + 4*i, aux, true);
            vector<uint8_t> bits;
            u322bits(aux, bits);
            zkassert(bits.size() == 32);
            for (uint64_t j=0; j<32; j++)
            {
                uint64_t ref = SHA256GateConfig.sinRef0 + (i*32 + j)*44;
                S.gate[ref].pin[pin_a].bit = bits[j];
                S.gate[ref].pin[pin_a].source = external;
                w[i].bit[j].ref = ref;
                w[i].bit[j].pin = pin_a;
            }
            //cout << "w[" << i << "]=" << w[i].toString(S) << endl;
        }

        // create a set of 8 variables of 32-bit words = 256 bits
        GateU32 h32[8] = {GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S), GateU32(S)};

        // copy the h[8] state (256 bits) into Sin[512..767] and into h32[0..7]
        for (uint64_t i=0; i<8; i++)
        {
            vector<uint8_t> bits;
            u322bits(h[i], bits);
            zkassert(bits.size() == 32);
            for (uint64_t j=0; j<32; j++)
            {
                uint64_t ref = SHA256GateConfig.sinRef0 + (512 + i*32 + j)*44;
                S.gate[ref].pin[pin_a].bit = bits[j];
                S.gate[ref].pin[pin_a].source = external;
                h32[i].bit[j].ref = ref;
                h32[i].bit[j].pin = pin_a;
            }
            //cout << "h32[" << i << "]=" << h32[i].toString(S) << endl;
        }

        // Extend the first 16 words into the remaining 48 words w[16..63] of the message schedule array:
        for (uint64_t i=16; i<64; i++)
        {
            GateU32 aux1(S), aux2(S), aux3(S), aux4(S);

            // uint32_t s0 = rotateRight32(w[i-15], 7) ^ rotateRight32(w[i-15], 18) ^ ( w[i-15] >> 3 );
            aux1 = w[i-15];
            aux1.rotateRight(7);
            aux2 = w[i-15];
            aux2.rotateRight(18);
            aux3 = w[i-15];
            aux3.shiftRight(3);
            GateU32_xor(S, aux1, aux2, aux4);
            GateU32 s0(S);
            GateU32_xor(S, aux4, aux3, s0);
            //cout << "s0[" << i << "]=" << s0.toString(S) << endl;

            // uint32_t s1 = rotateRight32(w[i-2], 17) ^ rotateRight32(w[i-2], 19) ^ ( w[i-2] >> 10 );
            aux1 = w[i-2];
            aux1.rotateRight(17);
            //cout << "w[i-2]=" << w[i-2].toString(S) << " aux1=" << aux1.toString(S) << endl;
            aux2 = w[i-2];
            aux2.rotateRight(19);
            //cout << "w[i-2]=" << w[i-2].toString(S) << " aux2=" << aux2.toString(S) << endl;
            aux3 = w[i-2];
            aux3.shiftRight(10);
            //cout << "w[i-2]=" << w[i-2].toString(S) << " aux3=" << aux3.toString(S) << endl;
            GateU32_xor(S, aux1, aux2, aux4);
            //cout << "aux4=" << aux4.toString(S) << endl;
            GateU32 s1(S);
            GateU32_xor(S, aux4, aux3, s1);
            //cout << "s1[" << i << "]=" << s1.toString(S) << endl;

            // w[i] = w[i-16] + s0 + w[i-7] + s1;
            GateU32_add(S, w[i-16], s0, aux1);
            //cout << "aux1=" << aux1.toString(S) << "=" << w[i-16].toString(S) << "+" << s0.toString(S) << endl;
            GateU32_add(S, aux1, w[i-7], aux2);
            //cout << "aux2=" << aux2.toString(S) << "=" << aux1.toString(S) << "+" << w[i-7].toString(S) << endl;
            GateU32_add(S, aux2, s1, w[i]);
            //cout << "w[i]=" << w[i].toString(S) << "=" << aux2.toString(S) << "+" << s1.toString(S) << endl;
            //cout << "w[" << i << "]=" << w[i-16].toString(S) << "+" << s0.toString(S) << "+" << w[i-7].toString(S) << "+" << s1.toString(S) << endl;

            //cout << "w[" << i << "]=" << w[i].toString(S) << endl;
        }

        // Initialize working variables to current hash value
        GateU32 a(S), b(S), c(S), d(S), e(S), f(S), g(S), hh(S);
        a = h32[0];
        b = h32[1];
        c = h32[2];
        d = h32[3];
        e = h32[4];
        f = h32[5];
        g = h32[6];
        hh = h32[7];

        // Compression function main loop
        for (uint64_t i=0; i<64; i++)
        {
            GateU32 aux1(S), aux2(S), aux3(S), aux4(S);

            // uint32_t S1 = rotateRight32(e, 6) ^ rotateRight32(e, 11) ^ rotateRight32(e, 25);
            GateU32 S1(S);
            aux1 = e;
            aux1.rotateRight(6);
            aux2 = e;
            aux2.rotateRight(11);
            aux3 = e;
            aux3.rotateRight(25);
            GateU32_xor(S, aux1, aux2, aux4);
            GateU32_xor(S, aux4, aux3, S1);

            // uint32_t ch = (e & f) ^ ((~e) & g);
            GateU32 ch(S);
            GateU32_and(S, e, f, aux1);
            GateU32_not(S, e, aux2);
            GateU32_and(S, aux2, g, aux3);
            GateU32_xor(S, aux1, aux3, ch);

            // uint32_t temp1 = h + S1 + ch + k[i] + w[i];
            GateU32 temp1(S);
            GateU32_add(S, hh, S1, aux1);
            GateU32_add(S, aux1, ch, aux2);
            GateU32_add(S, aux2, k[i], aux3);
            GateU32_add(S, aux3, w[i], temp1);
            //cout << "temp1[" << i << "]=" << temp1.toString(S) << endl;

            // uint32_t S0 = rotateRight32(a, 2) ^ rotateRight32(a, 13) ^ rotateRight32(a, 22);
            GateU32 S0(S);
            aux1 = a;
            aux1.rotateRight(2);
            aux2 = a;
            aux2.rotateRight(13);
            aux3 = a;
            aux3.rotateRight(22);
            GateU32_xor(S, aux1, aux2, aux4);
            GateU32_xor(S, aux4, aux3, S0);

            // uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            GateU32 maj(S);
            GateU32_and(S, a, b, aux1);
            GateU32_and(S, a, c, aux2);
            GateU32_and(S, b, c, aux3);
            GateU32_xor(S, aux1, aux2, aux4);
            GateU32_xor(S, aux4, aux3, maj);

            // uint32_t temp2 = S0 + maj;
            GateU32 temp2(S);
            GateU32_add(S, S0, maj, temp2);
            //cout << "temp2[" << i << "]=" << temp2.toString(S) << endl;
    
            hh = g;
            g = f;
            f = e;
            // e = d + temp1;
            GateU32_add(S, d, temp1, e);
            d = c;
            c = b;
            b = a;
            // a = temp1 + temp2;
            GateU32_add(S, temp1, temp2, a);
        }
        // Add the compressed chunk to the current hash value
        h[0] = h[0] + a.toU32();
        h[1] = h[1] + b.toU32();
        h[2] = h[2] + c.toU32();
        h[3] = h[3] + d.toU32();
        h[4] = h[4] + e.toU32();
        h[5] = h[5] + f.toU32();
        h[6] = h[6] + g.toU32();
        h[7] = h[7] + hh.toU32();

        // Make sure that Sout is located in the expected gates, both in pin a and r
        for (uint64_t i=0; i<8; i++)
        {
            GateU32 * pGateU32 = NULL;
            switch (i)
            {
                case 0: pGateU32 = &a; break;
                case 1: pGateU32 = &b; break;
                case 2: pGateU32 = &c; break;
                case 3: pGateU32 = &d; break;
                case 4: pGateU32 = &e; break;
                case 5: pGateU32 = &f; break;
                case 6: pGateU32 = &g; break;
                case 7: pGateU32 = &hh; break;
                default: zkassert(false);                
            }
            for (uint64_t j=0; j<32; j++)
            {
                uint64_t aux = SHA256GateConfig.soutRef0 + SHA256GateConfig.soutRefDistance*(32*i + j);
                S.XOR( pGateU32->bit[j].ref, pGateU32->bit[i].pin, SHA256GateConfig.zeroRef, pin_a, aux );
                S.SoutRefs[32*i + j] = aux;
                //cout << "SHA256() i=" << i << " aux=" << aux << " pin_a=" << (uint64_t)S.gate[S.SoutRefs[i]].pin[pin_a].bit << " pin_r=" << (uint64_t)S.gate[S.SoutRefs[i]].pin[pin_r].bit << endl;
            }
        }

        S.printCounters();
        if (chunk == 0 && scriptFile.size() > 0)
        {
            json j;
            S.saveScriptToJson(j);
            cout << "Generating SHA256 script file: " << scriptFile << endl;
            json2file(j, scriptFile);
            cout << "Generated SHA256 script file: " << scriptFile << endl;
            scriptFile = "";
        }

        if (chunk == 0 && polsFile.size() > 0)
        {
            json j;
            S.savePolsToJson(j);
            cout << "Generating SHA256 polynomials file: " << polsFile << endl;
            json2file(j, polsFile);
            cout << "Generated SHA256 polynomials file: " << polsFile << endl;
            polsFile = "";
        }

        if (chunk == 0 && connectionsFile.size() > 0)
        {
            json j;
            S.saveConnectionsToJson(j);
            cout << "Generating SHA256 connections file: " << connectionsFile << endl;
            json2file(j, connectionsFile);
            cout << "Generated SHA256 connections file: " << connectionsFile << endl;
            connectionsFile = "";
        }
    }

    mpz_class hashScalar;
    hashScalar = h[0];
    hashScalar = hashScalar << 32;
    hashScalar += h[1];
    hashScalar = hashScalar << 32;
    hashScalar += h[2];
    hashScalar = hashScalar << 32;
    hashScalar += h[3];
    hashScalar = hashScalar << 32;
    hashScalar += h[4];
    hashScalar = hashScalar << 32;
    hashScalar += h[5];
    hashScalar = hashScalar << 32;
    hashScalar += h[6];
    hashScalar = hashScalar << 32;
    hashScalar += h[7];

    hash = "0x" + hashScalar.get_str(16);
}
