#ifndef KECCAK_INPUT_HPP
#define KECCAK_INPUT_HPP

#include <stdint.h>
#include <cstring>
#include "scalar.hpp"

class KeccakInput
{
public:
    const uint8_t * pInput;
    uint64_t inputSize;
    uint64_t offset;
    uint8_t pad[136];
    uint64_t padSize;

    inline void init (const uint8_t * pIn, uint64_t inSizeInBytes)
    {
        pInput = pIn;
        inputSize = inSizeInBytes;
        offset = 0;
        memset(pad, 0, 136);
        pad[0] = 1; // bit 0 = 1
        padSize = 136 - inputSize%136;
        pad[padSize-1] |= 0b10000000; // bit 7 = 1
    }
    
    // Read next 1088 bits = 136 Bytes
    inline bool getNext (uint8_t * pNext)
    {
        if ((offset+136) <= inputSize)
        {
            memcpy(pNext, pInput+offset, 136);
            offset += 136;
            return true;
        }
        else if (offset <= inputSize) // If inputSize=0 or is multiple of 1088, also return data
        {
            uint64_t offset2 = inputSize - offset;
            if (offset2 > 0)
            {
                memcpy(pNext, pInput+offset, offset2);
            }
            if (offset2 < 136)
            {
                memcpy(pNext+offset2, pad, padSize);
            }
            offset += 136;
            return true;
        }

        // No more data to return
        return false;
    }

    inline bool getNextBits (uint8_t * pNext)
    {
        uint8_t r[136];
        if (!getNext(r)) return false;
        for (uint64_t i=0; i<136; i++)
        {
            byte2bits(r[i], pNext+i*8);
        }
        return true;
    }
};

#endif