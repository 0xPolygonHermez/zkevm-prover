#ifndef KECCAK_STATE_HPP
#define KECCAK_STATE_HPP

#include <stdint.h>
#include "config.hpp"

class KeccakState
{
public:
    uint8_t byte[200]; // 200 bytes * 8 = 1600 bits = b

    /*
        bit = 0 ... 1600
        returns 1 or 0
    */
    inline uint8_t getBit (uint64_t bit) const
    {
        zkassert(bit<1600);
        if ( ( byte[bit/8] & (1<<(bit%8)) ) == 0 )
        {
            return 0;
        }
        else
        {
            return 1;
        }
    };

    /*
        bit = 0 ... 1600
        value = 0 or 1
    */
    inline void setBit (uint64_t bit, uint8_t value)
    {
        zkassert(bit<1600);
        zkassert(value<=1);
        if ( value == 0 )
        {
            byte[bit/8] &= ~(1<<(bit%8));
        }
        else
        {
            byte[bit/8] |= 1<<(bit%8);
        }
    };

    /*
        A[x, y, z] = S [w(5y +x) + z] = S[64x+320y+z]
        x = 0 ... 4
        y = 0 ... 4
        z = 0 ... 63
    */
    inline uint8_t getBit (uint64_t x, uint64_t y, uint64_t z) const
    {
        zkassert(x<5);
        zkassert(y<5);
        zkassert(z<64);
        return getBit(64*x+320*y+z);
    };

    /*
        A[x, y, z] = S [w(5y +x) + z] = S[64x+320y+z]
        x = 0 ... 4
        y = 0 ... 4
        z = 0 ... 63
        value = 0 or 1
    */
    inline void setBit (uint64_t x, uint64_t y, uint64_t z, uint8_t value)
    {
        zkassert(x<5);
        zkassert(y<5);
        zkassert(z<64);
        zkassert(value<=1);
        setBit(64*x+320*y+z, value);
    };
};

#endif