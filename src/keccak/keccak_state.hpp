#ifndef KECCAK_STATE_HPP
#define KECCAK_STATE_HPP

#include <stdint.h>
#include "config.hpp"

class KeccakState
{
public:
    uint8_t byte[200]; // 200 bytes * 8 = 1600 bits = b
    uint64_t xors;
    uint64_t ands;
    uint64_t ors;
    uint64_t adds;
    uint64_t divs;
    uint64_t mults;
    uint64_t mods;
    uint64_t shifts;
    uint64_t bytexors;
    uint64_t bitsets;
    uint64_t bitgets;

    KeccakState ()
    {
        memset(byte, 0, 200);
        xors = 0;
        ands = 0;
        ors = 0;
        adds = 0;
        divs = 0;
        mults = 0;
        mods = 0;
        shifts = 0;
        bytexors = 0;
        bitsets = 0;
        bitgets = 0;
    }

    void copyCounters (KeccakState S){
        xors = S.xors;
        ands = S.ands;
        ors = S.ors;
        adds = S.adds;
        divs = S.divs;
        mults = S.mults;
        mods = S.mods;
        shifts = S.shifts;
        bytexors = S.bytexors;
        bitsets = S.bitsets;
        bitgets = S.bitgets;
    }

    void printCounters (void)
    {
        cout << "bit xors=" << to_string(xors) << endl;
        cout << "bit ands=" << to_string(ands) << endl;
        cout << "bit ors=" << to_string(ors) << endl;
        cout << "byte adds=" << to_string(adds) << endl;
        cout << "byte divs=" << to_string(divs) << endl;
        cout << "byte mults=" << to_string(mults) << endl;
        cout << "byte mods=" << to_string(mods) << endl;
        cout << "byte shifts=" << to_string(shifts) << endl;
        cout << "byte xors=" << to_string(bytexors) << endl;
        cout << "status bit sets=" << to_string(bitsets) << endl;
        cout << "status bit gets=" << to_string(bitgets) << endl;
    }

    /*
        bit = 0 ... 1600
        returns 1 or 0
    */
    inline uint8_t getBit (uint64_t bit)
    {
        zkassert(bit<1600);
        bitgets++;
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
        bitsets++;
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
    inline uint8_t getBit (uint64_t x, uint64_t y, uint64_t z)
    {
        zkassert(x<5);
        zkassert(y<5);
        zkassert(z<64);
        mults += 2;
        adds += 2;
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
        mults += 2;
        adds += 2;
        setBit(64*x+320*y+z, value);
    };
};

#endif