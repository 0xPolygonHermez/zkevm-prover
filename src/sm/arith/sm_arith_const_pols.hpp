#ifndef __SM_ARITH_CONST_POLS__H__
#define __SM_ARITH_CONST_POLS__H__

#include <stdint.h>

template <typename T, uint64_t period>
class ClockConstPolItem
{
    protected:
        uint64_t offset;
    public:
        ClockConstPolItem ( void ) { offset = 0; };
        void setup ( uint64_t offset ) { this->offset = offset; };
        const T operator[] (uint64_t index) const { return ((index % period) == offset) ? 1 : 0; }
};

template <typename T, uint64_t period>
class ClockConstPol
{
    protected:
        ClockConstPolItem<T, period> clocks[period];
    public:
        ClockConstPol ( void ) { for (uint64_t index = 0; index < period; ++index) clocks[index].setup(index); };
        const ClockConstPolItem<T, period> &operator[] (uint64_t index) { return clocks[index]; }
};

template <typename T, T fromValue, T toValue>
class RangeConstPol
{
    public:
        const T operator[] (uint64_t index) const { return fromValue + (index % (toValue - fromValue + 1)); }
};

template <typename T, int bits>
class BitsConstPol: public RangeConstPol<uint64_t, 0, (1 << bits)-1> {};

typedef BitsConstPol<uint64_t, 19> ArithBit19ConstPol;
typedef ClockConstPol<uint64_t, 32> ArithClockConstPol;
typedef RangeConstPol<int64_t, -16, 16> ArithSigned4BitsConstPol;
typedef RangeConstPol<int64_t, -(1 << 18), 1 << 18> ArithSigned18BitsConstPol;

#endif