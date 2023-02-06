#ifndef __RLP_TOOLS__
#define __RLP_TOOLS__

#include <stdint.h>
#include <string>
#include <gmp.h>
#include <gmpxx.h>

namespace rlp {

static inline uint8_t valueToByte(uint64_t value)
{
    return (0xFF & value);
}

static inline uint8_t valueToByte(mpz_class value)
{
    return (0xFF & value.get_ui());
}

template<typename T>
inline int coding(std::string &data, T value, uint8_t codingBase = 0)
{
    unsigned char blen[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

    int index;
    for (index = 0; (index < 32) && (value != 0); ++index) {
        blen[31-index] = valueToByte(value);
        value = value >> 8;
    }
    const char *pdata = (char *)blen;
    pdata += (32 - index);
    if (codingBase) {
        codingBase += index;
        data += codingBase;
    }
    data.append(pdata, index);
    return index;
}

inline void encodeLen(std::string &data, uint32_t len, bool composed = false)
{
    unsigned char encodeType = (composed ? 0xc0 : 0x80);
    if (len <= 55) {
        encodeType += len;
        data.push_back(encodeType);
        return;
    }
    coding<uint64_t>(data, len, encodeType + 55);
}

template<typename T>
inline void encode(std::string &data, T value)
{
    if (value && value <= 127) {
        data.push_back(valueToByte(value));
        return;
    }
    coding<T>(data, value, 0x80);
}

const int ASCIIHexToInt[] =
{
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
        0,     1,     2,    3,      4,     5,     6,     7,     8,     9, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000,    10,    11,    12,    13,    14,    15, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000,    10,    11,    12,    13,    14,    15, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,

    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
    -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000,
};

inline bool encodeHexValue(std::string &data, const std::string &hex)
{
    int len = hex.length();
    const char *phex = hex.c_str();
    if (len >= 2 && phex[0] == '0' && phex[1] == 'x') {
        len -= 2;
        phex += 2;
    }

    int index = 0;

    while (index < len)
    {
        int value = ASCIIHexToInt[(int)phex[index]];
        if (index || !(len % 2)) {
            ++index;
            value = value * 16 + ASCIIHexToInt[(int)phex[index]];
        }
        ++index;
        if (value < 0) return false;
        data.push_back(value);
    }
    return true;
}

inline int getHexValueLen(const std::string &hex)
{
    int len = hex.length();
    const char *phex = hex.c_str();
    if (len >= 2 && phex[0] == '0' && phex[1] == 'x') {
        len -= 2;
    }
    if (len % 2) {
        ++len;
    }
    return (len >> 1);
}

inline bool encodeHexData(string &data, const string &hex)
{
    int len = getHexValueLen(hex);

    if (len == 1) {
        string edata;

        bool res = encodeHexValue(edata, hex);
        if (edata.length() != 1 || edata[0] >= 0x80) {
            encodeLen(data, len);
        }
        data += edata;
        return res;
    }

    encodeLen(data, len);
    return encodeHexValue(data, hex);
}

}
#endif
