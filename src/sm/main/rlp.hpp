#ifndef __RLP_TOOLS__
#define __RLP_TOOLS__

#include <stdint.h>
#include <string>

inline int codingUInt64(std::string &data, uint64_t value, uint8_t codingBase = 0)
{
    unsigned char blen[9] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    int index;
    for (index = 0; index < 8 && value; ++index) {
        blen[7-index] = value & 0xFF;
        value = value >> 8;
    }
    const char *pdata = (char *)blen;
    pdata += (8 - index);
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
    int bytes = codingUInt64(data, len, encodeType + 55);
}

inline void encodeUInt64(std::string &data, uint64_t value)
{
    if (value && value <= 127) {
        data.push_back(value);
        return;
    }
    codingUInt64(data, value, 0x80);
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
        int value = ASCIIHexToInt[phex[index]];
        if (index || !(len % 2)) {
            ++index;
            value = value * 16 + ASCIIHexToInt[phex[index]];
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


#endif
