#ifndef BINARY_ACTION_BYTES_HPP
#define BINARY_ACTION_BYTES_HPP

#include <cstdint>

class BinaryActionBytes
{
public:
    uint8_t a_bytes[32];
    uint8_t b_bytes[32];
    uint8_t c_bytes[32];
    uint64_t opcode;
    uint64_t type;

    BinaryActionBytes()
    {
        memset(a_bytes, 0, 32);
        memset(b_bytes, 0, 32);
        memset(b_bytes, 0, 32);
        opcode = 0;
        type = 0;
    }
};

#endif