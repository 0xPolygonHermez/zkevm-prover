#include "keccak_wrapper.hpp"
#include "Keccak-more-compact.hpp"

int64_t keccak (void *data, int64_t dataSize, void *hash, int64_t hashSize)
{
    if (hashSize < 32) {
        return -1;
    }
    Keccak(1088, 512, (unsigned char *) data, dataSize, 0x01, (unsigned char *) hash, 32);
    return 32;
}

