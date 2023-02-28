#ifndef __KECCAK_WRAPPER_H__
#define __KECCAK_WRAPPER_H__

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int64_t keccak (void *data, int64_t dataSize, void *hash, int64_t HashSize);

#endif