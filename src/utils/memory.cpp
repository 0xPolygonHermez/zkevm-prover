#include <stdint.h>
#include <cstdlib>

#ifndef __USE_CUDA__
void *calloc_zkevm(uint64_t count, uint64_t size) { return calloc(count, size); }

void *malloc_zkevm(uint64_t size) { return malloc(size); }
void free_zkevm(void *ptr) { free(ptr); }
#endif
