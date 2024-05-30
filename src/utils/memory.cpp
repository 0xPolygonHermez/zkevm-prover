#include <stdint.h>
#include <cstdlib>

#ifndef __USE_CUDA__
void *calloc2(uint64_t count, uint64_t size) { return calloc(count, size); }

void *malloc2(uint64_t size) { return malloc(size); }
void free2(void *ptr) { free(ptr); }
#endif
