#include <stdint.h>
#include <cstdlib>

#if !defined(__USE_CUDA__) || !defined(ENABLE_EXPERIMENTAL_CODE)
void *calloc_zkevm(uint64_t count, uint64_t size) { return calloc(count, size); }

void *malloc_zkevm(uint64_t size) { return malloc(size); }
void free_zkevm(void *ptr) { free(ptr); }
#endif
