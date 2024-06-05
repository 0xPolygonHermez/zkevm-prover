#include "stdint.h"

void *calloc_zkevm(uint64_t count, uint64_t size);
void *malloc_zkevm(uint64_t size);

void free_zkevm(void *ptr);
