#include <stdint.h>

#ifdef __USE_CUDA__
void *calloc2(uint64_t count, uint64_t size) {
    char *a;
    cudaMallocManaged(&a, count*size);
#pragma omp parallel for
    for (uint64_t i = 0; i < count; i++) {
        memset(a+ i*size, 0, size);
    }
    return a;
}

void *malloc2(uint64_t size) {
    char *a;
    cudaMallocManaged(&a, size);
    return a;
}

void free2(void *ptr) { cudaFree(ptr); }
#endif
