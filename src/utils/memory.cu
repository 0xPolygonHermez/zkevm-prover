#include <stdint.h>

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
void *calloc_zkevm(uint64_t count, uint64_t size) {
    char *a;
    uint64_t total = count*size;
    cudaHostAlloc(&a, total, cudaHostAllocPortable);
    uint64_t segment = 1<<20;
    if (total > segment) {
        uint64_t nPieces = (total + segment - 1) / segment;
        uint64_t last_segment = total - segment*(nPieces-1);
#pragma omp parallel for
        for (int i = 0; i < nPieces; i++) {
            memset(a+segment*i, 0, i==nPieces-1?last_segment:segment);
        }
    } else {
        memset(a, 0, total);
    }
    return a;
}

void *malloc_zkevm(uint64_t size) {
    char *a;
    cudaHostAlloc(&a, size, cudaHostAllocPortable);
    return a;
}

void free_zkevm(void *ptr) { cudaFreeHost(ptr); }
#endif
