#include <stdint.h>

#ifdef __USE_CUDA__
void *calloc_zkevm(uint64_t count, uint64_t size) {
    char *a;
    uint64_t total = count*size;
    cudaMallocHost(&a, total);
    if (total > (1<<20)) {
        uint64_t nPieces = (1<<8);
        uint64_t segment = total/nPieces;
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
    cudaMallocHost(&a, size);
    return a;
}

void free_zkevm(void *ptr) { cudaFreeHost(ptr); }
#endif
