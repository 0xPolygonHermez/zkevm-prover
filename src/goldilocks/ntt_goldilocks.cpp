#include "ntt_goldilocks.hpp"

static inline u_int64_t BR(u_int64_t x, u_int64_t domainPow)
{
    x = (x >> 16) | (x << 16);
    x = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8);
    x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4);
    x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2);
    return (((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1)) >> (32 - domainPow);
}

void NTT_Goldilocks::NTT(Goldilocks::Element *_a, u_int64_t n)
{
    Goldilocks::Element *aux_a = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * n);
    Goldilocks::Element *a = _a;
    Goldilocks::Element *a2 = aux_a;
    Goldilocks::Element *tmp;

    reversePermutation(a2, a, n);

    tmp = a2;
    a2 = a;
    a = tmp;

    u_int64_t domainPow = log2(n);
    assert(((u_int64_t)1 << domainPow) == n);
    u_int64_t maxBatchPow = s / 4;
    u_int64_t batchSize = 1 << maxBatchPow;
    u_int64_t nBatches = n / batchSize;
    for (u_int64_t s = 1; s <= domainPow; s += maxBatchPow)
    {
        u_int64_t sInc = s + maxBatchPow <= domainPow ? maxBatchPow : domainPow - s + 1;
        omp_set_dynamic(0);
        omp_set_num_threads(nThreads);
#pragma omp parallel for
        for (u_int64_t b = 0; b < nBatches; b++)
        {
            u_int64_t rs = s - 1;
            u_int64_t re = domainPow - 1;
            u_int64_t rb = 1 << rs;
            u_int64_t rm = (1 << (re - rs)) - 1;
            for (u_int64_t si = 0; si < sInc; si++)
            {
                u_int64_t m = 1 << (s + si);
                u_int64_t mdiv2 = m >> 1;
                u_int64_t mdiv2i = 1 << si;
                u_int64_t mi = mdiv2i * 2;
                for (u_int64_t i = 0; i < (batchSize >> 1); i++)
                {
                    Goldilocks::Element t;
                    Goldilocks::Element u;
                    u_int64_t ki = b * batchSize + (i / mdiv2i) * mi;
                    u_int64_t ji = i % mdiv2i;

                    u_int64_t j = (b * batchSize / 2 + i);
                    j = (j & rm) * rb + (j >> (re - rs));
                    j = j % mdiv2;

                    // t = root(s + si, j) * a[ki + ji + mdiv2i];
                    Goldilocks::mul(t, root(s + si, j), a[ki + ji + mdiv2i]);
                    u = a[ki + ji];

                    Goldilocks::add(a[ki + ji], t, u);
                    // result[ki + ji] = t + u;
                    Goldilocks::sub(a[ki + ji + mdiv2i], u, t);
                    // result[ki + ji + mdiv2i] = u - t;
                }
            }
        }
        shuffle(a2, a, n, sInc);
        tmp = a2;
        a2 = a;
        a = tmp;
    }
    if (a != _a)
    {
        std::memcpy(_a, a, n * sizeof(u_int64_t));
    }
    free(aux_a);
}

void NTT_Goldilocks::reversePermutation(Goldilocks::Element *result, Goldilocks::Element *a, u_int64_t size)
{
    uint32_t domainSize = log2(size);
#pragma omp parallel for
    for (u_int64_t i = 0; i < size; i++)
    {
        u_int64_t r;
        r = BR(i, domainSize);
        result[i] = a[r];
    }
}

void NTT_Goldilocks::shuffle(Goldilocks::Element *result, Goldilocks::Element *src, u_int64_t size, u_int64_t s)
{
    u_int64_t srcRowSize = 1 << s;

    u_int64_t srcX = 0;
    u_int64_t srcWidth = 1 << s;
    u_int64_t srcY = 0;
    u_int64_t srcHeight = size / srcRowSize;

    u_int64_t dstRowSize = size / srcRowSize;
    u_int64_t dstX = 0;
    u_int64_t dstY = 0;

#pragma omp parallel
#pragma omp single
    traspose(result, src, srcRowSize, srcX, srcWidth, srcY, srcHeight, dstRowSize, dstX, dstY);
#pragma omp taskwait
}

void NTT_Goldilocks::traspose(
    Goldilocks::Element *dst,
    Goldilocks::Element *src,
    u_int64_t srcRowSize,
    u_int64_t srcX,
    u_int64_t srcWidth,
    u_int64_t srcY,
    u_int64_t srcHeight,
    u_int64_t dstRowSize,
    u_int64_t dstX,
    u_int64_t dstY)
{
    if ((srcWidth == 1) || (srcHeight == 1) || (srcWidth * srcHeight < CACHESIZE))
    {
#pragma omp task
        {
            for (u_int64_t x = 0; x < srcWidth; x++)
            {
                for (u_int64_t y = 0; y < srcHeight; y++)
                {
                    dst[(dstY + +x) * dstRowSize + (dstX + y)] = src[(srcY + +y) * srcRowSize + (srcX + x)];
                }
            }
        }
        return;
    }
    if (srcWidth > srcHeight)
    {
        traspose(dst, src, srcRowSize, srcX, srcWidth / 2, srcY, srcHeight, dstRowSize, dstX, dstY);
        traspose(dst, src, srcRowSize, srcX + srcWidth / 2, srcWidth / 2, srcY, srcHeight, dstRowSize, dstX, dstY + srcWidth / 2);
    }
    else
    {
        traspose(dst, src, srcRowSize, srcX, srcWidth, srcY, srcHeight / 2, dstRowSize, dstX, dstY);
        traspose(dst, src, srcRowSize, srcX, srcWidth, srcY + srcHeight / 2, srcHeight / 2, dstRowSize, dstX + srcHeight / 2, dstY);
    }
}

void NTT_Goldilocks::INTT(Goldilocks::Element *a, u_int64_t size)
{
    NTT_Goldilocks::NTT(a, size);
    u_int64_t domainPow = NTT_Goldilocks::log2(size);
    u_int64_t nDiv2 = size >> 1;
#pragma omp parallel for num_threads(nThreads)
    for (u_int64_t i = 1; i < nDiv2; i++)
    {
        Goldilocks::Element tmp;
        u_int64_t r = size - i;
        tmp = a[i];
        a[i] = a[r] * powTwoInv[domainPow];
        a[r] = tmp * powTwoInv[domainPow];
    }
    a[0] = a[0] * powTwoInv[domainPow];
    a[size >> 1] = a[size >> 1] * powTwoInv[domainPow];
}

/*
    Blocks implementation
*/
void NTT_Goldilocks::NTT_Block(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, u_int64_t nphase)
{
    Goldilocks::Element *aux_a = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * size * ncols);
    Goldilocks::Element *dst_;
    if (dst != NULL)
    {
        dst_ = dst;
    }
    else
    {
        dst_ = src;
    }
    Goldilocks::Element *a = dst_;
    Goldilocks::Element *a2 = aux_a;
    Goldilocks::Element *tmp;

    reversePermutation_block(a2, src, size, ncols);
    tmp = a2;
    a2 = a;
    a = tmp;

    u_int64_t domainPow = log2(size);
    assert(((u_int64_t)1 << domainPow) == size);
    u_int64_t maxBatchPow = s / nphase;

    u_int64_t batchSize = 1 << maxBatchPow;
    u_int64_t nBatches = size / batchSize;

    omp_set_dynamic(0);
    omp_set_num_threads(nThreads);
    for (u_int64_t s = 1; s <= domainPow; s += maxBatchPow)
    {

        u_int64_t sInc = s + maxBatchPow <= domainPow ? maxBatchPow : domainPow - s + 1;
#pragma omp parallel for
        for (u_int64_t b = 0; b < nBatches; b++)
        {
            u_int64_t rs = s - 1;
            u_int64_t re = domainPow - 1;
            u_int64_t rb = 1 << rs;
            u_int64_t rm = (1 << (re - rs)) - 1;
            for (u_int64_t si = 0; si < sInc; si++)
            {
                u_int64_t m = 1 << (s + si);
                u_int64_t mdiv2 = m >> 1;
                u_int64_t mdiv2i = 1 << si;
                u_int64_t mi = mdiv2i * 2;
                for (u_int64_t i = 0; i < (batchSize >> 1); i++)
                {
                    u_int64_t ki = b * batchSize + (i / mdiv2i) * mi;
                    u_int64_t ji = i % mdiv2i;

                    u_int64_t offset1 = (ki + ji + mdiv2i) * ncols;
                    u_int64_t offset2 = (ki + ji) * ncols;

                    u_int64_t j = (b * batchSize / 2 + i);
                    j = (j & rm) * rb + (j >> (re - rs));
                    j = j % mdiv2;

                    Goldilocks::Element w = root(s + si, j);
                    for (u_int64_t k = 0; k < ncols; ++k)
                    {
                        Goldilocks::Element t = w * a[offset1 + k];
                        Goldilocks::Element u = a[offset2 + k];

                        Goldilocks::add(a[offset2 + k], t, u);
                        Goldilocks::sub(a[offset1 + k], u, t);
                    }
                }
            }
            u_int64_t srcWidth = 1 << sInc;
            u_int64_t niters = batchSize / srcWidth;
            for (u_int64_t kk = 0; kk < niters; ++kk)
            {
                for (u_int64_t x = 0; x < srcWidth; x++)
                {
                    u_int64_t offset_dstY = (x * (nBatches * niters) + (b * niters + kk)) * ncols;
                    u_int64_t offset_src = ((b * niters + kk) * srcWidth + x) * ncols;
                    std::memcpy(&a2[offset_dstY], &a[offset_src], ncols * sizeof(Goldilocks::Element));
                }
            }
        }
        tmp = a2;
        a2 = a;
        a = tmp;
    }
    if (a != dst_)
    {
        std::memcpy(dst_, a, size * ncols * sizeof(Goldilocks::Element));
    }
    free(aux_a);
}

void NTT_Goldilocks::reversePermutation_block(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols)
{
    uint32_t domainSize = log2(size);
#pragma omp parallel for schedule(static)
    for (u_int64_t i = 0; i < size; i++)
    {
        u_int64_t r = BR(i, domainSize);
        u_int64_t offset_i = i * ncols;
        u_int64_t offset_r = r * ncols;
        std::memcpy(&dst[offset_i], &src[offset_r], ncols * sizeof(u_int64_t));
    }
}

void NTT_Goldilocks::INTT_Block(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, u_int64_t nphase)
{
    Goldilocks::Element *dst_;
    if (dst == NULL)
    {
        dst_ = src;
    }
    else
    {
        dst_ = dst;
    }
    NTT_Block(dst_, src, size, ncols, nphase);
    u_int64_t domainPow = log2(size);
    u_int64_t nDiv2 = size >> 1;

#pragma omp parallel for
    for (u_int64_t i = 1; i < nDiv2; i++)
    {
        Goldilocks::Element tmp;

        u_int64_t r = size - i;
        u_int64_t offset_r = ncols * r;
        u_int64_t offset_i = ncols * i;

        for (uint64_t k = 0; k < ncols; k++)
        {
            tmp = dst_[offset_i + k];
            Goldilocks::mul(dst_[offset_i + k], dst_[offset_r + k], powTwoInv[domainPow]);
            Goldilocks::mul(dst_[offset_r + k], tmp, powTwoInv[domainPow]);
        }
    }

    u_int64_t offset_n = ncols * (size >> 1);
    for (uint64_t k = 0; k < ncols; k++)
    {
        Goldilocks::mul(dst_[k], dst_[k], powTwoInv[domainPow]);
        Goldilocks::mul(dst_[offset_n + k], dst_[offset_n + k], powTwoInv[domainPow]);
    }
}

void NTT_Goldilocks::extendPol(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t N_Extended, uint64_t N, uint64_t ncols)
{
    NTT_Goldilocks ntt_extension(N_Extended);

    Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(N_Extended * ncols * sizeof(Goldilocks::Element));

    // TODO: Pre-compute r
    Goldilocks::Element *r;
    r = (Goldilocks::Element *)malloc(N * sizeof(Goldilocks::Element));
    r[0] = Goldilocks::one();

    for (uint64_t i = 1; i < N; i++)
    {
        Goldilocks::mul(r[i], r[i - 1], Goldilocks::SHIFT);
    }

    INTT_Block(tmp, input, N, ncols);

    //
    for (uint64_t j = 0; j < ncols; j++)
    {
#pragma omp parallel for
        for (uint64_t i = 0; i < N; i++)
        {

            Goldilocks::mul(tmp[i * ncols + j], tmp[ncols * i + j], r[i]);
        }
    }
#pragma omp parallel for schedule(static)
    for (uint64_t i = N * ncols; i < N_Extended * ncols; i++)
    {
        tmp[i] = Goldilocks::zero();
    }
    ntt_extension.NTT_Block(output, tmp, N_Extended, ncols);

    free(r);
    free(tmp);
}