#include "ntt_goldilocks.hpp"
#include <iostream>

static inline u_int64_t BR(u_int64_t x, u_int64_t domainPow)
{
    x = (x >> 16) | (x << 16);
    x = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8);
    x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4);
    x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2);
    return (((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1)) >> (32 - domainPow);
}

void NTT_Goldilocks::NTT_iters(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all, u_int64_t nphase, Goldilocks::Element *aux)
{
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
    Goldilocks::Element *a2 = aux;
    Goldilocks::Element *tmp;

    reversePermutation(a2, src, size, offset_cols, ncols, ncols_all);

    tmp = a2;
    a2 = a;
    a = tmp;

    u_int64_t domainPow = log2(size);
    assert(((u_int64_t)1 << domainPow) == size);
    if (nphase < 1 || domainPow == 0)
    {
        nphase = 1;
    }
    else if (nphase > domainPow)
    {
        nphase = domainPow;
    }
    u_int64_t maxBatchPow = s / nphase;
    u_int64_t res = s % nphase;
    if (res > 0)
    {
        maxBatchPow += 1;
    }
    u_int64_t batchSize = 1 << maxBatchPow;
    u_int64_t nBatches = size / batchSize;
    omp_set_dynamic(0);
    omp_set_num_threads(nThreads);
    for (u_int64_t s = 1; s <= domainPow; s += maxBatchPow)
    {
        if (res > 0 && s == res + 1 && maxBatchPow > 1)
        {
            maxBatchPow -= 1;
        }
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
#pragma omp parallel for schedule(static)
        for (u_int64_t ie = 0; ie < size; ++ie)
        {
            u_int64_t offset2 = ie * ncols;
            std::memcpy(&dst_[offset2], &a[offset2], ncols * sizeof(Goldilocks::Element));
        }
    }
}

void NTT_Goldilocks::NTT(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, u_int64_t nphase, u_int64_t nblock)
{
    if (nblock < 1)
    {
        nblock = 1;
    }
    if (nblock > ncols)
    {
        nblock = ncols;
    }

    u_int64_t offset_cols = 0;
    u_int64_t ncols_block = ncols / nblock;
    u_int64_t ncols_res = ncols % nblock;
    u_int64_t ncols_alloc = ncols_block;
    if (ncols_res > 0)
    {
        ncols_alloc += 1;
    }
    Goldilocks::Element *dst_ = NULL;
    Goldilocks::Element *aux = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * size * ncols_alloc);
    if (nblock > 1)
    {
        dst_ = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * size * ncols_alloc);
    }
    else
    {
        dst_ = dst;
    }
    for (u_int64_t ib = 0; ib < nblock; ++ib)
    {
        uint64_t aux_ncols = ncols_block;
        if (ib < ncols_res)
            aux_ncols += 1;
        NTT_Goldilocks::NTT_iters(dst_, src, size, offset_cols, aux_ncols, ncols, nphase, aux);
        if (nblock > 1)
        {
#pragma omp parallel for schedule(static)
            for (u_int64_t ie = 0; ie < size; ++ie)
            {
                u_int64_t offset2 = ie * ncols + offset_cols;
                std::memcpy(&dst[offset2], &dst_[ie * aux_ncols], aux_ncols * sizeof(Goldilocks::Element));
            }
        }
        offset_cols += aux_ncols;
    }
    if (nblock > 1)
    {
        free(dst_);
    }
    free(aux);
}

void NTT_Goldilocks::reversePermutation(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all)
{
    uint32_t domainSize = log2(size);
#pragma omp parallel for schedule(static)
    for (u_int64_t i = 0; i < size; i++)
    {
        u_int64_t r = BR(i, domainSize);
        u_int64_t offset_r = r * ncols_all + offset_cols;
        u_int64_t offset_i = i * ncols;
        std::memcpy(&dst[offset_i], &src[offset_r], ncols * sizeof(Goldilocks::Element));
    }
}

void NTT_Goldilocks::INTT(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, u_int64_t nphase, u_int64_t nblock)
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
    NTT(dst_, src, size, ncols, nphase, nblock);
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
        Goldilocks::mul(r[i], r[i - 1], Goldilocks::shift());
    }

    std::cout << "Starting INTT of " << ncols << " polinomials with " << N << " length" << std::endl;
    double st_intt_start = omp_get_wtime();
    INTT(tmp, input, N, ncols, 3, 1);
    double st_intt_end = omp_get_wtime();
    std::cout << "INTT finished!  " << st_intt_end - st_intt_start << std::endl;

    std::cout << "Starting polinomial extension..." << std::endl;
    double st_ext_start = omp_get_wtime();

#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < N; i++)
    {
        for (uint64_t j = 0; j < ncols; j++)
        {
            Goldilocks::mul(tmp[i * ncols + j], tmp[ncols * i + j], r[i]);
        }
    }

#pragma omp parallel for schedule(static)
    for (uint64_t i = N * ncols; i < N_Extended * ncols; i++)
    {
        tmp[i] = Goldilocks::zero();
    }

    double st_ext_end = omp_get_wtime();
    std::cout << "Polinomial extension finished!  " << st_ext_end - st_ext_start << std::endl;

    std::cout << "Starting NTT of " << N_Extended << " length polinomials and " << ncols << " polinomials" << std::endl;
    double st_ntt_start = omp_get_wtime();
    ntt_extension.NTT(output, tmp, N_Extended, ncols, 3, 1);
    double st_ntt_end = omp_get_wtime();
    std::cout << "NTT finished! " << st_ntt_end - st_ntt_start << std::endl;

    free(r);
    free(tmp);
}
