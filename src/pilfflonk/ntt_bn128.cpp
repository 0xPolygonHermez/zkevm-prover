#include "ntt_bn128.hpp"
#include <cstring>
#include <thread_utils.hpp>

static inline u_int64_t BR(u_int64_t x, u_int64_t domainPow)
{
    x = (x >> 16) | (x << 16); 
    x = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8);
    x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4);
    x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2);
    return (((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1)) >> (32 - domainPow);
}

void NTT_AltBn128::NTT_iters(AltBn128::FrElement *dst, AltBn128::FrElement *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all, u_int64_t nphase, AltBn128::FrElement *aux, bool inverse)
{
    AltBn128::FrElement *dst_;
    if (dst != NULL)
    {
        dst_ = dst;
    }
    else
    {
        dst_ = src; 
    } 
    AltBn128::FrElement *a = dst_;
    AltBn128::FrElement *a2 = aux;
    AltBn128::FrElement *tmp;

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
    bool iseven = true;
    tmp = a;
    if (nphase % 2 == 1)
    {
        iseven = false;
        tmp = a2;
    }
    reversePermutation(tmp, src, size, offset_cols, ncols, ncols_all);
    if (iseven == false)
    {
        tmp = a2;
        a2 = a;
        a = tmp;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(nThreads);
    uint64_t count = 1;
    for (u_int64_t s = 1; s <= domainPow; s += maxBatchPow, ++count)
    {
        if (res > 0 && count == res + 1 && maxBatchPow > 1)
        {
            maxBatchPow -= 1;
        }
        u_int64_t sInc = s + maxBatchPow <= domainPow ? maxBatchPow : domainPow - s + 1;
        u_int64_t rs = s - 1;
        u_int64_t re = domainPow - 1;
        u_int64_t rb = 1 << rs;
        u_int64_t rm = (1 << (re - rs)) - 1;
        u_int64_t batchSize = 1 << sInc;
        u_int64_t nBatches = size / batchSize;

        int chunk1 = nBatches / nThreads;
        if (chunk1 == 0)
        {
            chunk1 = 1;
        }

#pragma omp parallel for schedule(static, chunk1)
        for (u_int64_t b = 0; b < nBatches; b++)
        {
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

                    AltBn128::FrElement w = root(s + si, j);
                    for (u_int64_t k = 0; k < ncols; ++k)
                    {
                        AltBn128::FrElement t = E.fr.mul(w, a[offset1 + k]);
                        AltBn128::FrElement u = a[offset2 + k];

                        E.fr.add(a[offset2 + k], t, u);
                        E.fr.sub(a[offset1 + k], u, t);
                    }
                }
            }
            if (s + maxBatchPow <= domainPow || !inverse)
            {
                for (u_int64_t x = 0; x < batchSize; x++)
                {
                    u_int64_t offset_dstY = (x * nBatches + b) * ncols;
                    u_int64_t offset_src = (b * batchSize + x) * ncols;
                    std::memcpy(&a2[offset_dstY], &a[offset_src], ncols * sizeof(AltBn128::FrElement));
                }
            }
            else
            {
                for (u_int64_t x = 0; x < batchSize; x++)
                {
                    u_int64_t dsty = intt_idx((x * nBatches + b), size);
                    u_int64_t offset_dstY = dsty * ncols;
                    u_int64_t offset_src = (b * batchSize + x) * ncols;
                    for (uint64_t k = 0; k < ncols; k++)
                    {
                        E.fr.mul(a2[offset_dstY + k], a[offset_src + k], powTwoInv[domainPow]);
                    }
                }
            }
        }
        tmp = a2;
        a2 = a;
        a = tmp;
    }
    if (a != dst_)
    {
        if (size > 1)
        {
            assert(0); // should never need this copy...
        }
        ThreadUtils::parcpy(dst_, a, size * ncols, nThreads);
    }
}

void NTT_AltBn128::NTT(AltBn128::FrElement *dst, AltBn128::FrElement *src, u_int64_t size, u_int64_t ncols, AltBn128::FrElement *buffer, u_int64_t nphase, u_int64_t nblock, bool inverse)
{ 
    if (ncols == 0 || size == 0)
    {
        return;
    }
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
    AltBn128::FrElement *dst_ = NULL;
    AltBn128::FrElement *aux = NULL;
    if (buffer == NULL)
    {
        aux = (AltBn128::FrElement *)malloc(sizeof(AltBn128::FrElement) * size * ncols_alloc);
    }
    else
    {
        aux = buffer;
    }
    if (nblock > 1)
    {
        dst_ = (AltBn128::FrElement *)malloc(sizeof(AltBn128::FrElement) * size * ncols_alloc);
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
        NTT_AltBn128::NTT_iters(dst_, src, size, offset_cols, aux_ncols, ncols, nphase, aux, inverse);
        if (nblock > 1)
        {
#pragma omp parallel for schedule(static)
            for (u_int64_t ie = 0; ie < size; ++ie)
            {
                u_int64_t offset2 = ie * ncols + offset_cols;
                std::memcpy(&dst[offset2], &dst_[ie * aux_ncols], aux_ncols * sizeof(AltBn128::FrElement));
            }
        }
        offset_cols += aux_ncols;
    }
    if (nblock > 1)
    {
        free(dst_);
    }
    if (buffer == NULL)
    {
        free(aux);
    }
}
/**
 * @brief permutation of components of an array in bit-reversal order. If dst==src the permutation is performed on-site.
 *
 * @param dst destination pointer (may be equal to src)
 * @param src source pointer
 * @param size field size
 * @param offset_cols columns offset (for NTT wifh nblock>1)
 * @param ncols number of columns of destination array
 * @param ncols_all number of columns of source array (ncols = nocols_all if nblock == 1)
 */
void NTT_AltBn128::reversePermutation(AltBn128::FrElement *dst, AltBn128::FrElement *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all)
{
    uint32_t domainSize = log2(size);
    if (dst != src)
    {
        if (extension <= 1)
        {
#pragma omp parallel for schedule(static)
            for (u_int64_t i = 0; i < size; i++)
            {
                u_int64_t r = BR(i, domainSize);
                u_int64_t offset_r1 = r * ncols_all + offset_cols;
                u_int64_t offset_i1 = i * ncols;
                std::memcpy(&dst[offset_i1], &src[offset_r1], ncols * sizeof(AltBn128::FrElement));
            }
        }
        else
        {
            u_int64_t ext_ = (size / extension) * ncols_all;

#pragma omp parallel for schedule(static)
            for (u_int64_t i = 0; i < size; i++)
            {
                u_int64_t r = BR(i, domainSize);
                u_int64_t offset_r1 = r * ncols_all + offset_cols;
                u_int64_t offset_i1 = i * ncols;
                if (offset_r1 < ext_)
                {
                    std ::memcpy(&dst[offset_i1], &src[offset_r1], ncols * sizeof(AltBn128::FrElement));
                }
                else
                {
                    std::memset(&dst[offset_i1], 0, ncols * sizeof(AltBn128::FrElement));
                }
            }
        }
    }
    else
    {
        if (extension <= 1)
        {
            assert(offset_cols == 0 && ncols == ncols_all); // single block
#pragma omp parallel for schedule(static)
            for (u_int64_t i = 0; i < size; i++)
            {
                u_int64_t r = BR(i, domainSize);
                u_int64_t offset_r = r * ncols;
                u_int64_t offset_i = i * ncols;
                if (r < i)
                {
                    AltBn128::FrElement tmp[ncols];
                    memcpy(&tmp[0], &src[offset_r], ncols * sizeof(AltBn128::FrElement));
                    memcpy(&dst[offset_r], &src[offset_i], ncols * sizeof(AltBn128::FrElement));
                    memcpy(&dst[offset_i], &tmp[0], ncols * sizeof(AltBn128::FrElement));
                }
            }
        }
        else
        {
            assert(0); // Option not implemented yet
        }
    }
}

void NTT_AltBn128::INTT(AltBn128::FrElement *dst, AltBn128::FrElement *src, u_int64_t size, u_int64_t ncols, AltBn128::FrElement *buffer, u_int64_t nphase, u_int64_t nblock)
{

    if (ncols == 0 || size == 0)
    {
        return;
    }
    AltBn128::FrElement *dst_;
    if (dst == NULL)
    {
        dst_ = src;
    }
    else
    {
        dst_ = dst;
    }
    NTT(dst_, src, size, ncols, buffer, nphase, nblock, true);
}

void NTT_AltBn128::extendPol(AltBn128::FrElement *output, AltBn128::FrElement *input, uint64_t N_Extended, uint64_t N, uint64_t ncols, AltBn128::FrElement *buffer, u_int64_t nphase, u_int64_t nblock)
{
    NTT_AltBn128 ntt_extension(E, N_Extended, nThreads, N_Extended / N);

    AltBn128::FrElement *tmp = NULL;
    if (buffer == NULL)
    {
        tmp = (AltBn128::FrElement *)malloc(N_Extended * ncols * sizeof(AltBn128::FrElement));
    }
    else
    {
        tmp = buffer;
    }

    INTT(output, input, N, ncols, tmp, nphase, nblock);
    ntt_extension.NTT(output, output, N_Extended, ncols, tmp, nphase, nblock);

    if (buffer == NULL)
    {
        free(tmp);
    }
}