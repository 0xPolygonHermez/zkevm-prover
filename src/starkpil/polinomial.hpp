#ifndef POLINOMIAL
#define POLINOMIAL

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "compare_fe.hpp"
#include <math.h> /* log2 */
#include "zklog.hpp"
#include "zkassert.hpp"
#include "exit_process.hpp"

class Polinomial
{
private:
    Goldilocks::Element *_pAddress = NULL;
    uint64_t _degree = 0;
    uint64_t _dim = 0;
    uint64_t _offset = 0;
    bool _allocated = false;
    std::string _name = "";

public:
    Polinomial()
    {
        _pAddress = NULL;
        _degree = 0;
        _dim = 0;
        _offset = 0;
        _allocated = false;
    }
    Polinomial(void *pAddress,
               uint64_t degree,
               uint64_t dim,
               uint64_t offset = 0,
               std::string name = "") : _pAddress((Goldilocks::Element *)pAddress),
                                        _degree(degree),
                                        _dim(dim),
                                        _offset(offset),
                                        _name(name){};
    Polinomial(uint64_t degree,
               uint64_t dim,
               std::string name = "") : _degree(degree),
                                        _dim(dim),
                                        _name(name)
    {
        if (degree == 0 || dim == 0)
            return;
        _pAddress = (Goldilocks::Element *)calloc(_degree * _dim, sizeof(Goldilocks::Element));
        if (_pAddress == NULL)
        {
            zklog.error("Polinomial::Polinomial() failed allocating polinomial with size: " + to_string(_degree * _dim * sizeof(Goldilocks::Element)));
            exitProcess();
        }
        _offset = _dim;
        _allocated = true;
    };

    ~Polinomial()
    {
        if (_allocated)
            free(_pAddress);
    };

    void potConstruct(Goldilocks::Element *pAddress,
                      uint64_t degree,
                      uint64_t dim,
                      uint64_t offset = 0)
    {
        _pAddress = pAddress;
        _degree = degree;
        _dim = dim;
        _offset = offset;
        _allocated = false;
    }

    void potConstruct(uint64_t degree,
                      uint64_t dim
    )
    {
        if (degree == 0 || dim == 0)
            return;
        _pAddress = (Goldilocks::Element *)calloc(degree * dim, sizeof(Goldilocks::Element));
        if (_pAddress == NULL)
        {
            zklog.error("Polinomial::Polinomial() failed allocating polinomial with size: " + to_string(_degree * _dim * sizeof(Goldilocks::Element)));
            exitProcess();
        }
        _dim = dim;
        _degree = degree;
        _offset = dim;
        _allocated = true;
    }

    inline Goldilocks::Element *address(void) { return _pAddress; }
    inline uint64_t degree(void) { return _degree; }
    inline uint64_t dim(void) { return _dim; }
    inline uint64_t length(void) { return _degree * _dim; }
    inline uint64_t size(void) { return _degree * _dim * sizeof(Goldilocks::Element); }
    inline uint64_t offset(void) { return _offset; }

    Goldilocks::Element *operator[](uint64_t i) { return &_pAddress[i * _offset]; };

    std::string toString(uint numElements = 0, uint radix = 10)
    {
        uint64_t elements = (numElements != 0) ? numElements : _degree;
        std::string res = (_name != "") ? _name + ":\n" : "";
        for (uint i = 0; i < elements; i++)
        {
            if (_dim != 1)
            {
                res += "[ ";
                for (uint j = 0; j < _dim; j++)
                {
                    res += Goldilocks::toString(_pAddress[i * _offset + j]);
                    res += " ";
                }
                res += "]\n";
            }
            else
            {
                res += Goldilocks::toString(_pAddress[i * _offset]);
                if (i != elements - 1)
                    res += " ";
            }
        }
        return res;
    }

    static void copy(Polinomial &a, Polinomial &b)
    {
        assert(a.dim() == b.dim());
        assert(a.degree() == b.degree());
        if (b._offset == 1 && a._offset == 1)
        {
            std::memcpy(a.address(), b.address(), b.size());
        }
        else
        {
#pragma omp parallel for
            for (uint64_t i = 0; i < b.degree(); i++)
            {
                std::memcpy(a[i], b[i], b.dim() * sizeof(Goldilocks::Element));
            }
        }
    };

    static void copyElement(Polinomial &a, uint64_t idx_a, Polinomial &b, uint64_t idx_b)
    {
        assert(a.dim() == b.dim());
        std::memcpy(a[idx_a], b[idx_b], b.dim() * sizeof(Goldilocks::Element));
    };

    static void copyElement(Polinomial &a, uint64_t idx_a, std::vector<Goldilocks::Element> b)
    {
        assert(a.dim() == b.size());
        std::memcpy(a[idx_a], &b[0], a.dim() * sizeof(Goldilocks::Element));
    };

    static inline void addElement(Polinomial &out, uint64_t idx_out, Polinomial &in_a, uint64_t idx_a, Polinomial &in_b, uint64_t idx_b)
    {
        assert(out.dim() == in_a.dim());

        if (in_a.dim() == 1)
        {
            out[idx_out][0] = in_a[idx_a][0] + in_b[idx_b][0];
        }
        else
        {
            out[idx_out][0] = in_a[idx_a][0] + in_b[idx_b][0];
            out[idx_out][1] = in_a[idx_a][1] + in_b[idx_b][1];
            out[idx_out][2] = in_a[idx_a][2] + in_b[idx_b][2];
        }
    }

    static inline void mulElement(Polinomial &out, uint64_t idx_out, Polinomial &in_a, uint64_t idx_a, Goldilocks::Element &b)
    {
        Polinomial polB(&b, 1, 1);
        mulElement(out, idx_out, in_a, idx_a, polB, 0);
    }

    static inline void mulElement(Polinomial &out, uint64_t idx_out, Polinomial &in_a, uint64_t idx_a, Polinomial &in_b, uint64_t idx_b)
    {
        assert(out.dim() == in_a.dim());

        if (in_a.dim() == 1)
        {
            out[idx_out][0] = in_a[idx_a][0] * in_b[idx_b][0];
        }
        else if (in_a.dim() == 3 && in_b.dim() == 1)
        {

            out[idx_out][0] = in_a[idx_a][0] * in_b[idx_b][0];
            out[idx_out][1] = in_a[idx_a][1] * in_b[idx_b][0];
            out[idx_out][2] = in_a[idx_a][2] * in_b[idx_b][0];
        }
        else
        {
            Goldilocks::Element A = (in_a[idx_a][0] + in_a[idx_a][1]) * (in_b[idx_b][0] + in_b[idx_b][1]);
            Goldilocks::Element B = (in_a[idx_a][0] + in_a[idx_a][2]) * (in_b[idx_b][0] + in_b[idx_b][2]);
            Goldilocks::Element C = (in_a[idx_a][1] + in_a[idx_a][2]) * (in_b[idx_b][1] + in_b[idx_b][2]);
            Goldilocks::Element D = in_a[idx_a][0] * in_b[idx_b][0];
            Goldilocks::Element E = in_a[idx_a][1] * in_b[idx_b][1];
            Goldilocks::Element F = in_a[idx_a][2] * in_b[idx_b][2];
            Goldilocks::Element G = D - E;

            out[idx_out][0] = (C + G) - F;
            out[idx_out][1] = ((((A + C) - E) - E) - D);
            out[idx_out][2] = B - G;
        }
    };

    inline std::vector<Goldilocks::Element> toVector(uint64_t idx)
    {
        std::vector<Goldilocks::Element> result;
        result.assign(&_pAddress[idx * _offset], &_pAddress[idx * _offset] + _dim);
        return result;
    }
    inline void toVectorU64(uint64_t idx, uint64_t *vect)
    {
        for (uint32_t i = 0; i < _dim; ++i)
        {
            Goldilocks::toU64(vect[i], _pAddress[idx * _offset + i]);
        }
    }
    inline uint64_t firstValueU64(uint64_t idx)
    {
        return Goldilocks::toU64(_pAddress[idx * _offset]);
    }

    // compute the multiplications of the polynomials in src in parallel with partitions of size partitionSize
    // Every thread computes a partition of size partitionSize / (2 * nThreadsPartition)
    // after every computation the size of the partition is doubled until it reaches partitionSize
    inline static void computeMuls(Polinomial &src, uint64_t srcSize, uint64_t partitionSize, uint64_t nThreads)
    {
        if (partitionSize > srcSize)
        {
            return;
        }
        uint64_t nPartitions = srcSize / partitionSize;
        uint64_t nThreadsPartition = nThreads / nPartitions;

#pragma omp parallel for num_threads(nThreads)
        for (uint64_t i = 0; i < nThreads; i++)
        {
            uint64_t thread_idx = omp_get_thread_num();
            uint64_t offset = partitionSize / 2 + thread_idx / nThreadsPartition * partitionSize;
            uint64_t threadOffset = partitionSize / (2 * nThreadsPartition) * (thread_idx % nThreadsPartition) + offset;
            for (uint64_t j = 0; j < partitionSize / (2 * nThreadsPartition); j++)
            {
                Goldilocks3::mul((Goldilocks3::Element *)src[threadOffset + j], (Goldilocks3::Element *)src[threadOffset + j], (Goldilocks3::Element *)src[offset - 1]);
            }
        }
        computeMuls(src, srcSize, partitionSize * 2, nThreads);
    }

    inline static void batchInverseParallel(Polinomial &res, Polinomial &src)
    {
        uint64_t size = src.degree();
        Polinomial tmp(size, 3);

        double pow2thread = floor(log2(omp_get_max_threads()));
        uint64_t nThreads = (1 << (int)pow2thread) / 4;
        uint64_t partitionSize = size / nThreads;

        if(partitionSize < 2) {
            batchInverse(res, src);
            return;
        }
        // initalize tmp with src
        // | s_0 0 0 .. 0 | s_partitionSize 0 0 .. 0 | s_partitionSize+1 0 0 .. 0 | s_partitionSize*(nThreads-1) ... 0 |

#pragma omp parallel for num_threads(nThreads)
        for (uint64_t i = 0; i < nThreads; i++)
        {
            uint64_t thread_idx = omp_get_thread_num();
            Polinomial::copyElement(tmp, thread_idx * partitionSize, src, thread_idx * partitionSize);
        }

#pragma omp parallel for num_threads(nThreads)
        for (uint64_t j = 0; j < nThreads; j++)
        {
            uint64_t thread_idx = omp_get_thread_num();
            for (uint64_t i = 1; i < partitionSize; i++)
            {
                Polinomial::mulElement(tmp, i + thread_idx * partitionSize, tmp, i - 1 + thread_idx * partitionSize, src, i + thread_idx * partitionSize);
            }
        }

        computeMuls(tmp, size, 2 * partitionSize, nThreads);

        Polinomial z(size, 3);
        Goldilocks3::inv((Goldilocks3::Element *)z[0], (Goldilocks3::Element *)tmp[size - 1]);

        // calculo Z
#pragma omp parallel for num_threads(nThreads - 1)
        for (uint64_t i = 0; i < nThreads - 1; i++)
        {
            uint64_t thread_idx = omp_get_thread_num();
            Polinomial::copyElement(z, thread_idx * partitionSize + partitionSize, src, size - thread_idx * partitionSize - partitionSize);
        }
        Goldilocks3::inv((Goldilocks3::Element *)z[0], (Goldilocks3::Element *)tmp[size - 1]);

#pragma omp parallel for num_threads(nThreads)
        for (uint64_t j = 0; j < nThreads; j++)
        {
            uint64_t thread_idx = omp_get_thread_num();
            for (uint64_t i = 1; i < partitionSize; i++)
            {
                Polinomial::mulElement(z, i + thread_idx * partitionSize, z, i - 1 + thread_idx * partitionSize, src, size - i - thread_idx * partitionSize);
            }
        }

        computeMuls(z, size, 2 * partitionSize, nThreads);

#pragma omp parallel for num_threads(nThreads)
        for (uint64_t i = 0; i < size - 1; i++)
        {
            Polinomial::mulElement(res, size - 1 - i, z, i, tmp, size - 2 - i);
        }
        Polinomial::copyElement(res, 0, z, size - 1);
    }

    inline static void batchInverse(Polinomial &res, Polinomial &src)
    {
        uint64_t size = src.degree();
        Polinomial tmp(size, 3);
        Polinomial z(2, 3);

        Polinomial::copyElement(tmp, 0, src, 0);

        for (uint64_t i = 1; i < size; i++)
        {
            Polinomial::mulElement(tmp, i, tmp, i - 1, src, i);
        }

        Goldilocks3::inv((Goldilocks3::Element *)z[0], (Goldilocks3::Element *)tmp[size - 1]);

        for (uint64_t i = size - 1; i > 0; i--)
        {
            Polinomial::mulElement(z, 1, z, 0, src, i);
            Polinomial::mulElement(res, i, z, 0, tmp, i - 1);
            Polinomial::copyElement(z, 0, z, 1);
        }
        Polinomial::copyElement(res, 0, z, 0);
    }
};
#endif