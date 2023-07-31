#ifndef POLINOMIAL
#define POLINOMIAL

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "compare_fe.hpp"
#include <math.h> /* log2 */
#include "zklog.hpp"
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

    static void copy(Polinomial &a, Polinomial &b, uint64_t size = 0)
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

    static inline void subElement(Polinomial &out, uint64_t idx_out, Polinomial &in_a, uint64_t idx_a, Polinomial &in_b, uint64_t idx_b)
    {
        assert(out.dim() == in_a.dim());

        if (in_a.dim() == 1)
        {
            out[idx_out][0] = in_a[idx_a][0] - in_b[idx_b][0];
        }
        else
        {
            out[idx_out][0] = in_a[idx_a][0] - in_b[idx_b][0];
            out[idx_out][1] = in_a[idx_a][1] - in_b[idx_b][1];
            out[idx_out][2] = in_a[idx_a][2] - in_b[idx_b][2];
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

    static inline void divElement(Polinomial &out, uint64_t idx_out, Polinomial &in_a, uint64_t idx_a, Goldilocks::Element &b)
    {
        Polinomial polB(&b, 1, 1);
        divElement(out, idx_out, in_a, idx_a, polB, 0);
    }

    static inline void divElement(Polinomial &out, uint64_t idx_out, Polinomial &in_a, uint64_t idx_a, Polinomial &in_b, uint64_t idx_b)
    {
        assert(out.dim() == in_a.dim() && in_b.dim() == 1);

        if (in_a.dim() == 1)
        {
            out[idx_out][0] = in_a[idx_a][0] / in_b[idx_b][0];
        }
        else
        {
            Goldilocks::Element inv = Goldilocks::inv(*in_b[idx_b]);
            Polinomial polInv(&inv, 1, 1);
            mulElement(out, idx_out, in_a, idx_a, polInv, 0);
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

    static void calculateH1H2(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol)
    {
        map<std::vector<Goldilocks::Element>, uint64_t, CompareFe> idx_t;
        multimap<std::vector<Goldilocks::Element>, uint64_t, CompareFe> s;
        multimap<std::vector<Goldilocks::Element>, uint64_t>::iterator it;
        uint64_t i = 0;

        for (uint64_t i = 0; i < tPol.degree(); i++)
        {
            vector<Goldilocks::Element> key = tPol.toVector(i);
            std::pair<vector<Goldilocks::Element>, uint64_t> pr(key, i);

            auto const result = idx_t.insert(pr);
            if (not result.second)
            {
                result.first->second = i;
            }

            s.insert(pr);
        }

        for (uint64_t i = 0; i < fPol.degree(); i++)
        {
            vector<Goldilocks::Element> key = fPol.toVector(i);

            if (idx_t.find(key) == idx_t.end())
            {
                zklog.error("Polinomial::calculateH1H2() Number not included: " + Goldilocks::toString(fPol[i], 16));
                exitProcess();
            }
            uint64_t idx = idx_t[key];
            s.insert(pair<vector<Goldilocks::Element>, uint64_t>(key, idx));
        }

        multimap<uint64_t, vector<Goldilocks::Element>> s_sorted;
        multimap<uint64_t, vector<Goldilocks::Element>>::iterator it_sorted;

        for (it = s.begin(); it != s.end(); it++)
        {
            s_sorted.insert(make_pair(it->second, it->first));
        }

        for (it_sorted = s_sorted.begin(); it_sorted != s_sorted.end(); it_sorted++, i++)
        {
            if ((i & 1) == 0)
            {
                Polinomial::copyElement(h1, i / 2, it_sorted->second);
            }
            else
            {
                Polinomial::copyElement(h2, i / 2, it_sorted->second);
            }
        }
    };

    static void calculateH1H2_(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol, uint64_t pNumber)
    {
        map<std::vector<Goldilocks::Element>, uint64_t, CompareFe> idx_t;
        multimap<std::vector<Goldilocks::Element>, uint64_t, CompareFe> s;
        multimap<std::vector<Goldilocks::Element>, uint64_t>::iterator it;

        vector<int> counter(tPol.degree(), 1);

        for (uint64_t i = 0; i < tPol.degree(); i++)
        {
            vector<Goldilocks::Element> key = tPol.toVector(i);
            idx_t[key] = i + 1;
        }

        for (uint64_t i = 0; i < fPol.degree(); i++)
        {
            vector<Goldilocks::Element> key = fPol.toVector(i);
            uint64_t indx = idx_t[key];
            if (indx == 0)
            {
                zklog.error("Polynomial::calculateH1H2() Number not included: w=" + to_string(i) + " plookup_number=" + to_string(pNumber) + "\nPol:" + Goldilocks::toString(fPol[i], 16));
                exitProcess();
            }
            ++counter[indx - 1];
        }

        uint64_t id = 0;
        for (u_int64_t i = 0; i < tPol.degree(); ++i)
        {
            if (counter[id] == 0)
            {
                ++id;
            }
            counter[id] -= 1;
            Polinomial::copyElement(h1, i, tPol, id);

            if (counter[id] == 0)
            {
                ++id;
            }
            counter[id] -= 1;
            Polinomial::copyElement(h2, i, tPol, id);
        }
    }

    static void calculateH1H2_opt1(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol, uint64_t pNumber, uint64_t *buffer, uint64_t size_keys, uint64_t size_values)
    {
        vector<int> counter(tPol.degree(), 1);  // this 1 is important, space of the original buffer could be used
        vector<bool> touched(size_keys, false); // faster use this than initialize buffer, bitmask could be used
        uint32_t pos = 0;

        // double time1 = omp_get_wtime();
        for (uint64_t i = 0; i < tPol.degree(); i++)
        {
            uint64_t key = tPol.firstValueU64(i);
            uint64_t ind = key % size_keys;
            if (!touched[ind])
            {
                buffer[ind] = pos;
                uint32_t offset = size_keys + 3 * pos;
                buffer[offset] = key;
                buffer[offset + 1] = i;
                buffer[offset + 2] = 0;
                pos += 1;
                touched[ind] = true;
            }
            else
            {
                uint64_t pos_ = buffer[ind];
                bool exit_ = false;
                do
                {
                    uint32_t offset = size_keys + 3 * pos_;
                    if (key == buffer[offset])
                    {
                        buffer[offset + 1] = i;
                        exit_ = true;
                    }
                    else
                    {
                        if (buffer[offset + 2] != 0)
                        {
                            pos_ = buffer[offset + 2];
                        }
                        else
                        {
                            buffer[offset + 2] = pos;
                            // new offset
                            offset = size_keys + 3 * pos;
                            buffer[offset] = key;
                            buffer[offset + 1] = i;
                            buffer[offset + 2] = 0;
                            pos += 1;
                            exit_ = true;
                        }
                    }
                } while (!exit_);
            }
        }

        // double time2 = omp_get_wtime();

        for (uint64_t i = 0; i < fPol.degree(); i++)
        {
            uint64_t indx = 0;
            uint64_t key = fPol.firstValueU64(i);
            uint64_t ind = key % size_keys;
            if (!touched[ind])
            {
                zklog.error("Polynomial::calculateH1H2() Number not included: w=" + to_string(i) + " plookup_number=" + to_string(pNumber) + "\nPol:" + Goldilocks::toString(fPol[i], 16));
                exitProcess();
            }
            uint64_t pos_ = buffer[ind];
            bool exit_ = false;
            do
            {
                uint32_t offset = size_keys + 3 * pos_;
                if (key == buffer[offset])
                {
                    indx = buffer[offset + 1];
                    exit_ = true;
                }
                else
                {
                    if (buffer[offset + 2] != 0)
                    {
                        pos_ = buffer[offset + 2];
                    }
                    else
                    {
                        zklog.error("Polynomial::calculateH1H2() Number not included: w=" + to_string(i) + " plookup_number=" + to_string(pNumber) + "\nPol:" + Goldilocks::toString(fPol[i], 16));
                        exitProcess();
                    }
                }
            } while (!exit_);
            ++counter[indx];
        }

        // double time3 = omp_get_wtime();
        uint64_t id = 0;
        for (u_int64_t i = 0; i < tPol.degree(); ++i)
        {
            if (counter[id] == 0)
            {
                ++id;
            }

            counter[id] -= 1;
            Polinomial::copyElement(h1, i, tPol, id);

            if (counter[id] == 0)
            {
                ++id;
            }
            counter[id] -= 1;
            Polinomial::copyElement(h2, i, tPol, id);
        }
        // double time4 = omp_get_wtime();
        // std::cout << "holu: " << id << " " << pos << " times: " << time2 - time1 << " " << time3 - time2 << " " << time4 - time3 << " " << h2.dim() << std::endl;
    }

    static void calculateH1H2_opt3(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol, uint64_t pNumber, uint64_t *buffer, uint64_t size_keys, uint64_t size_values)
    {
        vector<int> counter(tPol.degree(), 1);  // this 1 is important, space of the original buffer could be used
        vector<bool> touched(size_keys, false); // faster use this than initialize buffer, bitmask could be used
        uint32_t pos = 0;
        uint64_t key[3];

        // double time1 = omp_get_wtime();
        for (uint64_t i = 0; i < tPol.degree(); i++)
        {
            tPol.toVectorU64(i, key);
            uint64_t ind = key[0] % size_keys;
            if (!touched[ind])
            {
                buffer[ind] = pos;
                uint32_t offset = size_keys + 5 * pos;
                buffer[offset] = key[0];
                buffer[offset + 1] = key[1];
                buffer[offset + 2] = key[2];
                buffer[offset + 3] = i;
                buffer[offset + 4] = 0;
                pos += 1;
                touched[ind] = true;
            }
            else
            {
                uint64_t pos_ = buffer[ind];
                bool exit_ = false;
                do
                {
                    uint32_t offset = size_keys + 5 * pos_;
                    if (key[0] == buffer[offset] && key[1] == buffer[offset + 1] && key[2] == buffer[offset + 2])
                    {
                        buffer[offset + 3] = i;
                        exit_ = true;
                    }
                    else
                    {
                        if (buffer[offset + 4] != 0)
                        {
                            pos_ = buffer[offset + 4];
                        }
                        else
                        {
                            buffer[offset + 4] = pos;
                            // new offset
                            offset = size_keys + 5 * pos;
                            buffer[offset] = key[0];
                            buffer[offset + 1] = key[1];
                            buffer[offset + 2] = key[2];
                            buffer[offset + 3] = i;
                            buffer[offset + 4] = 0;
                            pos += 1;
                            exit_ = true;
                        }
                    }
                } while (!exit_);
            }
        }

        // double time2 = omp_get_wtime();

        for (uint64_t i = 0; i < fPol.degree(); i++)
        {
            uint64_t indx = 0;
            fPol.toVectorU64(i, key);
            uint64_t ind = key[0] % size_keys;
            if (!touched[ind])
            {
                zklog.error("Polinomial::calculateH1H2() Number not included: w=" + to_string(i) + " plookup_number=" + to_string(pNumber) + "\nPol:" + Goldilocks::toString(fPol[i], 16));
                exitProcess();
            }
            uint64_t pos_ = buffer[ind];
            bool exit_ = false;
            do
            {
                uint32_t offset = size_keys + 5 * pos_;
                if (key[0] == buffer[offset] && key[1] == buffer[offset + 1] && key[2] == buffer[offset + 2])
                {
                    indx = buffer[offset + 3];
                    exit_ = true;
                }
                else
                {
                    if (buffer[offset + 4] != 0)
                    {
                        pos_ = buffer[offset + 4];
                    }
                    else
                    {
                        zklog.error("Polinomial::calculateH1H2() Number not included: w=" + to_string(i) + " plookup_number=" + to_string(pNumber) + "\nPol:" + Goldilocks::toString(fPol[i], 16));
                        exitProcess();
                    }
                }
            } while (!exit_);
            ++counter[indx];
        }

        // double time3 = omp_get_wtime();
        uint64_t id = 0;
        for (u_int64_t i = 0; i < tPol.degree(); ++i)
        {
            if (counter[id] == 0)
            {
                ++id;
            }

            counter[id] -= 1;
            Polinomial::copyElement(h1, i, tPol, id);

            if (counter[id] == 0)
            {
                ++id;
            }
            counter[id] -= 1;
            Polinomial::copyElement(h2, i, tPol, id);
        }
        // double time4 = omp_get_wtime();
        // std::cout << "holu: " << id << " " << pos << " times: " << time2 - time1 << " " << time3 - time2 << " " << time4 - time3 << " " << h2.dim() << std::endl;
    }

    static void calculateZ(Polinomial &z, Polinomial &num, Polinomial &den)
    {
        uint64_t size = num.degree();

        Polinomial denI(size, 3);
        Polinomial checkVal(1, 3);
        Goldilocks::Element *pZ = z[0];
        Goldilocks3::copy((Goldilocks3::Element *)&pZ[0], &Goldilocks3::one());

        batchInverse(denI, den);
        for (uint64_t i = 1; i < size; i++)
        {
            Polinomial tmp(1, 3);
            Polinomial::mulElement(tmp, 0, num, i - 1, denI, i - 1);
            Polinomial::mulElement(z, i, z, i - 1, tmp, 0);
        }
        Polinomial tmp(1, 3);
        Polinomial::mulElement(tmp, 0, num, size - 1, denI, size - 1);
        Polinomial::mulElement(checkVal, 0, z, size - 1, tmp, 0);

        zkassert(Goldilocks3::isOne((Goldilocks3::Element &)*checkVal[0]));
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
                Polinomial::mulElement(src, threadOffset + j, src, threadOffset + j, src, offset - 1);
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

    static inline void mulAddElement_adim3(Goldilocks::Element *out, Goldilocks::Element *in_a, Polinomial &in_b, uint64_t idx_b)
    {
        if (in_b.dim() == 1)
        {
            out[0] = out[0] + in_a[0] * in_b[idx_b][0];
            out[1] = out[1] + in_a[1] * in_b[idx_b][0];
            out[2] = out[2] + in_a[2] * in_b[idx_b][0];
        }
        else
        {
            Goldilocks::Element A = (in_a[0] + in_a[1]) * (in_b[idx_b][0] + in_b[idx_b][1]);
            Goldilocks::Element B = (in_a[0] + in_a[2]) * (in_b[idx_b][0] + in_b[idx_b][2]);
            Goldilocks::Element C = (in_a[1] + in_a[2]) * (in_b[idx_b][1] + in_b[idx_b][2]);
            Goldilocks::Element D = in_a[0] * in_b[idx_b][0];
            Goldilocks::Element E = in_a[1] * in_b[idx_b][1];
            Goldilocks::Element F = in_a[2] * in_b[idx_b][2];
            Goldilocks::Element G = D - E;
            out[0] = out[0] + (C + G) - F;
            out[1] = out[1] + ((((A + C) - E) - E) - D);
            out[2] = out[2] + B - G;
        }
    }
};
#endif