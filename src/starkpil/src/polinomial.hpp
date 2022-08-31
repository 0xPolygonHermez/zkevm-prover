#ifndef POLINOMIAL
#define POLINOMIAL

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "compare_fe.hpp"

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
            std::cerr << "Error allocating polinomial with size: " << _degree * _dim * sizeof(Goldilocks::Element) << std::endl;
            exit(-1);
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
    }

    Goldilocks::Element *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t dim(void) { return _dim; }
    uint64_t length(void) { return _degree * _dim; }
    uint64_t size(void) { return _degree * _dim * sizeof(Goldilocks::Element); }

    Goldilocks::Element *operator[](int i) { return &_pAddress[i * _offset]; };

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
                cerr << "Error: calculateH1H2() Number not included: " << Goldilocks::toString(fPol[i], 16) << endl;
                exit(-1);
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

    static void calculateH1H2_(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol)
    {
        map<std::vector<Goldilocks::Element>, uint64_t, CompareFe> idx_t;
        multimap<std::vector<Goldilocks::Element>, uint64_t, CompareFe> s;
        multimap<std::vector<Goldilocks::Element>, uint64_t>::iterator it;
        uint64_t i = 0;
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
                cerr << "Error: calculateH1H2() Number not included: " << Goldilocks::toString(fPol[i], 16) << endl;
                exit(-1);
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

    inline static void batchInverse(Polinomial &res, Polinomial &src)
    {
        uint64_t size = src.degree();
        Polinomial aux(size, 3);
        Polinomial tmp(size, 3);

        Polinomial::copyElement(tmp, 0, src, 0);

        for (uint64_t i = 1; i < size; i++)
        {
            Polinomial::mulElement(tmp, i, tmp, i - 1, src, i);
        }

        Polinomial z(1, 3);
        Goldilocks3::inv((Goldilocks3::Element *)z[0], (Goldilocks3::Element *)tmp[size - 1]);

        for (uint64_t i = size - 1; i > 0; i--)
        {
            Polinomial::mulElement(aux, i, z, 0, tmp, i - 1);
            Polinomial::mulElement(z, 0, z, 0, src, i);
        }
        Polinomial::copyElement(aux, 0, z, 0);
        Polinomial::copy(res, aux);
    }
};
#endif