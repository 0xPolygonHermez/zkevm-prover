#ifndef POLINOMIAL
#define POLINOMIAL

#include "goldilocks_base_field.hpp"

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
};
#endif