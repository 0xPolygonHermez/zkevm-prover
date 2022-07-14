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
               uint64_t offset = 0) : _pAddress((Goldilocks::Element *)pAddress),
                                      _degree(degree),
                                      _dim(dim),
                                      _offset(offset){};
    Polinomial(uint64_t degree,
               uint64_t dim) : _degree(degree),
                               _dim(dim)
    {
        if (degree == 0 || dim == 0)
            return;
        _pAddress = (Goldilocks::Element *)calloc(_degree * _dim, sizeof(Goldilocks::Element));
        if (_pAddress == NULL)
        {
            std::cerr << "Error allocating polinomial with size: " << _degree * _dim * sizeof(Goldilocks::Element) << std::endl;
            exit(-1);
        }
        _offset = 1;
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

    Goldilocks::Element &operator[](int i) { return _pAddress[i * _offset]; };

    std::string toString(uint numElements = 1, uint radix = 10)
    {
        std::string res = "";
        for (uint i = 0; i < numElements; i++)
        {
            res += Goldilocks::toString(_pAddress[i]);
            if (i != numElements - 1)
                res += " ";
        }
        return res;
    }
};
#endif