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
    bool allocated = false;

public:
    Polinomial()
    {
        _pAddress = NULL;
        _degree = 0;
        _dim = 0;
        _offset = 0;
        allocated = false;
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
        _pAddress = (Goldilocks::Element *)calloc(_degree * _dim, sizeof(Goldilocks::Element));
        _offset = 1;
        allocated = true;
    };
    ~Polinomial()
    {
        if (allocated)
            free(_pAddress);
    };
    Goldilocks::Element *address(void) { return _pAddress; }
    uint64_t degree(void) { return _degree; }
    uint64_t dim(void) { return _dim; }
    uint64_t length(void) { return _degree * _dim; }

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