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
};
#endif