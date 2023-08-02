#ifndef COMPARE_FE_FR_HPP
#define COMPARE_FE_FR_HPP

#include <alt_bn128.hpp>


struct CompareFeFr
{
    AltBn128::Engine& E;
    
    CompareFeFr(AltBn128::Engine& _E) : E(_E) {}
    
    bool operator()(const AltBn128::FrElement& a, const AltBn128::FrElement& b) const
    {
        std::string num1 = E.fr.toString(a);
        std::string num2 = E.fr.toString(b);

        if(num1.length() < num2.length()) {
            return true;
        } else if(num1.length() > num2.length()) {
            return false;
        } else {
            return num1 < num2;
        }
       
        // bool res = E.fr.lt(a, b);
        // std::cout << E.fr.toString(a) << " < " << E.fr.toString(b) << " = " << res << std::endl;
        // return res;
    }
};

#endif

