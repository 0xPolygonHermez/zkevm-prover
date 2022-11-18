#ifndef CONSTANT_POLS_ABSTARCT_HPP
#define CONSTANT_POLS_ABSTARCT_HPP

#include <string>

class ConstantPolsAbstarct
{
public:
    ConstantPolsAbstarct(void *pAddress, uint64_t degree){};
    ConstantPolsAbstarct(){};
    virtual ~ConstantPolsAbstarct(){};
    static void *address(void) { return nullptr; };
    static uint64_t degree(void) { return 0; };
    static uint64_t size(void) { return 0; };

    virtual Goldilocks::Element &getElement(uint64_t pol, uint64_t evaluation) = 0;
};

struct StarkFiles
{
    std::string zkevmConstPols;
    bool mapConstPolsFile;
    std::string zkevmConstantsTree;
    std::string zkevmStarkInfo;
};
#endif // CONSTANT_POLS_ABSTARCT_HPP
