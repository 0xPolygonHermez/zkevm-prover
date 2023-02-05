#ifndef MAIN_EXEC_REQUIRED_HPP_fork_0
#define MAIN_EXEC_REQUIRED_HPP_fork_0

#include <string>
#include "sm/storage/smt_action.hpp"
#include "goldilocks_base_field.hpp"
#include "sm/binary/binary_action.hpp"
#include "sm/arith/arith_action.hpp"
#include "sm/padding_kk/padding_kk_executor.hpp"
#include "sm/padding_kkbit/padding_kkbit_executor.hpp"
//#include "sm/nine2one/nine2one_executor.hpp"
#include "sm/memory/memory_executor.hpp"
#include "sm/padding_pg/padding_pg_executor.hpp"
#include "sm/mem_align/mem_align_executor.hpp"

using namespace std;

namespace fork_0
{

class MainExecRequired
{
public:
    vector<SmtAction> Storage;
    vector<MemoryAccess> Memory;
    vector<BinaryAction> Binary;
    vector<ArithAction> Arith;
    vector<PaddingKKExecutorInput> PaddingKK;
    vector<PaddingKKBitExecutorInput> PaddingKKBit;
    //vector<Nine2OneExecutorInput> Nine2One;
    vector<vector<Goldilocks::Element>> KeccakF;
    vector<PaddingPGExecutorInput> PaddingPG;
    vector<array<Goldilocks::Element, 17>> PoseidonG; // The 17th fe is the permutation
    vector<MemAlignAction> MemAlign;
};

} // namespace

#endif