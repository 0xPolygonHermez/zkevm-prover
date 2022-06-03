#ifndef MAIN_EXEC_REQUIRED_HPP
#define MAIN_EXEC_REQUIRED_HPP

#include <string>
#include "sm/storage/smt_action.hpp"
#include "ff/ff.hpp"
#include "sm/generated/commit_pols.hpp"
#include "sm/binary/binary_action.hpp"
#include "sm/arith/arith_action.hpp"
#include "sm/padding_kk/padding_kk_executor.hpp"
#include "sm/padding_kkbit/padding_kkbit_executor.hpp"
#include "sm/nine2one/nine2one_executor.hpp"
#include "sm/norm_gate9/norm_gate9_executor.hpp"
#include "sm/memory/memory_executor.hpp"
#include "sm/padding_pg/padding_pg_executor.hpp"
#include "sm/mem_align/mem_align_executor.hpp"

using namespace std;

class MainExecRequired
{
public:
    vector<SmtAction> Storage;
    vector<MemoryAccess> Memory;
    vector<BinaryAction> Binary;
    vector<ArithAction> Arith;
    vector<PaddingKKExecutorInput> PaddingKK;
    vector<PaddingKKBitExecutorInput> PaddingKKBit;
    vector<Nine2OneExecutorInput> Nine2One;
    vector<vector<FieldElement>> KeccakF;
    vector<NormGate9ExecutorInput> NormGate9;
    map<uint32_t, bool> Byte4;
    vector<PaddingPGExecutorInput> PaddingPG;
    vector<array<FieldElement, 16>> PoseidonG;
    vector<MemAlignAction> MemAlign;
};

#endif