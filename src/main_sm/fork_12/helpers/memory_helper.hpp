#ifndef MEMORY_HELPER_HPP_fork_12
#define MEMORY_HELPER_HPP_fork_12

#include "zkresult.hpp"
#include "main_sm/fork_12/main/context.hpp"
#include "main_sm/fork_12/main/main_exec_required.hpp"

namespace fork_12
{

void Memory_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7, int32_t memAddr);
zkresult Memory_verify (Context &ctx, Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7, MainExecRequired *required, int32_t memAddr);

} // namespace

#endif