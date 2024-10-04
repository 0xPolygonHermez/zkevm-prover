#ifndef ARITH_HELPER_HPP_fork_12
#define ARITH_HELPER_HPP_fork_12

#include "zkresult.hpp"
#include "main_sm/fork_12/main/context.hpp"
#include "main_sm/fork_12/main/main_exec_required.hpp"

namespace fork_12
{
    
zkresult Arith_verify (Context &ctx, Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7, MainExecRequired *required);

} // namespace

#endif