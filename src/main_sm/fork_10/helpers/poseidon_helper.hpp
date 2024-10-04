#ifndef POSEIDON_HELPER_HPP_fork_10
#define POSEIDON_HELPER_HPP_fork_10

#include "zkresult.hpp"
#include "main_sm/fork_10/main/context.hpp"
#include "main_sm/fork_10/main/main_exec_required.hpp"

namespace fork_10
{

zkresult HashP_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7, int32_t hashAddr);
zkresult HashP_verify (Context &ctx, Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7, MainExecRequired *required, int32_t hashAddr);

void HashPLen_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7, int32_t hashAddr);
zkresult HashPLen_verify (Context &ctx, Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7, MainExecRequired *required, int32_t hashAddr);

zkresult HashPDigest_calculate (Context &ctx, Goldilocks::Element &fi0, Goldilocks::Element &fi1, Goldilocks::Element &fi2, Goldilocks::Element &fi3, Goldilocks::Element &fi4, Goldilocks::Element &fi5, Goldilocks::Element &fi6, Goldilocks::Element &fi7, int32_t hashAddr);
zkresult HashPDigest_verify (Context &ctx, Goldilocks::Element &op0, Goldilocks::Element &op1, Goldilocks::Element &op2, Goldilocks::Element &op3, Goldilocks::Element &op4, Goldilocks::Element &op5, Goldilocks::Element &op6, Goldilocks::Element &op7, MainExecRequired *required, int32_t hashAddr);

} // namespace

#endif