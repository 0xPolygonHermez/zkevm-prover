#ifndef SSM_PROGRAM_LINE_HPP
#define SSM_PROGRAM_LINE_HPP

#include "config.hpp"
#include "smt_action_list.hpp"
#include "ff/ff.hpp"
#include "poseidon_opt/poseidon_goldilocks.hpp"

void StorageExecutor (FiniteField &fr, Poseidon_goldilocks &poseidon, const Config &config, vector<SmtAction> &action);

#endif