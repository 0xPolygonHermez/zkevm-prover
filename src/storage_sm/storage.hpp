#ifndef SSM_PROGRAM_LINE_HPP
#define SSM_PROGRAM_LINE_HPP

#include "config.hpp"
#include "smt_action_list.hpp"
#include "ffiasm/fr.hpp"

void StorageExecutor (RawFr &fr, const Config &config, SmtActionList &smtActionList);

#endif