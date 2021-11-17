#ifndef UTILS_HPP
#define UTILS_HPP

#include "ffiasm/fr.hpp"
#include "context.hpp"

/*********/
/* Print */
/*********/

void printReg  (Context &ctx, string name, RawFr::Element &V, bool h = false, bool bShort = false);
void printRegs (Context &ctx);
void printVars (Context &ctx);
void printMem  (Context &ctx);

#endif