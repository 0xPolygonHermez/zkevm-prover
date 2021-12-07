#ifndef UTILS_HPP
#define UTILS_HPP

#include "ffiasm/fr.hpp"
#include "context.hpp"

/*********/
/* Print */
/*********/

void printRegs    (Context &ctx);
void printVars    (Context &ctx);
void printMem     (Context &ctx);
void printStorage (Context &ctx);
void printDb      (Context &ctx);
void printDb      (RawFr &fr, map< RawFr::Element, vector<RawFr::Element>, CompareFe > &db);

void printReg  (Context &ctx, string name, RawFr::Element &V, bool h = false, bool bShort = false);
void printU64  (Context &ctx, string name, uint64_t v);
void printU32  (Context &ctx, string name, uint32_t v);
void printU16  (Context &ctx, string name, uint16_t v);

string printFea (Context &ctx, Fea &fea);

#endif