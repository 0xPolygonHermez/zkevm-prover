#ifndef MEM_HPP
#define MEM_HPP

#include <vector>
#include "reference.hpp"
#include "script.hpp"
#include "pols.hpp"
#include "ffiasm/fr.hpp"

using namespace std;

typedef vector<Reference> Mem;

void MemAlloc (Mem &mem, const Script &script);
void MemFree (Mem &mem);
void MemCopyPols (RawFr &fr, Mem &mem, const Pols &cmPols, const Pols &constPols);

#endif