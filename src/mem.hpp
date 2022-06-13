#ifndef MEM_HPP
#define MEM_HPP

#include <vector>
#include "reference.hpp"
#include "script.hpp"
#include "pols.hpp"
#include "goldilocks/goldilocks_base_field.hpp"

using namespace std;

typedef vector<Reference> Mem;

void MemAlloc(Mem &mem, Goldilocks &fr, const Script &script, const Pols &cmPols, const Reference *constRefs, const string &constTreePolsInputFile);
void MemFree(Mem &mem);

void Pols2Refs(Goldilocks &fr, const Pols &pol, Reference *ref);

#endif