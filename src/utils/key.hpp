#ifndef KEY_HPP
#define KEY_HPP

#include <vector>
#include "goldilocks_base_field.hpp"

using namespace std;

void splitKey (Goldilocks &fr, const Goldilocks::Element (&key)[4], bool (&result)[256]);
void joinKey (Goldilocks &fr, const vector<uint64_t> &bits, const Goldilocks::Element (&rkey)[4], Goldilocks::Element (&key)[4]);
void removeKeyBits (Goldilocks &fr, const Goldilocks::Element (&key)[4], uint64_t nBits, Goldilocks::Element (&rkey)[4]);
uint64_t getKeyChildren64Position (const bool (&keys)[256], uint64_t level);

#endif