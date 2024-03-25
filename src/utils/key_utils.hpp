#ifndef KEY_UTILS_HPP
#define KEY_UTILS_HPP

#include <vector>
#include "goldilocks_base_field.hpp"

using namespace std;

// Get 256 key bits in SMT order
void splitKey (Goldilocks &fr, const Goldilocks::Element (&key)[4], bool (&result)[256]);

// Get 256 key bits in SMT order, in sets of 6 bits
void splitKey6 (Goldilocks &fr, const Goldilocks::Element (&key)[4], uint8_t (&result)[43]);

// Get 64 key bits, in sets of 6 bits
void splitKey6 (Goldilocks &fr, const uint64_t key, uint8_t (&result)[11]);

// Split a generic-size key, in sets of 9 bits
void splitKey9 (const string &baString, vector<uint64_t> &result);

// Join bits in SMT order and a remaining key into a full key
void joinKey (Goldilocks &fr, const vector<uint64_t> &bits, const Goldilocks::Element (&rkey)[4], Goldilocks::Element (&key)[4]);

// Remove bits in SMT order from a key and get a remaining key
void removeKeyBits (Goldilocks &fr, const Goldilocks::Element (&key)[4], uint64_t nBits, Goldilocks::Element (&rkey)[4]);

#endif