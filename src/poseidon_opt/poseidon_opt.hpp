#ifndef POSEIDON_OPT
#define POSEIDON_OPT

#include <vector>
#include <string>
#include "ffiasm/fr.hpp"
#include "constants_opt.hpp"
#include <cassert>
using namespace std;

class Poseidon_opt
{
  typedef RawFr::Element FrElement;

  const static int N_ROUNDS_F = 8;
  const unsigned int N_ROUNDS_P[16] = {56, 57, 56, 60, 60, 63, 64, 63, 60, 66, 60, 65, 70, 60, 64, 68};

private:
  RawFr field;
  void ark(vector<FrElement> *state, const vector<FrElement> *c, const int ssize, int it);
  void sbox(vector<FrElement> *state, const vector<FrElement> *c, const int ssize, int it);
  void mix(vector<FrElement> *new_state, vector<FrElement> state, const vector<vector<FrElement>> *m, const int ssize);
  void exp5(FrElement &r);
  void stateExp5(vector<FrElement> *state, const int ssize);

public:
  void hash(vector<FrElement> &state);
  void hash(vector<FrElement> &state, FrElement *result);
  void gmimc(vector<FrElement>, FrElement *result);
};

#endif // POSEIDON_OPT