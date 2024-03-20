#ifndef KEY_VALUE_TREE_HPP
#define KEY_VALUE_TREE_HPP

#include <unistd.h>
#include <unordered_map>
#include <vector>
#include "zkresult.hpp"
#include "goldilocks_base_field.hpp"

using namespace std;

class KeyValueTree
{
private:
    //LevelTree levelTree;
    unordered_map< string, vector<mpz_class> > keys;
public:
    zkresult write   (const Goldilocks::Element (&key)[4], const mpz_class &value, uint64_t &level);
    zkresult write   (const string &keyString,             const mpz_class &value, uint64_t &level);
    zkresult read    (const Goldilocks::Element (&key)[4],       mpz_class &value, uint64_t &level);
    zkresult read    (const string &keyString,                   mpz_class &value, uint64_t &level);
    zkresult extract (const Goldilocks::Element (&key)[4], const mpz_class &value); // returns ZKR_DB_KEY_NOT_FOUND if key was not found; value is used to check that it matches the latest value
    zkresult extract (const string &keyString,             const mpz_class &value); // returns ZKR_DB_KEY_NOT_FOUND if key was not found; value is used to check that it matches the latest value
    uint64_t level   (const Goldilocks::Element (&key)[4]); // returns level; key might or might not exist in the tree
};

#endif