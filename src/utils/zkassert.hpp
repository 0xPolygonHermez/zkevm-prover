#ifndef ZKASSERT_HPP
#define ZKASSERT_HPP

#include <iostream>
#include "exit_process.hpp"

using namespace std;

/* zkassert() definition; unused in opt mode */
#ifdef DEBUG
#define zkassert(a) {if (!(a)) {cerr << "Error: assert failed: " << (#a) << endl; exitProcess();}}
#else
#define zkassert(a)
#endif

/* zkassertpermanent() definition */
#define zkassertpermanent(a) {if (!(a)) {cerr << "Error: assert failed: " << (#a) << endl; exitProcess();}}

#endif