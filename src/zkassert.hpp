#ifndef ZKASSERT_HPP
#define ZKASSERT_HPP

/* zkassert() definition; unused in debug mode */
#define DEBUG
#ifdef DEBUG
#define zkassert(a) {if (!(a)) {cerr << "Error: assert failed: " << (#a) << endl; exit(-1);}}
#else
#define zkassert(a)
#endif

#endif