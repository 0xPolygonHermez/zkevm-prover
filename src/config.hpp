#ifndef CONFIG_HPP
#define CONFIG_HPP

#define NEVALUATIONS 65536 //1<<16, i.e. 1<<NBITS
#define NPOLS 86 //Up to 512 in the future
#define NCONSTPOLS 7
#define ARITY 4
#define NBITS 16
#define EXTENDED_BITS 1

//#define LOG_STEPS
//#define LOG_INX
//#define LOG_ADDR
//#define LOG_NEG
//#define LOG_ASSERT
//#define LOG_SETX
//#define LOG_JMP
//#define LOG_STORAGE
//#define LOG_MEMORY
//#define LOG_HASH
#define LOG_POLS
//#define LOG_VARIABLES // If defined, logs variable declaration, get and set actions
//#define LOG_FILENAME // If defined, logs ROM compilation file name and line number
#define LOG_TIME // If defined, logs time differences to measure performance
//#define LOG_TXS

#define DEBUG
#ifdef DEBUG
#define zkassert(a) {if (!(a)) {cerr << "Error: assert failed: " << (#a) << endl; exit(-1);}}
#else
#define zkassert(a)
#endif

#define USE_LOCAL_DATABASE
//#define USE_LOCALHOST_DATABASE

#endif