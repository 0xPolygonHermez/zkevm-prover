#ifndef CONFIG_HPP
#define CONFIG_HPP

#define NEVALUATIONS 65536 //4096 //1<<23 // 8M
#define NPOLS 86 //512
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

#endif