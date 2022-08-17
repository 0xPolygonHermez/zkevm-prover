#ifndef __CIRCOM_C12_H
#define __CIRCOM_C12_H

#include <map>
#include <gmp.h>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "fr.hpp"

typedef unsigned long long u64;
typedef uint32_t u32;
typedef uint8_t u8;

//only for the main inputs
struct __attribute__((__packed__)) HashSignalInfoC12 {
    u64 hash;
    u64 signalid; 
    u64 signalsize; 
};

struct IODefC12 { 
    u32 offset;
    u32 len;
    u32 *lengths;
};

struct IODefC12Pair { 
    u32 len;
    IODefC12* defs;
};

struct Circom_CircuitC12 {
  //  const char *P;
  HashSignalInfoC12* InputHashMap;
  u64* witness2SignalList;
  FrElement* circuitConstants;  
  std::map<u32,IODefC12Pair> templateInsId2IOSignalInfo;
};


struct Circom_ComponentC12 {
  u32 templateId;
  u64 signalStart;
  u32 inputCounter;
  std::string templateName;
  std::string componentName;
  u64 idFather; 
  u32* subcomponents;
  bool *outputIsSet;  //one for each output
  std::mutex *mutexes;  //one for each output
  std::condition_variable *cvs;
  std::thread *sbct; //subcomponent threads
};

/*
For every template instantiation create two functions:
- name_create
- name_run

//PFrElement: pointer to FrElement

Every name_run or circom_function has:
=====================================

//array of PFrElements for auxiliars in expression computation (known size);
PFrElements expaux[];

//array of PFrElements for local vars (known size)
PFrElements lvar[];

*/

uint get_main_input_signal_start_C12();
uint get_main_input_signal_no_C12();
uint get_total_signal_no_C12();
uint get_number_of_components_C12();
uint get_size_of_input_hashmap_C12();
uint get_size_of_witness_C12();
uint get_size_of_constants_C12();
uint get_size_of_io_map_C12();

#endif  // __CIRCOM_H
