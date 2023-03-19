#ifndef __CIRCOM_RECURSIVE_1_H
#define __CIRCOM_RECURSIVE_1_H

#include <map>
#include <gmp.h>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "fr_goldilocks.hpp"

namespace CircomRecursive1
{
  typedef unsigned long long u64;
  typedef uint32_t u32;
  typedef uint8_t u8;

  // only for the main inputs
  struct __attribute__((__packed__)) HashSignalInfo
  {
    u64 hash;
    u64 signalid;
    u64 signalsize;
  };

  struct IODef
  {
    u32 offset;
    u32 len;
    u32 *lengths;
  };

  struct IODefPair
  {
    u32 len;
    IODef *defs;
  };

  struct Circom_Circuit
  {
    //  const char *P;
    HashSignalInfo *InputHashMap;
    u64 *witness2SignalList;
    FrGElement *circuitConstants;
    std::map<u32, IODefPair> templateInsId2IOSignalInfo;
  };

  struct Circom_Component
  {
    u32 templateId;
    u64 signalStart;
    u32 inputCounter;
    std::string templateName;
    std::string componentName;
    u64 idFather;
    u32 *subcomponents = NULL;
    bool *subcomponentsParallel = NULL;
    bool *outputIsSet = NULL;   // one for each output
    std::mutex *mutexes = NULL; // one for each output
    std::condition_variable *cvs = NULL;
    std::thread *sbct = NULL; // subcomponent threads
  };

  /*
  For every template instantiation create two functions:
  - name_create
  - name_run

  //PFrGElement: pointer to FrGElement

  Every name_run or circom_function has:
  =====================================

  //array of PFrGElements for auxiliars in expression computation (known size);
  PFrGElements expaux[];

  //array of PFrGElements for local vars (known size)
  PFrGElements lvar[];

  */

  uint get_main_input_signal_start();
  uint get_main_input_signal_no();
  uint get_total_signal_no();
  uint get_number_of_components();
  uint get_size_of_input_hashmap();
  uint get_size_of_witness();
  uint get_size_of_constants();
  uint get_size_of_io_map();

}
#endif // __CIRCOM_H
