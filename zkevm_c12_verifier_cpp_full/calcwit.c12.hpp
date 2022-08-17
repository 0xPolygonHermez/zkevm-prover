#ifndef CIRCOM_CALCWIT_C12_H
#define CIRCOM_CALCWIT_C12_H

#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <memory>

#include "circom.c12.hpp"
#include "fr.hpp"

#define NMUTEXES 12 //512

u64 fnv1aC12(std::string s);

class Circom_CalcWitC12 {

  bool *inputSignalAssigned;
  uint inputSignalAssignedCounter;

  Circom_CircuitC12 *circuit;

public:

  FrElement *signalValues;
  Circom_ComponentC12* componentMemory;
  FrElement* circuitConstants; 
  std::map<u32,IODefC12Pair> templateInsId2IOSignalInfo; 
  std::string* listOfTemplateMessages; 

  // parallelism
  std::mutex numThreadMutex;
  std::condition_variable ntcvs;
  uint numThread;

  uint maxThread;

  // Functions called by the circuit
  Circom_CalcWitC12(Circom_CircuitC12 *aCircuit, uint numTh = NMUTEXES);
  ~Circom_CalcWitC12();

  // Public functions
  void setInputSignal(u64 h, uint i, FrElement &val);
  
  u64 getInputSignalSize(u64 h);

  inline uint getRemaingInputsToBeSet() {
    return inputSignalAssignedCounter;
  }
  
  inline void getWitness(uint idx, PFrElement val) {
    Fr_copy(val, &signalValues[circuit->witness2SignalList[idx]]);
  }

  std::string getTrace(u64 id_cmp);

  std::string generate_position_array(uint* dimensions, uint size_dimensions, uint index);

private:
  
  uint getInputSignalHashPosition(u64 h);

};

typedef void (*Circom_TemplateFunctionC12)(uint __cIdx, Circom_CalcWitC12* __ctx); 

#endif // CIRCOM_CALCWIT_H
