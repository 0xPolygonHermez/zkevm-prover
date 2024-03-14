#ifndef CIRCOM_CALCWIT_RECURSIVE_2_H
#define CIRCOM_CALCWIT_RECURSIVE_2_H

#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <memory>

#include "circom.recursive2.hpp"
#include "fr_goldilocks.hpp"

#define NMUTEXES 160

namespace CircomRecursive2
{
  u64 fnv1a(std::string s);

  class Circom_CalcWit
  {

    bool *inputSignalAssigned;
    uint inputSignalAssignedCounter;

    Circom_Circuit *circuit;

  public:
    FrGElement *signalValues;
    Circom_Component *componentMemory;
    FrGElement *circuitConstants;
    std::map<u32, IODefPair> templateInsId2IOSignalInfo;
    std::string *listOfTemplateMessages;

    // parallelism
    std::mutex numThreadMutex;
    std::condition_variable ntcvs;
    int numThread;

    int maxThread;

    // Functions called by the circuit
    Circom_CalcWit(Circom_Circuit *aCircuit, uint numTh = NMUTEXES);
    ~Circom_CalcWit();

    // Public functions
    void setInputSignal(u64 h, uint i, FrGElement &val);
    void tryRunCircuit();

    u64 getInputSignalSize(u64 h);

    inline uint getRemaingInputsToBeSet()
    {
      return inputSignalAssignedCounter;
    }

    inline void getWitness(uint idx, PFrGElement val)
    {
      FrG_copy(val, &signalValues[circuit->witness2SignalList[idx]]);
    }

    std::string getTrace(u64 id_cmp);

    std::string generate_position_array(uint *dimensions, uint size_dimensions, uint index);

  private:
    uint getInputSignalHashPosition(u64 h);
  };

  typedef void (*Circom_TemplateFunction)(uint __cIdx, Circom_CalcWit *__ctx);
}
#endif // CIRCOM_CALCWIT_H
