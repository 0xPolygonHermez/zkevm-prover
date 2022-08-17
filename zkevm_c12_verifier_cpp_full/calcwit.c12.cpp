#include <iomanip>
#include <sstream>
#include <assert.h>
#include "calcwit.c12.hpp"

extern void run(Circom_CalcWitC12* ctx);

std::string int_to_hex_C12( u64 i )
{
  std::stringstream stream;
  stream << "0x"
         << std::setfill ('0') << std::setw(16)
         << std::hex << i;
  return stream.str();
}

u64 fnv1aC12(std::string s) {
  u64 hash = 0xCBF29CE484222325LL;
  for(char& c : s) {
    hash ^= u64(c);
    hash *= 0x100000001B3LL;
  }
  return hash;
}

Circom_CalcWitC12::Circom_CalcWitC12 (Circom_CircuitC12 *aCircuit, uint maxTh) {
  circuit = aCircuit;
  inputSignalAssignedCounter = get_main_input_signal_no_C12();
  inputSignalAssigned = new bool[inputSignalAssignedCounter];
  for (int i = 0; i< inputSignalAssignedCounter; i++) {
    inputSignalAssigned[i] = false;
  }
  signalValues = new FrElement[get_total_signal_no_C12()];
  Fr_str2element(&signalValues[0], "1");
  componentMemory = new Circom_ComponentC12[get_number_of_components_C12()];
  circuitConstants = circuit ->circuitConstants;
  templateInsId2IOSignalInfo = circuit -> templateInsId2IOSignalInfo;

  maxThread = maxTh;

  // parallelism
  numThread = 0;

}

Circom_CalcWitC12::~Circom_CalcWitC12() {
  // ...
}

uint Circom_CalcWitC12::getInputSignalHashPosition(u64 h) {
  uint n = get_size_of_input_hashmap_C12();
  uint pos = (uint)(h % (u64)n);
  if (circuit->InputHashMap[pos].hash!=h){
    uint inipos = pos;
    pos++;
    while (pos != inipos) {
      if (circuit->InputHashMap[pos].hash==h) return pos;
      if (circuit->InputHashMap[pos].hash==0) {
	fprintf(stderr, "Signal not found\n");
	assert(false);
      }
      pos = (pos+1)%n; 
    }
    fprintf(stderr, "Signals not found\n");
    assert(false);
  }
  return pos;
}

void Circom_CalcWitC12::setInputSignal(u64 h, uint i,  FrElement & val){
  if (inputSignalAssignedCounter == 0) {
    fprintf(stderr, "No more signals to be assigned\n");
    assert(false);
  }
  uint pos = getInputSignalHashPosition(h);
  if (i >= circuit->InputHashMap[pos].signalsize) {
    fprintf(stderr, "Input signal array access exceeds the size\n");
    assert(false);
  }
  
  uint si = circuit->InputHashMap[pos].signalid+i;
  if (inputSignalAssigned[si-get_main_input_signal_start_C12()]) {
    fprintf(stderr, "Signal assigned twice: %d\n", si);
    assert(false);
  }
  signalValues[si] = val;
  inputSignalAssigned[si-get_main_input_signal_start_C12()] = true;
  inputSignalAssignedCounter--;
  if (inputSignalAssignedCounter == 0) {
    run(this);
  }
}

u64 Circom_CalcWitC12::getInputSignalSize(u64 h) {
  uint pos = getInputSignalHashPosition(h);
  return circuit->InputHashMap[pos].signalsize;
}

std::string Circom_CalcWitC12::getTrace(u64 id_cmp){
  if (id_cmp == 0) return componentMemory[id_cmp].componentName;
  else{
    u64 id_father = componentMemory[id_cmp].idFather;
    std::string my_name = componentMemory[id_cmp].componentName;

    return Circom_CalcWitC12::getTrace(id_father) + "." + my_name;
  }


}

std::string Circom_CalcWitC12::generate_position_array(uint* dimensions, uint size_dimensions, uint index){
  std::string positions = "";

  for (uint i = 0 ; i < size_dimensions; i++){
    uint last_pos = index % dimensions[size_dimensions -1 - i];
    index = index / dimensions[size_dimensions -1 - i];
    std::string new_pos = "[" + std::to_string(last_pos) + "]";
    positions =  new_pos + positions;
  }
  return positions;
}

