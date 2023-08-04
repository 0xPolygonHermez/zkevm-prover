#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <vector>
#include <chrono>

using json = nlohmann::json;

#include "calcwit.hpp"
#include "circom.hpp"

#include "utils.hpp"
#include "timer.hpp"
#include "execFile.hpp"
#include "commit_pols_starks.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

using namespace std;

namespace Circom
{
#define handle_error(msg) \
  do                      \
  {                       \
    perror(msg);          \
    exit(EXIT_FAILURE);   \
  } while (0)

  Circom_Circuit *loadCircuit(std::string const &datFileName)
  {
    Circom_Circuit *circuit = new Circom_Circuit;

    int fd;
    struct stat sb;

    fd = open(datFileName.c_str(), O_RDONLY);
    if (fd == -1)
    {
      std::cout << ".dat file not found: " << datFileName << "\n";
      throw std::system_error(errno, std::generic_category(), "open");
    }

    if (fstat(fd, &sb) == -1)
    { /* To obtain file size */
      throw std::system_error(errno, std::generic_category(), "fstat");
    }

    u8 *bdata = (u8 *)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    circuit->InputHashMap = new HashSignalInfo[get_size_of_input_hashmap()];
    uint dsize = get_size_of_input_hashmap() * sizeof(HashSignalInfo);
    memcpy((void *)(circuit->InputHashMap), (void *)bdata, dsize);

    circuit->witness2SignalList = new u64[get_size_of_witness()];
    uint inisize = dsize;
    dsize = get_size_of_witness() * sizeof(u64);
    memcpy((void *)(circuit->witness2SignalList), (void *)(bdata + inisize), dsize);

    circuit->circuitConstants = new FrGElement[get_size_of_constants()];
    if (get_size_of_constants() > 0)
    {
      inisize += dsize;
      dsize = get_size_of_constants() * sizeof(FrGElement);
      memcpy((void *)(circuit->circuitConstants), (void *)(bdata + inisize), dsize);
    }

    std::map<u32, IODefPair> templateInsId2IOSignalInfo1;
    if (get_size_of_io_map() > 0)
    {
      u32 index[get_size_of_io_map()];
      inisize += dsize;
      dsize = get_size_of_io_map() * sizeof(u32);
      memcpy((void *)index, (void *)(bdata + inisize), dsize);
      inisize += dsize;
      assert(inisize % sizeof(u32) == 0);
      assert(sb.st_size % sizeof(u32) == 0);
      u32 dataiomap[(sb.st_size - inisize) / sizeof(u32)];
      memcpy((void *)dataiomap, (void *)(bdata + inisize), sb.st_size - inisize);
      u32 *pu32 = dataiomap;

      for (uint i = 0; i < get_size_of_io_map(); i++)
      {
        u32 n = *pu32;
        IODefPair p;
        p.len = n;
        IODef defs[n];
        pu32 += 1;
        for (u32 j = 0; j < n; j++)
        {
          defs[j].offset = *pu32;
          u32 len = *(pu32 + 1);
          defs[j].len = len;
          defs[j].lengths = new u32[len];
          memcpy((void *)defs[j].lengths, (void *)(pu32 + 2), len * sizeof(u32));
          pu32 += len + 2;
        }
        p.defs = (IODef *)calloc(10, sizeof(IODef));
        for (u32 j = 0; j < p.len; j++)
        {
          p.defs[j] = defs[j];
        }
        templateInsId2IOSignalInfo1[index[i]] = p;
      }
    }
    circuit->templateInsId2IOSignalInfo = move(templateInsId2IOSignalInfo1);

    munmap(bdata, sb.st_size);

    return circuit;
  }

  void freeCircuit(Circom_Circuit *circuit)
  {
    delete[] circuit->InputHashMap;
    delete[] circuit->witness2SignalList;
    delete[] circuit->circuitConstants;
    delete circuit;
  }

  bool check_valid_number(std::string &s, uint base)
  {
    bool is_valid = true;
    if (base == 16)
    {
      for (uint i = 0; i < s.size(); i++)
      {
        is_valid &= (('0' <= s[i] && s[i] <= '9') ||
                     ('a' <= s[i] && s[i] <= 'f') ||
                     ('A' <= s[i] && s[i] <= 'F'));
      }
    }
    else
    {
      for (uint i = 0; i < s.size(); i++)
      {
        is_valid &= ('0' <= s[i] && s[i] < char(int('0') + base));
      }
    }
    return is_valid;
  }

  void json2FrGElements(json val, std::vector<FrGElement> &vval)
  {
    if (!val.is_array())
    {
      FrGElement v;
      std::string s_aux, s;
      uint base;
      if (val.is_string())
      {
        s_aux = val.get<std::string>();
        std::string possible_prefix = s_aux.substr(0, 2);
        if (possible_prefix == "0b" || possible_prefix == "0B")
        {
          s = s_aux.substr(2, s_aux.size() - 2);
          base = 2;
        }
        else if (possible_prefix == "0o" || possible_prefix == "0O")
        {
          s = s_aux.substr(2, s_aux.size() - 2);
          base = 8;
        }
        else if (possible_prefix == "0x" || possible_prefix == "0X")
        {
          s = s_aux.substr(2, s_aux.size() - 2);
          base = 16;
        }
        else
        {
          s = s_aux;
          base = 10;
        }
        if (!check_valid_number(s, base))
        {
          std::ostringstream errStrStream;
          errStrStream << "Invalid number in JSON input: " << s_aux << "\n";
          throw std::runtime_error(errStrStream.str());
        }
      }
      else if (val.is_number())
      {
        double vd = val.get<double>();
        std::stringstream stream;
        stream << std::fixed << std::setprecision(0) << vd;
        s = stream.str();
        base = 10;
      }
      else
      {
        std::ostringstream errStrStream;
        errStrStream << "Invalid JSON type\n";
        throw std::runtime_error(errStrStream.str());
      }
      FrG_str2element(&v, s.c_str(), base);
      vval.push_back(v);
    }
    else
    {
      for (uint i = 0; i < val.size(); i++)
      {
        json2FrGElements(val[i], vval);
      }
    }
  }

  void loadJsonImpl(Circom_CalcWit *ctx, json &j)
  {

    u64 nItems = j.size();
    // printf("Items : %llu\n",nItems);
    if (nItems == 0)
    {
      ctx->tryRunCircuit();
    }
    for (json::iterator it = j.begin(); it != j.end(); ++it)
    {
      // std::cout << it.key() << " => " << it.value() << '\n';
      u64 h = fnv1a(it.key());
      std::vector<FrGElement> v;
      json2FrGElements(it.value(), v);
      uint signalSize = ctx->getInputSignalSize(h);
      if (v.size() < signalSize)
      {
        std::ostringstream errStrStream;
        errStrStream << "Error loading signal " << it.key() << ": Not enough values\n";
        throw std::runtime_error(errStrStream.str());
      }
      if (v.size() > signalSize)
      {
        std::ostringstream errStrStream;
        errStrStream << "Error loading signal " << it.key() << ": Too many values\n";
        throw std::runtime_error(errStrStream.str());
      }
      for (uint i = 0; i < v.size(); i++)
      {
        try
        {
          // std::cout << it.key() << "," << i << " => " << FrG_element2str(&(v[i])) << '\n';
          ctx->setInputSignal(h, i, v[i]);
        }
        catch (std::runtime_error &e)
        {
          std::ostringstream errStrStream;
          errStrStream << "Error setting signal: " << it.key() << "\n"
                       << e.what();
          throw std::runtime_error(errStrStream.str());
        }
      }
    }
  }

  void writeBinWitness(Circom_CalcWit *ctx, std::string wtnsFileName)
  {
    FILE *write_ptr;

    write_ptr = fopen(wtnsFileName.c_str(), "wb");

    fwrite("wtns", 4, 1, write_ptr);

    u32 version = 2;
    fwrite(&version, 4, 1, write_ptr);

    u32 nSections = 2;
    fwrite(&nSections, 4, 1, write_ptr);

    // Header
    u32 idSection1 = 1;
    fwrite(&idSection1, 4, 1, write_ptr);

    u32 n8 = FrG_N64 * 8;

    u64 idSection1length = 8 + n8;
    fwrite(&idSection1length, 8, 1, write_ptr);

    fwrite(&n8, 4, 1, write_ptr);

    fwrite(FrG_q.longVal, FrG_N64 * 8, 1, write_ptr);

    uint Nwtns = get_size_of_witness();

    u32 nVars = (u32)Nwtns;
    fwrite(&nVars, 4, 1, write_ptr);

    // Data
    u32 idSection2 = 2;
    fwrite(&idSection2, 4, 1, write_ptr);

    u64 idSection2length = (u64)n8 * (u64)Nwtns;
    fwrite(&idSection2length, 8, 1, write_ptr);

    FrGElement v;

    for (uint i = 0; i < Nwtns; i++)
    {
      ctx->getWitness(i, &v);
      FrG_toLongNormal(&v, &v);
      fwrite(v.longVal, FrG_N64 * 8, 1, write_ptr);
    }
    fclose(write_ptr);
  }

  void loadJson(Circom_CalcWit *ctx, std::string filename)
  {
    std::ifstream inStream(filename);
    json j;
    inStream >> j;
    inStream.close();
    loadJsonImpl(ctx, j);
  }
  
  void getCommitedPols(CommitPolsStarks *commitPols, const std::string zkevmVerifier, const std::string execFile, nlohmann::json &zkin, uint64_t N, uint64_t nCols)
  {
    //-------------------------------------------
    // Verifier stark proof
    //-------------------------------------------
    TimerStart(CIRCOM_LOAD_CIRCUIT_BATCH_PROOF_ZKEVM);
    Circom_Circuit *circuit = loadCircuit(zkevmVerifier);
    TimerStopAndLog(CIRCOM_LOAD_CIRCUIT_BATCH_PROOF_ZKEVM);
    TimerStart(CIRCOM_LOAD_JSON_BATCH_PROOF);
    Circom_CalcWit *ctx = new Circom_CalcWit(circuit);

    loadJsonImpl(ctx, zkin);
    if (ctx->getRemaingInputsToBeSet() != 0)
    {
      zklog.error("Prover::genBatchProof() Not all inputs have been set. Only " + to_string(get_main_input_signal_no() - ctx->getRemaingInputsToBeSet()) + " out of " + to_string(get_main_input_signal_no()));
      exitProcess();
    }
    TimerStopAndLog(CIRCOM_LOAD_JSON_BATCH_PROOF);
    //-------------------------------------------
    // Compute witness and commited pols
    //-------------------------------------------
    TimerStart(STARK_WITNESS_AND_COMMITED_POLS_BATCH_PROOF);

    ExecFile exec(execFile, nCols);
    uint64_t sizeWitness = get_size_of_witness();
    Goldilocks::Element *tmp = new Goldilocks::Element[exec.nAdds + sizeWitness];
    for (uint64_t i = 0; i < sizeWitness; i++)
    {
      FrGElement aux;
      ctx->getWitness(i, &aux);
      FrG_toLongNormal(&aux, &aux);
      tmp[i] = Goldilocks::fromU64(aux.longVal[0]);
    }
    delete ctx;
    for (uint64_t i = 0; i < exec.nAdds; i++)
    {
      FrG_toLongNormal(&exec.p_adds[i * 4], &exec.p_adds[i * 4]);
      FrG_toLongNormal(&exec.p_adds[i * 4 + 1], &exec.p_adds[i * 4 + 1]);
      FrG_toLongNormal(&exec.p_adds[i * 4 + 2], &exec.p_adds[i * 4 + 2]);
      FrG_toLongNormal(&exec.p_adds[i * 4 + 3], &exec.p_adds[i * 4 + 3]);

      uint64_t idx_1 = exec.p_adds[i * 4].longVal[0];
      uint64_t idx_2 = exec.p_adds[i * 4 + 1].longVal[0];

      Goldilocks::Element c = tmp[idx_1] * Goldilocks::fromU64(exec.p_adds[i * 4 + 2].longVal[0]);
      Goldilocks::Element d = tmp[idx_2] * Goldilocks::fromU64(exec.p_adds[i * 4 + 3].longVal[0]);
      tmp[sizeWitness + i] = c + d;
    }

    // #pragma omp parallel for
    for (uint i = 0; i < exec.nSMap; i++)
    {
      for (uint j = 0; j < nCols; j++)
      {
        FrGElement aux;
        FrG_toLongNormal(&aux, &exec.p_sMap[nCols * i + j]);
        uint64_t idx_1 = aux.longVal[0];
        if (idx_1 != 0)
        {
          uint64_t idx_2 = Goldilocks::toU64(tmp[idx_1]);
          commitPols->Compressor.a[j][i] = Goldilocks::fromU64(idx_2);
        }
        else
        {
          commitPols->Compressor.a[j][i] = Goldilocks::zero();
        }
      }
    }
    for (uint i = exec.nSMap; i < N; i++)
    {
      for (uint j = 0; j < nCols; j++)
      {
        commitPols->Compressor.a[j][i] = Goldilocks::zero();
      }
    }
    delete[] tmp;
    freeCircuit(circuit);
    TimerStopAndLog(STARK_WITNESS_AND_COMMITED_POLS_BATCH_PROOF);
  }

}
