#ifndef EXEC_PILFFLONK_FILE
#define EXEC_PILFFLONK_FILE

#define EXEC_FILE_SIZE
#include <iostream>
#include <alt_bn128.hpp>
#include <binfile_utils.hpp>
#include "thread_utils.hpp"

#include "zklog.hpp"

class ExecFilePilFflonk
{
    const int EXEC_PF_NSECTIONS = 5;

    const int EXEC_INFO_SECTION = 2;
    const int EXEC_ADDS_BIGINT_SECTION = 3;
    const int EXEC_ADDS_FR_SECTION = 4;
    const int EXEC_SMAP_SECTION = 5;

public:
    uint64_t nAdds;
    uint64_t nSMap;

    std::unique_ptr<BinFileUtils::BinFile> execFile;

    u_int64_t *p_adds;
    AltBn128::FrElement *p_adds_fr;
    u_int64_t *p_sMap;

    ExecFilePilFflonk(AltBn128::Engine &E, std::string execFileName, uint64_t nCommittedPols)
    {
        execFile = BinFileUtils::openExisting(execFileName, "exec", 1);

        auto fdExec = execFile.get();

        fdExec->startReadSection(EXEC_INFO_SECTION);

        nAdds = fdExec->readU64LE();
        nSMap = fdExec->readU64LE();

        fdExec->endReadSection();

        uint64_t lenAdds = nAdds * 2 * sizeof(u_int64_t);
        p_adds = new u_int64_t[nAdds * 2];
        ThreadUtils::parcpy(p_adds, (u_int64_t *)fdExec->getSectionData(EXEC_ADDS_BIGINT_SECTION), lenAdds, omp_get_num_threads() / 2);
        
        uint64_t lenAddsFr = nAdds * 2 * sizeof(AltBn128::FrElement);
        p_adds_fr = new AltBn128::FrElement[nAdds * 2];
        ThreadUtils::parcpy(p_adds_fr, (u_int64_t *)fdExec->getSectionData(EXEC_ADDS_FR_SECTION), lenAddsFr, omp_get_num_threads() / 2);

        uint64_t lenSMap = nSMap * nCommittedPols * sizeof(u_int64_t);
        p_sMap = new u_int64_t[nSMap * nCommittedPols];
        ThreadUtils::parcpy(p_sMap, (u_int64_t *)fdExec->getSectionData(EXEC_SMAP_SECTION), lenSMap, omp_get_num_threads() / 2);
    }
    ~ExecFilePilFflonk()
    {
        delete[] p_adds;
        delete[] p_sMap;
    }
};
#endif