#ifndef EXEC_FILE
#define EXEC_FILE

#define EXEC_FILE_SIZE
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "fr.hpp"

class ExecFile
{
public:
    uint64_t nAdds;
    uint64_t nSMap;

    FrElement *p_adds;
    FrElement *p_sMap;

    ExecFile(std::string execFile, uint64_t nCommitedPols)
    {
        int fd;
        struct stat sb;
        uint64_t *p_data;

        fd = open(execFile.c_str(), O_RDONLY);
        if (fd == -1)
        {
            // zklog.error("ExecFile::ExecFile() .exec file not found: " + execFile);
            throw std::system_error(errno, std::generic_category(), "open");
        }

        if (fstat(fd, &sb) == -1)
        { /* To obtain file size */
            throw std::system_error(errno, std::generic_category(), "fstat");
        }

        p_data = (uint64_t *)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);

        nAdds = (uint64_t)p_data[0];
        nSMap = (uint64_t)p_data[1];

        p_adds = new FrElement[nAdds * 4];
        p_sMap = new FrElement[nSMap * nCommitedPols];
        for (uint64_t i = 0; i < nAdds * 4; i++)
        {
            p_adds[i].shortVal = 0;
            p_adds[i].type = Fr_LONG;
            p_adds[i].longVal[0] = p_data[i + 2];
        }
        for (uint64_t j = 0; j < nSMap * nCommitedPols; j++)
        {
            p_sMap[j].shortVal = 0;
            p_sMap[j].type = Fr_LONG;
            p_sMap[j].longVal[0] = p_data[2 + nAdds * 4 + j];
        }
        munmap(p_data, sb.st_size);
    }
    ~ExecFile()
    {
        delete[] p_adds;
        delete[] p_sMap;
    }
};
#endif
