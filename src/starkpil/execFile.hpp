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

#include "goldilocks_base_field.hpp"
#include "fr_goldilocks.hpp"

class ExecFile
{
public:
    uint64_t nAdds;
    uint64_t nSMap;

    FrGElement *p_adds;
    FrGElement *p_sMap;

    ExecFile(std::string execFile)
    {
        int fd;
        struct stat sb;
        uint64_t *p_data;

        fd = open(execFile.c_str(), O_RDONLY);
        if (fd == -1)
        {
            std::cout << ".exec file not found: " << execFile << "\n";
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

        p_adds = new FrGElement[nAdds * 4];
        p_sMap = new FrGElement[nSMap * 12];
        for (uint64_t i = 0; i < nAdds * 4; i++)
        {
            p_adds[i].shortVal = 0;
            p_adds[i].type = FrG_LONG;
            p_adds[i].longVal[0] = p_data[i + 2];
        }
        for (uint64_t j = 0; j < nSMap * 12; j++)
        {
            p_sMap[j].shortVal = 0;
            p_sMap[j].type = FrG_LONG;
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
