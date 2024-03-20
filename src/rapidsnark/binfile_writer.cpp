#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <system_error>
#include <string>
#include <memory.h>
#include <stdexcept>
#include <iostream>
#include "binfile_writer.hpp"
#include "thread_utils.hpp"
#include <omp.h>

namespace BinFileUtils
{
    BinFileWriter::BinFileWriter(std::string filename, std::string type, uint32_t version, uint32_t nSections)
    {
        this->filename = filename;
        this->type = type;
        this->version = version;
        this->nSections = nSections;

        file.open(filename, std::ios::out | std::ios::binary);

        std::vector<char> bytes(4);
        for (int i = 0; i < 4; i++) bytes[i] = type.at(i);
        
        write(bytes.data(), 4);

        writeU32LE(version);

        writeU32LE(nSections);

        writingSection = NULL;
    }

    BinFileWriter::~BinFileWriter()
    {
        this->close();

        for (auto const &x : sections) delete[] x.second;
        sections.clear();
    }

    void BinFileWriter::close()
    {
        if(nSections != sections.size())
        {
            std::cerr << "ERROR: Warning!!! Not all sections written" << std::endl;
        }

        if (file.is_open()) file.close();
    }

    off_t BinFileWriter::getCurrentPosition()
    {
        return file.tellp();
    }

    void BinFileWriter::startWriteSection(u_int32_t sectionId)
    {
        if (writingSection != NULL)
        {
            throw new std::range_error("Already writing a section");
        }

        auto s = new Section(getCurrentPosition(), 0);
        sections[sectionId] = s;

        writingSection = sections[sectionId];

        writeU32LE(sectionId);
        writeU64LE(0); // Dummy size, to be updated in endWriteSection()
    }

    void BinFileWriter::endWriteSection()
    {
        if (writingSection == NULL)
        {
            throw new std::range_error("Not writing a section");
        }

        writingSection->size = getCurrentPosition() - writingSection->start - 12;

        off_t current_pos = getCurrentPosition();
        file.seekp(writingSection->start + 4);
        writeU64LE(writingSection->size);
        file.seekp(current_pos);

        writingSection = NULL;
    }

    void BinFileWriter::writeU32LE(u_int32_t value)
    {
        file.write(reinterpret_cast<char *>(&value), sizeof(value));
    }

    void BinFileWriter::writeU64LE(u_int64_t value)
    {
        file.write(reinterpret_cast<char *>(&value), sizeof(value));
    }

    void BinFileWriter::write(void *buffer, u_int64_t len)
    {
        file.write(reinterpret_cast<char *>(buffer), len);
    }

    void BinFileWriter::writeString(const std::string &str)
    {
        file.write(str.c_str(), str.length());
        file.write("\0", 1);
    }
}