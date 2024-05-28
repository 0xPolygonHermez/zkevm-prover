#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <system_error>
#include <string>
#include <memory.h>
#include <stdexcept>

#include "binfile_utils.hpp"
#include "thread_utils.hpp"
#include <omp.h>
#include <iostream>

namespace BinFileUtils
{
    BinFile::BinFile(void *data, uint64_t size, std::string _type, uint32_t maxVersion)
    {
        addr = malloc(size);
        int nThreads = omp_get_max_threads() / 2;
        ThreadUtils::parcpy(addr, data, size, nThreads);

        type.assign((const char *)addr, 4);
        pos = 4;

        if (type != _type)
        {
            throw new std::invalid_argument("Invalid file type. It should be " + _type + " and it us " + type);
        }

        version = readU32LE();
        if (version > maxVersion)
        {
            throw new std::invalid_argument("Invalid version. It should be <=" + std::to_string(maxVersion) + " and it us " + std::to_string(version));
        }

        u_int32_t nSections = readU32LE();

        for (u_int32_t i = 0; i < nSections; i++)
        {
            u_int32_t sType = readU32LE();
            u_int64_t sSize = readU64LE();

            if (sections.find(sType) == sections.end())
            {
                sections.insert(std::make_pair(sType, std::vector<Section>()));
            }

            sections[sType].push_back(Section((void *)((u_int64_t)addr + pos), sSize));

            pos += sSize;
        }

        pos = 0;
        readingSection = NULL;
    }

    BinFile::BinFile(std::string fileName, std::string _type, uint32_t maxVersion)
    {

        int fd;
        struct stat sb;

        fd = open(fileName.c_str(), O_RDONLY);
        if (fd == -1)
            throw std::system_error(errno, std::generic_category(), "open");

        if (fstat(fd, &sb) == -1) /* To obtain file size */
            throw std::system_error(errno, std::generic_category(), "fstat");

        size = sb.st_size;
        /*close(fd);

        void * addr3 = malloc(size);
        std::cout << "debub 0" << std::endl;
        std::cout << "size " <<size<< std::endl;
        size_t nThreads = 2;
        if(nThreads == 0) nThreads = 1;
        size_t chunkSize = size / nThreads;
        size_t chunkRemainder = size - nThreads*chunkSize;
        std::cout << "chunkSize " <<chunkSize<< std::endl;
        std::cout << "chunkRemainder " <<chunkRemainder<< std::endl;
        //#pragma omp parallel for num_threads(nThreads)
        for(size_t i=0; i<nThreads; i++){
            // Open the file
            FILE * fd_ = fopen(fileName.c_str(), "rb");
            if(fd_ == NULL){
                throw std::system_error(errno, std::generic_category(), "open");
            }
            size_t offset = i * chunkSize;
            size_t chunkSize_ = chunkSize;
            if(i == nThreads - 1){
                chunkSize_ += chunkRemainder;
            }
            std::cout << "offset " <<offset<<" "<<omp_get_thread_num()<<std::endl;
            std::cout << "chunkSize_ " <<chunkSize_<<" "<<omp_get_thread_num()<<std::endl;
            //std::cout << "pointer " << &(((u_int8_t*)addr)[offset]) <<" "<<omp_get_thread_num()<<std::endl;
            size_t readed = fread(addr3 + offset, chunkSize_, 1,fd_);
            if(readed != chunkSize_){
                throw std::system_error(errno, std::generic_category(), "readed");
            }
            fclose(fd_);
        }
        */

        void *addrmm = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        addr = malloc(sb.st_size);

        int nThreads = omp_get_max_threads() / 2;
        ThreadUtils::parcpy(addr, addrmm, sb.st_size, nThreads);
        //    memcpy(addr, addrmm, sb.st_size);

        munmap(addrmm, sb.st_size);
        close(fd);

        /*for(size_t k=0; k<size; ++k){
            if(((u_int8_t*)addr3)[k] != ((u_int8_t*)addr2)[k]){
                std::cout << "diff " <<k<<std::endl;
                exit(0);
            }
        }
        std::cout << "debub 1" << std::endl;*/
        type.assign((const char *)addr, 4);
        pos = 4;
        //std::cout << "debub 2" << std::endl;
        if (type != _type)
        {
            throw new std::invalid_argument("Invalid file type. It should be " + _type + " and it us " + type);
        }
        //std::cout << "debub 3" << std::endl;
        version = readU32LE();
        //std::cout << "version " <<version<< std::endl;
        if (version > maxVersion)
        {
            throw new std::invalid_argument("Invalid version. It should be <=" + std::to_string(maxVersion) + " and it us " + std::to_string(version));
        }
        //std::cout << "debub 4" << std::endl;
        u_int32_t nSections = readU32LE();
        //std::cout << "debub 5" << nSections<<std::endl;

        for (u_int32_t i = 0; i < nSections; i++)
        {
            u_int32_t sType = readU32LE();
            //std::cout << "sType " <<sType<< std::endl;
            u_int64_t sSize = readU64LE();
            //std::cout << "sSize " <<sSize<< std::endl;

            if (sections.find(sType) == sections.end())
            {
                sections.insert(std::make_pair(sType, std::vector<Section>()));
            }

            sections[sType].push_back(Section((void *)((u_int64_t)addr + pos), sSize));

            pos += sSize;
        }
        //std::cout << "debub 6" << std::endl;
        pos = 0;
        readingSection = NULL;
    }

    BinFile::~BinFile()
    {
        free(addr);
    }

    void BinFile::startReadSection(u_int32_t sectionId, u_int32_t sectionPos)
    {

        if (sections.find(sectionId) == sections.end())
        {
            throw new std::range_error("Section does not exist: " + std::to_string(sectionId));
        }

        if (sectionPos >= sections[sectionId].size())
        {
            throw new std::range_error("Section pos too big. There are " + std::to_string(sections[sectionId].size()) + " and it's trying to access section: " + std::to_string(sectionPos));
        }

        if (readingSection != NULL)
        {
            throw new std::range_error("Already reading a section");
        }

        pos = (u_int64_t)(sections[sectionId][sectionPos].start) - (u_int64_t)addr;

        readingSection = &sections[sectionId][sectionPos];
    }

    void BinFile::endReadSection(bool check)
    {
        if (check)
        {
            if ((u_int64_t)addr + pos - (u_int64_t)(readingSection->start) != readingSection->size)
            {
                throw new std::range_error("Invalid section size");
            }
        }
        readingSection = NULL;
    }

    void *BinFile::getSectionData(u_int32_t sectionId, u_int32_t sectionPos)
    {

        if (sections.find(sectionId) == sections.end())
        {
            throw new std::range_error("Section does not exist: " + std::to_string(sectionId));
        }

        if (sectionPos >= sections[sectionId].size())
        {
            throw new std::range_error("Section pos too big. There are " + std::to_string(sections[sectionId].size()) + " and it's trying to access section: " + std::to_string(sectionPos));
        }

        return sections[sectionId][sectionPos].start;
    }

    u_int64_t BinFile::getSectionSize(u_int32_t sectionId, u_int32_t sectionPos)
    {

        if (sections.find(sectionId) == sections.end())
        {
            throw new std::range_error("Section does not exist: " + std::to_string(sectionId));
        }

        if (sectionPos >= sections[sectionId].size())
        {
            throw new std::range_error("Section pos too big. There are " + std::to_string(sections[sectionId].size()) + " and it's trying to access section: " + std::to_string(sectionPos));
        }

        return sections[sectionId][sectionPos].size;
    }

    u_int8_t BinFile::readU8LE()
    {
        u_int8_t res = *((u_int8_t *)((u_int64_t)addr + pos));
        pos += 1;
        return res;
    }


    u_int16_t BinFile::readU16LE()
    {
        u_int16_t res = *((u_int16_t *)((u_int64_t)addr + pos));
        pos += 2;
        return res;
    }


    u_int32_t BinFile::readU32LE()
    {
        u_int32_t res = *((u_int32_t *)((u_int64_t)addr + pos));
        pos += 4;
        return res;
    }

    u_int64_t BinFile::readU64LE()
    {
        u_int64_t res = *((u_int64_t *)((u_int64_t)addr + pos));
        pos += 8;
        return res;
    }

    bool BinFile::sectionExists(u_int32_t sectionId) {
        return sections.find(sectionId) != sections.end();
    }

    void *BinFile::read(u_int64_t len)
    {
        void *res = (void *)((u_int64_t)addr + pos);
        pos += len;
        return res;
    }

    std::unique_ptr<BinFile> openExisting(std::string filename, std::string type, uint32_t maxVersion)
    {
        return std::unique_ptr<BinFile>(new BinFile(filename, type, maxVersion));
    }

} // Namespace
