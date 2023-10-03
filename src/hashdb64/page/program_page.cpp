#include <string.h>
#include "program_page.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include "page_manager.hpp"
#include "scalar.hpp"
#include "raw_data_page.hpp"
#include "header_page.hpp"

zkresult ProgramPage::InitEmptyPage (const uint64_t pageNumber)
{
    ProgramStruct * page = (ProgramStruct *)pageManager.getPage(pageNumber);
    memset((void *)page, 0, 4096);
    return ZKR_SUCCESS;
}

zkresult ProgramPage::Read (const uint64_t pageNumber, const string &key, string &program, const uint64_t level)
{
    zkassert(key.size() == 32);
    zkassert(level < 32);

    zkresult zkr;
    ProgramStruct * page = (ProgramStruct *)pageManager.getPage(pageNumber);
    uint8_t index = key[level];

    uint64_t control = page->key[index][0] >> 48;

    switch (control)
    {
        // Empty slot
        case 0:
        {
            zklog.error("ProgramPage::Read() found empty slot pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
            return ZKR_DB_KEY_NOT_FOUND;
        }

        // Leaf node
        case 1:
        {
            uint64_t length = page->key[index][0] & 0xFFFFFF;
            uint64_t rawDataPage = page->key[index][1] & 0xFFFFFF;
            uint64_t rawDataOffset = page->key[index][1] >> 48;
            string rawdata;
            zkr = RawDataPage::Read(rawDataPage, rawDataOffset, length, rawdata);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("ProgramPage::Read() failed calling RawDataPage::Read() result=" + zkresult2string(zkr) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return zkr;
            }

            if (rawdata.size() < 32)
            {
                zklog.error("ProgramPage::Read() called RawDataPage::Read() and got invalid raw data size=" + to_string(rawdata.size()) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return ZKR_DB_ERROR;
            }

            if (memcmp(rawdata.c_str(), key.c_str(), 32) != 0)
            {
                zklog.error("ProgramPage::Read() called RawDataPage::Read() and got different keys pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return ZKR_DB_KEY_NOT_FOUND;
            }
            
            zklog.info("ProgramPage::Read() read length=" + to_string(length) +
                " result=" + zkresult2string(zkr) +
                " pageNumber=" + to_string(pageNumber) +
                " index=" + to_string(index) +
                " level=" + to_string(level) +
                " key=" + ba2string(key) +
                " rawDataPage=" + to_string(rawDataPage) +
                " rawDataOffset=" + to_string(rawDataOffset));
            
            program = rawdata.substr(32);

            return ZKR_SUCCESS;
        }

        // Intermediate node
        case 2:
        {
            uint64_t nextProgramPage = page->key[index][0] & 0xFFFFFF;
            return Read(nextProgramPage, key, program, level+1);
        }

        // Default
        default:
        {
            zklog.error("ProgramPage::Read() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
            exitProcess();
        }
    }

    return ZKR_DB_ERROR;
}

zkresult ProgramPage::Read (const uint64_t pageNumber, const string &key, string &program)
{
    zkassert(key.size() == 32);
    zkassert(program.size() == 0);
    return Read(pageNumber, key, program, 0);
}

zkresult ProgramPage::Write (const uint64_t pageNumber, const string &key, const string &program, const uint64_t level, const uint64_t headerPageNumber)
{
    zkassert(key.size() == 32);
    zkassert(level < 32);
    zkassert(program.size() + 32 <= 0xFFFFFF);

    zkresult zkr;
    ProgramStruct * page = (ProgramStruct *)pageManager.getPage(pageNumber);
    uint8_t index = key[level];

    uint64_t control = page->key[index][0] >> 48;

    switch (control)
    {
        // Empty slot: insert the new key here
        case 0:
        {
            //zklog.error("ProgramPage::Write() found empty slot pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
            string keyAndProgram = key + program;
            uint64_t length = keyAndProgram.size();

            // Get header page data
            HeaderStruct * headerPage = (HeaderStruct *)pageManager.getPage(headerPageNumber);
            uint64_t rawDataPage = headerPage->rawDataPage;
            uint64_t rawDataOffset = headerPage->rawDataOffset;

            // Write to raw data page list
            zkr = RawDataPage::Write(rawDataPage, rawDataOffset, keyAndProgram);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("ProgramPage::Write() failed calling RawDataPage::Write() result=" + zkresult2string(zkr) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return zkr;
            }
            else
            {
                zklog.info("ProgramPage::Write() wrote length=" + to_string(length) +
                    " result=" + zkresult2string(zkr) +
                    " pageNumber=" + to_string(pageNumber) +
                    " index=" + to_string(index) +
                    " level=" + to_string(level) +
                    " key=" + ba2string(key) +
                    " rawDataPage=" + to_string(headerPage->rawDataPage) +
                    " rawDataOffset=" + to_string(headerPage->rawDataOffset) +
                    " leaving headerPage->rawDataPage=" + to_string(rawDataPage) +
                    " headerPage->rawDataOffset=" + to_string(rawDataOffset));
            }
            

            // Update this entry as a leaf node (control = 1)
            page->key[index][0] = length | (uint64_t(1)<<48);
            page->key[index][1] = headerPage->rawDataPage | (headerPage->rawDataOffset << 48);

            // Update header page data
            headerPage->rawDataPage = rawDataPage;
            headerPage->rawDataOffset = rawDataOffset;


            return ZKR_SUCCESS;
        }

        // Leaf node
        case 1:
        {
            // Read the key from the raw data page
            uint64_t length = page->key[index][0] & 0xFFFFFF;
            uint64_t rawPageNumber = page->key[index][1] & 0xFFFFFF;
            uint64_t rawPageOffset = page->key[index][1] >> 48;

            if (length < 32)
            {
                zklog.error("ProgramPage::Write() found invalid length=" + to_string(length) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                exitProcess();
            }

            // Get the existing key
            string existingKey;
            zkr = RawDataPage::Read(rawPageNumber, rawPageOffset, 32, existingKey);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("ProgramPage::Write() failed calling RawDataPage::Read() result=" + zkresult2string(zkr) + " length=" + to_string(length) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                return zkr;
            }

            // If both keys are identical, values should be identical, too
            if (existingKey == key)
            {
                // Compare total length
                if (length != program.size() + 32)
                {
                    zklog.error("ProgramPage::Write() found not matching length=" + to_string(length) + " program.size()+32=" + to_string(program.size() + 32) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                    exitProcess();
                }

                // Get the existing key + program
                string existingKeyAndProgram;
                zkr = RawDataPage::Read(rawPageNumber, rawPageOffset, length, existingKeyAndProgram);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("ProgramPage::Write() failed calling RawDataPage::Read() result=" + zkresult2string(zkr) + " length=" + to_string(length) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                    return zkr;
                }

                // Compare programs
                if (existingKeyAndProgram.size() != program.size() + 32)
                {
                    zklog.error("ProgramPage::Write() found not matching existingKeyAndProgram.size()=" + to_string(existingKeyAndProgram.size()) + " program.size()+32=" + to_string(program.size() + 32) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                    exitProcess();
                }
                if (program.compare(existingKeyAndProgram.substr(32)) != 0)
                {
                    zklog.error("ProgramPage::Write() found not matching program of size=" + to_string(program.size()) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
                    exitProcess();
                }

                return ZKR_SUCCESS;
            }

            // If keys are different, create a new page, move the existing key one level down, and call Write(level+1)
            uint64_t newPageNumber = pageManager.getFreePage();
            ProgramPage::InitEmptyPage(newPageNumber);
            ProgramStruct * newPage = (ProgramStruct *)pageManager.getPage(newPageNumber);
            uint8_t newIndex = existingKey[level+1];
            newPage->key[newIndex][0] = page->key[index][0];
            newPage->key[newIndex][1] = page->key[index][1];

            // Set this page entry as a intermeiate node
            page->key[index][0] = newPageNumber | (uint64_t(2) << 48);
            page->key[index][1] = 0;

            // Write the new key in the newly created page at the next level
            return ProgramPage::Write(newPageNumber, key, program, level+1, headerPageNumber);
        }

        // Intermediate node
        case 2:
        {
            uint64_t nextProgramPage = page->key[index][0] & 0xFFFFFF;
            return Write(nextProgramPage, key, program, level+1, headerPageNumber);
        }

        // Default
        default:
        {
            zklog.error("ProgramPage::Write() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber) + " index=" + to_string(index) + " level=" + to_string(level) + " key=" + ba2string(key));
            exitProcess();
        }
    }

    return ZKR_DB_ERROR;
}

zkresult ProgramPage::Write (const uint64_t pageNumber, const string &key, const string &program, const uint64_t headerPageNumber)
{
    zkassert(key.size() == 32);
    zkassert(program.size() != 0);
    return Write(pageNumber, key, program, 0, headerPageNumber);
}

void ProgramPage::Print (const uint64_t pageNumber, bool details)
{
    zklog.info("ProgramPage::Print() pageNumber=" + to_string(pageNumber));
    if (details)
    {
        vector<uint64_t> nextProgramPages;

        // For each entry of the page
        ProgramStruct * page = (ProgramStruct *)pageManager.getPage(pageNumber);
        for (uint64_t i=0; i<256; i++)
        {
            uint64_t control = page->key[i][0] >> 48;

            switch (control)
            {
                // Empty slot
                case 0:
                {
                    continue;
                }

                // Leaf node
                case 1:
                {
                    uint64_t length = page->key[i][0] & 0xFFFFFF;
                    uint64_t rawPageNumber = page->key[i][1] & 0xFFFFFF;
                    uint64_t rawPageOffset = page->key[i][1] >> 48;
                    zklog.info("  i=" + to_string(i) + " length=" + to_string(length) + " rawPageNumber=" + to_string(rawPageNumber) + " rawPageOffset=" + to_string(rawPageOffset));
                    continue;
                }

                // Intermediate node
                case 2:
                {
                    uint64_t nextProgramPage = page->key[i][0] & 0xFFFFFF;
                    nextProgramPages.emplace_back(nextProgramPage);
                    zklog.info("  i=" + to_string(i) + " nextProgramPage=" + to_string(nextProgramPage));
                    continue;
                }

                // Default
                default:
                {
                    zklog.error("ProgramPage::Print() found invalid control=" + to_string(control) + " pageNumber=" + to_string(pageNumber) + " i=" + to_string(i));
                    exitProcess();
                }

            }      
        }

        for (uint64_t i=0; i<nextProgramPages.size(); i++)
        {
            Print(nextProgramPages[i], details);
        }
    }
}