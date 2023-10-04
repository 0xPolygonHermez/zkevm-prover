#include <unistd.h>
#include "database.hpp"
#include "hashdb_singleton.hpp"
#include "poseidon_goldilocks.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "page_manager.hpp"
#include <unordered_set>
#include <fcntl.h>

uint64_t PageManagerTest (void)
{
    TimerStart(PAGE_MANAGER_TEST);

    std::cout << "PageManagerTest" << std::endl;
    //
    // Memory version
    //
    PageManager pageManagerMem(100);
    uint32_t page1 = pageManagerMem.getFreePage();
    zkassertpermanent(page1 == 2);
    zkassertpermanent(pageManagerMem.getNumFreePages() == 97);
    uint32_t page2 = pageManagerMem.getFreePage();
    zkassertpermanent(page2 == 3);
    zkassertpermanent(pageManagerMem.getNumFreePages() == 96);
    pageManagerMem.releasePage(page1);
    zkassertpermanent(pageManagerMem.getNumFreePages() == 97);
    pageManagerMem.releasePage(page2);
    zkassertpermanent(pageManagerMem.getNumFreePages() == 98);

    unordered_set<uint32_t> pages; 
    for(int i=0; i<98;++i){
        pages.insert(pageManagerMem.getFreePage());
    }
    zkassertpermanent(pages.size() == 98);
    zkassertpermanent(pageManagerMem.getNumFreePages() == 0);

    for(int i=2; i<100;++i){
        pageManagerMem.releasePage(i);
    }
    zkassertpermanent(pageManagerMem.getNumFreePages() == 98);
    
    uint32_t page3=pageManagerMem.getFreePage();
    uint64_t * page3Data = (uint64_t *)pageManagerMem.getPageAddress(page3);
    for(uint64_t i=0; i<256;++i){
        page3Data[i] = i;
    }
    uint32_t page4 = pageManagerMem.editPage(page3);
    zkassertpermanent(page4 != page3);
    uint64_t * page4Data = (uint64_t *)pageManagerMem.getPageAddress(page4);
    for(uint64_t i=0; i<256;++i){
        zkassertpermanent(page4Data[i] == i);
    }
    zkassertpermanent(pageManagerMem.getNumFreePages() == 96);

    zkassertpermanent(pageManagerMem.editPage(page3) == page4);
    zkassertpermanent(pageManagerMem.editPage(page4) == page4);
    zkassertpermanent(pageManagerMem.editPage(0) == 1);
    zkassertpermanent(pageManagerMem.editPage(1) == 1);


    pageManagerMem.flushPages();
    zkassertpermanent(pageManagerMem.getNumFreePages() == 97);
    for(uint64_t i=0; i<256;++i){
        zkassertpermanent(page3Data[i] == 0);
    }
    for(uint64_t i=0; i<256;++i){
        zkassertpermanent(page4Data[i] == i);
    }
    //
    // File version
    //

    //generate a file with 100 pages
    const string fileName = "page_manager_test.bin";
    const int file_size = 4096 * 100;  

    // Create a binary file and fill it with zeros
    int fd = open(fileName.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd == -1) {
        zklog.error("Failed to open file: " + (string)strerror(errno));
        exitProcess();
    }
    //fill file with zeros
    char *buffer = (char *)calloc(file_size, sizeof(char));
    if (buffer == NULL) {
        zklog.error("Failed to allocate buffer: " + (string)strerror(errno));
        exitProcess();
    }
    ssize_t  wirten_bytes = write(fd, buffer, file_size);
    zkassertpermanent(wirten_bytes == file_size);
    close(fd);
    free(buffer);

    // Same tests than with memory version:
    PageManager pageManagerFile(fileName);
    page1 = pageManagerFile.getFreePage();
    zkassertpermanent(page1 == 2);
    zkassertpermanent(pageManagerFile.getNumFreePages() == 97);
    page2 = pageManagerFile.getFreePage();
    zkassertpermanent(page2 == 3);
    zkassertpermanent(pageManagerFile.getNumFreePages() == 96);
    pageManagerFile.releasePage(page1);
    zkassertpermanent(pageManagerFile.getNumFreePages() == 97);
    pageManagerFile.releasePage(page2);
    zkassertpermanent(pageManagerFile.getNumFreePages() == 98);

    pages.clear(); 
    for(int i=0; i<98;++i){
        pages.insert(pageManagerFile.getFreePage());
    }
    zkassertpermanent(pages.size() == 98);
    zkassertpermanent(pageManagerFile.getNumFreePages() == 0);

    for(int i=2; i<100;++i){
        pageManagerFile.releasePage(i);
    }
    zkassertpermanent(pageManagerFile.getNumFreePages() == 98);
    
    page3=pageManagerFile.getFreePage();
    page3Data = (uint64_t *)pageManagerFile.getPageAddress(page3);
    for(uint64_t i=0; i<256;++i){
        page3Data[i] = i;
    }
    page4 = pageManagerFile.editPage(page3);
    zkassertpermanent(page4 != page3);
    page4Data = (uint64_t *)pageManagerFile.getPageAddress(page4);
    for(uint64_t i=0; i<256;++i){
        zkassertpermanent(page4Data[i] == i);
    }
    zkassertpermanent(pageManagerFile.getNumFreePages() == 96);

    zkassertpermanent(pageManagerFile.editPage(page3) == page4);
    zkassertpermanent(pageManagerFile.editPage(0) == 1);
    zkassertpermanent(pageManagerFile.editPage(1) == 1);


    pageManagerFile.flushPages();
    zkassertpermanent(pageManagerFile.getNumFreePages() == 97);
    for(uint64_t i=0; i<256;++i){
        zkassertpermanent(page3Data[i] == 0);
    }
    for(uint64_t i=0; i<256;++i){
        zkassertpermanent(page4Data[i] == i);
    }

    //Let's check persistence of file
    PageManager pageManagerFile2(fileName);
    page3Data = (uint64_t *)pageManagerFile2.getPageAddress(page3);
    page4Data = (uint64_t *)pageManagerFile2.getPageAddress(page4);
    for(uint64_t i=0; i<256;++i){
        zkassertpermanent(page3Data[i] == 0);
    }
    for(uint64_t i=0; i<256;++i){
        zkassertpermanent(page4Data[i] == i);
    }

    //delete file
    std::remove(fileName.c_str());

    TimerStopAndLog(PAGE_MANAGER_TEST);
    return 0;   
}