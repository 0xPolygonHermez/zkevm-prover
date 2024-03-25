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
#include "page_manager_test.hpp"
#include "omp.h"
#include <random>
#include <unordered_set>
#include "page_manager.hpp"
#include "config.hpp"
#include "header_page.hpp"
#include "zkglobals.hpp"
#include <bitset>


uint64_t PageManagerTest (void)
{
    TimerStart(PAGE_MANAGER_TEST);
    PageManagerAccuracyTest();
    PageManagerDBResizeTest();
    PageManagerDBResetTest();
    //PageManagerPerformanceTest();
    TimerStopAndLog(PAGE_MANAGER_TEST);
    return 0;
}

uint64_t PageManagerPerformanceTest (void)
{

    string fileName = "benchmark_file";
    uint64_t fileSize = 128;
    string folderName = "pmtest";
    uint64_t numPositions = 20000;
    uint64_t numReps = 100;
    uint64_t printFreq = 10;

    //remove folder if exists
    std::string command = "rm -rf " + folderName;
    int r=system(command.c_str());
    if(r!=0){
        zklog.info("Error removing folder");
        return 1;
    }

    // Create the state manager
    double start = omp_get_wtime();
    
    PageManager pageManagerFile;
    Config configPM;
    configPM.hashDBFileName = fileName;
    configPM.hashDBFileSize = fileSize;
    configPM.hashDBFolder = folderName;
    PageContext ctx(pageManagerFile, configPM);
    pageManagerFile.init(ctx);

    double end = omp_get_wtime();
    std::cout << std::endl << "Time to construct the PageManager: " << end - start << " seconds" << std::endl;

    // Evaluate numPositions different random positions in the range [0,numPages)
    uint64_t numPages = pageManagerFile.getNumFreePages() +2;
    uint64_t *position = (uint64_t *)malloc(numPositions * sizeof(uint64_t));
    std::random_device rd;
    std::mt19937 rng(rd());

    std::cout  << "Total number of pages: " << numPages << std::endl;
    std::cout  << "Number of pages modified: " << numPositions << std::endl << std::endl;
    
    double numGBytes = (numPositions * 4096.0) / (1024.0  * 1024.0);
    double avgTimeFistWrite = 0;
    double avgTimeSecondWrite = 0;
    double avgTimeFlush = 0;
    double avgThroughputFirstWrite = 0;
    double avgThroughputSecondWrite = 0;
    double avgThroughputFlush = 0;
    double avgThroughputFirstWriteLast = 0;
    double avgThroughputSecondWriteLast = 0;
    double avgThroughputFlushLast = 0;
    double throughput = 0;
    
    for(uint64_t k=0; k<numReps;++k){

        //Generate randoms
        for (uint64_t i = 0; i < numPositions; i++) {
            
            position[i] = std::uniform_int_distribution<uint64_t>(0, numPages - 1)(rng);
            assert(position[i] < numPages);
        }

        //Change first value of each page
        start = omp_get_wtime();
        for (uint64_t i = 0; i < numPositions; ++i) {
            uint64_t* pageData = (uint64_t *)pageManagerFile.getPageAddress(position[i]);
            pageData[0] = pageData[2] + position[i];
        }
        end = omp_get_wtime();

        throughput = numGBytes / (end-start);
        avgThroughputFirstWrite += throughput;
        avgThroughputFirstWriteLast += throughput;
        avgTimeFistWrite += end - start;

        //Change second value of each page
        start = omp_get_wtime();
        for (uint64_t i = 0; i < numPositions; ++i) {
            uint64_t* pageData = (uint64_t *)pageManagerFile.getPageAddress(position[i]);
            assert(pageData[0] == (pageData[2] + position[i]) );
            pageData[1] = position[i];    
        }
        end = omp_get_wtime();

        throughput = numGBytes / (end-start);
        avgThroughputSecondWrite += throughput;
        avgThroughputSecondWriteLast += throughput;
        avgTimeSecondWrite += end - start;

        //flushPAges
        start = omp_get_wtime();
        pageManagerFile.flushPages(ctx);
        end = omp_get_wtime();
        
        throughput = numGBytes / (end-start);
        avgThroughputFlush += throughput;
        avgThroughputFlushLast += throughput;
        avgTimeFlush += end - start;
        
        //Check that positions are in the file
        PageManager pageManagerFile2;
        PageContext ctx2(pageManagerFile2, configPM);
        pageManagerFile2.init(ctx2);
        for (uint64_t i = 0; i < numPositions; ++i) {
            uint64_t* pageData = (uint64_t *)pageManagerFile2.getPageAddress(position[i]);
            if(position[i] != 0){
                assert(pageData[0] == (pageData[2] + position[i]) );
                assert(pageData[1] == position[i] );
            }
        }
        
        if(k%printFreq == 0){

            std::cout << "Iteration: " << k << std::endl;
            std::cout << "Average time first read/write: " << avgTimeFistWrite/(k+1) << " seconds" << std::endl;
            std::cout << "Average throughput first read/write (last): " << avgThroughputFirstWrite/(k+1) << "( "<< avgThroughputFirstWriteLast/printFreq <<") MBytes/s" << std::endl;
            std::cout << "Average time second read/write: " << avgTimeSecondWrite/(k+1) << " seconds" << std::endl;
            std::cout << "Average throughput second read/write (last): " << avgThroughputSecondWrite/(k+1) << "( "<< avgThroughputSecondWriteLast/printFreq <<") MBytes/s" << std::endl;
            std::cout << "Average time flush: " << avgTimeFlush/k << " seconds" << std::endl;
            std::cout << "Average throughput flush (last): " << avgThroughputFlush/(k+1) << "( "<< avgThroughputFlushLast/printFreq <<") MBytes/s" << std::endl << std::endl;
            avgThroughputFirstWriteLast = 0;
            avgThroughputSecondWriteLast = 0;
            avgThroughputFlushLast = 0;
        }
    }

    //delete folder
    command = "rm -rf " + folderName;
    r = system(command.c_str());
    if(r!=0){
        zklog.info("Error removing folder");
        return 1;
    }
    return 0;
}

uint64_t PageManagerAccuracyTest (void)
{

    //
    // Memory version
    //
    PageManager pageManagerMem;
    Config configPM;
    PageContext ctx(pageManagerMem, configPM);
    pageManagerMem.init(ctx);
    uint64_t initialFreePages = pageManagerMem.getNumFreePages();
    
    uint64_t page1 = pageManagerMem.getFreePage();
    zkassertpermanent(page1 == 8);
    zkassertpermanent(pageManagerMem.getNumFreePages() == initialFreePages-1);
    uint64_t page2 = pageManagerMem.getFreePage();
    zkassertpermanent(page2 == 9 );
    zkassertpermanent(pageManagerMem.getNumFreePages() == initialFreePages-2);
    pageManagerMem.releasePage(page1);
    zkassertpermanent(pageManagerMem.getNumFreePages() == initialFreePages-1);
    pageManagerMem.releasePage(page2);
    zkassertpermanent(pageManagerMem.getNumFreePages() == initialFreePages);

    unordered_set<uint64_t> pages; 
    for(uint64_t i=0; i<initialFreePages;++i){
        pages.insert(pageManagerMem.getFreePage());
    }
    zkassertpermanent(pages.size() == initialFreePages);
    zkassertpermanent(pageManagerMem.getNumFreePages() == 0);

    for(uint64_t i=2; i<initialFreePages+2;++i){
        pageManagerMem.releasePage(i);
    }
    zkassertpermanent(pageManagerMem.getNumFreePages() == initialFreePages);

    uint64_t page3=pageManagerMem.getFreePage();
    uint64_t * page3Data = (uint64_t *)pageManagerMem.getPageAddress(page3);
    for(uint64_t i=0; i<256;++i){
        page3Data[i] = i;
    }
    pageManagerMem.flushPages(ctx);
    uint64_t freePagesAfterFlush = pageManagerMem.getNumFreePages();
    for(uint64_t i=0; i<256;++i){
        zkassertpermanent(page3Data[i] == i);
    }
    uint64_t page4 = pageManagerMem.editPage(page3);
    zkassertpermanent(page4 != page3);
    uint64_t * page4Data = (uint64_t *)pageManagerMem.getPageAddress(page4);
    for(uint64_t i=0; i<256;++i){
        zkassertpermanent(page4Data[i] == i);
    }
    zkassertpermanent(pageManagerMem.getNumFreePages() == freePagesAfterFlush-1);

    zkassertpermanent(pageManagerMem.editPage(page3) == page4);
    zkassertpermanent(pageManagerMem.editPage(page4) == page4);
    zkassertpermanent(pageManagerMem.editPage(0) == 1);
    zkassertpermanent(pageManagerMem.editPage(1) == 1);


    pageManagerMem.flushPages(ctx);
    for(uint64_t i=0; i<256;++i){
        zkassertpermanent(page4Data[i] == i);
    }

    //
    // File version
    //
    
    //generate a file with 100 pages
    const string fileName = "page_manager_test";
    const string folderName = "pmtest";
    const int file_size = 1;  //in GB
    
    //delete folder (is exists)
    std::string command = "rm -rf " + folderName;
    int r=system(command.c_str());
    if(r!=0){
        zklog.info("Error removing folder");
        return 1;
    }
    

    // Same tests than with memory version:
    PageManager pageManagerFile;
    Config configPMFile;
    configPMFile.hashDBFileName = fileName;
    configPMFile.hashDBFileSize = file_size;
    configPMFile.hashDBFolder = folderName;
    PageContext ctxf(pageManagerFile, configPMFile);
    pageManagerFile.init(ctxf);
    initialFreePages = pageManagerFile.getNumFreePages();


    page1 = pageManagerFile.getFreePage();
    zkassertpermanent(page1 == 8);
    zkassertpermanent(pageManagerFile.getNumFreePages() == initialFreePages-1);
    page2 = pageManagerFile.getFreePage();
    zkassertpermanent(page2 == 9);
    zkassertpermanent(pageManagerFile.getNumFreePages() == initialFreePages-2);
    pageManagerFile.releasePage(page1);
    zkassertpermanent(pageManagerFile.getNumFreePages() == initialFreePages-1);
    pageManagerFile.releasePage(page2);
    zkassertpermanent(pageManagerFile.getNumFreePages() == initialFreePages);

    pages.clear(); 
    for(uint64_t i=0; i<initialFreePages;++i){
        pages.insert(pageManagerFile.getFreePage());
    }
    zkassertpermanent(pages.size() == initialFreePages);
    zkassertpermanent(pageManagerFile.getNumFreePages() == 0);
    for(uint64_t i=2; i<initialFreePages+2ULL;++i){
        pageManagerFile.releasePage(i);
    }
    zkassertpermanent(pageManagerFile.getNumFreePages() == initialFreePages);
    
    page3=pageManagerFile.getFreePage();
    page3Data = (uint64_t *)pageManagerFile.getPageAddress(page3);
    for(uint64_t i=0; i<256;++i){
        page3Data[i] = i;
    }
    pageManagerFile.flushPages(ctxf);
    freePagesAfterFlush = pageManagerFile.getNumFreePages();
    for(uint64_t i=0; i<256;++i){
        zkassertpermanent(page3Data[i] == i);
    }
    page4 = pageManagerFile.editPage(page3);
    zkassertpermanent(page4 != page3);
    page4Data = (uint64_t *)pageManagerFile.getPageAddress(page4);
    for(uint64_t i=0; i<256;++i){
        zkassertpermanent(page4Data[i] == i);
    }
    zkassertpermanent(pageManagerFile.getNumFreePages() == freePagesAfterFlush-1);

    zkassertpermanent(pageManagerFile.editPage(page3) == page4);
    zkassertpermanent(pageManagerFile.editPage(page4) == page4);
    zkassertpermanent(pageManagerFile.editPage(0) == 1);
    zkassertpermanent(pageManagerFile.editPage(1) == 1);


    pageManagerFile.flushPages(ctxf);
    for(uint64_t i=0; i<256;++i){
        zkassertpermanent(page4Data[i] == i);
    }

    //Let's check persistence of file
    PageManager pageManagerFile2;
    PageContext ctxf2(pageManagerFile2, configPMFile);
    pageManagerFile2.init(ctxf2);
    page3Data = (uint64_t *)pageManagerFile2.getPageAddress(page3);
    page4Data = (uint64_t *)pageManagerFile2.getPageAddress(page4);
    for(uint64_t i=0; i<256;++i){
        zkassertpermanent(page4Data[i] == i);
    }

    uint64_t freePagesBeforeAddFile = pageManagerFile.getNumFreePages();
    pageManagerFile.addFile();
    uint64_t freePagesAfterAddFile = pageManagerFile.getNumFreePages();
    zkassertpermanent(freePagesAfterAddFile - freePagesBeforeAddFile == (file_size*1024*1024*1024)/4096);
    pages.clear(); 
    for(uint64_t i=0; i<freePagesAfterAddFile;++i){
        pages.insert(pageManagerFile.getFreePage());
    }
    zkassertpermanent(pages.size() == freePagesAfterAddFile);
    zkassertpermanent(pageManagerFile.getNumFreePages() == 0);
    for(uint64_t i=2; i<freePagesAfterAddFile+2;++i){
        pageManagerFile.releasePage(i);
    }
    zkassertpermanent(pageManagerFile.getNumFreePages() == freePagesAfterAddFile);

    //delete folder
    command = "rm -rf " + folderName;
    r = system(command.c_str());
    if(r!=0){
        zklog.info("Error removing folder");
        return 1;
    }
    

    return 0;   
}

uint64_t PageManagerDBResizeTest (void)
{
    //
    // Memory version
    //
    PageManager pageManagerMem;
    Config configPM;
    PageContext ctx(pageManagerMem, configPM);
    pageManagerMem.init(ctx);
    uint64_t initialFreePages = pageManagerMem.getNumFreePages();
    for(uint64_t i=0; i<initialFreePages;++i){
        pageManagerMem.getFreePage();
    }
    assert(pageManagerMem.getNumFreePages() == 0);
    pageManagerMem.getFreePage();
    assert(pageManagerMem.getNumFreePages() == initialFreePages+7);

    //
    // File version
    //
    
    //generate a file with
    const string fileName = "page_manager_test";
    const string folderName = "pmtest";
    const int file_size = 1;  //in GB
    
    //delete folder (is exists)
    std::string command = "rm -rf " + folderName;
    int r = system(command.c_str());
    if(r!=0){
        zklog.info("Error removing folder");
        return 1;
    }

    PageManager pageManagerFile;
    Config configPMFile;
    configPMFile.hashDBFileName = fileName;
    configPMFile.hashDBFileSize = file_size;
    configPMFile.hashDBFolder = folderName;
    PageContext ctxf(pageManagerFile, configPMFile);
    pageManagerFile.init(ctxf);

    initialFreePages = pageManagerFile.getNumFreePages();
    for(uint64_t i=0; i<initialFreePages;++i){
        pageManagerFile.getFreePage();
    }
    assert(pageManagerFile.getNumFreePages() == 0);
    pageManagerFile.getFreePage();
    assert(pageManagerFile.getNumFreePages() == initialFreePages+7);
    
    //delete folder
    command = "rm -rf " + folderName;
    r = system(command.c_str());
    if(r!=0){
        zklog.info("Error removing folder");
        return 1;
    }
    
    return 0;
}

uint64_t PageManagerDBResetTest (void){

    //
    // Memory version
    //
    PageManager pageManagerMem;
    Config configPM;
    PageContext ctx(pageManagerMem, configPM);
    pageManagerMem.init(ctx);
    uint64_t initialFreePages = pageManagerMem.getNumFreePages();
    for(uint64_t i=0; i<initialFreePages;++i){
        pageManagerMem.getFreePage();
    }
    assert(pageManagerMem.getNumFreePages() == 0);
    pageManagerMem.reset(ctx);
    assert(pageManagerMem.getNumFreePages() == initialFreePages);

    //
    // File version
    //
    
    //generate a file
    const string fileName = "page_manager_test";
    const string folderName = "pmtest";
    const int file_size = 1;  //in GB
    
    //delete folder (is exists)
    std::string command = "rm -rf " + folderName;
    int r = system(command.c_str());
    if(r!=0){
        zklog.info("Error removing folder");
        return 1;
    }

    PageManager pageManagerFile;
    Config configPMFile;
    configPMFile.hashDBFileName = fileName;
    configPMFile.hashDBFileSize = file_size;
    configPMFile.hashDBFolder = folderName;
    PageContext ctxf(pageManagerFile, configPMFile);
    pageManagerFile.init(ctxf);

    initialFreePages = pageManagerFile.getNumFreePages();
    for(uint64_t i=0; i<initialFreePages;++i){
        pageManagerFile.getFreePage();
    }
    assert(pageManagerFile.getNumFreePages() == 0);
    pageManagerFile.getFreePage();
    assert(pageManagerFile.getNumFreePages() == initialFreePages+7); 

    pageManagerFile.reset(ctxf);
    assert(pageManagerFile.getNumFreePages() == initialFreePages+262144);
    
    //delete folder
    command = "rm -rf " + folderName;
    r = system(command.c_str());
    if(r!=0){
        zklog.info("Error removing folder");
        return 1;
    }

    return 0;
}

