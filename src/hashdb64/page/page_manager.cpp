#include "page_manager.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "zkassert.hpp"
#include <cstring>
#include <cassert>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>

PageManager pageManager;
PageManager::PageManager()
{
    nPages = 0;
    mappedFile = false;
    pages.resize(1);
    firstUnusedPage = 2;
    numFreePages = 0;
    freePages.resize(16384);
    AddPages(131072);
}
PageManager::PageManager(const uint64_t nPages_)
{
    zkassertpermanent(nPages_ > 2);
    firstUnusedPage = 2;
    mappedFile = false;
    nPages = 0;
    pages.resize(1);
    numFreePages = 0;
    freePages.resize(16384);
    AddPages(nPages_);
}
PageManager::PageManager(const string fileName_, const uint64_t fileSize_, const uint64_t nFiles_, const string folderName_){
    
    fileName = fileName_;
    fileSize = fileSize_;
    nFiles = nFiles_;
    folderName = folderName_;
    pagesPerFile = fileSize >> 12;
    numFreePages = 0;
    freePages.resize(16384);

    mappedFile = true;
    struct stat file_stat;
    int fd;
    nPages = 0;
    uint64_t pagesPerFile = fileSize >> 12;

    for(uint64_t k = 0; k< nFiles; ++k){
        string file = "";
        if(folderName != "")
            file = folderName + "/";
        file += (fileName + "_" + to_string(k)+".db");
        fd = open(file.c_str(),O_RDWR);
        if(k==0) file0Descriptor = fd;
        if (fd == -1) {
            zklog.error("Failed to open file: " + (string)strerror(errno));
            exitProcess();
        }
        // Get the file size
        if (fstat(fd, &file_stat) == -1) {
            zklog.error("Failed to get file size: " + (string)strerror(errno));
            close(fd);
        }
        zkassertpermanent(fileSize == (uint64_t)file_stat.st_size);
        nPages += pagesPerFile;
        pages.push_back(NULL);
        pages[k] = (char *)mmap(NULL, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); //MAP_POPULATE
        if (pages[k] == MAP_FAILED) {
            zklog.error("Failed to mmap file: " + (string)strerror(errno));
        }
        fileDescriptors.push_back(fd);
        //if(k!=0) close(fd);
    }
    zkassertpermanent(nPages > 2);
    firstUnusedPage = 2;
}
PageManager::~PageManager(void)
{
    if(mappedFile){
        for(uint64_t i=0; i< pages.size(); i++){
            munmap(pages[i], fileSize);
        }
        close(file0Descriptor);
    }else{
        if (pages[0] != NULL)
            free(pages[0]);
    }
}

zkresult PageManager::AddPages(const uint64_t nPages_)
{
    unique_lock<shared_mutex> guard(pagesLock);
    zkassertpermanent(mappedFile == false);
    char *auxPages = NULL;
    auxPages = (char *)realloc(pages[0], (nPages + nPages_) * 4096);
    if (auxPages == NULL)
    {
        zklog.error("PageManager::AddPages() failed calling realloc()");
        exitProcess();
    }
    pages[0] = auxPages;
    memset(pages[0] + nPages * 4096, 0, nPages_ * 4096);
    nPages += nPages_;
    pagesPerFile = nPages;
    return zkresult::ZKR_SUCCESS;
}

uint64_t PageManager::getFreePage(void)
{
#if MULTIPLE_WRITES
    lock_guard<mutex> guard(freePagesLock);
#endif
    uint64_t pageNumber;
    if(numFreePages > 0){
        pageNumber = freePages[numFreePages-1];
        --numFreePages;
    }else{
        pageNumber = firstUnusedPage;
        firstUnusedPage++;
#if MULTIPLE_WRITES
        shared_lock<shared_mutex> guard(pagesLock);
#endif
        if(pageNumber >= nPages){
            zklog.error("PageManager::getFreePage() failed, no more pages available");
            exitProcess();
        }
    }
    #if MULTIPLE_WRITES
        std::lock_guard<std::mutex> lock(editedPagesLock);
    #endif
    editedPages[pageNumber] = pageNumber;
    return pageNumber;
}
void PageManager::releasePage(const uint64_t pageNumber)
{
    zkassertpermanent(pageNumber >= 2);  //first two pages cannot be released
    memset(getPageAddress(pageNumber), 0, 4096);
#if MULTIPLE_WRITES
    std::lock_guard<std::mutex> lock(freePagesLock);
#endif
    zkassertpermanent(pageNumber<firstUnusedPage);
    if(numFreePages == freePages.size()){
        freePages.resize(freePages.size()*2);
    }
    freePages[numFreePages++]=pageNumber;
}
uint64_t PageManager::editPage(const uint64_t pageNumber)
{
    uint32_t pageNumber_;
#if MULTIPLE_WRITES
    std::lock_guard<std::mutex> lock(editedPagesLock);
#endif
    unordered_map<uint64_t, uint64_t>::const_iterator it = editedPages.find(pageNumber);
    if(it == editedPages.end()){
        pageNumber_ = ( pageNumber <= 1 ? 1 : getFreePage() );
        memcpy(getPageAddress(pageNumber_),getPageAddress(pageNumber) , 4096);
        editedPages[pageNumber] = pageNumber_;
    }else{
        pageNumber_ = it->second;
    }
    return pageNumber_;
}
void PageManager::flushPages(){
    
    if(!mappedFile){
        unique_lock<shared_mutex> guard2(headerLock);
        memcpy(getPageAddress(0), getPageAddress(1), 4096); // copy tmp header to header
    }else{
#if MULTIPLE_WRITES
        shared_lock<shared_mutex> guard(pagesLock);
#endif
        msync(getPageAddress(1), fileSize-4096, MS_SYNC);
        for(uint64_t k=1; k< pages.size(); ++k){
            msync(pages[k], fileSize, MS_SYNC);
        }
        off_t offset = 0;
        if(lseek(file0Descriptor, offset, SEEK_SET) == -1) {
            zklog.error("Failed to seek file: " + (string)strerror(errno));
            exitProcess();
        }
        unique_lock<shared_mutex> guard2(headerLock);
        ssize_t write_size =write(file0Descriptor, getPageAddress(1), 4096); //how transactional is this?
        zkassertpermanent(write_size == 4096);
    }
#if MULTIPLE_WRITES
    std::lock_guard<std::mutex> lock(editedPagesLock);
#endif
    for(unordered_map<uint64_t, uint64_t>::const_iterator it = editedPages.begin(); it != editedPages.end(); it++){
        if(it->first != it->second && it->first >= 2){
            releasePage(it->first);
        }
    }
    editedPages.clear();
}
