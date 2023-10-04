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
    pages = NULL;
    mappedFile = false;
    AddPages(131072);
}
PageManager::PageManager(const uint64_t nPages_)
{
    assert(nPages_ >= 2);
    mappedFile = false;
    nPages = 0;
    pages = NULL;
    AddPages(nPages_);
}
PageManager::PageManager(const string fileName_){
   
    mappedFile = true;
    fileName = fileName_;

    struct stat file_stat;

    fd = open(fileName.c_str(),O_RDWR);
    if (fd == -1) {
         zklog.error("Failed to open file: " + (string)strerror(errno));
         exitProcess();
    }

    // Get the file size
    if (fstat(fd, &file_stat) == -1) {
        zklog.error("Failed to get file size: " + (string)strerror(errno));
        close(fd);
    }

    fileSize = file_stat.st_size;
    nPages = fileSize / 4096;
    assert(nPages >= 2);
    pages = (char *)mmap(NULL, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
    if (pages == MAP_FAILED) {
        zklog.error("Failed to mmap file: " + (string)strerror(errno));
    }
}
PageManager::~PageManager(void)
{
    if(mappedFile){
        munmap(pages, fileSize);
        close(fd);
    }else{
        if (pages != NULL)
            free(pages);
    }
}
zkresult PageManager::AddPages(const uint64_t nPages_)
{
    assert(mappedFile == false);
    char *auxPages = NULL;
    auxPages = (char *)realloc(pages, (nPages + nPages_) * 4096);
    if (auxPages != NULL)
    {
        pages = auxPages;
        memset(pages + nPages * 4096, 0, nPages_ * 4096);
        for (uint64_t i = 2; i < nPages_; i++) //0 and 1 are reserved pages
            freePages.push_back(nPages + i);
        nPages += nPages_;
    }
    else
    {
        zklog.error("PageManager::AddPages() failed calling realloc()");
        exitProcess();
    }
    return zkresult::ZKR_SUCCESS;
}

uint64_t PageManager::getFreePage(void)
{
    
    uint32_t pageNumber = freePages.front();
    freePages.pop_front();
    return pageNumber;
}
void PageManager::releasePage(const uint64_t pageNumber)
{
    assert(pageNumber >= 2); //first two pages cannot be released
    memset(getPageAddress(pageNumber), 0, 4096);
    freePages.push_back(pageNumber);
}
uint32_t PageManager::editPage(const uint32_t pageNumber)
{
    if(pageNumber <= 1){
        return 1;
    }
    uint32_t pageNumber_;
    unordered_map<uint32_t, uint32_t>::const_iterator it = editedPages.find(pageNumber);
    if(it == editedPages.end()){
        pageNumber_ = getFreePage();
        memcpy(getPageAddress(pageNumber_),getPageAddress(pageNumber) , 4096);
        editedPages[pageNumber] = pageNumber_;
    }else{
        pageNumber_ = it->second;
    }
    return pageNumber_;
}
void PageManager::flushPages(){

    if(!mappedFile){
        memcpy(getPageAddress(0), getPageAddress(1), 4096); // copy tmp header to header
    }else{
        msync(getPageAddress(1), fileSize-4096, MS_SYNC);
        off_t offset = 0;
        if(lseek(fd, offset, SEEK_SET) == -1) {
            zklog.error("Failed to seek file: " + (string)strerror(errno));
            exitProcess();
        }
        ssize_t write_size =write(fd, getPageAddress(1), 4096); //how transactional is this?
        assert(write_size == 4096);
    }
    for(unordered_map<uint32_t, uint32_t>::const_iterator it = editedPages.begin(); it != editedPages.end(); it++){
        releasePage(it->second);
    }
    editedPages.clear();
}
