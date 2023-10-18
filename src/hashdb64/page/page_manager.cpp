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
#include <omp.h>
#include "header_page.hpp"
#include "page_list_page.hpp"

PageManager pageManager;
PageManager::PageManager()
{
    nPages = 0;
    mappedFile = false;
    pages.resize(1);
    firstUnusedPage = 2;
    numFreePages = 0;
    freePages.resize(16384);
    addPages(131072);
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
    addPages(nPages_);
}
PageManager::PageManager(const string fileName_, const uint64_t fileSize_, const uint64_t nFiles_, const string folderName_){
    
    fileName = fileName_;
    fileSize = fileSize_;
    nFiles = nFiles_;
    folderName = folderName_;
    pagesPerFile = fileSize >> 12;
    

    mappedFile = true;
    int fd;
    nPages = 0;
    uint64_t pagesPerFile = fileSize >> 12;

    for(uint64_t k = 0; k< nFiles; ++k){
        string file = "";
        if(folderName != "")
            file = folderName + "/";
        file += (fileName + "_" + to_string(k)+".db");
        fd = open(file.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
        if(k==0) file0Descriptor = fd;
        if (fd == -1) {
            zklog.error("PageManager: failed to open file.");
            exitProcess();
        }
        if (ftruncate(fd, fileSize) == -1) {
            zklog.error("PageManager: failed to truncate to file.");
            close(fd);
        }
        nPages += pagesPerFile;
        pages.push_back(NULL);
        pages[k] = (char *)mmap(NULL, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); //MAP_POPULATE
        if (pages[k] == MAP_FAILED) {
            zklog.error("Failed to mmap file: " + (string)strerror(errno));
        }
#if USE_FILE_IO
        fileDescriptors.push_back(fd);
#else
        if(k!=0) close(fd);
#endif
    }
    zkassertpermanent(nPages > 2);
    firstUnusedPage = 2;

    //Free pages
    /*vector<uint64_t> freePagesDB;
    HeaderPage::GetFreePages(0, freePagesDB); 
    numFreePages = freePagesDB.size();
    freePages.resize((numFreePages)*2+1);
    memccpy(freePages.data(), freePagesDB.data(), 0, numFreePages*sizeof(uint64_t));*/
    numFreePages = 0;
    freePages.resize(16384);
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

zkresult PageManager::addPages(const uint64_t nPages_)
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
zkresult PageManager::addFile(){

    zkassertpermanent(mappedFile == true);
    string file = "";
    if(folderName != "")
        file = folderName + "/";
    file += (fileName + "_" + to_string(nFiles)+".db");
    int fd = open(file.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd == -1) {
        zklog.error("PageManager: failed to open file.");
        exitProcess();
    }
    if (ftruncate(fd, fileSize) == -1) {
        zklog.error("PageManager: failed to truncate to file.");
        close(fd);
    }
    unique_lock<shared_mutex> guard(pagesLock);
    nPages += pagesPerFile;
    pages.push_back(NULL);
    pages[nFiles] = (char *)mmap(NULL, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); //MAP_POPULATE
    if (pages[nFiles] == MAP_FAILED) {
        zklog.error("Failed to mmap file: " + (string)strerror(errno));
    }
    ++nFiles;
#if USE_FILE_IO
        fileDescriptors.push_back(fd);
#endif
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
        memset(getPageAddress(pageNumber), 0, 4096);
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

#if MULTIPLE_WRITES
        lock_guard<mutex> guard(freePagesLock);
        shared_lock<shared_mutex> guard(pagesLock);
        std::lock_guard<std::mutex> lock(editedPagesLock);
#endif

    //1// get list of previous used pages as freePages container
    uint64_t headerPageNum = 0;
    vector<uint64_t> prevFreePagesContainer;
    HeaderPage::GetFreePagesContainer(headerPageNum, prevFreePagesContainer);

    //2// get list of edited pages
    vector<uint64_t> copiedPages;
    for(unordered_map<uint64_t, uint64_t>::const_iterator it = editedPages.begin(); it != editedPages.end(); it++){
        if(it->first != it->second && it->first >= 2){
            copiedPages.emplace_back(it->first);
        }
    }

    //3// generate new list of freePages
    uint64_t nPrevFreePagesContainer = prevFreePagesContainer.size();
    uint64_t nEditedPages = copiedPages.size();
    uint64_t entriesPerPage = (PageListPage::maxOffset - PageListPage::minOffset) / PageListPage::entrySize; 
    uint64_t nNewFreePages = numFreePages + nEditedPages + nPrevFreePagesContainer;
    uint64_t nNewContainerPages = (nNewFreePages + entriesPerPage) / (entriesPerPage +1);
    if(nNewContainerPages > numFreePages){
        uint64_t nNewFreePages_ = nNewFreePages - numFreePages*(entriesPerPage +1);
        uint64_t nAddedContanerPages = (nNewFreePages_ + entriesPerPage - 1) / entriesPerPage;
        nNewContainerPages = numFreePages + nAddedContanerPages;
    }
    if(nNewContainerPages == 0) nNewContainerPages = 1; //at least one container page that will be empty

    vector<uint64_t> newContainerPages(nNewContainerPages);
    for(uint64_t i=0; i< nNewContainerPages; ++i){
        newContainerPages[i]=getFreePage();
    }

    vector<uint64_t> newFreePages(numFreePages+nEditedPages+nPrevFreePagesContainer);
    memcpy(newFreePages.data(), freePages.data(), numFreePages*sizeof(uint64_t));
    memcpy(newFreePages.data()+numFreePages, prevFreePagesContainer.data(), nPrevFreePagesContainer*sizeof(uint64_t));
    memcpy(newFreePages.data()+numFreePages+nPrevFreePagesContainer, copiedPages.data(), nEditedPages*sizeof(uint64_t));

    HeaderPage::CreateFreePages(headerPageNum, newFreePages, newContainerPages);
    HeaderPage::setFirstUnusedPage(headerPageNum, firstUnusedPage);

    //4// sync all pages
    if(mappedFile){
        #pragma omp parallel for schedule(static,1) num_threads(omp_get_num_threads()/2)
        for(uint64_t k=0; k< pages.size(); ++k){
            if(k==0){
                msync(getPageAddress(1), fileSize-4096, MS_SYNC);
            }else{
                msync(pages[k], fileSize, MS_SYNC);
            }
        }
    }

    //5// write header
    {
        unique_lock<shared_mutex> guard_header(headerLock);
        if(mappedFile){
            off_t offset = 0;
            if(lseek(file0Descriptor, offset, SEEK_SET) == -1) {
                zklog.error("Failed to seek file.");
                exitProcess();
            }
            ssize_t write_size =write(file0Descriptor, getPageAddress(1), 4096); //how transactional is this?
            zkassertpermanent(write_size == 4096);
        }else{
            memcpy(getPageAddress(0), getPageAddress(1), 4096);
        }
    }

    //6// Release freePagesContainers 
    for(vector<uint64_t>::const_iterator it = prevFreePagesContainer.begin(); it != prevFreePagesContainer.end(); it++){
        releasePage(*it);
    }

    //7// Release edited pages
    for(vector<uint64_t>::const_iterator it = copiedPages.begin(); it != copiedPages.end(); it++){
        releasePage(*it);
    }     
    editedPages.clear();

}
