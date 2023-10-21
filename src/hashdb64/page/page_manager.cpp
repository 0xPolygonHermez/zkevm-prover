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
#include <dirent.h>
#include <regex>

PageManager::PageManager() 
{
    mappedFile=false;
    fileSize=0;
    pagesPerFile=0;
    nFiles=0;
    nPages=0;
    numFreePages=0;
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

zkresult PageManager::init(PageContext &ctx)
{

    if(ctx.config.hashDBFileName == "" ){
        
        //In-memory initailization
        
        mappedFile = false;
        nPages = 0;
        pages.resize(1);
        firstUnusedPage = 2;
        numFreePages = 0;
        freePages.resize(512);
        addPages(1024);

    }else{

        //File-mapped initialization
        fileName = ctx.config.hashDBFileName;
        fileSize = ctx.config.hashDBFileSize*1ULL<<30;
        folderName = ctx.config.hashDBFolder;
        pagesPerFile = fileSize >> 12;
        mappedFile = true;

        //Create the folder if it does not exist
        if(folderName != ""){
            struct stat st = {0};
            if (stat(folderName.c_str(), &st) == -1) {
                int mkdir_result = mkdir(folderName.c_str(), S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
                if (mkdir_result == -1) {
                    zklog.error("PageManager: failed to create directory");
                    exitProcess();
                }
            }
        }

        //Revise syntaxis of the existing files and create new ones if needed
        DIR *dir;
        struct dirent *ent;
        std::regex rgx(fileName + "_(\\d+)\\.db");
        bool onlyNewFiles = true;

        if((dir=opendir(folderName.c_str())) != nullptr){
            nFiles = 0;
            while((ent = readdir(dir)) != nullptr){

                onlyNewFiles = false;

                //check if the file name matches the pattern and is consecutive
                std::string file_name = ent->d_name;
                std::smatch match;
                if(std::regex_search(file_name, match, rgx)){
                    uint64_t file_number = std::stoull(match[1]);
                    if(file_number != nFiles){
                        zklog.error("PageManager: found non consecuitve db file: " + file_name + "in folder: " + folderName);
                        exitProcess();
                    }
                    
                }else{
                    zklog.error("PageManager: found db file with wrong name: " + file_name + "in folder: " + folderName);
                    exitProcess();
                }

                //check that file size is correct
                struct stat st;
                string file = "";   
                if(folderName != "")
                    file = folderName + "/";
                file += file_name;
                if(stat(file.c_str(), &st) == -1){
                    zklog.error("PageManager: failed to stat file: " + file);
                    exitProcess();
                }
                if((uint64_t)st.st_size != fileSize){
                    zklog.error("PageManager: found db file with wrong size: " + file_name + "in folder: " + folderName);
                    exitProcess();
                }

                //map the file and increase the number of pages
                int fd = open(file.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
                if (fd == -1) {
                    zklog.error("PageManager: failed to open file.");
                    exitProcess();
                }
                nPages += pagesPerFile;
                pages.push_back(NULL);
                pages[nFiles] = (char *)mmap(NULL, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); //MAP_POPULATE
                if (pages[nFiles] == MAP_FAILED) {
                    zklog.error("Failed to mmap file: " + (string)strerror(errno));
                    exitProcess();
                }
                
                if(nFiles!=0){ 
                    close(fd);
                }else{
                    file0Descriptor = fd;
                }
                nFiles++;
            }
            //add new file if needed
            if(nFiles != ctx.config.hashDBMinFilesNum){
                for(uint64_t i=nFiles; i<ctx.config.hashDBMinFilesNum; ++i){
                    addFile();
                }
            }
        }else{

            zklog.error("PageManager: failed to open directory: " + folderName);
            exitProcess();

        }
        closedir(dir);
        
        zkassertpermanent(nPages > 2);

        //Free pages
        if(onlyNewFiles){
            numFreePages = 0;
            freePages.resize(16384);
            firstUnusedPage = 2;
            HeaderPage::InitEmptyPage(ctx,0);
            msync(getPageAddress(0), 4096, MS_SYNC);

        }else{
            HeaderPage::Check(ctx,0);
            vector<uint64_t> freePagesDB;
            HeaderPage::GetFreePages(ctx,0, freePagesDB); 
            numFreePages = freePagesDB.size();
            freePages.resize((numFreePages)*2+1);
            memcpy(freePages.data(), freePagesDB.data(), numFreePages*sizeof(uint64_t));
            HeaderPage::GetFirstUnusedPage(ctx,0, firstUnusedPage);
        }

    }
    return zkresult::ZKR_SUCCESS;

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
        if(pageNumber >= nPages && mappedFile){
            if(mappedFile){
                zklog.info("PageManager: adding file");
                addFile();
            }else{
                zklog.info("PageManager: adding pages to memory");
                addPages(nPages);
            }
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

void PageManager::flushPages(PageContext &ctx){

#if MULTIPLE_WRITES
        lock_guard<mutex> guard(freePagesLock);
        shared_lock<shared_mutex> guard(pagesLock);
        std::lock_guard<std::mutex> lock(editedPagesLock);
#endif
    zkassertpermanent(&ctx.pageManager == this); 
    //1// get list of previous used pages as freePages container
    uint64_t headerPageNum = 0;
    vector<uint64_t> prevFreePagesContainer;
    HeaderPage::GetFreePagesContainer(ctx, headerPageNum, prevFreePagesContainer);

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

    HeaderPage::CreateFreePages(ctx, headerPageNum, newFreePages, newContainerPages);
    HeaderPage::SetFirstUnusedPage(ctx, headerPageNum, firstUnusedPage);

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
