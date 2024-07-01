#include <stdexcept>

#include "zkey.hpp"
#include "zkey_fflonk.hpp"
#include <iostream>
#include "zklog.hpp"
#include "exit_process.hpp"


namespace Zkey {
    FflonkZkeyHeader::FflonkZkeyHeader() {
        this->protocolId = Zkey::FFLONK_PROTOCOL_ID;
    }

    FflonkZkeyHeader::~FflonkZkeyHeader() {
        mpz_clear(qPrime);
        mpz_clear(rPrime);

        // Free memory allocated with malloc
        free(k1);
        free(k2);
        free(w3);
        free(w4);
        free(w8);
        free(wr);
        free(X2);
        free(C0);
    }

    FflonkZkeyHeader* FflonkZkeyHeader::loadFflonkZkeyHeader(BinFileUtils::BinFile *f) {
        auto fflonkZkeyHeader = new FflonkZkeyHeader();

        f->startReadSection(Zkey::ZKEY_FF_HEADER_SECTION);

        fflonkZkeyHeader->n8q = f->readU32LE();
        mpz_init(fflonkZkeyHeader->qPrime);
        mpz_import(fflonkZkeyHeader->qPrime, fflonkZkeyHeader->n8q, -1, 1, -1, 0, f->read(fflonkZkeyHeader->n8q));

        fflonkZkeyHeader->n8r = f->readU32LE();
        mpz_init(fflonkZkeyHeader->rPrime);
        mpz_import(fflonkZkeyHeader->rPrime, fflonkZkeyHeader->n8r, -1, 1, -1, 0, f->read(fflonkZkeyHeader->n8r));

        fflonkZkeyHeader->nVars = f->readU32LE();
        fflonkZkeyHeader->nPublic = f->readU32LE();
        fflonkZkeyHeader->domainSize = f->readU32LE();
        fflonkZkeyHeader->nAdditions = f->readU32LE();
        fflonkZkeyHeader->nConstraints = f->readU32LE();

         // Memory allocation for void* members (assuming FrElement is a known size)
        fflonkZkeyHeader->k1 = malloc(fflonkZkeyHeader->n8r);
        if(fflonkZkeyHeader->k1 == NULL){
            zklog.error("Memory allocation failed for k1");
            exitProcess();
        } 
        fflonkZkeyHeader->k2 = malloc(fflonkZkeyHeader->n8r);
        if(fflonkZkeyHeader->k2 == NULL){
            zklog.error("Memory allocation failed for k2");
            exitProcess();
        }
        fflonkZkeyHeader->w3 = malloc(fflonkZkeyHeader->n8r);
        if(fflonkZkeyHeader->w3 == NULL){
            zklog.error("Memory allocation failed for w3");
            exitProcess();
        }
        fflonkZkeyHeader->w4 = malloc(fflonkZkeyHeader->n8r);
        if(fflonkZkeyHeader->w4 == NULL){
            zklog.error("Memory allocation failed for w4");
            exitProcess();
        }
        fflonkZkeyHeader->w8 = malloc(fflonkZkeyHeader->n8r);
        if(fflonkZkeyHeader->w8 == NULL){
            zklog.error("Memory allocation failed for w8");
            exitProcess();
        }
        fflonkZkeyHeader->wr = malloc(fflonkZkeyHeader->n8r);
        if(fflonkZkeyHeader->wr == NULL){
            zklog.error("Memory allocation failed for wr");
            exitProcess();
        }
        fflonkZkeyHeader->X2 = malloc(fflonkZkeyHeader->n8q * 4);
        if(fflonkZkeyHeader->X2 == NULL){
            zklog.error("Memory allocation failed for X2");
            exitProcess();
        }
        fflonkZkeyHeader->C0 = malloc(fflonkZkeyHeader->n8q * 2);
        if(fflonkZkeyHeader->C0 == NULL){
            zklog.error("Memory allocation failed for C0");
            exitProcess();
        }

        memcpy(fflonkZkeyHeader->k1, f->read(fflonkZkeyHeader->n8r), fflonkZkeyHeader->n8r);
        memcpy(fflonkZkeyHeader->k2, f->read(fflonkZkeyHeader->n8r), fflonkZkeyHeader->n8r);

        memcpy(fflonkZkeyHeader->w3, f->read(fflonkZkeyHeader->n8r), fflonkZkeyHeader->n8r);
        memcpy(fflonkZkeyHeader->w4, f->read(fflonkZkeyHeader->n8r), fflonkZkeyHeader->n8r);
        memcpy(fflonkZkeyHeader->w8, f->read(fflonkZkeyHeader->n8r), fflonkZkeyHeader->n8r);
        memcpy(fflonkZkeyHeader->wr, f->read(fflonkZkeyHeader->n8r), fflonkZkeyHeader->n8r);

        memcpy(fflonkZkeyHeader->X2, f->read(fflonkZkeyHeader->n8q * 4), fflonkZkeyHeader->n8q * 4);

        memcpy(fflonkZkeyHeader->C0, f->read(fflonkZkeyHeader->n8q * 2), fflonkZkeyHeader->n8q * 2);

        f->endReadSection();

        return fflonkZkeyHeader;
    }
}

