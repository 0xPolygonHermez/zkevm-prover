#ifndef RAPIDSNARK_PROVER_HPP
#define RAPIDSNARK_PROVER_HPP

#include <string>

int rapidsnark_prover ( std::string zkeyFilename,
                        std::string wtnsFilename,
                        json &jsonProof,
                        json &jsonPublic );
#endif