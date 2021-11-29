#ifndef ECRECOVER_HPP
#define ECRECOVER_HPP

#include <string>

#define keccak_256(a,b,c,d) sha3_256(a,b,c,d)

std::string ecrecover(std::string sig, std::string msg); // hex-encoded sig, plain text msg

#endif