#ifndef ECRECOVER_HPP
#define ECRECOVER_HPP

#include <string>

std::string ecrecover(std::string sig, std::string msg); // hex-encoded sig, plain text msg

#endif