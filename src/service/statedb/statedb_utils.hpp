#ifndef STATEDB_UTILS_HPP
#define STATEDB_UTILS_HPP

#include "statedb.grpc.pb.h"
#include "goldilocks/goldilocks_base_field.hpp"

void fea2grpc (Goldilocks &fr, const Goldilocks::Element (&fea)[4], ::statedb::v1::fea* grpcFea);
void grpc2fea (Goldilocks &fr, const ::statedb::v1::fea& grpcFea, Goldilocks::Element (&fea)[4]);

#endif
