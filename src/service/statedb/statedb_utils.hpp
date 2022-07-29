#ifndef STATEDB_UTILS_HPP
#define STATEDB_UTILS_HPP

#include "statedb.grpc.pb.h"
#include "goldilocks_base_field.hpp"

void fea2grpc (Goldilocks &fr, const Goldilocks::Element (&fea)[4], ::statedb::v1::Fea* grpcFea);
void grpc2fea (Goldilocks &fr, const ::statedb::v1::Fea& grpcFea, Goldilocks::Element (&fea)[4]);

#endif
