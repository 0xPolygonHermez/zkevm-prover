#include "statedb.grpc.pb.h"
#include "goldilocks/goldilocks_base_field.hpp"

void fea2grpc (Goldilocks &fr, const Goldilocks::Element (&fea)[4], ::statedb::v1::Fea* grpcFea) {
    grpcFea->set_fe0(fr.toU64(fea[0]));
    grpcFea->set_fe1(fr.toU64(fea[1]));
    grpcFea->set_fe2(fr.toU64(fea[2]));
    grpcFea->set_fe3(fr.toU64(fea[3]));
}

void grpc2fea (Goldilocks &fr, const ::statedb::v1::Fea& grpcFea, Goldilocks::Element (&fea)[4]) {
    fea[0] = fr.fromU64(grpcFea.fe0());
    fea[1] = fr.fromU64(grpcFea.fe1());
    fea[2] = fr.fromU64(grpcFea.fe2());
    fea[3] = fr.fromU64(grpcFea.fe3());
}