#ifndef PROVER_UTILS_HPP
#define PROVER_UTILS_HPP

#include "zk-prover.grpc.pb.h"
#include "goldilocks/goldilocks_base_field.hpp"
#include "input.hpp"
#include "proof.hpp"

// Converts grpc objects
void inputProver2Input (Goldilocks &fr, const zkprover::v1::InputProver &inputProver, Input &input);
void input2InputProver (Goldilocks &fr, const Input &input, zkprover::v1::InputProver &inputProver);
void proof2ProofProver (Goldilocks &fr, const Proof &proof, zkprover::v1::Proof &proofProver);

#endif