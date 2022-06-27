#ifndef EVAL_COMMAND_HPP
#define EVAL_COMMAND_HPP

#include <gmpxx.h>
#include "context.hpp"
#include "rom_command.hpp"
#include "goldilocks/goldilocks_base_field.hpp"

// Enumerates the possible types of command results
typedef enum {
    crt_unknown = 0,
    crt_scalar = 1,
    crt_fe = 2,
    crt_fea = 3,
    crt_string = 4,
    crt_u64 = 5,
    crt_u32 = 6,
    crt_u16 = 7
} CommandResultType;

class CommandResult
{
public:
    CommandResultType type;
    mpz_class         scalar; // used if type==crt_scalar
    Goldilocks::Element      fe;     // used if type==crt_fe
    Goldilocks::Element      fea0;   // used if type==crt_fea
    Goldilocks::Element      fea1;   // used if type==crt_fea
    Goldilocks::Element      fea2;   // used if type==crt_fea
    Goldilocks::Element      fea3;   // used if type==crt_fea
    Goldilocks::Element      fea4;   // used if type==crt_fea
    Goldilocks::Element      fea5;   // used if type==crt_fea
    Goldilocks::Element      fea6;   // used if type==crt_fea
    Goldilocks::Element      fea7;   // used if type==crt_fea
    string            str;    // used if type==crt_string
    uint64_t          u64;    // used if type==crt_u64
    uint32_t          u32;    // used if type==crt_u32
    uint16_t          u16;    // used if type==crt_u16
    bool beforeLast;
    CommandResult() : type(crt_unknown), beforeLast(false) {}
};

// Evaluates a ROM command, and returns command result
void evalCommand (Context &ctx, const RomCommand &cmd, CommandResult &cr);

// Converts a returned command result into a field element
void cr2fe (Goldilocks &fr, const CommandResult &cr, Goldilocks::Element &fe);

// Converts a returned command result into a scalar
void cr2scalar (Goldilocks &fr, const CommandResult &cr, mpz_class &s);

#endif