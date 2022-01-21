#ifndef EVAL_COMMAND_HPP
#define EVAL_COMMAND_HPP

#include <gmpxx.h>
#include "context.hpp"
#include "rom_command.hpp"
#include "ffiasm/fr.hpp"

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
    RawFr::Element    fe;     // used if type==crt_fe
    RawFr::Element    fea0;   // used if type==crt_fea
    RawFr::Element    fea1;   // used if type==crt_fea
    RawFr::Element    fea2;   // used if type==crt_fea
    RawFr::Element    fea3;   // used if type==crt_fea
    string            str;    // used if type==crt_string
    uint64_t          u64;    // used if type==crt_u64
    uint32_t          u32;    // used if type==crt_u32
    uint16_t          u16;    // used if type==crt_u16
    CommandResult() { type = crt_unknown; }
};

// Evaluates a ROM command, and returns command result
void evalCommand (Context &ctx, const RomCommand &cmd, CommandResult &cr);

// Converts a returned command result into a field element
void cr2fe (RawFr &fr, const CommandResult &cr, RawFr::Element &fe);

// Converts a returned command result into a scalar
void cr2scalar (RawFr &fr, const CommandResult &cr, mpz_class &s);

#endif