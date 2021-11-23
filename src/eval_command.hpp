#ifndef EVAL_COMMAND_HPP
#define EVAL_COMMAND_HPP

#include <gmpxx.h>

#include "context.hpp"
#include "rom_command.hpp"
#include "ffiasm/fr.hpp"

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
    mpz_class scalar;
    RawFr::Element fe;
    RawFr::Element fea0;
    RawFr::Element fea1;
    RawFr::Element fea2;
    RawFr::Element fea3;
    string str;
    uint64_t u64;
    uint32_t u32;
    uint16_t u16;
    CommandResult() {type=crt_unknown;}
};

//RawFr::Element evalCommand (Context &ctx, RomCommand &cmd);

void evalCommand(Context &ctx, RomCommand &cmd, CommandResult &cr);

#endif