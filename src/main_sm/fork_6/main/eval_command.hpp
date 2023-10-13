#ifndef EVAL_COMMAND_HPP_fork_6
#define EVAL_COMMAND_HPP_fork_6

#include <gmpxx.h>
#include "main_sm/fork_6/main/context.hpp"
#include "main_sm/fork_6/main/rom_command.hpp"
#include "goldilocks_base_field.hpp"
#include "zkresult.hpp"
#include "ecrecover.hpp"

namespace fork_6
{

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
    CommandResultType   type;
    mpz_class           scalar;     // used if type==crt_scalar
    Goldilocks::Element fe;         // used if type==crt_fe
    Goldilocks::Element fea0;       // used if type==crt_fea
    Goldilocks::Element fea1;       // used if type==crt_fea
    Goldilocks::Element fea2;       // used if type==crt_fea
    Goldilocks::Element fea3;       // used if type==crt_fea
    Goldilocks::Element fea4;       // used if type==crt_fea
    Goldilocks::Element fea5;       // used if type==crt_fea
    Goldilocks::Element fea6;       // used if type==crt_fea
    Goldilocks::Element fea7;       // used if type==crt_fea
    string              str;        // used if type==crt_string
    uint64_t            u64;        // used if type==crt_u64
    uint32_t            u32;        // used if type==crt_u32
    uint16_t            u16;        // used if type==crt_u16
    zkresult            zkResult;   // used to return external errors, e.g. database errors
    CommandResult() { reset(); }
    void reset(void)
    {
        type = crt_unknown;
        zkResult = ZKR_SUCCESS;
    }
};

// Evaluates a ROM command, and returns command result
void evalCommand (Context &ctx, const RomCommand &cmd, CommandResult &cr);

// Converts a returned command result into a field element
void cr2fe (Context &ctx, const CommandResult &cr, Goldilocks::Element &fe);

// Converts a returned command result into a scalar
void cr2scalar (Context &ctx, const CommandResult &cr, mpz_class &s);

// Forwar declarations of internal operation functions
void eval_number              (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_declareVar          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_setVar              (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getVar              (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getReg              (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_add                 (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_sub                 (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_neg                 (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_mul                 (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_div                 (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_mod                 (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_logical_or          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_logical_and         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_logical_gt          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_logical_ge          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_logical_lt          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_logical_le          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_logical_eq          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_logical_ne          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_logical_not         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_bit_and             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_bit_or              (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_bit_xor             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_bit_not             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_bit_shl             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_bit_shr             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_if                  (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getMemValue         (Context &ctx, const RomCommand &cmd, CommandResult &cr);

// Forward declaration of internal callable functions
void eval_getGlobalExitRoot   (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getNTxs             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getRawTx            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getSequencerAddr    (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getTxs              (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getTxsLen           (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_eventLog            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getTimestamp        (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_cond                (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_inverseFpEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_inverseFnEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_sqrtFpEc            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_sqrtFpEcParity      (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_xAddPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_yAddPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_xDblPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_yDblPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_beforeLast          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_bitwise_and         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_bitwise_or          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_bitwise_xor         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_bitwise_not         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_comp_lt             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_comp_gt             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_comp_eq             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_loadScalar          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_log                 (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_exp                 (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_storeLog            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_memAlignWR_W0       (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_memAlignWR_W1       (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_memAlignWR8_W0      (Context &ctx, const RomCommand &cmd, CommandResult &cr);

zkresult AddPointEc (Context &ctx, bool dbl, const RawFec::Element &x1, const RawFec::Element &y1, const RawFec::Element &x2, const RawFec::Element &y2, RawFec::Element &x3, RawFec::Element &y3);

zkresult eval_addReadWriteAddress (Context &ctx, const mpz_class value);

mpz_class sqrtTonelliShanks ( const mpz_class &n, const mpz_class &p );


} // namespace

#endif