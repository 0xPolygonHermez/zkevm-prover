#ifndef EVAL_COMMAND_HPP
#define EVAL_COMMAND_HPP

#include "context.hpp"
#include "rom_command.hpp"

RawFr::Element evalCommand (Context &ctx, RomCommand &cmd);

#endif