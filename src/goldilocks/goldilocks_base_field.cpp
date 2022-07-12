#include "goldilocks_base_field.hpp"

const Goldilocks::Element Goldilocks::ZR = {(uint64_t)0x0000000000000000LL};
const Goldilocks::Element Goldilocks::Q = {(uint64_t)0xFFFFFFFF00000001LL};
const Goldilocks::Element Goldilocks::MM = {(uint64_t)0xFFFFFFFeFFFFFFFFLL};
const Goldilocks::Element Goldilocks::CQ = {(uint64_t)0x00000000FFFFFFFFLL};
const Goldilocks::Element Goldilocks::R2 = {(uint64_t)0xFFFFFFFe00000001LL};

const Goldilocks::Element Goldilocks::W[33] = {
    Goldilocks::fromU64(0x1),
    Goldilocks::fromU64(0xFFFFFFFF00000000),
    Goldilocks::fromU64(0x1000000000000),
    Goldilocks::fromU64(0xFFFFFFFEFF000001),
    Goldilocks::fromU64(0xEFFFFFFF00000001),
    Goldilocks::fromU64(0x3FFFFFFFC000),
    Goldilocks::fromU64(0x8000000000),
    Goldilocks::fromU64(0xF80007FF08000001),
    Goldilocks::fromU64(0xBF79143CE60CA966),
    Goldilocks::fromU64(0x1905D02A5C411F4E),
    Goldilocks::fromU64(0x9D8F2AD78BFED972),
    Goldilocks::fromU64(0x653B4801DA1C8CF),
    Goldilocks::fromU64(0xF2C35199959DFCB6),
    Goldilocks::fromU64(0x1544EF2335D17997),
    Goldilocks::fromU64(0xE0EE099310BBA1E2),
    Goldilocks::fromU64(0xF6B2CFFE2306BAAC),
    Goldilocks::fromU64(0x54DF9630BF79450E),
    Goldilocks::fromU64(0xABD0A6E8AA3D8A0E),
    Goldilocks::fromU64(0x81281A7B05F9BEAC),
    Goldilocks::fromU64(0xFBD41C6B8CAA3302),
    Goldilocks::fromU64(0x30BA2ECD5E93E76D),
    Goldilocks::fromU64(0xF502AEF532322654),
    Goldilocks::fromU64(0x4B2A18ADE67246B5),
    Goldilocks::fromU64(0xEA9D5A1336FBC98B),
    Goldilocks::fromU64(0x86CDCC31C307E171),
    Goldilocks::fromU64(0x4BBAF5976ECFEFD8),
    Goldilocks::fromU64(0xED41D05B78D6E286),
    Goldilocks::fromU64(0x10D78DD8915A171D),
    Goldilocks::fromU64(0x59049500004A4485),
    Goldilocks::fromU64(0xDFA8C93BA46D2666),
    Goldilocks::fromU64(0x7E9BD009B86A0845),
    Goldilocks::fromU64(0x400A7F755588E659),
    Goldilocks::fromU64(0x185629DCDA58878C)};

#if USE_MONTGOMERY == 0
const Goldilocks::Element Goldilocks::ONE = {(uint64_t)0x0000000000000001LL};
const Goldilocks::Element Goldilocks::ZERO = {(uint64_t)0x0000000000000000LL};
const Goldilocks::Element Goldilocks::NEGONE = {(uint64_t)0xFFFFFFFF00000000LL};
const Goldilocks::Element Goldilocks::TWO32 = {0x0000000100000000LL};
const Goldilocks::Element Goldilocks::SHIFT = Goldilocks::fromU64(49);

#else
const Goldilocks::Element Goldilocks::ONE = {(uint64_t)0x00000000FFFFFFFFLL};
const Goldilocks::Element Goldilocks::ZERO = {(uint64_t)0x0000000000000000LL};
const Goldilocks::Element Goldilocks::NEGONE = {(uint64_t)0XFFFFFFFE00000002LL};
const Goldilocks::Element Goldilocks::SHIFT = Goldilocks::fromU64(49);

#endif