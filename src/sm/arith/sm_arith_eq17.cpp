/* code generated with arith_eq_gen.js
* equation: x1-x2-x3+p*q1-p*offset
* 
* p=0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
* offset=0x10
*/

#include <stdint.h>
#include "definitions.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "goldilocks_base_field.hpp"

USING_PROVER_FORK_NAMESPACE;

int64_t eq17 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o)
{
	switch(step) {
		case 0: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0xffaaab0 + (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.x2[0][_o]) - (int64_t)fr.toU64(p.x3[0][_o]));

		case 1: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0xffffff * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0xffffff0 + (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.x2[1][_o]) - (int64_t)fr.toU64(p.x3[1][_o]));

		case 2: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0xffffff * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0xffb9fe * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0xffb9fe0 + (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.x2[2][_o]) - (int64_t)fr.toU64(p.x3[2][_o]));

		case 3: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0xffffff * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0xffb9fe * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0xb153ff * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0xb153ff0 + (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.x2[3][_o]) - (int64_t)fr.toU64(p.x3[3][_o]));

		case 4: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0xffffff * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0xffb9fe * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0xb153ff * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0xabfffe * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0xabfffe0 + (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.x2[4][_o]) - (int64_t)fr.toU64(p.x3[4][_o]));

		case 5: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0xffffff * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0xffb9fe * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0xb153ff * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0xabfffe * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0xf6241e * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0xf6241e0 + (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.x2[5][_o]) - (int64_t)fr.toU64(p.x3[5][_o]));

		case 6: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0xffffff * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0xffb9fe * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0xb153ff * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0xabfffe * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0xf6241e * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0xa0f6b00 + (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.x2[6][_o]) - (int64_t)fr.toU64(p.x3[6][_o]));

		case 7: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0xffffff * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0xffb9fe * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0xb153ff * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0xabfffe * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0xf6241e * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x6730d2 * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x6730d20 + (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.x2[7][_o]) - (int64_t)fr.toU64(p.x3[7][_o]));

		case 8: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0xffffff * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0xffb9fe * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0xb153ff * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0xabfffe * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0xf6241e * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x6730d2 * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x8512bf * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x8512bf0 + (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.x2[8][_o]) - (int64_t)fr.toU64(p.x3[8][_o]));

		case 9: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0xffffff * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0xffb9fe * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0xb153ff * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0xabfffe * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0xf6241e * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x6730d2 * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x8512bf * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x4b84f30 + (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.x2[9][_o]) - (int64_t)fr.toU64(p.x3[9][_o]));

		case 10: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0xffffff * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0xffb9fe * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0xb153ff * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0xabfffe * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0xf6241e * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x6730d2 * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x8512bf * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0xd76477 * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0xd764770 + (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.x2[10][_o]) - (int64_t)fr.toU64(p.x3[10][_o]));

		case 11: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0xffffff * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0xffb9fe * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0xb153ff * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0xabfffe * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0xf6241e * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x6730d2 * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x8512bf * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0xd76477 * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x434bac * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x434bac0 + (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.x2[11][_o]) - (int64_t)fr.toU64(p.x3[11][_o]));

		case 12: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0xffffff * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0xffb9fe * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0xb153ff * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0xabfffe * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0xf6241e * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x6730d2 * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x8512bf * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0xd76477 * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x434bac * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x1ba7b60 + (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.x2[12][_o]) - (int64_t)fr.toU64(p.x3[12][_o]));

		case 13: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0xffffff * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0xffb9fe * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0xb153ff * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0xabfffe * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0xf6241e * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x6730d2 * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x8512bf * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0xd76477 * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x434bac * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0xe69a4b * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0xe69a4b0 + (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.x2[13][_o]) - (int64_t)fr.toU64(p.x3[13][_o]));

		case 14: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0xffffff * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0xffb9fe * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0xb153ff * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0xabfffe * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0xf6241e * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x6730d2 * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x8512bf * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0xd76477 * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x434bac * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0xe69a4b * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0xea397f * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0xea397f0 + (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.x2[14][_o]) - (int64_t)fr.toU64(p.x3[14][_o]));

		case 15: 
		return (
		(0xffaaab * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0xffffff * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0xffb9fe * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0xb153ff * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0xabfffe * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0xf6241e * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x6730d2 * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x8512bf * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0xd76477 * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x434bac * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0xe69a4b * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0xea397f * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x1a0111 * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x1a01110 + (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.x2[15][_o]) - (int64_t)fr.toU64(p.x3[15][_o]));

		case 16: 
		return (
		(0xffffff * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0xffb9fe * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0xb153ff * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0xabfffe * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0xf6241e * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x6730d2 * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x8512bf * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0xd76477 * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x434bac * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0xe69a4b * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0xea397f * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x1a0111 * (int64_t)fr.toU64(p.q1[1][_o])));

		case 17: 
		return (
		(0xffb9fe * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0xb153ff * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0xabfffe * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0xf6241e * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x6730d2 * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x8512bf * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0xd76477 * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x434bac * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0xe69a4b * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0xea397f * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x1a0111 * (int64_t)fr.toU64(p.q1[2][_o])));

		case 18: 
		return (
		(0xb153ff * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0xabfffe * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0xf6241e * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x6730d2 * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x8512bf * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0xd76477 * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x434bac * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0xe69a4b * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0xea397f * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x1a0111 * (int64_t)fr.toU64(p.q1[3][_o])));

		case 19: 
		return (
		(0xabfffe * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0xf6241e * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x6730d2 * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x8512bf * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0xd76477 * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x434bac * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0xe69a4b * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0xea397f * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x1a0111 * (int64_t)fr.toU64(p.q1[4][_o])));

		case 20: 
		return (
		(0xf6241e * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x6730d2 * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x8512bf * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0xd76477 * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x434bac * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0xe69a4b * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0xea397f * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x1a0111 * (int64_t)fr.toU64(p.q1[5][_o])));

		case 21: 
		return (
		(0xa0f6b0 * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x6730d2 * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x8512bf * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0xd76477 * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x434bac * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0xe69a4b * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0xea397f * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x1a0111 * (int64_t)fr.toU64(p.q1[6][_o])));

		case 22: 
		return (
		(0x6730d2 * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x8512bf * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0xd76477 * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x434bac * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0xe69a4b * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0xea397f * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x1a0111 * (int64_t)fr.toU64(p.q1[7][_o])));

		case 23: 
		return (
		(0x8512bf * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0xd76477 * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x434bac * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0xe69a4b * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0xea397f * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x1a0111 * (int64_t)fr.toU64(p.q1[8][_o])));

		case 24: 
		return (
		(0x4b84f3 * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0xd76477 * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x434bac * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0xe69a4b * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0xea397f * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x1a0111 * (int64_t)fr.toU64(p.q1[9][_o])));

		case 25: 
		return (
		(0xd76477 * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x434bac * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0xe69a4b * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0xea397f * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x1a0111 * (int64_t)fr.toU64(p.q1[10][_o])));

		case 26: 
		return (
		(0x434bac * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0xe69a4b * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0xea397f * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x1a0111 * (int64_t)fr.toU64(p.q1[11][_o])));

		case 27: 
		return (
		(0x1ba7b6 * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0xe69a4b * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0xea397f * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x1a0111 * (int64_t)fr.toU64(p.q1[12][_o])));

		case 28: 
		return (
		(0xe69a4b * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0xea397f * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x1a0111 * (int64_t)fr.toU64(p.q1[13][_o])));

		case 29: 
		return (
		(0xea397f * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x1a0111 * (int64_t)fr.toU64(p.q1[14][_o])));

		case 30: 
		return ( (0x1a0111 * (int64_t)fr.toU64(p.q1[15][_o])));

		case 31: 
		return (
		0x0     );
	}
	return 0;
}
