/* code generated with arith_eq_gen.js
* equation: x1*y1-p2_256*y2+x2-y3
* 
* p2_256=0x10000000000000000000000000000000000000000000000000000000000000000
*/

#include <stdint.h>
#include "definitions.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "goldilocks_base_field.hpp"

USING_PROVER_FORK_NAMESPACE;

int64_t eq0 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o)
{
	switch(step) {
		case 0: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[0][_o]) - (int64_t)fr.toU64(p.y3[0][_o]));

		case 1: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[1][_o])) +
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[1][_o]) - (int64_t)fr.toU64(p.y3[1][_o]));

		case 2: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[2][_o])) +
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[1][_o])) +
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[2][_o]) - (int64_t)fr.toU64(p.y3[2][_o]));

		case 3: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[3][_o])) +
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[2][_o])) +
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[1][_o])) +
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[3][_o]) - (int64_t)fr.toU64(p.y3[3][_o]));

		case 4: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[4][_o])) +
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[3][_o])) +
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[2][_o])) +
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[1][_o])) +
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[4][_o]) - (int64_t)fr.toU64(p.y3[4][_o]));

		case 5: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[5][_o])) +
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[4][_o])) +
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[3][_o])) +
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[2][_o])) +
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[1][_o])) +
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[5][_o]) - (int64_t)fr.toU64(p.y3[5][_o]));

		case 6: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[6][_o])) +
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[5][_o])) +
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[4][_o])) +
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[3][_o])) +
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[2][_o])) +
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[1][_o])) +
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[6][_o]) - (int64_t)fr.toU64(p.y3[6][_o]));

		case 7: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[7][_o])) +
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[6][_o])) +
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[5][_o])) +
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[4][_o])) +
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[3][_o])) +
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[2][_o])) +
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[1][_o])) +
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[7][_o]) - (int64_t)fr.toU64(p.y3[7][_o]));

		case 8: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[8][_o])) +
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[7][_o])) +
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[6][_o])) +
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[5][_o])) +
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[4][_o])) +
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[3][_o])) +
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[2][_o])) +
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[1][_o])) +
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[8][_o]) - (int64_t)fr.toU64(p.y3[8][_o]));

		case 9: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[9][_o])) +
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[8][_o])) +
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[7][_o])) +
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[6][_o])) +
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[5][_o])) +
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[4][_o])) +
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[3][_o])) +
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[2][_o])) +
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[1][_o])) +
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[9][_o]) - (int64_t)fr.toU64(p.y3[9][_o]));

		case 10: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[10][_o])) +
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[9][_o])) +
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[8][_o])) +
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[7][_o])) +
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[6][_o])) +
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[5][_o])) +
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[4][_o])) +
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[3][_o])) +
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[2][_o])) +
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[1][_o])) +
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[10][_o]) - (int64_t)fr.toU64(p.y3[10][_o]));

		case 11: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[11][_o])) +
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[10][_o])) +
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[9][_o])) +
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[8][_o])) +
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[7][_o])) +
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[6][_o])) +
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[5][_o])) +
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[4][_o])) +
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[3][_o])) +
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[2][_o])) +
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[1][_o])) +
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[11][_o]) - (int64_t)fr.toU64(p.y3[11][_o]));

		case 12: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[12][_o])) +
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[11][_o])) +
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[10][_o])) +
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[9][_o])) +
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[8][_o])) +
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[7][_o])) +
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[6][_o])) +
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[5][_o])) +
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[4][_o])) +
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[3][_o])) +
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[2][_o])) +
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[1][_o])) +
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[12][_o]) - (int64_t)fr.toU64(p.y3[12][_o]));

		case 13: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[13][_o])) +
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[12][_o])) +
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[11][_o])) +
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[10][_o])) +
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[9][_o])) +
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[8][_o])) +
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[7][_o])) +
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[6][_o])) +
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[5][_o])) +
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[4][_o])) +
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[3][_o])) +
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[2][_o])) +
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[1][_o])) +
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[13][_o]) - (int64_t)fr.toU64(p.y3[13][_o]));

		case 14: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[14][_o])) +
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[13][_o])) +
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[12][_o])) +
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[11][_o])) +
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[10][_o])) +
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[9][_o])) +
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[8][_o])) +
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[7][_o])) +
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[6][_o])) +
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[5][_o])) +
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[4][_o])) +
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[3][_o])) +
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[2][_o])) +
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[1][_o])) +
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[14][_o]) - (int64_t)fr.toU64(p.y3[14][_o]));

		case 15: 
		return (
		((int64_t)fr.toU64(p.x1[0][_o]) * (int64_t)fr.toU64(p.y1[15][_o])) +
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[14][_o])) +
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[13][_o])) +
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[12][_o])) +
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[11][_o])) +
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[10][_o])) +
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[9][_o])) +
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[8][_o])) +
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[7][_o])) +
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[6][_o])) +
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[5][_o])) +
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[4][_o])) +
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[3][_o])) +
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[2][_o])) +
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[1][_o])) +
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[0][_o]))
		 + (int64_t)fr.toU64(p.x2[15][_o]) - (int64_t)fr.toU64(p.y3[15][_o]));

		case 16: 
		return (
		((int64_t)fr.toU64(p.x1[1][_o]) * (int64_t)fr.toU64(p.y1[15][_o])) +
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[14][_o])) +
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[13][_o])) +
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[12][_o])) +
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[11][_o])) +
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[10][_o])) +
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[9][_o])) +
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[8][_o])) +
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[7][_o])) +
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[6][_o])) +
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[5][_o])) +
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[4][_o])) +
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[3][_o])) +
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[2][_o])) +
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[1][_o]))
		    - (int64_t)fr.toU64(p.y2[0][_o]));

		case 17: 
		return (
		((int64_t)fr.toU64(p.x1[2][_o]) * (int64_t)fr.toU64(p.y1[15][_o])) +
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[14][_o])) +
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[13][_o])) +
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[12][_o])) +
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[11][_o])) +
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[10][_o])) +
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[9][_o])) +
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[8][_o])) +
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[7][_o])) +
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[6][_o])) +
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[5][_o])) +
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[4][_o])) +
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[3][_o])) +
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[2][_o]))
		    - (int64_t)fr.toU64(p.y2[1][_o]));

		case 18: 
		return (
		((int64_t)fr.toU64(p.x1[3][_o]) * (int64_t)fr.toU64(p.y1[15][_o])) +
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[14][_o])) +
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[13][_o])) +
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[12][_o])) +
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[11][_o])) +
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[10][_o])) +
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[9][_o])) +
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[8][_o])) +
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[7][_o])) +
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[6][_o])) +
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[5][_o])) +
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[4][_o])) +
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[3][_o]))
		    - (int64_t)fr.toU64(p.y2[2][_o]));

		case 19: 
		return (
		((int64_t)fr.toU64(p.x1[4][_o]) * (int64_t)fr.toU64(p.y1[15][_o])) +
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[14][_o])) +
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[13][_o])) +
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[12][_o])) +
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[11][_o])) +
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[10][_o])) +
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[9][_o])) +
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[8][_o])) +
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[7][_o])) +
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[6][_o])) +
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[5][_o])) +
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[4][_o]))
		    - (int64_t)fr.toU64(p.y2[3][_o]));

		case 20: 
		return (
		((int64_t)fr.toU64(p.x1[5][_o]) * (int64_t)fr.toU64(p.y1[15][_o])) +
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[14][_o])) +
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[13][_o])) +
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[12][_o])) +
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[11][_o])) +
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[10][_o])) +
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[9][_o])) +
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[8][_o])) +
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[7][_o])) +
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[6][_o])) +
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[5][_o]))
		    - (int64_t)fr.toU64(p.y2[4][_o]));

		case 21: 
		return (
		((int64_t)fr.toU64(p.x1[6][_o]) * (int64_t)fr.toU64(p.y1[15][_o])) +
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[14][_o])) +
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[13][_o])) +
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[12][_o])) +
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[11][_o])) +
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[10][_o])) +
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[9][_o])) +
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[8][_o])) +
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[7][_o])) +
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[6][_o]))
		    - (int64_t)fr.toU64(p.y2[5][_o]));

		case 22: 
		return (
		((int64_t)fr.toU64(p.x1[7][_o]) * (int64_t)fr.toU64(p.y1[15][_o])) +
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[14][_o])) +
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[13][_o])) +
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[12][_o])) +
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[11][_o])) +
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[10][_o])) +
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[9][_o])) +
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[8][_o])) +
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[7][_o]))
		    - (int64_t)fr.toU64(p.y2[6][_o]));

		case 23: 
		return (
		((int64_t)fr.toU64(p.x1[8][_o]) * (int64_t)fr.toU64(p.y1[15][_o])) +
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[14][_o])) +
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[13][_o])) +
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[12][_o])) +
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[11][_o])) +
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[10][_o])) +
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[9][_o])) +
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[8][_o]))
		    - (int64_t)fr.toU64(p.y2[7][_o]));

		case 24: 
		return (
		((int64_t)fr.toU64(p.x1[9][_o]) * (int64_t)fr.toU64(p.y1[15][_o])) +
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[14][_o])) +
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[13][_o])) +
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[12][_o])) +
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[11][_o])) +
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[10][_o])) +
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[9][_o]))
		    - (int64_t)fr.toU64(p.y2[8][_o]));

		case 25: 
		return (
		((int64_t)fr.toU64(p.x1[10][_o]) * (int64_t)fr.toU64(p.y1[15][_o])) +
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[14][_o])) +
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[13][_o])) +
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[12][_o])) +
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[11][_o])) +
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[10][_o]))
		    - (int64_t)fr.toU64(p.y2[9][_o]));

		case 26: 
		return (
		((int64_t)fr.toU64(p.x1[11][_o]) * (int64_t)fr.toU64(p.y1[15][_o])) +
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[14][_o])) +
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[13][_o])) +
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[12][_o])) +
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[11][_o]))
		    - (int64_t)fr.toU64(p.y2[10][_o]));

		case 27: 
		return (
		((int64_t)fr.toU64(p.x1[12][_o]) * (int64_t)fr.toU64(p.y1[15][_o])) +
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[14][_o])) +
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[13][_o])) +
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[12][_o]))
		    - (int64_t)fr.toU64(p.y2[11][_o]));

		case 28: 
		return (
		((int64_t)fr.toU64(p.x1[13][_o]) * (int64_t)fr.toU64(p.y1[15][_o])) +
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[14][_o])) +
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[13][_o]))
		    - (int64_t)fr.toU64(p.y2[12][_o]));

		case 29: 
		return (
		((int64_t)fr.toU64(p.x1[14][_o]) * (int64_t)fr.toU64(p.y1[15][_o])) +
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[14][_o]))
		    - (int64_t)fr.toU64(p.y2[13][_o]));

		case 30: 
		return (
		((int64_t)fr.toU64(p.x1[15][_o]) * (int64_t)fr.toU64(p.y1[15][_o]))
		    - (int64_t)fr.toU64(p.y2[14][_o]));

		case 31: 
		return ( - (int64_t)fr.toU64(p.y2[15][_o]));
	}
	return 0;
}
