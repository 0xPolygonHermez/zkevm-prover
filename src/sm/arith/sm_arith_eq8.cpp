/* code generated with arith_eq_gen.js
* equation: y1+y2-y3-p*q2+p*offset
* 
* p=0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
* offset=0x8
*/

#include <stdint.h>
#include "definitions.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "goldilocks_base_field.hpp"

USING_PROVER_FORK_NAMESPACE;

int64_t eq8 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o)
{
	switch(step) {
		case 0: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x7ea38  + (int64_t)fr.toU64(p.y1[0][_o]) + (int64_t)fr.toU64(p.y2[0][_o]) - (int64_t)fr.toU64(p.y3[0][_o]));

		case 1: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[1][_o])) +
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x6c3e0  + (int64_t)fr.toU64(p.y1[1][_o]) + (int64_t)fr.toU64(p.y2[1][_o]) - (int64_t)fr.toU64(p.y3[1][_o]));

		case 2: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[2][_o])) +
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[1][_o])) +
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x460b0  + (int64_t)fr.toU64(p.y1[2][_o]) + (int64_t)fr.toU64(p.y2[2][_o]) - (int64_t)fr.toU64(p.y3[2][_o]));

		case 3: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[3][_o])) +
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[2][_o])) +
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[1][_o])) +
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x1e100  + (int64_t)fr.toU64(p.y1[3][_o]) + (int64_t)fr.toU64(p.y2[3][_o]) - (int64_t)fr.toU64(p.y3[3][_o]));

		case 4: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[4][_o])) +
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[3][_o])) +
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[2][_o])) +
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[1][_o])) +
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x65468  + (int64_t)fr.toU64(p.y1[4][_o]) + (int64_t)fr.toU64(p.y2[4][_o]) - (int64_t)fr.toU64(p.y3[4][_o]));

		case 5: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[5][_o])) +
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[4][_o])) +
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[3][_o])) +
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[2][_o])) +
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[1][_o])) +
		( - 0x6871   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x34388  + (int64_t)fr.toU64(p.y1[5][_o]) + (int64_t)fr.toU64(p.y2[5][_o]) - (int64_t)fr.toU64(p.y3[5][_o]));

		case 6: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[6][_o])) +
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[5][_o])) +
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[4][_o])) +
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[3][_o])) +
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[2][_o])) +
		( - 0x6871   * (int64_t)fr.toU64(p.q2[1][_o])) +
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x35488  + (int64_t)fr.toU64(p.y1[6][_o]) + (int64_t)fr.toU64(p.y2[6][_o]) - (int64_t)fr.toU64(p.y3[6][_o]));

		case 7: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[7][_o])) +
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[6][_o])) +
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[5][_o])) +
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[4][_o])) +
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[3][_o])) +
		( - 0x6871   * (int64_t)fr.toU64(p.q2[2][_o])) +
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[1][_o])) +
		( - 0x9781   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x4bc08  + (int64_t)fr.toU64(p.y1[7][_o]) + (int64_t)fr.toU64(p.y2[7][_o]) - (int64_t)fr.toU64(p.y3[7][_o]));

		case 8: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[8][_o])) +
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[7][_o])) +
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[6][_o])) +
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[5][_o])) +
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[4][_o])) +
		( - 0x6871   * (int64_t)fr.toU64(p.q2[3][_o])) +
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[2][_o])) +
		( - 0x9781   * (int64_t)fr.toU64(p.q2[1][_o])) +
		( - 0x585d   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x2c2e8  + (int64_t)fr.toU64(p.y1[8][_o]) + (int64_t)fr.toU64(p.y2[8][_o]) - (int64_t)fr.toU64(p.y3[8][_o]));

		case 9: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[9][_o])) +
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[8][_o])) +
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[7][_o])) +
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[6][_o])) +
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[5][_o])) +
		( - 0x6871   * (int64_t)fr.toU64(p.q2[4][_o])) +
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[3][_o])) +
		( - 0x9781   * (int64_t)fr.toU64(p.q2[2][_o])) +
		( - 0x585d   * (int64_t)fr.toU64(p.q2[1][_o])) +
		( - 0x8181   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x40c08  + (int64_t)fr.toU64(p.y1[9][_o]) + (int64_t)fr.toU64(p.y2[9][_o]) - (int64_t)fr.toU64(p.y3[9][_o]));

		case 10: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[10][_o])) +
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[9][_o])) +
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[8][_o])) +
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[7][_o])) +
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[6][_o])) +
		( - 0x6871   * (int64_t)fr.toU64(p.q2[5][_o])) +
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[4][_o])) +
		( - 0x9781   * (int64_t)fr.toU64(p.q2[3][_o])) +
		( - 0x585d   * (int64_t)fr.toU64(p.q2[2][_o])) +
		( - 0x8181   * (int64_t)fr.toU64(p.q2[1][_o])) +
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x22db0  + (int64_t)fr.toU64(p.y1[10][_o]) + (int64_t)fr.toU64(p.y2[10][_o]) - (int64_t)fr.toU64(p.y3[10][_o]));

		case 11: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[11][_o])) +
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[10][_o])) +
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[9][_o])) +
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[8][_o])) +
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[7][_o])) +
		( - 0x6871   * (int64_t)fr.toU64(p.q2[6][_o])) +
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[5][_o])) +
		( - 0x9781   * (int64_t)fr.toU64(p.q2[4][_o])) +
		( - 0x585d   * (int64_t)fr.toU64(p.q2[3][_o])) +
		( - 0x8181   * (int64_t)fr.toU64(p.q2[2][_o])) +
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[1][_o])) +
		( - 0xb850   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x5c280  + (int64_t)fr.toU64(p.y1[11][_o]) + (int64_t)fr.toU64(p.y2[11][_o]) - (int64_t)fr.toU64(p.y3[11][_o]));

		case 12: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[12][_o])) +
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[11][_o])) +
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[10][_o])) +
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[9][_o])) +
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[8][_o])) +
		( - 0x6871   * (int64_t)fr.toU64(p.q2[7][_o])) +
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[6][_o])) +
		( - 0x9781   * (int64_t)fr.toU64(p.q2[5][_o])) +
		( - 0x585d   * (int64_t)fr.toU64(p.q2[4][_o])) +
		( - 0x8181   * (int64_t)fr.toU64(p.q2[3][_o])) +
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[2][_o])) +
		( - 0xb850   * (int64_t)fr.toU64(p.q2[1][_o])) +
		( - 0xa029   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x50148  + (int64_t)fr.toU64(p.y1[12][_o]) + (int64_t)fr.toU64(p.y2[12][_o]) - (int64_t)fr.toU64(p.y3[12][_o]));

		case 13: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[13][_o])) +
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[12][_o])) +
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[11][_o])) +
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[10][_o])) +
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[9][_o])) +
		( - 0x6871   * (int64_t)fr.toU64(p.q2[8][_o])) +
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[7][_o])) +
		( - 0x9781   * (int64_t)fr.toU64(p.q2[6][_o])) +
		( - 0x585d   * (int64_t)fr.toU64(p.q2[5][_o])) +
		( - 0x8181   * (int64_t)fr.toU64(p.q2[4][_o])) +
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[3][_o])) +
		( - 0xb850   * (int64_t)fr.toU64(p.q2[2][_o])) +
		( - 0xa029   * (int64_t)fr.toU64(p.q2[1][_o])) +
		( - 0xe131   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x70988  + (int64_t)fr.toU64(p.y1[13][_o]) + (int64_t)fr.toU64(p.y2[13][_o]) - (int64_t)fr.toU64(p.y3[13][_o]));

		case 14: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[14][_o])) +
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[13][_o])) +
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[12][_o])) +
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[11][_o])) +
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[10][_o])) +
		( - 0x6871   * (int64_t)fr.toU64(p.q2[9][_o])) +
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[8][_o])) +
		( - 0x9781   * (int64_t)fr.toU64(p.q2[7][_o])) +
		( - 0x585d   * (int64_t)fr.toU64(p.q2[6][_o])) +
		( - 0x8181   * (int64_t)fr.toU64(p.q2[5][_o])) +
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[4][_o])) +
		( - 0xb850   * (int64_t)fr.toU64(p.q2[3][_o])) +
		( - 0xa029   * (int64_t)fr.toU64(p.q2[2][_o])) +
		( - 0xe131   * (int64_t)fr.toU64(p.q2[1][_o])) +
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x27390  + (int64_t)fr.toU64(p.y1[14][_o]) + (int64_t)fr.toU64(p.y2[14][_o]) - (int64_t)fr.toU64(p.y3[14][_o]));

		case 15: 
		return (
		( - 0xfd47   * (int64_t)fr.toU64(p.q2[15][_o])) +
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[14][_o])) +
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[13][_o])) +
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[12][_o])) +
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[11][_o])) +
		( - 0x6871   * (int64_t)fr.toU64(p.q2[10][_o])) +
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[9][_o])) +
		( - 0x9781   * (int64_t)fr.toU64(p.q2[8][_o])) +
		( - 0x585d   * (int64_t)fr.toU64(p.q2[7][_o])) +
		( - 0x8181   * (int64_t)fr.toU64(p.q2[6][_o])) +
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[5][_o])) +
		( - 0xb850   * (int64_t)fr.toU64(p.q2[4][_o])) +
		( - 0xa029   * (int64_t)fr.toU64(p.q2[3][_o])) +
		( - 0xe131   * (int64_t)fr.toU64(p.q2[2][_o])) +
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[1][_o])) +
		( - 0x3064   * (int64_t)fr.toU64(p.q2[0][_o]))
		    + 0x18320  + (int64_t)fr.toU64(p.y1[15][_o]) + (int64_t)fr.toU64(p.y2[15][_o]) - (int64_t)fr.toU64(p.y3[15][_o]));

		case 16: 
		return (
		( - 0xd87c   * (int64_t)fr.toU64(p.q2[15][_o])) +
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[14][_o])) +
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[13][_o])) +
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[12][_o])) +
		( - 0x6871   * (int64_t)fr.toU64(p.q2[11][_o])) +
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[10][_o])) +
		( - 0x9781   * (int64_t)fr.toU64(p.q2[9][_o])) +
		( - 0x585d   * (int64_t)fr.toU64(p.q2[8][_o])) +
		( - 0x8181   * (int64_t)fr.toU64(p.q2[7][_o])) +
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[6][_o])) +
		( - 0xb850   * (int64_t)fr.toU64(p.q2[5][_o])) +
		( - 0xa029   * (int64_t)fr.toU64(p.q2[4][_o])) +
		( - 0xe131   * (int64_t)fr.toU64(p.q2[3][_o])) +
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[2][_o])) +
		( - 0x3064   * (int64_t)fr.toU64(p.q2[1][_o])));

		case 17: 
		return (
		( - 0x8c16   * (int64_t)fr.toU64(p.q2[15][_o])) +
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[14][_o])) +
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[13][_o])) +
		( - 0x6871   * (int64_t)fr.toU64(p.q2[12][_o])) +
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[11][_o])) +
		( - 0x9781   * (int64_t)fr.toU64(p.q2[10][_o])) +
		( - 0x585d   * (int64_t)fr.toU64(p.q2[9][_o])) +
		( - 0x8181   * (int64_t)fr.toU64(p.q2[8][_o])) +
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[7][_o])) +
		( - 0xb850   * (int64_t)fr.toU64(p.q2[6][_o])) +
		( - 0xa029   * (int64_t)fr.toU64(p.q2[5][_o])) +
		( - 0xe131   * (int64_t)fr.toU64(p.q2[4][_o])) +
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[3][_o])) +
		( - 0x3064   * (int64_t)fr.toU64(p.q2[2][_o])));

		case 18: 
		return (
		( - 0x3c20   * (int64_t)fr.toU64(p.q2[15][_o])) +
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[14][_o])) +
		( - 0x6871   * (int64_t)fr.toU64(p.q2[13][_o])) +
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[12][_o])) +
		( - 0x9781   * (int64_t)fr.toU64(p.q2[11][_o])) +
		( - 0x585d   * (int64_t)fr.toU64(p.q2[10][_o])) +
		( - 0x8181   * (int64_t)fr.toU64(p.q2[9][_o])) +
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[8][_o])) +
		( - 0xb850   * (int64_t)fr.toU64(p.q2[7][_o])) +
		( - 0xa029   * (int64_t)fr.toU64(p.q2[6][_o])) +
		( - 0xe131   * (int64_t)fr.toU64(p.q2[5][_o])) +
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[4][_o])) +
		( - 0x3064   * (int64_t)fr.toU64(p.q2[3][_o])));

		case 19: 
		return (
		( - 0xca8d   * (int64_t)fr.toU64(p.q2[15][_o])) +
		( - 0x6871   * (int64_t)fr.toU64(p.q2[14][_o])) +
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[13][_o])) +
		( - 0x9781   * (int64_t)fr.toU64(p.q2[12][_o])) +
		( - 0x585d   * (int64_t)fr.toU64(p.q2[11][_o])) +
		( - 0x8181   * (int64_t)fr.toU64(p.q2[10][_o])) +
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[9][_o])) +
		( - 0xb850   * (int64_t)fr.toU64(p.q2[8][_o])) +
		( - 0xa029   * (int64_t)fr.toU64(p.q2[7][_o])) +
		( - 0xe131   * (int64_t)fr.toU64(p.q2[6][_o])) +
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[5][_o])) +
		( - 0x3064   * (int64_t)fr.toU64(p.q2[4][_o])));

		case 20: 
		return (
		( - 0x6871   * (int64_t)fr.toU64(p.q2[15][_o])) +
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[14][_o])) +
		( - 0x9781   * (int64_t)fr.toU64(p.q2[13][_o])) +
		( - 0x585d   * (int64_t)fr.toU64(p.q2[12][_o])) +
		( - 0x8181   * (int64_t)fr.toU64(p.q2[11][_o])) +
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[10][_o])) +
		( - 0xb850   * (int64_t)fr.toU64(p.q2[9][_o])) +
		( - 0xa029   * (int64_t)fr.toU64(p.q2[8][_o])) +
		( - 0xe131   * (int64_t)fr.toU64(p.q2[7][_o])) +
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[6][_o])) +
		( - 0x3064   * (int64_t)fr.toU64(p.q2[5][_o])));

		case 21: 
		return (
		( - 0x6a91   * (int64_t)fr.toU64(p.q2[15][_o])) +
		( - 0x9781   * (int64_t)fr.toU64(p.q2[14][_o])) +
		( - 0x585d   * (int64_t)fr.toU64(p.q2[13][_o])) +
		( - 0x8181   * (int64_t)fr.toU64(p.q2[12][_o])) +
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[11][_o])) +
		( - 0xb850   * (int64_t)fr.toU64(p.q2[10][_o])) +
		( - 0xa029   * (int64_t)fr.toU64(p.q2[9][_o])) +
		( - 0xe131   * (int64_t)fr.toU64(p.q2[8][_o])) +
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[7][_o])) +
		( - 0x3064   * (int64_t)fr.toU64(p.q2[6][_o])));

		case 22: 
		return (
		( - 0x9781   * (int64_t)fr.toU64(p.q2[15][_o])) +
		( - 0x585d   * (int64_t)fr.toU64(p.q2[14][_o])) +
		( - 0x8181   * (int64_t)fr.toU64(p.q2[13][_o])) +
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[12][_o])) +
		( - 0xb850   * (int64_t)fr.toU64(p.q2[11][_o])) +
		( - 0xa029   * (int64_t)fr.toU64(p.q2[10][_o])) +
		( - 0xe131   * (int64_t)fr.toU64(p.q2[9][_o])) +
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[8][_o])) +
		( - 0x3064   * (int64_t)fr.toU64(p.q2[7][_o])));

		case 23: 
		return (
		( - 0x585d   * (int64_t)fr.toU64(p.q2[15][_o])) +
		( - 0x8181   * (int64_t)fr.toU64(p.q2[14][_o])) +
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[13][_o])) +
		( - 0xb850   * (int64_t)fr.toU64(p.q2[12][_o])) +
		( - 0xa029   * (int64_t)fr.toU64(p.q2[11][_o])) +
		( - 0xe131   * (int64_t)fr.toU64(p.q2[10][_o])) +
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[9][_o])) +
		( - 0x3064   * (int64_t)fr.toU64(p.q2[8][_o])));

		case 24: 
		return (
		( - 0x8181   * (int64_t)fr.toU64(p.q2[15][_o])) +
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[14][_o])) +
		( - 0xb850   * (int64_t)fr.toU64(p.q2[13][_o])) +
		( - 0xa029   * (int64_t)fr.toU64(p.q2[12][_o])) +
		( - 0xe131   * (int64_t)fr.toU64(p.q2[11][_o])) +
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[10][_o])) +
		( - 0x3064   * (int64_t)fr.toU64(p.q2[9][_o])));

		case 25: 
		return (
		( - 0x45b6   * (int64_t)fr.toU64(p.q2[15][_o])) +
		( - 0xb850   * (int64_t)fr.toU64(p.q2[14][_o])) +
		( - 0xa029   * (int64_t)fr.toU64(p.q2[13][_o])) +
		( - 0xe131   * (int64_t)fr.toU64(p.q2[12][_o])) +
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[11][_o])) +
		( - 0x3064   * (int64_t)fr.toU64(p.q2[10][_o])));

		case 26: 
		return (
		( - 0xb850   * (int64_t)fr.toU64(p.q2[15][_o])) +
		( - 0xa029   * (int64_t)fr.toU64(p.q2[14][_o])) +
		( - 0xe131   * (int64_t)fr.toU64(p.q2[13][_o])) +
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[12][_o])) +
		( - 0x3064   * (int64_t)fr.toU64(p.q2[11][_o])));

		case 27: 
		return (
		( - 0xa029   * (int64_t)fr.toU64(p.q2[15][_o])) +
		( - 0xe131   * (int64_t)fr.toU64(p.q2[14][_o])) +
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[13][_o])) +
		( - 0x3064   * (int64_t)fr.toU64(p.q2[12][_o])));

		case 28: 
		return (
		( - 0xe131   * (int64_t)fr.toU64(p.q2[15][_o])) +
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[14][_o])) +
		( - 0x3064   * (int64_t)fr.toU64(p.q2[13][_o])));

		case 29: 
		return (
		( - 0x4e72   * (int64_t)fr.toU64(p.q2[15][_o])) +
		( - 0x3064   * (int64_t)fr.toU64(p.q2[14][_o])));

		case 30: 
		return ( ( - 0x3064   * (int64_t)fr.toU64(p.q2[15][_o])));

		case 31: 
		return (
		0x0     );
	}
	return 0;
}
