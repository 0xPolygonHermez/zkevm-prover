#include "arith.hpp"

#include <nlohmann/json.hpp>
#include "arith.hpp"
#include "arith_pols.hpp"
#include "arith_action_bytes.hpp"
//#include "arith_defines.hpp"
#include "utils.hpp"
#include "scalar.hpp"

using json = nlohmann::json;

void ArithExecutor::execute (vector<ArithAction> &action)
{
    // Allocate polynomials
    ArithPols pols(config);
    //pols.alloc(polSize, pilJson); TODO: uncomment when available

    // Split actions into bytes
    vector<ArithActionBytes> input;
    for (uint64_t i=0; i<action.size(); i++)
    {
        uint64_t dataSize;
        ArithActionBytes actionBytes;
        actionBytes.x1 = action[i].x1;
        actionBytes.y1 = action[i].y1;
        actionBytes.x2 = action[i].x2;
        actionBytes.y2 = action[i].y2;
        actionBytes.x3 = action[i].x3;
        actionBytes.y3 = action[i].y3;
        actionBytes.selEq0 = action[i].selEq0;
        actionBytes.selEq1 = action[i].selEq1;
        actionBytes.selEq2 = action[i].selEq2;
        actionBytes.selEq3 = action[i].selEq3;
        dataSize = 16;
        scalar2ba16(actionBytes._x1, dataSize, action[i].x1);
        dataSize = 16;
        scalar2ba16(actionBytes._y1, dataSize, action[i].y1);
        dataSize = 16;
        scalar2ba16(actionBytes._x2, dataSize, action[i].x2);
        dataSize = 16;
        scalar2ba16(actionBytes._y2, dataSize, action[i].y2);
        dataSize = 16;
        scalar2ba16(actionBytes._x3, dataSize, action[i].x3);
        dataSize = 16;
        scalar2ba16(actionBytes._y3, dataSize, action[i].y3);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq0, dataSize, action[i].selEq0);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq1, dataSize, action[i].selEq1);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq2, dataSize, action[i].selEq2);
        dataSize = 16;
        scalar2ba16(actionBytes._selEq3, dataSize, action[i].selEq3);
        input.push_back(actionBytes);
    }

    //uint64_t N = polSize;

    // Process all the inputs
    for (uint64_t i = 0; i < input.size(); i++)
    {
#ifdef LOG_BINARY_EXECUTOR
        if (i%10000 == 0)
        {
            cout << "Computing binary pols " << i << "/" << input.size() << endl;
        }
#endif
    }
}

/*

module.exports.execute = async function (pols, polsDef, input) {
    // Get N from definitions
    const N = Object.entries(polsDef)[0][1]['polDeg'] || Object.entries(polsDef)[0][1][0]['polDeg']; 

    let pFr = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2fn;
    const Fr = new F1Field(pFr);

    // Split the input in little-endian bytes 
    console.log(N);
    prepareInput256bits(input, N);
    let eqCalculates = [arithEq0.calculate, arithEq1.calculate, arithEq2.calculate, arithEq3.calculate, arithEq4.calculate];

    // Initialization
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < 16; j++) {            
            pols.x1[j].push(0n);
            pols.y1[j].push(0n);
            pols.x2[j].push(0n);
            pols.y2[j].push(0n);
            pols.x3[j].push(0n);
            pols.y3[j].push(0n);
            pols.q0[j].push(0n);           
            pols.q1[j].push(0n);
            pols.q2[j].push(0n);
            pols.s[j].push(0n);
            if (j < pols.carryL.length) pols.carryL[j].push(0n);
            if (j < pols.carryH.length) pols.carryH[j].push(0n);
            if (j < pols.selEq.length) pols.selEq[j].push(0n);
        }
    }
    let s, q0, q1, q2;
    for (let i = 0; i < input.length; i++) {
        // TODO: if not have x1, need to componse it

        let x1 = BigInt(input[i]["x1"]);
        let y1 = BigInt(input[i]["y1"]);        
        let x2 = BigInt(input[i]["x2"]);
        let y2 = BigInt(input[i]["y2"]);
        let x3 = BigInt(input[i]["x3"]);
        let y3 = BigInt(input[i]["y3"]);

        if (input[i].selEq1) {
            s = Fr.div(Fr.sub(y2, y1), Fr.sub(x2, x1));
            let pq0 = s * x2 - s * x1 - y2 + y1;
            q0 = -(pq0/pFr);
            if ((pq0 + pFr*q0) != 0n) {
                throw new Error(`For input ${i}, with the calculated q0 the residual is not zero (diff point)`);
            }
            q0 += 2n ** 258n;
        }
        else if (input[i].selEq2) {
            s = Fr.div(Fr.mul(3n, Fr.mul(x1, x1)), Fr.add(y1, y1));
            let pq0 = s * 2n * y1 - 3n * x1 * x1;
            q0 = -(pq0/pFr);
            if ((pq0 + pFr*q0) != 0n) {
                throw new Error(`For input ${i}, with the calculated q0 the residual is not zero (same point)`);
            }
            q0 += 2n ** 258n;
        }
        else {
            s = 0n;
            q0 = 0n;
        }

        if (input[i].selEq3) {
            let pq1 = s * s - x1 - x2 - x3;
            q1 = -(pq1/pFr);
            if ((pq1 + pFr*q1) != 0n) {
                throw new Error(`For input ${i}, with the calculated q1 the residual is not zero`);
            }
            q1 += 2n ** 258n;

            let pq2 = s * x1 - s * x3 - y1 - y3;
            q2 = -(pq2/pFr);
            if ((pq2 + pFr*q2) != 0n) {
                throw new Error(`For input ${i}, with the calculated q2 the residual is not zero`);
            }
            q2 += 2n ** 258n;
        }
        else {
            q1 = 0n;
            q2 = 0n;
        }
        input[i]['_s'] = to16bitsRegisters(s);
        input[i]['_q0'] = to16bitsRegisters(q0);
        input[i]['_q1'] = to16bitsRegisters(q1);
        input[i]['_q2'] = to16bitsRegisters(q2);
    }

    for (let i = 0; i < input.length; i++) {
        let offset = i * 32;
        for (let step = 0; step < 32; ++step) {
            for (let j = 0; j < 16; j++) {
                pols.x1[j][offset + step] = BigInt(input[i]["_x1"][j])
                pols.y1[j][offset + step] = BigInt(input[i]["_y1"][j])
                pols.x2[j][offset + step] = BigInt(input[i]["_x2"][j])
                pols.y2[j][offset + step] = BigInt(input[i]["_y2"][j])
                pols.x3[j][offset + step] = BigInt(input[i]["_x3"][j])
                pols.y3[j][offset + step] = BigInt(input[i]["_y3"][j])
                pols.s[j][offset + step]  = BigInt(input[i]["_s"][j])
                pols.q0[j][offset + step] = BigInt(input[i]["_q0"][j])
                pols.q1[j][offset + step] = BigInt(input[i]["_q1"][j])
                pols.q2[j][offset + step] = BigInt(input[i]["_q2"][j])
            }
            pols.selEq[0][offset + step] = BigInt(input[i].selEq0);
            pols.selEq[1][offset + step] = BigInt(input[i].selEq1);
            pols.selEq[2][offset + step] = BigInt(input[i].selEq2);
            pols.selEq[3][offset + step] = BigInt(input[i].selEq3);
        }
        let carry = [0n, 0n, 0n];
        const eqIndexToCarryIndex = [0, 0, 0, 1, 2];
        let eq = [0n, 0n , 0n, 0n, 0n]

        let eqIndexes = [];
        if (pols.selEq[0][offset]) eqIndexes.push(0); -> arith normal -> 0
        if (pols.selEq[1][offset]) eqIndexes.push(1); -> suma punts diferents -> 1, 3, 4
        if (pols.selEq[2][offset]) eqIndexes.push(2); -> 2, 3, 4
        if (pols.selEq[3][offset]) eqIndexes = eqIndexes.concat([3, 4]); -> 3, 4

        for (let step = 0; step < 32; ++step) {
            eqIndexes.forEach((eqIndex) => {
                let carryIndex = eqIndexToCarryIndex[eqIndex];
                eq[eqIndex] = eqCalculates[eqIndex](pols, step, offset);
                pols.carryL[carryIndex][offset + step] = ((carry[carryIndex]) % (2n**18n));
                pols.carryH[carryIndex][offset + step] = ((carry[carryIndex]) / (2n**18n));
                carry[carryIndex] = (eq[eqIndex] + carry[carryIndex]) / (2n ** 16n);
            });
        }
    }
}

function prepareInput256bits(input, N) {
    for (let i = 0; i < input.length; i++) {
        for (var key of Object.keys(input[i])) {
            input[i][`_${key}`] = to16bitsRegisters(input[i][key]);
        }
    }
}

function to16bitsRegisters(value) {
    if (typeof value !== 'bigint') {
        value = BigInt(value);
    }

    let parts = [];
    for (let part = 0; part < 16; ++part) {
        parts.push(value & (part < 15 ? 0xFFFFn:0xFFFFFn));
        value = value >> 16n;
    }
    return parts;
}

*/