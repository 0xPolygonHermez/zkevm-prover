#ifndef ARITH_ACTION_HPP
#define ARITH_ACTION_HPP

#include <gmpxx.h>
#include "zkglobals.hpp"
#include "arith_equation.hpp"

class ArithAction
{
public:
    Goldilocks::Element x1[8];
    Goldilocks::Element y1[8];
    Goldilocks::Element x2[8];
    Goldilocks::Element y2[8];
    Goldilocks::Element x3[8];
    Goldilocks::Element y3[8];
    uint64_t equation;

    ArithAction() : equation(0)
    {
        for (uint64_t i=0; i<8; i++)
        {
            x1[i] = fr.zero();
            y1[i] = fr.zero();
            x2[i] = fr.zero();
            y2[i] = fr.zero();
            x3[i] = fr.zero();
            y3[i] = fr.zero();
        }
    };

    string toString (void)
    {
        return
            "x1=" + fea2stringchain(fr, x1[0], x1[1], x1[2], x1[3], x1[4], x1[5], x1[6], x1[7]) +
            " y1=" + fea2stringchain(fr, y1[0], y1[1], y1[2], y1[3], y1[4], y1[5], y1[6], y1[7]) +
            " x2=" + fea2stringchain(fr, x2[0], x2[1], x2[2], x2[3], x2[4], x2[5], x2[6], x2[7]) +
            " y2=" + fea2stringchain(fr, y2[0], y2[1], y2[2], y2[3], y2[4], y2[5], y2[6], y2[7]) +
            " x3=" + fea2stringchain(fr, x3[0], x3[1], x3[2], x3[3], x3[4], x3[5], x3[6], x3[7]) +
            " y3=" + fea2stringchain(fr, y3[0], y3[1], y3[2], y3[3], y3[4], y3[5], y3[6], y3[7]) +
            " equation=" + to_string(equation) + "=" + arith2string(equation);
    }
};

#endif