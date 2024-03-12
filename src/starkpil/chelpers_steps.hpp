#ifndef CHELPERS_STEPS_HPP
#define CHELPERS_STEPS_HPP
#include "chelpers.hpp"
#include "steps.hpp"

class CHelpersSteps {
    public:
        virtual void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {};
};

class BlobInnerSteps : public CHelpersSteps {
public:
    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {
        // Calculate expressions
        zklog.info("Calculating expressions for BlobInnerSteps");
    }
};

class BlobInnerCompressorSteps : public CHelpersSteps {
public:
    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {
        // Calculate expressions
        zklog.info("Calculating expressions for BlobInnerCompressorSteps");
    }
};

class BlobInnerRecursive1Steps : public CHelpersSteps {
public:
    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {
        // Calculate expressions
        zklog.info("Calculating expressions for BlobInnerRecursive1Steps");
    }
};

class BlobOuterSteps : public CHelpersSteps {
public:
    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {
        // Calculate expressions
        zklog.info("Calculating expressions for BlobOuterSteps");
    }
};

class BlobOuterRecursive2Steps : public CHelpersSteps {
public:
    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {
        // Calculate expressions
        zklog.info("Calculating expressions for BlobOuterRecursive2Steps");
    }
};

#endif