#include "subproof_value_hint_handler.hpp"

namespace Hints
{
    std::string SubproofValueHintHandler::getName()
    {
        return "subproofValue";
    }

    std::vector<std::string> SubproofValueHintHandler::getSources() const
    {
        return {"expression"};
    }

    std::vector<std::string> SubproofValueHintHandler::getDestinations() const
    {
        return {"reference"};
    }

    void SubproofValueHintHandler::resolveHint(int N, StepsParams &params, Hint hint, const std::map<std::string, Polinomial *> &polynomials) const
    {
        assert(polynomials.size() == 1);

        auto expression = polynomials.find("expression");
        assert(expression != polynomials.end());

        auto expressionPol = *expression->second;

        uint64_t row_index = hint.fields["row_index"].value;
        uint64_t subproofValueId = hint.fields["reference"].id;

        params.subproofValues[subproofValueId * FIELD_EXTENSION] = expressionPol[row_index][0];
        params.subproofValues[subproofValueId * FIELD_EXTENSION + 1] = expressionPol[row_index][1];
        params.subproofValues[subproofValueId * FIELD_EXTENSION + 2] = expressionPol[row_index][2];
    }

    std::shared_ptr<HintHandler> SubproofValueHintHandlerBuilder::build() const
    {
        return std::make_unique<SubproofValueHintHandler>();
    }
}