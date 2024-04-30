#ifndef H1H2_HINT_H
#define H1H2_HINT_H

#include "hint_handler.hpp"
#include "hint_handler_builder.hpp"

namespace Hints
{
    class H1H2HintHandler : public HintHandler
    {
    public:
        // Return the name of the hint
        static std::string getName();

        // Return the source names of the hint, so the fields needed to resolve the hint
        virtual std::vector<std::string> getSources() const override;

        // Return the destination names of the hint, so the fields that will be updated
        virtual std::vector<std::string> getDestinations() const override;

        // Returns the extra memory needed in bytes to resolve the hint
        virtual size_t getMemoryNeeded(uint64_t N) const override;

        // Resolve the hint
        virtual void resolveHint(int N, StepsParams &params, Hint hint, const std::map<std::string, Polinomial *> &polynomials, void *ptr_extra_mem) const override;
        
        void calculateH1H2(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol) const;
   
        void calculateH1H2_(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol, uint64_t pNumber) const;

        void calculateH1H2_opt1(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol, uint64_t pNumber, uint64_t *buffer, uint64_t size_keys) const;

        void calculateH1H2_opt3(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol, uint64_t pNumber, uint64_t *buffer, uint64_t size_keys) const;
    };

    class H1H2HintHandlerBuilder : public HintHandlerBuilder
    {
    public:
        std::shared_ptr<HintHandler> build() const override;
    };
}

#endif
