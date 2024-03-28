// #include <iostream>
// #include <vector>
// #include <string>
// #include <memory>
// #include "hint.hpp"
// #include "hint_builder.hpp"
// #include "h1h2_hint.hpp"

// using namespace Hints;

// int main()
// {
//     using namespace Hints;

//     // Register H1H2HintBuilder
//     HintBuilder::registerBuilder("h1h2", std::make_unique<H1H2HintBuilder>());

//     // Create H1H2Hint using the HintBuilder
//     std::unique_ptr<Hint> hint = HintBuilder::create("h1h2")->build();

//     // Use the Hint object as before
//     std::cout << "Name: " << hint->getName() << std::endl;

//     std::vector<std::string> polynomials = {"x^2", "2x+3"};
//     // hint->resolveHint(5, polynomials);

//     std::cout << "Fields: ";
//     for (const auto &field : hint->getFields())
//     {
//         std::cout << field << " ";
//     }
//     std::cout << std::endl;

//     std::cout << "Destinations: ";
//     for (const auto &destination : hint->getDestination())
//     {
//         std::cout << destination << " ";
//     }
//     std::cout << std::endl;

//     return 0;
// }
