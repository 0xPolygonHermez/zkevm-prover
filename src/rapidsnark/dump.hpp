#ifndef __DUMP__DUMP_H__
#define __DUMP__DUMP_H__ 

#include <string>
#include <vector>

namespace Dump {
    template <typename Engine>
    class Dump {
       protected:
        Engine &E;
      public:
        bool showValues;
        Dump(Engine &engine):E(engine), showValues(false) {};
        void dump(const std::string &label, const u_int8_t *elements, int count);
        void dump(const std::string &label, u_int32_t *elements, int count);
        void dump(const std::string &label, u_int32_t element);
        void dump(const std::string &label, typename Engine::FrElement *elements, u_int32_t count);
        void dump(const std::string &label, std::vector<typename Engine::FrElement> &elements, u_int64_t offset = 0);
        void dump(const std::string &label, const typename Engine::FrElement &element);
        void dump(const std::string &label, std::vector<typename Engine::G1PointAffine> &points);
        void dump(const std::string &label, typename Engine::G1PointAffine &point);
        void dump(const std::string &label, typename Engine::G1Point &point);
        std::string getColorLiteHash(std::string &data);

      protected:
        void setShowValues ( bool value ) { showValues = value; };
        std::string getLiteHash(std::string &data);
        std::string getHash(const void *data, u_int32_t len);
    };
}

#include "dump.c.hpp"

#endif