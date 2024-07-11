#include "h1h2_hint_handler.hpp"

namespace Hints
{
    std::string H1H2HintHandler::getName()
    {
        return "h1h2";
    }

    std::vector<std::string> H1H2HintHandler::getSources() const
    {
        return {"f", "t"};
    }

    std::vector<std::string> H1H2HintHandler::getDestinations() const
    {
        return {"referenceH1", "referenceH2"};
    }

    void H1H2HintHandler::resolveHint(int N, StepsParams &params, Hint hint, const std::map<std::string, Polinomial *> &polynomials) const
    {
        Goldilocks::Element* ptr_extra_mem = new Goldilocks::Element[8*N];

        assert(polynomials.size() == 4);

        auto h1 = polynomials.find("referenceH1");
        auto h2 = polynomials.find("referenceH2");
        auto f = polynomials.find("f");
        auto t = polynomials.find("t");

        assert(h1 != polynomials.end());
        assert(h2 != polynomials.end());
        assert(f != polynomials.end());
        assert(t != polynomials.end());

        auto h1Pol = *h1->second;
        auto h2Pol = *h2->second;
        auto fPol = *f->second;
        auto tPol = *t->second;

        assert(h1Pol.dim() == 1 || h1Pol.dim() == 3);
        assert(h1Pol.dim() == h2Pol.dim());
        assert(h1Pol.dim() == fPol.dim());
        assert(h1Pol.dim() == tPol.dim());

        if (h1Pol.dim() == 1)
        {
            calculateH1H2_opt1(h1Pol, h2Pol, fPol, tPol, 0, (uint64_t *) ptr_extra_mem, 5 * N);
        }
        else if (h1Pol.dim() == 3)
        {
            calculateH1H2_opt3(h1Pol, h2Pol, fPol, tPol, 0, (uint64_t *) ptr_extra_mem, 3 * N);
        }

        delete ptr_extra_mem;
    }

    void H1H2HintHandler::calculateH1H2(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol) const
    {
        map<std::vector<Goldilocks::Element>, uint64_t, CompareFe> idx_t;
        multimap<std::vector<Goldilocks::Element>, uint64_t, CompareFe> s;
        multimap<std::vector<Goldilocks::Element>, uint64_t>::iterator it;
        uint64_t i = 0;

        for (uint64_t i = 0; i < tPol.degree(); i++)
        {
            vector<Goldilocks::Element> key = tPol.toVector(i);
            std::pair<vector<Goldilocks::Element>, uint64_t> pr(key, i);

            auto const result = idx_t.insert(pr);
            if (not result.second)
            {
                result.first->second = i;
            }

            s.insert(pr);
        }

        for (uint64_t i = 0; i < fPol.degree(); i++)
        {
            vector<Goldilocks::Element> key = fPol.toVector(i);

            if (idx_t.find(key) == idx_t.end())
            {
                zklog.error("Polinomial::calculateH1H2() Number not included: " + Goldilocks::toString(fPol[i], 16));
                exitProcess();
            }
            uint64_t idx = idx_t[key];
            s.insert(pair<vector<Goldilocks::Element>, uint64_t>(key, idx));
        }

        multimap<uint64_t, vector<Goldilocks::Element>> s_sorted;
        multimap<uint64_t, vector<Goldilocks::Element>>::iterator it_sorted;

        for (it = s.begin(); it != s.end(); it++)
        {
            s_sorted.insert(make_pair(it->second, it->first));
        }

        for (it_sorted = s_sorted.begin(); it_sorted != s_sorted.end(); it_sorted++, i++)
        {
            if ((i & 1) == 0)
            {
                Polinomial::copyElement(h1, i / 2, it_sorted->second);
            }
            else
            {
                Polinomial::copyElement(h2, i / 2, it_sorted->second);
            }
        }
    };

    void H1H2HintHandler::calculateH1H2_(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol, uint64_t pNumber) const
    {
        map<std::vector<Goldilocks::Element>, uint64_t, CompareFe> idx_t;
        multimap<std::vector<Goldilocks::Element>, uint64_t, CompareFe> s;
        multimap<std::vector<Goldilocks::Element>, uint64_t>::iterator it;

        vector<int> counter(tPol.degree(), 1);

        for (uint64_t i = 0; i < tPol.degree(); i++)
        {
            vector<Goldilocks::Element> key = tPol.toVector(i);
            idx_t[key] = i + 1;
        }

        for (uint64_t i = 0; i < fPol.degree(); i++)
        {
            vector<Goldilocks::Element> key = fPol.toVector(i);
            uint64_t indx = idx_t[key];
            if (indx == 0)
            {
                zklog.error("Polinomial::calculateH1H2() Number not included: w=" + to_string(i) + " plookup_number=" + to_string(pNumber) + "\nPol:" + Goldilocks::toString(fPol[i], 16));
                exitProcess();
            }
            ++counter[indx - 1];
        }

        uint64_t id = 0;
        for (u_int64_t i = 0; i < tPol.degree(); ++i)
        {
            if (counter[id] == 0)
            {
                ++id;
            }
            counter[id] -= 1;
            Polinomial::copyElement(h1, i, tPol, id);

            if (counter[id] == 0)
            {
                ++id;
            }
            counter[id] -= 1;
            Polinomial::copyElement(h2, i, tPol, id);
        }
    }

    void H1H2HintHandler::calculateH1H2_opt1(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol, uint64_t pNumber, uint64_t *buffer, uint64_t size_keys) const
    {
        vector<int> counter(tPol.degree(), 1);  // this 1 is important, space of the original buffer could be used
        vector<bool> touched(size_keys, false); // faster use this than initialize buffer, bitmask could be used
        uint32_t pos = 0;

        // double time1 = omp_get_wtime();
        for (uint64_t i = 0; i < tPol.degree(); i++)
        {
            uint64_t key = tPol.firstValueU64(i);
            uint64_t ind = key % size_keys;
            if (!touched[ind])
            {
                buffer[ind] = pos;
                uint32_t offset = size_keys + 3 * pos;
                buffer[offset] = key;
                buffer[offset + 1] = i;
                buffer[offset + 2] = 0;
                pos += 1;
                touched[ind] = true;
            }
            else
            {
                uint64_t pos_ = buffer[ind];
                bool exit_ = false;
                do
                {
                    uint32_t offset = size_keys + 3 * pos_;
                    if (key == buffer[offset])
                    {
                        buffer[offset + 1] = i;
                        exit_ = true;
                    }
                    else
                    {
                        if (buffer[offset + 2] != 0)
                        {
                            pos_ = buffer[offset + 2];
                        }
                        else
                        {
                            buffer[offset + 2] = pos;
                            // new offset
                            offset = size_keys + 3 * pos;
                            buffer[offset] = key;
                            buffer[offset + 1] = i;
                            buffer[offset + 2] = 0;
                            pos += 1;
                            exit_ = true;
                        }
                    }
                } while (!exit_);
            }
        }

        // double time2 = omp_get_wtime();

        for (uint64_t i = 0; i < fPol.degree(); i++)
        {
            uint64_t indx = 0;
            uint64_t key = fPol.firstValueU64(i);
            uint64_t ind = key % size_keys;
            if (!touched[ind])
            {
                zklog.error("Polynomial::calculateH1H2() Number not included: w=" + to_string(i) + " plookup_number=" + to_string(pNumber) + "\nPol:" + Goldilocks::toString(fPol[i], 16));
                exitProcess();
            }
            uint64_t pos_ = buffer[ind];
            bool exit_ = false;
            do
            {
                uint32_t offset = size_keys + 3 * pos_;
                if (key == buffer[offset])
                {
                    indx = buffer[offset + 1];
                    exit_ = true;
                }
                else
                {
                    if (buffer[offset + 2] != 0)
                    {
                        pos_ = buffer[offset + 2];
                    }
                    else
                    {
                        zklog.error("Polynomial::calculateH1H2() Number not included: w=" + to_string(i) + " plookup_number=" + to_string(pNumber) + "\nPol:" + Goldilocks::toString(fPol[i], 16));
                        exitProcess();
                    }
                }
            } while (!exit_);
            ++counter[indx];
        }

        // double time3 = omp_get_wtime();
        uint64_t id = 0;
        for (u_int64_t i = 0; i < tPol.degree(); ++i)
        {
            if (counter[id] == 0)
            {
                ++id;
            }

            counter[id] -= 1;
            Polinomial::copyElement(h1, i, tPol, id);

            if (counter[id] == 0)
            {
                ++id;
            }
            counter[id] -= 1;
            Polinomial::copyElement(h2, i, tPol, id);
        }
        // double time4 = omp_get_wtime();
        // std::cout << "holu: " << id << " " << pos << " times: " << time2 - time1 << " " << time3 - time2 << " " << time4 - time3 << " " << h2.dim() << std::endl;
    }

    void H1H2HintHandler::calculateH1H2_opt3(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol, uint64_t pNumber, uint64_t *buffer, uint64_t size_keys) const
    {
        vector<int> counter(tPol.degree(), 1);  // this 1 is important, space of the original buffer could be used
        vector<bool> touched(size_keys, false); // faster use this than initialize buffer, bitmask could be used
        uint32_t pos = 0;
        uint64_t key[3];

        // double time1 = omp_get_wtime();
        for (uint64_t i = 0; i < tPol.degree(); i++)
        {
            tPol.toVectorU64(i, key);
            uint64_t ind = key[0] % size_keys;
            if (!touched[ind])
            {
                buffer[ind] = pos;
                uint32_t offset = size_keys + 5 * pos;
                buffer[offset] = key[0];
                buffer[offset + 1] = key[1];
                buffer[offset + 2] = key[2];
                buffer[offset + 3] = i;
                buffer[offset + 4] = 0;
                pos += 1;
                touched[ind] = true;
            }
            else
            {
                uint64_t pos_ = buffer[ind];
                bool exit_ = false;
                do
                {
                    uint32_t offset = size_keys + 5 * pos_;
                    if (key[0] == buffer[offset] && key[1] == buffer[offset + 1] && key[2] == buffer[offset + 2])
                    {
                        buffer[offset + 3] = i;
                        exit_ = true;
                    }
                    else
                    {
                        if (buffer[offset + 4] != 0)
                        {
                            pos_ = buffer[offset + 4];
                        }
                        else
                        {
                            buffer[offset + 4] = pos;
                            // new offset
                            offset = size_keys + 5 * pos;
                            buffer[offset] = key[0];
                            buffer[offset + 1] = key[1];
                            buffer[offset + 2] = key[2];
                            buffer[offset + 3] = i;
                            buffer[offset + 4] = 0;
                            pos += 1;
                            exit_ = true;
                        }
                    }
                } while (!exit_);
            }
        }

        // double time2 = omp_get_wtime();

        for (uint64_t i = 0; i < fPol.degree(); i++)
        {
            uint64_t indx = 0;
            fPol.toVectorU64(i, key);
            uint64_t ind = key[0] % size_keys;
            if (!touched[ind])
            {
                zklog.error("Polinomial::calculateH1H2() Number not included: w=" + to_string(i) + " plookup_number=" + to_string(pNumber) + "\nPol:" + Goldilocks::toString(fPol[i], 16));
                exitProcess();
            }
            uint64_t pos_ = buffer[ind];
            bool exit_ = false;
            do
            {
                uint32_t offset = size_keys + 5 * pos_;
                if (key[0] == buffer[offset] && key[1] == buffer[offset + 1] && key[2] == buffer[offset + 2])
                {
                    indx = buffer[offset + 3];
                    exit_ = true;
                }
                else
                {
                    if (buffer[offset + 4] != 0)
                    {
                        pos_ = buffer[offset + 4];
                    }
                    else
                    {
                        zklog.error("Polinomial::calculateH1H2() Number not included: w=" + to_string(i) + " plookup_number=" + to_string(pNumber) + "\nPol:" + Goldilocks::toString(fPol[i], 16));
                        exitProcess();
                    }
                }
            } while (!exit_);
            ++counter[indx];
        }

        // double time3 = omp_get_wtime();
        uint64_t id = 0;
        for (u_int64_t i = 0; i < tPol.degree(); ++i)
        {
            if (counter[id] == 0)
            {
                ++id;
            }

            counter[id] -= 1;
            Polinomial::copyElement(h1, i, tPol, id);

            if (counter[id] == 0)
            {
                ++id;
            }
            counter[id] -= 1;
            Polinomial::copyElement(h2, i, tPol, id);
        }
        // double time4 = omp_get_wtime();
        // std::cout << "holu: " << id << " " << pos << " times: " << time2 - time1 << " " << time3 - time2 << " " << time4 - time3 << " " << h2.dim() << std::endl;
    }

    std::shared_ptr<HintHandler> H1H2HintHandlerBuilder::build() const
    {
        return std::make_unique<H1H2HintHandler>();
    }
}