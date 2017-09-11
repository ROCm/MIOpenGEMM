/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#ifndef GUARD_PAR_FOR_HPP
#define GUARD_PAR_FOR_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#ifdef __MINGW32__
#include <mingw.thread.h>
#else
#include <thread>
#endif

namespace MIOpenGEMM {

struct joinable_thread : std::thread
{
    template <class... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...)
    {
    }

    joinable_thread& operator=(joinable_thread&& other) = default;
    joinable_thread(joinable_thread&& other)            = default;

    ~joinable_thread()
    {
        if(this->joinable())
            this->join();
    }
};

struct thread_factory
{
    template <class F>
    joinable_thread operator()(std::size_t& work, std::size_t n, std::size_t grainsize, F f) const
    {
        auto result = joinable_thread([=] {
            std::size_t start = work;
            std::size_t last  = std::min(n, work + grainsize);
            for(std::size_t i = start; i < last; i++)
            {
                f(i);
            }
        });
        work += grainsize;
        return result;
    }
};

template <class F>
void par_for_impl(std::size_t n, std::size_t threadsize, F f)
{
    if(threadsize <= 1)
    {
        for(std::size_t i = 0; i < n; i++)
            f(i);
    }
    else
    {
        std::vector<joinable_thread> threads(threadsize);
        const std::size_t grainsize = std::ceil(static_cast<double>(n) / threads.size());

        std::size_t work = 0;
        std::generate(threads.begin(),
                      threads.end(),
                      std::bind(thread_factory{}, std::ref(work), n, grainsize, f));
        assert(work >= n);
    }
}

template <class F>
void par_for(std::size_t n, std::size_t min_grain, F f)
{
    const auto threadsize =
        std::min<std::size_t>(std::thread::hardware_concurrency(), n / min_grain);
    par_for_impl(n, threadsize, f);
}

template <class F>
void par_for(std::size_t n, F f)
{
    const int min_grain = 8;
    par_for(n, min_grain, f);
}

}

#endif
