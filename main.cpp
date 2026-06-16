#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "ticktock.h"

#include <tbb/tbb.h>

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func)
{
    TICK(fill);
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, arr.size()),
        [&](tbb::blocked_range<size_t> r)
        {
            for (size_t i = r.begin(); i < r.end(); i++)
            {
                arr[i] = func(i);
            }
        });
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y)
{
    TICK(saxpy);
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, x.size()),
        [&](tbb::blocked_range<size_t> r)
        {
            for (size_t i = r.begin(); i < r.end(); i++)
            {
                x[i] = a * x[i] + y[i];
            }
        });
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y)
{
    TICK(sqrtdot);
    // parallel_reduce(range, identity, body, combiner)
    T ret = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())),
        T(0),
        [&](tbb::blocked_range<size_t> r, T local) { // T local is local accumulator for every thread
            for (size_t i = r.begin(); i < r.end(); i++)
            {
                local += x[i] * y[i];
            }
            return local;
        },
        std::plus<T>());
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x)
{
    TICK(minvalue);
    T ret = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, x.size()),
        std::numeric_limits<T>::max(), // set to infinity
        [&](tbb::blocked_range<size_t> r, T local)
        {
            for (size_t i = r.begin(); i < r.end(); i++)
            {
                if (x[i] < local)
                {
                    local = x[i];
                }
            }
            return local;
        },
        [](T a, T b)
        { return std::min(a, b); });
    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y)
{
    TICK(magicfilter);
    std::vector<T> res;
    std::mutex mtx;
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())),
        [&](tbb::blocked_range<size_t> r)
        {
            std::vector<T> local;
            local.reserve(r.size()*2);
            for (size_t i = r.begin(); i < r.end(); i++)
            {
                if (x[i] > y[i])
                {
                    local.push_back(x[i]);
                }
                else if (y[i] > x[i] && y[i] > 0.5f)
                {
                    local.push_back(y[i]);
                    local.push_back(x[i] * y[i]);
                }
            }
            std::lock_guard<std::mutex> lock(mtx);
            res.insert(res.end(), local.begin(), local.end());
        });
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x)
{
    TICK(scanner);
    T ret=tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, x.size()),
        T(0),
        [&](tbb::blocked_range<size_t> r, T sum, bool is_final)
        {
            T tmp = sum;
            for (size_t i = r.begin(); i < r.end(); i++)
            {
                tmp += x[i];
                if (is_final)
                    x[i] = tmp;
            }
            return tmp;
        },
        std::plus<T>()
    );
    TOCK(scanner);
    return ret;
}


int main()
{
    size_t n = 1 << 26;
    std::vector<float> x(n);
    std::vector<float> y(n);

    fill(x, [&](size_t i)
         { return std::sin(i); });
    fill(y, [&](size_t i)
         { return std::cos(i); });

    saxpy(0.5f, x, y);

    std::cout << sqrtdot(x, y) << std::endl;
    std::cout << minvalue(x) << std::endl;

    auto arr = magicfilter(x, y);
    std::cout << arr.size() << std::endl;

    scanner(x);
    std::cout << std::reduce(x.begin(), x.end()) << std::endl;

    return 0;
}
