#ifndef _MOLSIM_UTIL_H
#define _MOLSIM_UTIL_H

#include <cstddef>
#include <initializer_list>
#include <stdlib.h>
#include <vector>
#include <print>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#if defined(__AVX512F__)
#define MOLSIM_ALIGN 512
#elif defined(__ARM_NEON) || defined(__AVX2__)
#define MOLSIM_ALIGN 256
#endif

#define ERROR(...) \
do { \
    std::print(stderr, "{}({}): error: ", __FILE__, __LINE__); \
    std::print(stderr, __VA_ARGS__); \
    std::print(stderr, "\n"); \
    exit(1); \
} while(0);

template <typename T, size_t AlignedAs>
class AlignedAllocator
{
    public:
        using value_type = T;
        using size_type = std::ptrdiff_t;
        using difference_type = std::ptrdiff_t;

        AlignedAllocator() = default;

        constexpr AlignedAllocator(const AlignedAllocator&) noexcept {}

        [[nodiscard]] T* allocate(size_type n)
        {
            void* allocation = nullptr;
            if (n > std::allocator_traits<AlignedAllocator>::max_size(*this))
                throw std::bad_alloc();
            static_assert(AlignedAs % sizeof(void *) == 0);
            int err = posix_memalign(&allocation, AlignedAs, n*sizeof(T));
            if (allocation && !err)
                return static_cast<T*>(allocation);
            else
                throw std::bad_alloc();
        }

        void deallocate(T* p, std::size_t) noexcept
        {
            std::free(p);
        }

        template<typename _Tp1>
        struct rebind { typedef AlignedAllocator<_Tp1, AlignedAs> other; };

};

template <class T, size_t Tsize, class U, size_t Usize>
bool operator==(const AlignedAllocator<T, Tsize>& a, const AlignedAllocator<U, Usize>& b) { return true; }

template <class T, size_t Tsize, class U, size_t Usize>
bool operator!=(const AlignedAllocator<T, Tsize>& a, const AlignedAllocator<U, Usize>& b) { return false; }

template <typename T>
class AlignedVector
{
    /*
        Wrapper class for std::vector as of C++20, forcing storage optimally aligned for SIMD access and signed sizes.
        Only partially implemented for now; will add functionality as needed.
    */
    using value_type = T;
    using size_type = std::ptrdiff_t;
    using difference_type = std::ptrdiff_t;

    using iterator = typename std::vector<T, AlignedAllocator<T,MOLSIM_ALIGN>>::iterator;
    using const_iterator = typename std::vector<T, AlignedAllocator<T,MOLSIM_ALIGN>>::const_iterator;
    using reverse_iterator = typename std::vector<T>::reverse_iterator;
    using const_reverse_iterator = typename std::vector<T>::const_reverse_iterator;


    private:
        std::vector<T, AlignedAllocator<T, MOLSIM_ALIGN>> _v;

    public:        
        constexpr AlignedVector(size_type count, const T& value) { _v = std::vector<T, AlignedAllocator<T, MOLSIM_ALIGN>>(count, value); }
        
        template<class InputIt>
        constexpr AlignedVector(InputIt first, InputIt last) : _v(first, last) {}

        constexpr AlignedVector() noexcept : _v() { }
        explicit AlignedVector(size_type count) : _v(count) { }
        constexpr AlignedVector(const AlignedVector& other) : _v(other._v) { }
        constexpr AlignedVector(AlignedVector&& other) noexcept : _v(std::move(other._v)) { }
        constexpr AlignedVector(std::initializer_list<T> init) : _v(init) { }

        constexpr AlignedVector(const std::vector<T>& other) : _v(other._v) { }
        AlignedVector(const pybind11::array_t<T>& other) 
        {
            if (other.is_none()) _v = {};
            else _v = std::vector<T, AlignedAllocator<T, MOLSIM_ALIGN>>(other.data(), other.data()+other.size());
        }

        constexpr AlignedVector& operator=(const AlignedVector& other) { _v = other._v; return *this; }
        AlignedVector& operator=(AlignedVector&& other) noexcept { _v = std::move(other._v); return *this; }
        constexpr AlignedVector& operator=(std::initializer_list<T> ilist) { _v = ilist; return *this; }

        constexpr T& operator[](size_type pos) { return _v[pos]; }
        constexpr const T& operator[](size_type pos) const { return _v[pos]; }
        constexpr T& front(){ return _v.front(); }
        constexpr const T& front() const { return _v.front(); }
        T& back(){ return _v.back(); }
        const T& back() const { return _v.back(); }
        constexpr T* data() noexcept { return _v.data(); }
        constexpr const T* data() const { return _v.data(); }

        constexpr iterator begin() noexcept { return _v.begin(); }
        constexpr const_iterator begin() const noexcept { return _v.begin(); }
        constexpr const_iterator cbegin() const noexcept { return _v.cbegin(); }
        constexpr iterator end() noexcept { return _v.end(); }
        constexpr const_iterator end() const noexcept { return _v.end(); }
        constexpr const_iterator cend() const noexcept { return _v.cend(); }
        constexpr reverse_iterator rbegin() noexcept { return _v.rbegin(); }
        constexpr const_reverse_iterator rbegin() const noexcept { return _v.rbegin(); }
        constexpr const_reverse_iterator crbegin() const noexcept { return _v.crbegin(); }
        constexpr reverse_iterator rend() noexcept { return _v.rend(); }
        constexpr const_reverse_iterator rend() const noexcept { return _v.rend(); }
        constexpr const_reverse_iterator crend() const noexcept { return _v.crend(); }

        bool empty() const noexcept { return _v.empty(); }
        constexpr size_type size() const noexcept { return static_cast<size_type>(_v.size()); }
        constexpr size_type max_size() noexcept { return _v.max_size(); }
        constexpr void reserve(size_type new_cap) { return _v.reserve(new_cap); }
        constexpr size_type capacity() const noexcept { return static_cast<size_type>(_v.capacity()); }
        constexpr void shrink_to_fit() { _v.shrink_to_fit(); }

        constexpr void clear() noexcept { _v.clear(); }
        constexpr iterator insert(const_iterator pos, const T& value) { _v.insert(pos, value); }
        constexpr iterator insert(const_iterator pos, T&& value) { _v.insert(pos, std::move(value)); }
        constexpr iterator insert(const_iterator pos, size_type count, const T& value) { _v.insert(pos, count, value); }
        template <class InputIt>
        constexpr iterator insert(const_iterator pos, InputIt first, InputIt last) { _v.insert(pos, first, last); }
        constexpr iterator insert(const_iterator pos, std::initializer_list<T> ilist) { _v.insert(pos, ilist); }
        template <class... Args>
        constexpr iterator emplace(const iterator pos, Args&&... args) {_v.emplace(pos, std::forward<Args>(args)...); }
        iterator erase(iterator pos) { _v.erase(pos); }
        constexpr iterator erase(const_iterator pos) { _v.erase(pos); }
        iterator erase(iterator first, iterator last) { _v.erase(first, last); }
        constexpr iterator erase(const_iterator first, const_iterator last) { _v.erase(first, last); }
        constexpr void push_back(const T& value) { _v.push_back(value); }
        constexpr void push_back(T&& value) { _v.push_back(std::move(value)); }
        template <class... Args>
        T& emplace_back(Args&&... args) { return _v.emplace_back(std::forward<Args>(args)...); }
        constexpr void pop_back() { _v.pop_back(); }
        constexpr void resize(size_type count) { _v.resize(count); }
        constexpr void resize(size_type count, const T& value) { _v.resize(count, value); }
        constexpr void swap(AlignedVector& other) noexcept { _v.swap(other); }
};

template <class T>
constexpr bool operator==(const AlignedVector<T>& lhs,
                          const AlignedVector<T>& rhs) { return lhs._v == rhs._v; }

#endif
