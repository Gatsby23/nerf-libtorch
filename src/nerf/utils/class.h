#ifndef CLASS_H
#define CLASS_H
namespace nerf {
template <class T>
class NonCopyable {
    NonCopyable() = default;
    ~NonCopyable() = default;
    NonCopyable(const NonCopyable &) = delete;
    T &operator=(const T &) = delete;
    NonCopyable(const NonCopyable &&) = delete;
    T &operator=(const T &&) = delete;
};
}  // namespace nerf
#endif