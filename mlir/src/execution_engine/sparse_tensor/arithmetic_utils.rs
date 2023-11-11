/*!
This header is not part of the public API. It is placed in the includes directory only because that's required by the implementations of template-classes.

This file is part of the lightweight runtime support library for sparse tensor manipulations.  The functionality of the support library is meant to simplify benchmarking, testing, and debugging MLIR code operating on sparse tensors.  However, the provided functionality is **not** part of core MLIR itself.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/ExecutionEngine/SparseTensor/ArithmeticUtils.h>
*/

/*
Safe comparison functions.

Variants of the `==`, `!=`, `<`, `<=`, `>`, and `>=` operators which are careful to ensure that negatives are always considered strictly less than non-negatives regardless of the signedness of the types of the two arguments.  They are 'safe' in that they guarantee to *always* give an output and that that output is correct; in particular this means they never use assertions or other mechanisms for 'returning an error'.

These functions are C++17-compatible backports of the safe comparison functions added in C++20, and the implementations are based on the sample implementations provided by the standard: <https://en.cppreference.com/w/cpp/utility/intcmp>.
*/

// // #[noexcept]
// pub const fn safely_eq<T, U>(t: T, u: U) -> bool {
//     using UT = std::make_unsigned_t<T>;
//     using UU = std::make_unsigned_t<U>;
//     if constexpr (std::is_signed_v<T> == std::is_signed_v<U>) {
//         return t == u;
//     } else if constexpr (std::is_signed_v<T>) {
//         return t < 0 ? false : static_cast<UT>(t) == u;
//     } else {
//         return u < 0 ? false : t == static_cast<UU>(u);
//     }
// }

/*
Overflow checking functions.

These functions use assertions to ensure correctness with respect to overflow/underflow.  Unlike the 'safe' functions above, these 'checked' functions only guarantee that *if* they return an answer then that answer is correct.  When assertions are enabled, they do their best to remain as fast as possible (since MLIR keeps assertions enabled by default, even for optimized builds).  When assertions are disabled, they use the standard unchecked implementations.
*/

/*
TODO: we would like to be able to pass in custom error messages, to improve the user experience. We should be able to use something like `assert(((void)(msg ? msg : defaultMsg), cond))`; but I'm not entirely sure that'll work as intended when done within a function-definition rather than within a macro-definition.
*/

// /**
// A version of `static_cast<To>` which checks for overflow/underflow.
// The implementation avoids performing runtime assertions whenever
// the types alone are sufficient to statically prove that overflow
// cannot happen.
// */
// #[must_use]
// #[inline]
// pub fn check_overflow_cast<To, From>(x: From) -> To {
//     // Check the lower bound. (For when casting from signed types.)
//     let min_to = To::MIN;
//     let min_from = From::MIN;
//     if constexpr (!safely_ge(min_from, min_to)) {
//         assert!(safely_ge(x, min_to), "Cast would underflow.");
//     }
//     // Check the upper bound.
//     let max_to = To::MAX;
//     let max_from = From::MAX;
//     if constexpr (!safely_le(max_from, max_to)) {
//         assert!(safely_le(x, max_to), "Cast would overflow.");
//     }
//     // Now do the cast itself.
//     return static_cast<To>(x);
// }
