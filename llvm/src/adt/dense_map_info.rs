/*!
This file defines DenseMapInfo traits for DenseMap.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/DenseMapInfo.h>
*/

pub struct DenseMapInfo<T> {
    _unimplemented: std::marker::PhantomData<T>,
}

// // Provide DenseMapInfo for all pointers. Come up with sentinel pointer values
// // that are aligned to alignof(T) bytes, but try to avoid requiring T to be
// // complete. This allows clients to instantiate DenseMap<T*, ...> with forward
// // declared key types. Assume that no pointer key type requires more than 4096
// // bytes of alignment.
// impl<T> DenseMapInfo<T*> {
//     /**
//     The following should hold, but it would require T to be complete:
//     static_assert(alignof(T) <= (1 << Log2MaxAlign),
//                   "DenseMap does not support pointer keys requiring more than "
//                   "Log2MaxAlign bits of alignment");
//     */
//     static constexpr Log2MaxAlign: usize = 12;

//     #[inline]
//     static T* get_empty_key() {
//         usize Val = static_cast<usize>(-1);
//         Val <<= Log2MaxAlign;
//         return reinterpret_cast<T*>(Val);
//     }

//     #[inline]
//     static inline T* get_tombstone_key() {
//         usize Val = static_cast<usize>(-2);
//         Val <<= Log2MaxAlign;
//         return reinterpret_cast<T*>(Val);
//     }

//     static unsigned get_hash_value(const T *PtrVal) {
//         return (unsigned((usize)PtrVal) >> 4) ^
//                 (unsigned((usize)PtrVal) >> 9);
//     }

//     static bool is_equal(const T *LHS, const T *RHS) { return LHS == RHS; }
// }

// // Provide DenseMapInfo for chars.
// impl DenseMapInfo<char> {
//     #[inline]
//     static char get_empty_key() { return ~0; }

//     #[inline]
//     static char get_tombstone_key() { return ~0 - 1; }
    
//     static unsigned get_hash_value(const char& Val) { return Val * 37U; }
  
//     static bool is_equal(const char &LHS, const char &RHS) {
//       return LHS == RHS;
//     }
// }

// // Provide DenseMapInfo for unsigned chars.
// impl DenseMapInfo<u8> {
//     static inline u8 get_empty_key() { return ~0; }
//     static inline u8 get_tombstone_key() { return ~0 - 1; }
//     static unsigned get_hash_value(const u8 &Val) { return Val * 37U; }
  
//     static bool is_equal(const u8 &LHS, const u8 &RHS) {
//       return LHS == RHS;
//     }
// };
  
// // Provide DenseMapInfo for unsigned shorts.
// impl DenseMapInfo<u16> {
//     #[inline]
//     static u16 get_empty_key() { u16::MAX }

//     #[inline]
//     static u16 get_tombstone_key() { return 0xFFFF - 1; }

//     static unsigned get_hash_value(const u16 &Val) { return Val * 37U; }

//     static bool is_equal(const u16 &LHS, const u16 &RHS) {
//         return LHS == RHS;
//     }
// };

// // Provide DenseMapInfo for unsigned ints.
// impl DenseMapInfo<unsigned> {
//     static inline unsigned get_empty_key() { return ~0U; }
//     static inline unsigned get_tombstone_key() { return ~0U - 1; }
//     static unsigned get_hash_value(const unsigned& Val) { return Val * 37U; }
  
//     static bool is_equal(const unsigned& LHS, const unsigned& RHS) {
//       return LHS == RHS;
//     }
// }
  
// // Provide DenseMapInfo for unsigned longs.
// impl DenseMapInfo<u32> {
    
//     static inline u32 get_empty_key() { return ~0UL; }
//     static inline u32 get_tombstone_key() { return ~0UL - 1L; }
  
//     static unsigned get_hash_value(const u32& Val) {
//       return (unsigned)(Val * 37UL);
//     }
  
//     static bool is_equal(const u32& LHS, const u32& RHS) {
//       return LHS == RHS;
//     }
// }
  
// // Provide DenseMapInfo for unsigned long longs.
// impl DenseMapInfo<u64> {
//     static inline u64 get_empty_key() { return ~0ULL; }
//     static inline u64 get_tombstone_key() { return ~0ULL - 1ULL; }
  
//     static unsigned get_hash_value(const u64& Val) {
//       return (unsigned)(Val * 37ULL);
//     }
  
//     static bool is_equal(const u64& LHS,
//                         const u64& RHS) {
//       return LHS == RHS;
//     }
// }

// // Provide DenseMapInfo for shorts.
// impl DenseMapInfo<i16> {
//     #[inline]
//     static i16 get_empty_key() { i16::MAX }

//     #[inline]
//     static i16 get_tombstone_key() { i16::MIN }

//     static unsigned get_hash_value(const i16 &Val) { return Val * 37U; }

//     static bool is_equal(const i16 &LHS, const i16 &RHS) { return LHS == RHS; }
// }
  
// // Provide DenseMapInfo for ints.
// impl DenseMapInfo<i32> {
//     #[inline]
//     static i32 get_empty_key() {
//         i32::MAX
//     }

//     #[inline]
//     static i32 get_tombstone_key() {
//         i32::MIN
//     }

//     static unsigned get_hash_value(const i32& Val) {
//         (unsigned)(Val * 37U)
//     }
  
//     static bool is_equal(const i32& LHS, const i32& RHS) {
//       return LHS == RHS;
//     }
// }

// // Provide DenseMapInfo for long longs.
// impl DenseMapInfo<i64> {
//     #[inline]
//     static i64 get_empty_key() { i64::MAX }

//     #[inline]
//     static i64 get_tombstone_key() { i64::MIX }
  
//     static unsigned get_hash_value(const i64& Val) {
//       return (unsigned)(Val * 37ULL);
//     }
  
//     static bool is_equal(const i64& LHS, const i64& RHS) {
//         return LHS == RHS;
//     }
// }
