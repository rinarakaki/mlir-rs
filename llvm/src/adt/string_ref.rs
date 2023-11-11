/*!
- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/StringRef.h>
- manual <https://llvm.org/docs/ProgrammersManual.html#llvm-adt-stringref-h>
*/
  
// use crate::llvm::{
//     adt::{
//         dense_map_info,
//         stl_functional_extras,
//         iterator_range
//     },
//     support::compiler
// };
    
pub type StringRef = &'static str;

// /**
// StringRef - Represent a constant reference to a string, i.e. a character
// array and a length, which need not be null terminated.

// This class does not own the string data, it is expected to be used in
// situations where the character data resides in some other buffer, whose
// lifetime extends past that of the StringRef. For this reason, it is not in
// general safe to store a StringRef.
// */
// #[derive(Clone, Copy)]
// pub struct StringRef {
// }
