/*!
- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/TypeName.h>
*/

pub use std::any::type_name;

// /**
// We provide a function which tries to compute the (demangled) name of a type
// statically.

// This routine may fail on some platforms or for particularly unusual types.
// Do not use it for anything other than logging and debugging aids. It isn't
// portable or dependendable in any real sense.

// The returned StringRef will point into a static storage duration string.
// However, it may not be null terminated and may be some strangely aligned
// inner substring of a larger string.
// */
// #[inline]
// pub fn get_type_name<DesiredTypeName>() -> &'static str {
//     if cfg!(__clang__) || cfg!(__GNUC__) {
//         let mut name = __PRETTY_FUNCTION__;

//         let key = "DesiredTypeName = ";
//         name = name.substr(name.find(key));
//         assert!(!name.is_empty(), "Unable to find the template parameter!");
//         name = name.drop_front(key.len());
      
//         assert!(name.endswith("]"), "Name doesn't end in the substitution key!");
//         name.drop_back(1)
//     } else {
//         "UNKNOWN_TYPE"
//     }
// }
