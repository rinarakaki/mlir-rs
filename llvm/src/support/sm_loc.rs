/*!
This file declares the SMLoc class.  This class encapsulates a location in
source code for use in diagnostics.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/SMLoc.h>
*/

/// Represents a location in source code.
pub struct SMLoc {
}

/**
Represents a range in source code.

SMRange is implemented using a half-open range, as is the convention in C++.
In the string "abc", the range [1,3) represents the substring "bc", and the
range [2,2) represents an empty range between the characters "b" and "c".
*/
pub struct SMRange {
}
