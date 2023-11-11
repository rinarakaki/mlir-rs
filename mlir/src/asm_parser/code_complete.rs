/*!
- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/AsmParser/CodeComplete.h>
*/

use crate::{
    mlir::support::llvm,
    llvm::{
        adt::{
            string_map::StringMap
        },
        support::{
            sm_loc::SMLoc,
            source_manager
        }
    }
};

/**
This class provides an abstract interface into the parser for hooking in
code completion events. This class is only really useful for providing
language tooling for MLIR, general clients should not need to use this
class.
*/
pub struct AsmParserCodeCompleteContext {
    /// The location used to code complete.
    code_complete_loc: SMLoc
}
