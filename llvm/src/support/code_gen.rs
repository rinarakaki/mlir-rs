/*!
This file define some types which define code generation concepts. For example, relocation model.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/CodeGen.h>
*/

use std::default;

/// Relocation model types.
// Cannot be named PIC due to collision with -DPIC
pub enum Relocation {
    Static,
    PIC_,
    DynamicNoPIC,
    ROPI,
    RWPI,
    ROPI_RWPI
}

/// Code model types.
// Sync changes with CodeGenCWrappers.h.
pub enum CodeModel {
    Tiny,
    Small,
    Kernel,
    Medium,
    Large
}

/// This is used to map -fpic/-fPIC.
pub enum PICLevel {
    NotPIC = 0,
    SmallPIC = 1,
    BigPIC = 2
}

pub enum PIELevel {
    Default = 0,
    Small = 1,
    Large = 2
}

/// TLS models.
pub enum TLSModel {
    GeneralDynamic,
    LocalDynamic,
    InitialExec,
    LocalExec
}

//// Code generation optimisation level.
#[derive(Default)]
pub enum CodeGenOptLevel {
    None = 0,      // -O0
    Less = 1,      // -O1
    #[default]
    Default = 2,   // -O2, -Os
    Aggressive = 3 // -O3
}

/**
These enums are meant to be passed into addPassesToEmitFile to indicate what type of file to emit, and returned by it to indicate what type of file could actually be made.
*/
pub enum CodeGenFileType {
    AssemblyFile,
    ObjectFile,
    Null         // Do not emit any output.
}

/// Specify what functions should keep the frame pointer.
pub enum FramePointerKind {
    None,
    NonLeaf,
    All
}

const ONLY_USED: usize = 1usize << 1;
const ONLY_GPR: usize = 1usize << 2;
const ONLY_ARG: usize = 1usize << 3;

#[repr(align(8))]  // usize int
pub enum ZeroCallUsedRegsKind {
  /// Don't zero any call-used regs.
  Skip = (1 as usize) << 0,
  /// Only zeros call-used GPRs used in the fn and pass args.
  UsedGPRArg = ONLY_USED | ONLY_GPR | ONLY_ARG,
  /// Only zeros call-used GPRs used in the fn.
  UsedGPR = ONLY_USED | ONLY_GPR,
  /// Only zeros call-used regs used in the fn and pass args.
  UsedArg = ONLY_USED | ONLY_ARG,
  /// Only zeros call-used regs used in the fn.
  Used = ONLY_USED,
  /// Zeros all call-used GPRs that pass args.
  AllGPRArg = ONLY_GPR | ONLY_ARG,
  /// Zeros all call-used GPRs.
  AllGPR = ONLY_GPR,
  /// Zeros all call-used regs that pass args.
  AllArg = ONLY_ARG,
  /// Zeros all call-used regs.
  All = 0,
}

#[derive(Default)]
pub enum UWTableKind {
    /// No unwind table requested
    None = 0,
    /// "Synchronous" unwind tables
    Sync = 1,
    /// 'Asynchronous' unwind tables (instr precise)
    #[default]
    Async = 2,
}

#[repr(align(8))]  // usize int
pub enum FunctionReturnThunksKind {
     /// No function return thunk.
    Keep = 0,
    /// Replace returns with jump to thunk, don't emit thunk.
    Extern = 1,  
    /// Not used.
    Invalid = 2,
}
