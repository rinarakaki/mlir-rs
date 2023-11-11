//! - include <https://github.com/llvm/llvm-project/tree/main/llvm/include/llvm>
//! - lib <https://github.com/llvm/llvm-project/tree/main/llvm/lib>

#![feature(strict_provenance)]
#![feature(rustc_private)]
#![feature(generic_const_exprs)]
#![feature(const_for)]
#![feature(return_position_impl_trait_in_trait)]
#![feature(type_alias_impl_trait)]
#![feature(const_trait_impl)]
#![feature(ptr_metadata)]
#![feature(core_intrinsics)]
#![feature(proc_macro_diagnostic)]
#![feature(let_chains)]
#![feature(inherent_associated_types)]
// TODO Remove when translation is finished.
#![allow(unused)]

pub mod adt;
pub mod code_gen;
pub mod config;
pub mod execution_engine;
pub mod ir;
pub mod object;
pub mod support;
pub mod table_gen;
pub mod target;
pub mod target_parser;
