//! <https://github.com/llvm/llvm-project/tree/main/mlir>

#![feature(strict_provenance)]
#![feature(rustc_private)]
#![feature(generic_const_exprs)]
#![feature(const_evaluatable_checked)]
#![feature(const_for)]
#![feature(return_position_impl_trait_in_trait)]
#![feature(type_alias_impl_trait)]
#![feature(const_trait_impl)]
#![feature(ptr_metadata)]
#![feature(core_intrinsics)]
#![feature(proc_macro_diagnostic)]
#![feature(let_chains)]
#![feature(inherent_associated_types)]
// TODO Remove when translation work is done.
#![allow(unused)]

pub mod analysis;
pub mod asm_parser;
pub mod bytecode;
pub mod conversion;
pub mod dialect;
pub mod execution_engine;
pub mod interfaces;
pub mod ir;
pub mod parser;
pub mod pass;
pub mod reducer;
pub mod rewrite;
pub mod support;
pub mod table_gen;
pub mod target;
pub mod transforms;
