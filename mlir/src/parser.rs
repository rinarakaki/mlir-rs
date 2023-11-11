//! # MLIR Parser Library Interface
//!
//! This file is contains a unified interface for parsing serialised MLIR.
//!
//! This file implements the parser for the MLIR textual form.
//!
//! - include <https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Parser/Parser.h>
//! - lib <https://github.com/llvm/llvm-project/tree/main/mlir/lib/Parser/Parser.cpp>

use llvm::{
    adt::stl_extras::has_single_element,
    support::{
        memory_buffer::MemoryBuffer,
        sm_loc::SMLoc,
        source_manager::SourceManager
    }
};
use crate::{
    asm_parser::asm_parser::parse_asm_source_file,
    bytecode::reader::{is_bytecode, read_bytecode_file},
    ir::{
        asm_state::ParserConfig,
        block::Block,
        builders::Builder,
        builtins::location_attributes::{
            FileLineColLocation, UnknownLocation
        },
        location::{Location, LocationAttribute},
        mlir_context::MLIRContext,
        operation::Operation,
        owning_op_ref::OwningOpRef
    },
    support::logical_result::LogicalResult
};


// /**
// Given a block containing operations that have just been parsed, if the block contains a single operation of `ContainerOpT` type then remove it from the block and return it. If the block does not contain just that operation, create a new operation instance of `ContainerOpT` and move all of the operations within `parsed_block` into the first block of the first region.
// `ContainerOpT` is required to have a single region containing a single block, and must implement the `SingleBlockImplicitTerminator` trait.
// */
// #[inline] 
// pub fn construct_container_op_for_parser_if_necessary<ContainerOpT>(
//     parsed_block: *mut Block,
//     context: *mut MLIRContext,
//     source_file_loc: Location
// ) -> OwningOpRef<ContainerOpT>
// {

//     // Check to see if we parsed a single instance of this operation.
//     if has_single_element(*parsed_block) {
//         let op = dyn_cast::<ContainerOpT>(&parsed_block.front());
//         if op {
//             op.remove();
//             return op;
//         }
//     }

//     /*
//     If not, then build a new top-level op if a concrete operation type was specified.
//     */
//     if std::is_same_v<ContainerOpT, Option<Operation>> {
//         (void)context;
//         return emit_error(source_file_loc)
//                 << "Source must contain a single top-level operation, found: "
//                 << parsed_block.operations().len(),
//             nullptr;
//     } else {
//         assert!(
//             ContainerOpT::template has_trait::<OneRegion>()
//             && (
//                 ContainerOpT::template has_trait<NoTerminator>()
//                 || template hasSingleBlockImplicitTerminator<
//                     ContainerOpT>::value
//             ),
//             "Expected `ContainerOpT` to have a single region with a single "
//             "block that has an implicit terminator or does not require one");

//         let builder = Builder::new(context);
//         let op = builder.create::<ContainerOpT>(source_file_loc);
//         let op_ref = OwningOpRef::<ContainerOpT>::new(op);
//         assert!(
//             op.num_regions() == 1 && has_single_element(op.region(0)),
//             "Expected generated operation to have a single region with a single block");
//         let op_block = &op.region(0).front();
//         op_block.operations().splice(op_block.begin(),
//                                      parsed_block.operations());

//         /*
//         After splicing, verify just this operation to ensure it can properly contain the operations inside of it.
//         */
//         if let Err(()) = op.verify_invariants() {
//             return OwningOpRef::<ContainerOpT>::new();
//         }
        
//         return op_ref;
//     }
// }

// /**
// This parses the file specified by the indicated SourceManager and appends parsed operations to the given block. If the block is non-empty, the operations are placed before the current terminator. If parsing is successful, success is returned. Otherwise, an error message is emitted through the error handler registered in the context, and failure is returned. If `source_file_loc` is non-null, it is populated with a file location representing the start of the source file that is being parsed.
// */
// pub fn parse_source_file(
//     source_manager: &SourceManager,
//     block: *mut Block,
//     config: &ParserConfig,
//     source_file_loc: *mut LocationAttribute  // = nullptr
// ) -> Result<(), Box<dyn std::error::Error>>
// {
//     let source_buf = source_manager.memory_buffer(source_manager.main_file_id());
//     if source_file_loc.is_some() {
//         *source_file_loc
//             = FileLineColLocation::get(
//                 config.context(),
//                 source_buf.get_buffer_identifier(),
//                 /*line=*/0, /*column=*/0);
//     }
//     if is_bytecode(*source_buf) {
//         return read_bytecode_file(*source_buf, block, config);
//     }
//     return parse_asm_source_file(source_manager, block, config);
// }

// /**
// An overload with a source manager that may have references taken during the parsing process, and whose lifetime can be freely extended (such that the source manager is not destroyed before the parsed IR). This is useful, for example, to avoid copying some large resources into the MLIRContext and instead referencing the data directly from the input buffers.
// */
// pub fn parse_source_file(
//     source_manager: &Box<SourceManager>,
//     block: Option<Block>,
//     config: &ParserConfig,
//     source_file_loc: Option<LocationAttribute>  //= nullptr
// ) -> Result<(), Box<dyn std::error::Error>>
// {
//     let source_buf =
//         source_manager.get_memory_buffer(source_manager.get_main_file_id());
//     if source_file_loc.is_some() {
//         *source_file_loc = FileLineColLocation::get(config.context(),
//                                             source_buf.get_buffer_identifier(),
//                                             /*line=*/0, /*column=*/0);
//     }
//     if is_bytecode(*source_buf) {
//         return read_bytecode_file(source_manager, block, config);
//     }
//     return parse_asm_source_file(*source_manager, block, config);
// }

// /**
// This parses the file specified by the indicated filename and appends parsed operations to the given block. If the block is non-empty, the operations are placed before the current terminator. If parsing is successful, success is returned. Otherwise, an error message is emitted through the error handler registered in the context, and failure is returned. If `source_file_loc` is non-null, it is populated with a file location representing the start of the source file that is being parsed.
// */
// pub fn parse_source_file(
//     filename: &str,
//     block: Option<Block>,
//     config: &ParserConfig,
//     source_file_loc: Option<LocationAttribute>  // = nullptr
// ) -> Result<(), Box<dyn std::error::Error>>
// {
//     let source_manager = std::make_shared::<SourceManager>();
//     return parse_source_file(filename, source_manager, block, config, source_file_loc);
// } 

// /**
// This parses the file specified by the indicated filename using the provided SourceManager and appends parsed operations to the given block. If the block is non-empty, the operations are placed before the current terminator. If parsing is successful, success is returned. Otherwise, an error message is emitted through the error handler registered in the context, and failure is returned. If `source_file_loc` is non-null, it is populated with a file location representing the start of the source file that is being parsed.
// */
// pub fn parse_source_file(
//     filename: &str,
//     source_manager: &SourceManager, block: Option<Block>,
//     config: &ParserConfig,
//     source_file_loc: Option<LocationAttribute>  //..= nullptr
// ) -> Result<(), Box<dyn std::error::Error>>
// {
//     load_source_file_buffer(filename, source_manager, config.context())?;
//     parse_source_file(source_manager, block, config, source_file_loc)
// }

// /**
// An overload with a source manager that may have references taken during the parsing process, and whose lifetime can be freely extended (such that the source manager is not destroyed before the parsed IR). This is useful, for example, to avoid copying some large resources into the MLIRContext and instead referencing the data directly from the input buffers.
// */
// pub fn parse_source_file(
//     filename: &str,
//     source_manager: &Box<SourceManager>,
//     block: Option<Block>,
//     config: &ParserConfig,
//     source_file_loc: Option<LocationAttribute>  //= nullptr
// ) -> Result<(), Box<dyn std::error::Error>>
// {
//     load_source_file_buffer(filename, *source_manager, config.context())?;
//     parse_source_file(source_manager, block, config, source_file_loc)
// }

/**
This parses the IR string and appends parsed operations to the given block.
If the block is non-empty, the operations are placed before the current terminator. If parsing is successful, success is returned. Otherwise, an error message is emitted through the error handler registered in the context, and failure is returned. If `source_file_loc` is non-null, it is populated with a file location representing the start of the source file that is being parsed.
*/
pub fn parse_source_string(
    source_str: &str,
    block: Option<Block>,
    config: &ParserConfig,
    source_file_loc: Option<LocationAttribute> //= nullptr
) -> Result<(), Box<dyn std::error::Error>>
{
    let mem_buffer = MemoryBuffer::mem_buffer(source_str);
    if !mem_buffer {
        return Err(());
    }

  let source_manager = SourceManager::new();
  source_manager.add_new_source_buffer(mem_buffer, SMLoc::new());  // std::move(mum_buffer)
  return parse_source_file(source_manager, block, config, source_file_loc);
}

// /**
// The internal implementation of the templated `parse_source_file` methods below, that simply forwards to the non-templated version.
// */
// #[inline]
// pub fn parse_source_file<ContainerOpT, ParserArgs>(
//     config: &ParserConfig,
//     args: &&ParserArgs
// ) -> OwningOpRef<ContainerOpT>
// {
//     let source_file_loc = LocationAttribute::new();
//     let block = Block::new();
//     match parse_source_file(
//         std::forward::<ParserArgs>(args), &block, config, &source_file_loc)
//     {
//         Ok(()) =>
//             construct_container_op_for_parser_if_necessary::<ContainerOpT>(
//                 &block, config.context(), source_file_loc),
//         Err(()) => OwningOpRef::<ContainerOpT>::new()
//     }
// }

// /**
// This parses the file specified by the indicated SourceManager. If the source IR contained a single instance of `ContainerOpT`, it is returned. Otherwise, a new instance of `ContainerOpT` is constructed containing all of the parsed operations. If parsing was not successful, null is returned and an error message is emitted through the error handler registered in the context, and failure is returned. `ContainerOpT` is required to have a single region containing a single block, and must implement the `SingleBlockImplicitTerminator` trait.
// */
// #[inline]
// pub fn parse_source_file<ContainerOpT = Option<Operation>>(
//     source_manager: &SourceManager, config: &ParserConfig
// ) -> OwningOpRef<ContainerOpT>
// {
//     parse_source_file::<ContainerOpT>(config, source_manager)
// }

// /**
// An overload with a source manager that may have references taken during the parsing process, and whose lifetime can be freely extended (such that the source manager is not destroyed before the parsed IR). This is useful, for example, to avoid copying some large resources into the MLIRContext and instead referencing the data directly from the input buffers.
// */
// #[inline] 
// pub fn parse_source_file<ContainerOpT = Option<Operation>>(
//     source_manager: &Box<SourceManager>,
//     config: &ParserConfig
// ) -> OwningOpRef<ContainerOpT>
// {
//     parse_source_file::<ContainerOpT>(config, source_manager)
// }

// /**
// This parses the file specified by the indicated filename. If the source IR contained a single instance of `ContainerOpT`, it is returned. Otherwise, a new instance of `ContainerOpT` is constructed containing all of the parsed operations. If parsing was not successful, null is returned and an error message is emitted through the error handler registered in the context, and failure is returned. `ContainerOpT` is required to have a single region containing a single block, and must implement the `SingleBlockImplicitTerminator` trait.
// */
// #[inline] 
// pub fn parse_source_file<ContainerOpT = Option<Operation>>(
//     filename: &str,
//     config: &ParserConfig
// ) -> OwningOpRef<ContainerOpT>
// {
//     parse_source_file::<ContainerOpT>(config, filename)
// }

// /**
// This parses the file specified by the indicated filename using the provided SourceManager. If the source IR contained a single instance of `ContainerOpT`, it is returned. Otherwise, a new instance of `ContainerOpT` is constructed containing all of the parsed operations. If parsing was not successful, null is returned and an error message is emitted through the error handler registered in the context, and failure is returned. `ContainerOpT` is required to have a single region containing a single block, and must implement the `SingleBlockImplicitTerminator` trait.
// */
// #[inline] 
// pub fn parse_source_file<ContainerOpT = Option<Operation>>(
//     filename: &str,
//     source_manager: &SourceManager,
//     config: &ParserConfig
// ) -> OwningOpRef<ContainerOpT>
// {
//     parse_source_file::<ContainerOpT>(config, filename, source_manager)
// }

// /**
// An overload with a source manager that may have references taken during the parsing process, and whose lifetime can be freely extended (such that the source manager is not destroyed before the parsed IR). This is useful, for example, to avoid copying some large resources into the MLIRContext and instead referencing the data directly from the input buffers.
// */
// #[inline]
// pub fn parse_source_file<ContainerOpT = Option<Operation>>(
//     filename: &str,
//     source_manager: &Box<SourceManager>,
//     config: &ParserConfig
// ) -> OwningOpRef<ContainerOpT>
// {
//     parse_source_file::<ContainerOpT>(config, filename, source_manager)
// }

// /**
// This parses the provided string containing MLIR. If the source IR contained a single instance of `ContainerOpT`, it is returned. Otherwise, a new instance of `ContainerOpT` is constructed containing all of the parsed operations. If parsing was not successful, null is returned and an error message is emitted through the error handler registered in the context, and failure is returned. `ContainerOpT` is required to have a single region containing a single block, and must implement the `SingleBlockImplicitTerminator` trait.
// */
// #[inline] 
// pub fn parse_source_string<ContainerOpT = Option<Operation>>(
//     source_str: &str,
//     config: &ParserConfig
// ) -> OwningOpRef<ContainerOpT>
// {
//     let source_file_loc = LocationAttribute::new;
//     let block = Block::new();
//     match parse_source_string(source_str, &block, config, &source_file_loc) {
//         Ok(()) =>
//             construct_container_op_for_parser_if_necessary::<ContainerOpT>(
//                 &block, config.context(), source_file_loc),
//         Err(()) => OwningOpRef::<ContainerOpT>::new()
//     }
// }

pub fn load_source_file_buffer(
    filename: &str,
    source_manager: &SourceManager,
    context: Option<MLIRContext>
) -> LogicalResult
{
    if source_manager.num_buffers() != 0 {
        // TODO: Extend to support multiple buffers.
        return emit_error(UnknownLocation::get(context),
            "Only main buffer parsed at the moment");
    }
    match MemoryBuffer::file_or_stdin(filename, false, true, None) {
        Ok(file) => {
            // Load the MLIR source file.
            source_manager.add_new_source_buffer(file, SMLoc::new());  // std::move(*file_or_err)
            Ok(())
        },
        Err(error) => emit_error(UnknownLocation::get(context),
        "Could not open input file " + filename)
    }
}
