//! MLIR Diagnostics
//!
//! This file defines utilities for emitting diagnostics.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Diagnostics.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/Diagnostics.cpp>

use core::{
    default::Default,
    fmt::Arguments
};
use rustc_errors::diagnostic::{Diagnostic, Level};
use std::backtrace;
use llvm::{
    adt::{
        map_vector,
        small_string,
        small_vector::SmallVector,
        string_map,
        twine::Twine,
        type_switch,
    },
    support::{
        mutex,
        pretty_stack_trace,
        signals,
        source_manager,
        raw_ostream
    }
};
use crate::{
    ir::{
        attribute,
        location::Location,
        mlir_context,
        operation::{
            Operation,
            support::PrintingFlags
        },
        r#type
    },
    support::logical_result::LogicalResult
};

/// Append an operation with the given printing flags.
pub fn append_op(operation: &Operation, flags: &PrintingFlags, level: Level) -> String {
    let mut string = String::new();
    operation.print(string, adjust_printing_flags(flags, level));
    // Print on a new line for better readability if the op will be printed on multiple lines.
    if string.find('\n').is_some() {
        string.push("\n");
        string
    }
}

/// Adjusts operation printing flags used in diagnostics for the given severity level.
pub fn adjust_printing_flags(flags: PrintingFlags, level: Level)
    -> PrintingFlags
{
    flags.use_local_scope();
    flags.elide_large_elements_attrs();
    if level == Level::Error {
        flags.print_generic_op_form();
    }
    flags
}

/// This class is the main interface for diagnostics. The DiagnosticEngine manages the registration of diagnostic handlers as well as the core API for diagnostic emission. This class should not be constructed directly, but instead interfaced with via an MLIRContext instance.
pub struct DiagnosticEngine {
    /// A mutex to ensure that diagnostics emission is thread-safe.
    mutex: SmartMutex<true>,

    /// These are the handlers used to report diagnostics.
    handlers: SmallMapVector<u64, Fn(&Diagnostic) -> LogicalResult, 2>,

    /// This is a unique identifier counter for diagnostic handlers in the context. This id starts at 1 to allow for 0 to be used as a sentinel.
    unique_handler_id: u64  //  = 1;
}

impl DiagnosticEngine {
    /// Create a new inflight diagnostic with the given location and severity.
    pub fn emit(&self, location: Location, level: Level) -> Diagnostic {
        assert!(level != Level::Note,
            "Notes should not be emitted directly");
            Diagnostic::new(location, level)
    }
}

/// Helper function used to emit a diagnostic with an optionally empty twine message. If the message is empty, then it is not inserted into the diagnostic.
pub fn emit_diag(location: Location, severity: Level, message: &Twine) -> Diagnostic {
    let context = location.context();
    let diag_engine = context.diag_engine();
    let diag = diag_engine.emit(location, severity);
    if !message.is_trivially_empty() {
        diag << message;
    }
    // Add the stack trace as a note if necessary.
    if context.should_print_stack_trace_on_diagnostic() {
        let mut bt = String::new();
        {
            PrintStackTrace(bt);
        }
        if !bt.is_empty() {
            diag.attach_note() << "diagnostic emitted with trace:\n" << bt;
        }
    }
    diag
}

pub fn emit_error(location: Location, message: &Twine) -> Diagnostic {
    emit_diag(location, Level::Error, message)
}

/*
Diagnostic handler registration and use. MLIR supports the ability for the
IR to carry arbitrary metadata about operation location information. If a
problem is detected by the compiler, it can invoke the emitError /
emitWarning / emitRemark method on an Operation and have it get reported
through this interface.

Tools using MLIR are encouraged to register error handlers and define a
schema for their location information.  If they don't, then warnings and
notes will be dropped and errors will be emitted to errs.
*/

/// The handler type for MLIR diagnostics. This function takes a diagnostic as input, and returns success if the handler has fully processed this diagnostic. Returns failure otherwise.
//llvm::unique_function<Result(&Diagnostic)>;
type Handler = Box<dyn Fn(&Diagnostic) -> Result>; 

/// This diagnostic handler is a simple RAII class that registers and erases a diagnostic handler on a given context. This class can be either be used directly, or in conjunction with a derived diagnostic handler.
pub struct ScopedDiagnosticHandler {
}
