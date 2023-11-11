//! Assembly State Utilities
//!
//! This file defines various classes and utilites for interacting with the MLIR assembly formats.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/AsmState.h>

use llvm::adt::{
    dense_map::DenseMap,
    map_vector::MapVector,
    string_map::StringMap,
};
use crate::{
    ir::{
        mlir_context::MLIRContext,
        operation::support
    },
    support::llvm,
};

/// This class represents a processed binary blob of data. A resource blob is essentially a collection of data, potentially mutable, with an associated deleter function (used if the data needs to be destroyed).
pub struct AsmResourceBlob {
    /// The raw, properly aligned, blob data.
    data: &'static [u8],

    /// The alignment of the data.
    data_alignment: usize,  // = 0

    /// An optional deleter function used to deallocate the underlying data when
    /// necessary.
    deleter: DeleterFn,

    /// Whether the data is mutable.
    data_is_mutable: bool
}

type DeleterFn = fn(Option<()>, usize, usize);  // unique_function

/// This class provides a simple utility wrapper for creating heap allocated AsmResourceBlobs.
pub struct HeapAsmResourceBlob {
}

/// This class provides a simple utility wrapper for creating 'unmanaged' AsmResourceBlobs. The lifetime of the data provided to these blobs is guaranteed to persist beyond the lifetime of this reference.
pub struct UnmanagedAsmResourceBlob {
}

/// This class is used to build resource entries for use by the printer. Each resource entry is represented using a key/value pair. The provided key must be unique within the current context, which allows for a client to provide resource entries without worrying about overlap with other clients.
pub struct AsmResourceBuilder {
}

/// This enum represents the different kinds of resource values.
pub enum AsmResourceEntryKind {
    /// A blob of data with an accompanying alignment.
    Blob,
    /// A boolean value.
    Bool,
    /// A string value.
    String,
}

/// This class represents a single parsed resource entry.
pub struct AsmParsedResourceEntry {
}

/// This class represents an instance of a resource parser. This class should be implemented by non-dialect clients that want to inject additional resources into MLIR assembly formats.
pub struct AsmResourceParser {
    name: String
}

/// This class represents an instance of a resource printer. This class should be implemented by non-dialect clients that want to inject additional resources into MLIR assembly formats.
pub struct AsmResourcePrinter {
    name: String
}

/// A fallback map containing external resources not explicitly handled by another parser/printer.
pub struct FallbackAsmResourceMap {
    /// The set of opaque resources.
    key_to_resources: MapVector<
        String, Box<ResourceCollection>,
        StringMap<usize>
    >
}

/// This class represents an opaque resource.
struct OpaqueAsmResource {
    /// The key identifying the resource.
    key: String,
    /// An opaque value for the resource, whose variant values align 1-1 with the kinds defined in AsmResourceEntryKind.
    value: OpaqueAsmResourceValue
}

enum OpaqueAsmResourceValue {
    AsmResourceBlob(AsmResourceBlob),
    Bool(bool),
    String(String)
}

struct ResourceCollection {
    /// The set of resources parsed into this collection.
    resources: Vec<OpaqueAsmResource>
}

/// This class represents a configuration for the MLIR assembly parser. It contains all of the necessary state to parse a MLIR source file.
pub struct ParserConfig {
    context: *mut MLIRContext,
    verify_after_parse: bool,
    resource_parsers: DenseMap<&'static str, Box<AsmResourceParser>>,
    fallback_resource_map: *mut FallbackAsmResourceMap
}

/// This class provides management for the lifetime of the state used when printing the IR. It allows for alleviating the cost of recomputing the internal state of the asm printer.
///
/// The IR should not be mutated in-between invocations using this state, and the IR being printed must not be a parent of the IR originally used to initialise this state. This means that if a child operation is provided, a parent operation cannot reuse this state.
pub struct AsmState {
    // 
    // AsmStateImpl (https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/AsmPrinter.cpp#L1686)
    //
 
    /// Collection of OpAsm interfaces implemented in the context.
    interfaces: DialectInterfaceCollection<OpAsmDialectInterface>,

    /// A collection of non-dialect resource printers.
    external_resource_printers: SmallVector<std::unique_ptr<AsmResourcePrinter>>,

    /// A set of dialect resources that were referenced during printing.
    dialect_resources: DenseMap<*mut Dialect, SetVector<AsmDialectResourceHandle>>,

    /// The state used for attribute and type aliases.
    alias_state: AliasState,

    /// The state used for SSA value names.
    name_state: SSANameState,

    /// Flags that control op output.
    printer_flags: OpPrintingFlags,

    /// An optional location map to be populated.
    location_map: *mut AsmState::LocationMap
}

impl AsmState {
    /**
    This map represents the raw locations of operations within the output
    stream. This maps the original pointer to the operation, to a pair of line
    and column in the output stream.
    */
    using LocationMap = DenseMap<*mut Operation, (usize, usize)>;

    /// Initialize the asm state at the level of the given operation. A location
    /// map may optionally be provided to be populated when printing. `map` is an
    /// optional fallback resource map, which when provided will attach resource
    /// printers for the fallback resources within the map.
    AsmState(Operation *op,
            const OpPrintingFlags &printerFlags = OpPrintingFlags(),
            LocationMap *locationMap = nullptr,
            FallbackAsmResourceMap *map = nullptr);
    AsmState(MLIRContext *ctx,
            const OpPrintingFlags &printerFlags = OpPrintingFlags(),
            LocationMap *locationMap = nullptr,
            FallbackAsmResourceMap *map = nullptr);
    ~AsmState();

    /// Get the printer flags.
    const OpPrintingFlags &getPrinterFlags() const;

    // --------------------------------------------------------------------===//
    // Resources
    // --------------------------------------------------------------------===//

    /// Attach the given resource printer to the AsmState.
    void attach_resource_printer(std::unique_ptr<AsmResourcePrinter> printer);

    /// Attach an resource printer, in the form of a callable, to the AsmState.
    template <typename CallableT>
    std::enable_if_t<std::is_convertible<
        CallableT, function_ref<void(Operation *, AsmResourceBuilder &)>>::value>
    attach_resource_printer(StringRef name, CallableT &&printFn) {
        attach_resource_printer(AsmResourcePrinter::fromCallable(
            name, std::forward<CallableT>(printFn)));
    }

    /// Attach resource printers to the AsmState for the fallback resources
    /// in the given map.
    void attach_fallback_resource_printer(FallbackAsmResourceMap &map) {
        for (auto &printer : map.getPrinters())
            attach_resource_printer(std::move(printer));
    }

    /// Returns a map of dialect resources that were referenced when using this
    /// state to print IR.
    DenseMap<Dialect *, SetVector<AsmDialectResourceHandle>> &
    get_dialect_resources() const;
}
