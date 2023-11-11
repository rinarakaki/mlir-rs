//! MLIR Symbol Table Class
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/SymbolTable.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/SymbolTable.cpp>

use crate::{
    mlir::ir::{
        attribute::Attribute,
        builders,
        operation::{
            Operation,
            base::SymbolRefAttr,
            definition,
            implementation,
        }
    },
    llvm::adt::{
        dense_map::DenseMap,
        set_vector::SetVector,
        small_ptr_set,
        small_string,
        string_map,
        string_switch
    }
};

/// This class allows for representing and managing the symbol table used by operations with the 'SymbolTable' trait. Inserting into and erasing from this SymbolTable will also insert and erase from the Operation given to it at construction.
pub struct SymbolTable {
    symbol_table_op: *mut Operation,

    /// This is a mapping from a name to the symbol with that name. They key is always known to be a StringAttribute.
    symbol_table: DenseMap<dyn Attribute, *mut Operation>,
  
    /// This is used when name conflicts are detected.
    uniquing_counter: usize  //= 0;
}

/// An enumeration detailing the different visibility types that a symbol may have.
enum Visibility {
    /// The symbol is public and may be referenced anywhere internal or external to the visible references in the IR.
    Public,

    /// The symbol is private and may only be referenced by SymbolRefAttrs local to the operations within the current symbol table.
    Private,

    /**
    The symbol is visible to the current IR, which may include operations in symbol tables above the one that owns the current symbol. `Nested` visibility allows for referencing a symbol outside of its current symbol table, while retaining the ability to observe all uses.
    */
    Nested,
}

/// This class represents a specific symbol use.
struct SymbolUse {
    /// The operation that this access is held by.
    owner: *mut Operation,

    /// The symbol reference that this use represents.
    symbol_ref: SymbolRefAttribute
}

/// This class implements a range of SymbolRef uses.
struct UseRange {
    uses: Vec<SymbolUse> 
}

/// This class represents a collection of `SymbolTable`s. This simplifies certain algorithms that run recursively on nested symbol tables. Symbol tables are constructed lazily to reduce the upfront cost of constructing unnecessary tables.
pub struct SymbolTableCollection {
    /// The constructed symbol tables nested within this table.
    symbol_tables: DenseMap<*mut Operation, Box<SymbolTable>>
}

/// This class represents a map of symbols to users, and provides efficient implementations of symbol queries related to users; such as collecting the users of a symbol, replacing all uses, etc.
pub struct SymbolUserMap<'a> {
    /// A reference to the symbol table used to construct this map.
    symbol_table: &'a SymbolTableCollection,

    /// A map of symbol operations to symbol users.
    symbol_to_users: DenseMap<*mut Operation, SetVector<*mut Operation>>
}

impl<'a> SymbolUserMap<'a> {
    
}
