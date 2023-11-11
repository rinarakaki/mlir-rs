/*!
- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/AsmParser/AsmParserState.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/AsmParser/AsmParserState.cpp>
*/

use core::slice::Iter;

use crate::{
    mlir::{
        ir::{
            attribute::Attribute,
            block::Block,
            operation::Operation,
            symbol_table::SymbolTableCollection,
            value::Value
        },
        support::llvm
    },
    llvm::{
        adt::{
            dense_map::DenseMap,
            iterator::PointeeIterator,
            small_vector::SmallVector,
            string_extras
        },
        support::sm_loc::{SMLoc, SMRange}
    }
};

type SymbolUseMap = DenseMap<
    dyn Attribute,
    SmallVector<[SmallVector<[SMRange]>; 0]>
>;

/**
This class represents state from a parsed MLIR textual format string. It is useful for building additional analysis and language utilities on top of textual MLIR. This should generally not be used for traditional compilation.
*/
pub struct AsmParserState {
    // ======================================================================
    // AsmParserState::Impl
    // ======================================================================

    /// A mapping from operations in the input source file to their parser state.
    operations: SmallVector<[Box<OperationDefinition>]>,
    operation_to_idx: DenseMap<*mut Operation, usize>,

    /// A mapping from blocks in the input source file to their parser state.
    blocks: SmallVector<[Box<BlockDefinition>]>,
    blocks_to_idx: DenseMap<*mut Block, usize>,

    /**
    A set of value definitions that are placeholders for forward references.
    This map should be empty if the parser finishes successfully.
    */
    placeholder_value_uses: DenseMap<Value, SmallVector<SMLoc>>,

    /// The symbol table operations within the IR.
    symbol_table_operations:
        SmallVector<[(*mut Operation, Box<SymbolUseMap>)]>,

    /**
    A stack of partial operation definitions that have been started but not yet finalised.
    */
    partial_operations: SmallVector<PartialOpDef>,

    /**
    A stack of symbol use scopes. This is used when collecting symbol table uses during parsing.
    */
    symbol_use_scopes: SmallVector<[Option<SymbolUseMap>]>,

    /// A symbol table containing all of the symbol table operations in the IR.
    symbol_table: SymbolTableCollection
}

type BlockDefIterator<'a> = PointeeIterator<Iter<'a, Box<BlockDefinition>>>;
type OperationDefIterator<'a> = PointeeIterator<
      Iter<'a, Box<OperationDefinition>>>;

impl AsmParserState {    
    /// Return a range of the BlockDefinitions held by the current parser state.
    pub const fn block_defs(&self) -> BlockDefIterator {
        self.blocks
    }

    /**
    Return the definition for the given block, or nullptr if the given block does not have a definition.
    */
    pub const fn block_def(&self, block: *mut Block) -> *mut BlockDefinition {
        match self.blocks_to_idx.find(block) {
            None => None,
            Some(index) => self.blocks[index]
        }
    }

    // /// Return a range of the OperationDefinitions held by the current parser
    // /// state.
    // pub const fn get_op_defs(&self) -> OperationDefIterator {

    // }

    /**
    Return the definition for the given operation, or nullptr if the given operation does not have a definition.
    */
    pub const fn op_def(&self, operation: *mut Operation) -> *const OperationDefinition {
        match self.operation_to_idx.find(operation) {
            None => None,
            Some(index) => self.operations[index]
        }
    }

    /**
    Returns (heuristically) the range of an identifier given a SMLoc corresponding to the start of an identifier location.
    */
    pub fn convert_id_loc_to_range(&self, location: SMLoc) -> SMRange {
        if !location.is_valid() {
            return SMRange();
        }
        let cur_ptr = location.get_pointer();
    
        // Check if this is a string token.
        if *cur_ptr == '"' {
            cur_ptr = lex_loc_string_tok(cur_ptr + 1);
        
            // Otherwise, default to handling an identifier.
        } else {
            // Return if the given character is a valid identifier character.
            let is_identifier_char = |c: char| {
                return isalnum(c) || c == '$' || c == '.' || c == '_' || c == '-';
            };
        
            while *cur_ptr && is_identifier_char(*(++cur_ptr)) {
                continue;
            }
        }
        
        return SMRange(location, SMLoc::getFromPointer(cur_ptr));
    }

    // --------------------------------------------------------------------
    // Populate State
    // --------------------------------------------------------------------

    /**
    Initialise the state in preparation for populating more parser state under the given top-level operation.
    */
    pub fn initialise(&self, top_level_op: *mut Operation) {
        self.start_operation_definition(top_level_op.name());

        // If the top-level operation is a symbol table, push a new symbol scope.
        let partial_op_def = self.partial_operations.back();
        if partial_op_def.is_symbol_table() {
            self.symbol_use_scopes.push(partial_op_def.symbol_table.get());
        }
    }

    /**
    Finalise any in-progress parser state under the given top-level operation.
    */
    pub fn finalise(&self, top_level_op: *mut Operation) {
        assert!(
            !self.partial_operations.is_empty(),
            "Expected valid partial operation definition");
        let partial_op_def = self.partial_operations.pop_back_val();

        // If this operation is a symbol table, resolve any symbol uses.
        if partial_op_def.is_symbol_table() {
            self.symbol_table_operations.emplace_back(
                top_level_op, std::move(partial_op_def.symbol_table));
        }
        self.resolve_symbol_uses();
    }

    /// Start a definition for an operation with the given name.
    pub fn start_operation_definition(&self, op_name: &OperationName) {
        self.partial_operations.emplace_back(op_name);
    }

    /// Finalise the most recently started operation definition.
    pub fn finalise_operation_definition(
        &self,
        operation: *mut Operation, name_loc: SMRange, end_loc: SMLoc,
        result_groups: [&(usize, SMLoc)] /* nullptr */)
    {
        assert!(
            !self.partial_operations.is_empty(),
            "Expected valid partial operation definition");
        let partial_op_def = self.partial_operations.pop_back_val();

        // Build the full operation definition.
        let definition = OperationDefinition::new(operation, name_loc, end_loc);
        for result_group in result_groups {
            definition.result_groups.emplace_back(
                result_group.first,
                self.convert_id_loc_to_range(result_group.second));
        }
        self.operation_to_idx.try_emplace(operation, self.operations.len());
        self.operations.emplace_back(definition);

        // If this operation is a symbol table, resolve any symbol uses.
        if partial_op_def.is_symbol_table() {
            self.symbol_table_operations.emplace_back(
                operation, std::move(partial_op_def.symbol_table));
        }
    }

    /// Start a definition for a region nested under the current operation.
    pub fn start_region_definition(&self) {
        assert!(
            !self.partial_operations.is_empty(),
            "Expected valid partial operation definition");

        /*
        If the parent operation of this region is a symbol table, we also push a new symbol scope.
        */
        let partial_op_def = self.partial_operations.back();
        if partial_op_def.is_symbol_table(){
            self.symbol_use_scopes.push(partial_op_def.symbol_table.get());}
    }

    /// Finalise the most recently started region definition.
    pub fn finalise_region_definition(&self) {
        assert!(
            !self.partial_operations.is_empty(),
            "Expected valid partial operation definition");

        /*
        If the parent operation of this region is a symbol table, pop the symbol scope for this region.
        */
        let partial_op_def = self.partial_operations.back();
        if partial_op_def.is_symbol_table() {
            self.symbol_use_scopes.pop();
        }
    }

    /// Add a definition of the given entity.
    pub fn add_definition(&mut self, block: *mut Block, location: SMLoc) {
        let index = match self.blocks_to_idx.find(block) {
            None => {
                self.blocks_to_idx.try_emplace(block, self.blocks.len());
                self.blocks.emplace_back(std::make_unique<BlockDefinition>(
                    block, self.convert_id_loc_to_range(location)));
                return;
            },
            Some(index) => index
        };

        /*
        If an entry already exists, this was a forward declaration that now has a proper definition.
        */
        self.blocks[index].definition.location = self.convert_id_loc_to_range(location);
    }
    // void add_definition(Input blockArg, location: SMLoc);

    /// Add a source uses of the given value.
    pub fn add_uses(&mut self, value: Value, locations: [&SMLoc]) {
        // Handle the case where the value is an operation result.
        if let result = dyn_cast<Output>(value) {
            /*
            Check to see if a definition for the parent operation has been recorded.
            If one hasn't, we treat the provided value as a placeholder value that will be refined further later.
            */
            let parent_op = result.owner();
            let index = match self.operation_to_idx.find(parent_op) {
                None => {
                    self.placeholder_value_uses[value].append(locations);
                    return;
                },
                Some(index) => index
            };

            /*
            If a definition does exist, locate the value's result group and add the use. The result groups are ordered by increasing start index, so we just need to find the last group that has a smaller/equal start index.
            */
            let result_no = result.get_result_number();
            let def = *self.operations[index];
            for result_group in def.result_groups.rev() {
                if result_no >= result_group.start_index {
                    for location in locations {
                        result_group.definition.uses.push(
                            self.convert_id_loc_to_range(location));
                    }
                    return;
                }
            }
            unreachable!("expected valid result group for value use");
        }

        // Otherwise, this is a block argument.
        let arg = value.cast<Input>();
        let index = match self.blocks_to_idx.find(arg.owner()) {
            None => panic!("Expected valid block definition for block argument"),
            Some(index) => index
        }
        let block_def = *self.blocks[index];
        let arg_def = block_def.arguments[arg.get_arg_number()];
        for location in locations {
            arg_def.uses.emplace_back(self.convert_id_loc_to_range(location));
        }
    }
    // void add_uses(block: *mut Block, ArrayRef<SMLoc> locations);

    // /// Add source uses for all the references nested under `refAttr`. The
    // /// provided `locations` should match 1-1 with the number of references in
    // /// `refAttr`, i.e.:
    // ///   nestedReferences.len() + /*leafReference=*/1 == refLocations.len()
    // void add_uses(SymbolRefAttr refAttr, ArrayRef<SMRange> refLocations);

    /**
    Refine the `old_value` to the `new_value`. This is used to indicate that `old_value` was a placeholder, and the uses of it should really refer to
    `new_value`.
    */
    pub fn refine_definition(&self, old_value: Value, new_value: Value) {
        let index = match self.placeholder_value_uses.find(old_value) {
            None => panic!("Expected `old_value` to be a placeholder"),
            Some(index) => index
        };
        self.add_uses(new_value, index);
        self.placeholder_value_uses.erase(old_value);
    }
}

/**
This class represents a definition within the source manager, containing it's defining location and locations of any uses. SMDefinitions are only provided for entities that have uses within an input file, e.g. SSA values, Blocks, and Symbols.
*/
struct SMDefinition {
}

/**
This class represents the information for an operation definition within an input file.
*/
struct OperationDefinition {
    /// The operation representing this definition.
    operation: *mut Operation,

    /// The source location for the operation, i.e. the location of its name.
    location: SMRange,

    /**
    The full source range of the operation definition, i.e. a range encompassing the start and end of the full operation definition.
    */
    scope_loc: SMRange,

    /// Source definitions for any result groups of this operation.
    result_groups: Vec<ResultGroupDefinition>,

    /**
    If this operation is a symbol operation, this vector contains symbol uses of this operation.
    */
    symbol_uses: Vec<SMRange>
}

impl OperationDefinition {
    pub fn new(operation: *mut Operation, location: SMRange, end_location: SMLoc) -> Self {
        Self {
            operation,
            location,
            scope_loc: SMRange::new(location.start, end_location),
            ..Default::default()
        }
    }
}

struct ResultGroupDefinition {
}

/**
This class represents the information for a block definition within the input file.
*/
struct BlockDefinition {
    /// The block representing this definition.
    block: *mut Block,

    /**
    The source location for the block, i.e. the location of its name, and any uses it has.
    */
    definition: SMDefinition,

    /// Source definitions for any arguments of this block.
    arguments: Vec<SMDefinition>
}

struct PartialOpDef {
    /**
    If this operation is a symbol table, the following contains symbol uses within this operation.
    */
    symbol_table: Box<SymbolUseMap>
}
