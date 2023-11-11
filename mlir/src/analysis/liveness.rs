//! This file contains an analysis for computing liveness information from a
//! given top-level operation. The current version of the analysis uses a
//! traditional algorithm to resolve detailed live-range information about all
//! values within the specified regions. It is also possible to query liveness
//! information on block level.
//!
//! Implementation of the liveness analysis.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Analysis/Liveness.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Analysis/Liveness.cpp>

use llvm::{
    adt::{
        array_ref,
        dense_map::DenseMap,
        small_ptr_set::SmallPtrSet,
        set_operations,
        set_vector,
        stl_extras
    },
    support::raw_ostream::{errs, RawOStream}
};
use crate::{
    ir::{
        block::Block,
        operation::Operation,
        region,
        value::{Input, Value}
    },
    support::llvm
};

/**
Represents an analysis for computing liveness information from a
given top-level operation. The analysis iterates over all associated
regions that are attached to the given top-level operation. It
computes liveness information for every value and block that are
included in the mentioned regions. It relies on a fixpoint iteration
to compute all live-in and live-out values of all included blocks.
Sample usage:

```cpp
Liveness liveness(topLevelOp);
auto &allInValues = liveness.getLiveIn(block);
auto &allOutValues = liveness.getLiveOut(block);
auto allOperationsInWhichValueIsLive = liveness.resolveLiveness(value);
bool isDeafAfter = liveness.isDeadAfter(value, operation);
```
*/
pub struct Liveness {
    /// The operation this analysis was constructed from.
    operation: Option<Operation>,

    /// Maps blocks to internal liveness information.
    block_mapping: BlockMapT
}

type OperationListT = Vec<Option<Operation>>;
type BlockMapT = DenseMap<Option<Block>, LivenessBlockInfo>;
type ValueSet = SmallPtrSet<Value, 16>;

impl Liveness {
    /**
    Creates a new Liveness analysis that computes liveness
    information for all associated regions.
    */
    pub fn new(operation: Option<Operation>) -> Self {
        Self {
            operation
        }
        // TODO
    }

    /// Returns the operation this analysis was constructed from.
    pub const fn operation(&self) -> Option<Operation> {
        self.operation
    }

    /**
    Gets liveness info (if any) for the given value.
    This includes all operations in which the given value is live.
    Note that the operations in this list are not ordered and the current
    implementation is computationally expensive (as it iterates over all
    blocks in which the given value is live).
    */
    pub const fn resolve_liveness(&self, value: Value) -> OperationListT {

    }

    /// Gets liveness info (if any) for the block.
    pub const fn get_liveness(&self, block: Option<Block>
    ) -> Option<LivenessBlockInfo>
    {
        match self.block_mapping.find(block) {
            None => None,
            Some(it) => it
        }
    }

    /// Returns a reference to a set containing live-in values (unordered).
    pub const fn get_live_in(&self, block: Option<Block>) -> &ValueSet {
        self.get_liveness(block).r#in()
    }

    /// Returns a reference to a set containing live-out values (unordered).
    pub const fn get_live_out(&self, block: Option<Block>) -> &ValueSet {
        self.get_liveness(block).out()
    }

    /// Returns true if `value` is not live after `operation`.
    pub const fn is_dead_after(
        &self, value: Value, operation: Option<Operation>
    ) -> bool
    {
        let block = operation.block();
        let block_info = self.get_liveness(block);
      
        // The given value escapes the associated block.
        if block_info.is_live_out(value) {
            return false;
        }
      
        let end_operation = block_info.end_operation(value, operation);
        /*
        If the operation is a real user of `value` the first check is sufficient.
        If not, we will have to test whether the end operation is executed before
        the given operation in the block.
        */
        end_operation == operation || end_operation.is_before_in_block(operation)
    }

    /// Dumps the liveness information in a human readable format.
    pub const fn dump(&self) {
        self.print(errs());
    }

    /// Dumps the liveness information to the given stream.
    pub const fn print(&self, os: &RawOStream) {

    }

    /// Initialises the internal mappings.
    fn build(&self) {
        // Build internal block mapping.
        let builders = DenseMap::<Option<Block>, BlockInfoBuilder>::new();
        self.build_block_mapping(self.operation, builders);

        // Store internal block data.
        for &entry in builders {
            let builder = entry.second;
            let mut info = self.block_mapping[entry.first];

            info.block = builder.block;
            info.in_values = builder.in_values;  // move
            info.out_values = builder.out_values;  // move
        }
    }
}

/// This class represents liveness information on block level.
pub struct LivenessBlockInfo {
    /// The underlying block.
    block: Option<Block>,  // = nullptr;

    /// The set of all live in values.
    in_values: ValueSet,

    /// The set of all live out values.
    out_values: ValueSet
}

impl LivenessBlockInfo {
    // /// A typedef declaration of a value set.
    // type ValueSet = Liveness::ValueSet;

    /// Returns the underlying block.
    pub const fn block(&self) -> Option<Block> {
        self.block
    }

    /// Returns all values that are live at the beginning
    /// of the block (unordered).
    pub const fn r#in(&self) -> &ValueSet {
        &self.in_values
    }

    /// Returns all values that are live at the end
    /// of the block (unordered).
    pub const fn out(&self)  -> &ValueSet {
        &self.out_values
    }

    /// Returns true if the given value is in the live-in set.
    pub const fn is_live_in(&self, value: Value) -> bool {
        self.in_values.count(value)
    }

    /// Returns true if the given value is in the live-out set.
    pub const fn is_live_out(&self, value: Value) -> bool {
        self.out_values.count(value)
    }

    /**
    Gets the start operation for the given value. This is the first operation
    the given value is considered to be live. This could either be the start
    operation of the current block (in case the value is live-in) or the
    operation that defines the given value (must be referenced in this block).
    */
    pub const fn start_operation(&self, value: Value) -> Option<Operation> {
        let defining_operation = value.defining_operation();
        // The given value is either live-in or is defined
        // in the scope of this block.
        if self.is_live_in(value) || !defining_operation {
            return &self.block.first();
        }
            
        defining_operation
    }

    /// Gets the end operation for the given value using the start operation
    /// provided (must be referenced in this block).
    pub const fn end_operation(
        &self, value: Value, start_operation: Option<Operation>
    ) -> Option<Operation>
    {
        // The given value is either dying in this block or live-out.
        if self.is_live_out(value) {
            return &self.block.last();
        }
        
        // Resolve the last operation (must exist by definition).
        let end_operation = start_operation;
        for use_op in value.users() {
            // Find the associated operation in the current block (if any).
            use_op = self.block.find_ancestor_op_in_block(*use_op);
            // Check whether the use is in our block and after the current end
            // operation.
            if use_op && end_operation.is_before_in_block(use_op) {
                end_operation = use_op;
            }
                
        }
        end_operation
    }

    /**
    Get the set of values that are currently live (if any) for the current op.
    This analysis takes an expansive view of "live" in that if a value is
    defined by or within the operation or is fully consumed (as in last user)
    by or within the operation the value is considered "live". The values in
    the list are not ordered.

    This check is quite expensive as it does not cache the results of the
    computation, so the currently live values have to be recomputed for each
    op.
    */
    pub const fn currently_live_values(&self, op: Option<Operation>
    ) -> ValueSet
    {
        let mut live_set = ValueSet::new();

        // Given a value, check which ops are within its live range. For each of
        // those ops, add the value to the set of live values as-of that op.
        let add_value_to_currently_live_sets = |value: Value| {
            // Determine the live range of this value inside this block.
            let start_of_live_range = value.defining_operation();
            let end_of_live_range = None;
            /*
            If it's a live in or a block argument, then the start is the
            beginning of the block.
            */
            if self.is_live_in(value) || value.isa::<Input>() {
                start_of_live_range = &self.block.first();
            } else {
                start_of_live_range = self.block.find_ancestor_op_in_block(*start_of_live_range);
            }
        
            // If it's a live out, then the end is the back of the block.
            if self.is_live_out(value) {
                end_of_live_range = &self.block.last();
            }
        
            /*
            We must have at least a start_of_live_range at this point. Given this, we
            can use the existing end_operation to find the end of the live range.
            */
            if start_of_live_range && !end_of_live_range {
                end_of_live_range = self.end_operation(value, start_of_live_range);
            }
                
            assert!(end_of_live_range, "Must have end_of_live_range at this point!");
            // If this op is within the live range, insert the value into the set.
            if    !(op.is_before_in_block(start_of_live_range)
                || end_of_live_range.is_before_in_block(op))
            {
                live_set.insert(value);
            }
                
        };
      
        // Handle block arguments if any.
        for arg in self.block.arguments() {
            add_value_to_currently_live_sets(arg);
        }
      
        /*
        Handle live-ins. Between the live ins and all the op results that gives us
        every value in the block.
        */
        for r#in in self.in_values {
            add_value_to_currently_live_sets(r#in);
        }
      
        // Now walk the block and handle all values used in the block and values
        // defined by the block.
        for walk_op in llvm::make_range(self.block.begin(), ++op.iterator()) {
            for result in walk_op.results() {
                add_value_to_currently_live_sets(result);
            }
        }
          
        return live_set;
    }
}

/// Builds and holds block information during the construction phase.
struct BlockInfoBuilder {
    /// The current block.
    block: Option<Block>,  // None

    /// The set of all live in values.
    in_values: ValueSet,

    /// The set of all live out values.
    out_values: ValueSet,

    /// The set of all defined values.
    def_values: ValueSet,

    /// The set of all used values.
    use_values: ValueSet,
}
