//! Verifier Analysis for MLIR Structures 
//!
//! This file implements the verify() methods on the various IR types, performing (potentially expensive) checks on the holistic structure of the code. This can be used for detecting bugs in compiler transformations and hand written .mlir files.
//!
//! The checks in this file are only for things that can occur as part of IR transformations: e.g. violation of dominance information, malformed operation attributes, etc. MLIR supports transformations moving IR through locally invalid states (e.g. unlinking an operation from a block before re-inserting it in a new place), but each transformation must complete with the IR in a valid form.
//!
//! This should not check for things that are always wrong by construction (e.g. attributes or other immutable structures that are incorrect), because those are not mutable and can be checked at time of construction.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Verifier.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/Verifier.cpp>

use llvm::{
    adt::{
        small_vector::SmallVector,
        string_map
    },
    support::{
        format_variadic,
        pretty_stack_trace,
        regex
    }
};
use crate::{
    ir::{
        attribute,
        block::Block,
        dialect,
        dominance,
        operation::Operation,
        region_kind_interface,
        threading,
    },
    support::logical_result::LogicalResult
};

/// Perform (potentially expensive) checks of invariants, used to detect compiler bugs, on this operation and any nested operations. On error, this reports the error through the MLIRContext and returns failure. If `verify_recursively` is false, this assumes that nested operations have already been properly verified, and does not recursively invoke the verifier on nested operations.
pub fn verify(operation: *mut Operation, verify_recursively: bool /* true */) -> LogicalResult {
    let verifier = OperationVerifier::new(verify_recursively);
    verifier.verify_op_and_dominance(operation)
}

/// This class encapsulates all the state used to verify an operation region.
pub struct OperationVerifier {
    /// A flag indicating if this verifier should recursively verify nested operations.
    verify_recursively: bool
}

impl OperationVerifier {
    /// Verify the given operation.
    pub fn verify_op_and_dominance(&self, operation: &Operation) -> LogicalResult {
        // Verify the operation first, collecting any IsolatedFromAbove operations.
        if self.verify_operation(operation).is_err() {
            return Err(());
        }
        // Since everything looks structurally ok to this point, we do a dominance check for any nested regions. We do this as a second pass since malformed CFG's can cause dominator analysis construction to crash and we want the verifier to be resilient to malformed code.
        if operation.num_regions() != 0 {
            let dom_info = DominanceInfo;
            if self.verify_dominance_of_contained_regions(
                operation, dom_info).is_err()
            {
                return Err(());
            }
        }

        Ok(())
    }

    /// Any ops that have regions and are marked as "isolated from above" will be returned in the ops_with_isolated_regions vector.
    fn verify_block(
        &self,
        block: &Block,
        ops_with_isolated_regions: &SmallVector<*mut Operation>
    ) -> LogicalResult {
        Ok(())
    }

    /// Verify the properties and dominance relationships of this operation.
    fn verify_operation(&self, operation: &Operation) -> LogicalResult {
        Ok(())
    }

    /// Verify the dominance property of regions contained within the given
    /// Operation.
    fn verify_dominance_of_contained_regions(
        &self,
        operation: &Operation,
        dom_info: &DominanceInfo
    ) -> LogicalResult
    {
        for region in operation.regions() {
            // Verify the dominance of each of the held operations.
            for block in region {
                // Dominance is only meaningful inside reachable blocks.
                let is_reachable = dom_info.is_reachable_from_entry(&block);

                for operation in block {
                    if is_reachable {
                        // Check that operands properly dominate this use.
                        for operand in operation.operands() {
                            if dom_info.properly_dominates(
                                operand.value(), &operation) {
                                continue;
                            }

                            diagnose_invalid_operand_dominance(
                                operation, operand.index());
                            return Err(());
                        }
                    }

                    // Recursively verify dominance within each operation in the block, even if the block itself is not reachable, or we are in a region which doesn't respect dominance.
                    if self.verify_recursively && operation.num_regions() != 0 {
                        // If this operation is IsolatedFromAbove, then we'll handle it in the outer verification loop.
                        if operation.has_trait::<IsIsolatedFromAbove>() {
                            continue;
                        }
                        if self.verify_dominance_of_contained_regions(
                            operation, dom_info).is_err()
                        {
                            return Err(());
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
