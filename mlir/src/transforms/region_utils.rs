/*!
# Region-Related Transformation Utilities

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Transforms/RegionUtils.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Transforms/Utils/RegionUtils.cpp>
*/

use crate::{
    mlir::{
        interfaces::control_flow_interfaces::BranchOpInterface,
        ir::{
            block::Block,
            operation::Operation,
            pattern_match::Rewriter,
            region::Region,
            value::{Input, Value}
        },
        support::logical_result::LogicalResult
    },
    llvm::adt::{
        set_vector::{SetVector, SmallSetVector},
        small_ptr_set::SmallPtrSet,
        small_vector::SmallVector
    }
};

/**
Check if all values in the provided range are defined above the `limit`
region. That is, if they are defined in a region that is a proper ancestor
of `limit`.
*/
pub fn are_values_defined_above<Range>(values: Range, limit: &Region) -> bool {
    for value in values {
        if !value.parent_region().is_proper_ancestor(&limit) {
            return false;
        }
    }
    return true
}

/// Replace all uses of `orig` within the given region with `replacement`.
pub fn replace_all_uses_in_region_with(
    orig: Value,
    replacement: Value,
    region: &Region
) {
    for r#use in llvm::make_early_inc_range(orig.uses()) {
        if region.is_ancestor(r#use.owner().parent_region()) {
            r#use.set(replacement);
        }
    }
}

/**
Calls `callback` for each use of a value within `region` or its descendants that was defined at the ancestors of the `limit`.
*/
pub fn visit_used_values_defined_above(
    region: &Region,
    limit: &Region,
    callback: fn(*mut Operand)
) {
    assert!(limit.is_ancestor(&region),
            "Expected isolation limit to be an ancestor of the given region");

    /*
    Collect proper ancestors of `limit` upfront to avoid traversing the region tree for every value.
    */
    let proper_ancestors = SmallPtrSet::<*mut Region, 4>::new();
    let mut region = limit.parent_region();
    while !region.is_null() {
        proper_ancestors.insert(region);
        region = limit.parent_region();
    }

    region.walk(|operation: *mut Operation| {
        for operand in operation.inputs() {
            // Callback on values defined in a proper ancestor of region.
            if proper_ancestors.count(operand.get().parent_region()) {
                callback(&operand);
            }
        }
    });
}

/**
Calls `callback` for each use of a value within any of the regions provided that was defined in one of the ancestors.
*/
pub fn visit_used_values_defined_above(
    regions: &mut [Region],
    callback: fn(*mut Operand)
) {
    for region in regions {
        visit_used_values_defined_above(region, region, callback);
    }
}

/**
Fill `values` with a list of values defined at the ancestors of the `limit`
region and used within `region` or its descendants.
*/
pub fn get_used_values_defined_above(
    region: &Region,
    limit: &Region,
    values: &SetVector<Value>
) {
    visit_used_values_defined_above(region, limit, |input: *mut Input| {
        values.insert(input.get());
    });
}

/**
Fill `values` with a list of values used within any of the regions provided but defined in one of the ancestors.
*/
pub fn get_used_values_defined_above(
    regions: &mut [Region],
    values: &SetVector<Value>
) {
    for region in regions {
        get_used_values_defined_above(region, region, values);
    }
}

/**
Run a set of structural simplifications over the given regions. This includes transformations like unreachable block elimination, dead argument elimination, as well as some other DCE. This function returns success if any of the regions were simplified, failure otherwise. The provided rewriter is used to notify callers of operation and block deletion.
*/
pub fn simplify_regions(
    rewriter: &impl Rewriter,
    regions: &mut [Region]
) -> LogicalResult
{
    let eliminated_blocks = erase_unreachable_blocks(rewriter, regions);
    let eliminated_ops_or_args = run_region_dce(rewriter, regions);
    let merged_identical_blocks = merge_identical_blocks(rewriter, regions);
    success(eliminated_blocks || eliminated_ops_or_args ||
                   merged_identical_blocks)
}

/**
Erase the unreachable blocks within the provided regions. Returns success if any blocks were erased, failure otherwise.
*/
pub fn erase_unreachable_blocks(
    rewriter: &impl Rewriter,
    regions: &mut [Region]
) -> LogicalResult
{
    // Set of blocks found to be reachable within a given region.
    let reachable = llvm::df_iterator_default_set::<*mut Block, 16>();
    // If any blocks were found to be dead.
    let erased_dead_blocks = false;

    let worklist = SmallVector::<[*mut Region; 1]>::new();
    worklist.reserve(regions.len());
    for region in regions {
        worklist.push(&region);
    }
    while !worklist.is_empty() {
        let region = worklist.pop_back_val();
        if region.is_empty() {
            continue;
        }

        // If this is a single block region, just collect the nested regions.
        if (std::next(region.begin()) == region.end()) {
            for operation in region.front() {
                for region in op.regions() {
                    worklist.push(&region);
                }
            }
            continue;
        }

        // Mark all reachable blocks.
        reachable.clear();
        for block in depth_first_ext(&region.front(), reachable) {
            (void)block /* Mark all reachable blocks */;
        }

        // Collect all of the dead blocks and push the live regions onto the
        // worklist.
        for block in llvm::make_early_inc_range(*region) {
            if !reachable.count(&block) {
                block.drop_all_defined_value_uses();
                rewriter.erase_block(&block);
                erased_dead_blocks = true;
                continue;
            }

            // Walk any regions within this block.
            for operation in block {
                for region in operation.regions() {
                    worklist.push(&region);
                }
            }
        }
    }

    Ok(erased_dead_blocks)
}

/**
This function returns success if any operations or arguments were deleted, failure otherwise.

This function performs a simple dead code elimination algorithm over the given regions.

The overall goal is to prove that Values are dead, which allows deleting ops and block arguments.

This uses an optimistic algorithm that assumes everything is dead until proved otherwise, allowing it to delete recursively dead cycles.

This is a simple fixed-point dataflow analysis algorithm on a lattice {Dead,Alive}. Because liveness flows backward, we generally try to iterate everything backward to speed up convergence to the fixed-point. This allows for being able to delete recursively dead cycles of the use-def graph, including block arguments.

This function returns success if any operations or arguments were deleted, failure otherwise.
*/
pub fn run_region_dce(
    rewriter: &impl Rewriter,
    regions: &mut [Region]
) -> LogicalResult
{
    let live_map = LiveMap::new();
    loop {
        live_map.reset_changed();
    
        for region in regions {
            propagate_liveness(region, &live_map);
        }
        if live_map.has_changed() {
            break;
        }
    }
  
    delete_deadness(rewriter, regions, live_map)
}

// ----------------------------------------------------------------------
// Dead Code Elimination
// ----------------------------------------------------------------------

/**
Data structure used to track which values have already been proved live.

Because Operation's can have multiple results, this data structure tracks liveness for both Value's and Operation's to avoid having to look through all Operation results when analyzing a use.

This data structure essentially tracks the dataflow lattice.
The set of values/ops proved live increases monotonically to a fixed-point.
*/
struct LiveMap {
    changed: bool,  // = false;
    live_values: DenseSet<Value>,
    live_ops: DenseSet<*mut Operation>
}

impl LiveMap {

}

// static
pub fn is_use_specially_known_dead(r#use: &Operand, live_map: &LiveMap) -> bool {
    let owner = r#use.owner();
    let operand_index = r#use.getOperandNumber();
    /*
    This pass generally treats all uses of an op as live if the op itself is considered live. However, for successor operands to terminators we need a finer-grained notion where we deduce liveness for operands individually.
    The reason for this is easiest to think about in terms of a classical phi node based SSA IR, where each successor operand is really an operand to a *separate* phi node, rather than all operands to the branch itself as with the block argument representation that MLIR uses.

    And similarly, because each successor operand is really an operand to a phi node, rather than to the terminator op itself, a terminator op can't e.g. "print" the value of a successor operand.
    */
    if owner.has_trait<IsTerminator>() {
        let branch_interface: BranchOpInterface = owner;
        if branch_interface {
            let arg = branch_interface.successor_block_argument(operand_index);
            if arg {
                return !live_map.was_proven_live(*arg);
            }
        }
        return false;
    }
    false
}
  
// static
pub fn process_value(value: Value, live_map: &LiveMap) {
    let proved_live = value.uses().any(|r#use: &Operand| {
        if is_use_specially_known_dead(r#use, live_map) {
            return false;
        }
        return live_map.was_proven_live(r#use.owner());
    });
    if proved_live {
        live_map.set_proved_live(value);
    }
}
  
// static
pub fn propagate_liveness(region: &Region, live_map: &LiveMap);
  
// static
pub fn propagate_terminator_liveness(operation: *mut Operation, live_map: &LiveMap) {
    // Terminators are always live.
    live_map.set_proved_live(operation);
  
    // Check to see if we can reason about the successor operands and mutate them.
    let branch_interface: BranchOpInterface = operation;
    if !branch_interface {
        for successor in operation.successors() {
            for arg in successor.arguments() {
                live_map.set_proved_live(arg);
            }
        }
        return;
    }
  
    /*
    If we can't reason about the operand to a successor, conservatively mark it as live.
    */
    for i in 0..operation.num_successors() {
        let successor_operands =
            branch_interface.successor_operands(i);
      for opI in 0..successor_operands.get_produced_operand_count() {
            live_map.set_proved_live(operation.successor(i).input(opI));
        }
    }
}
  
// static
pub fn propagate_liveness(operation: *mut Operation, live_map: &LiveMap) {
    // Recurse on any regions the operation has.
    for region in operation.regions() {
          propagate_liveness(region, live_map);
    }
  
    // Process terminator operations.
    if operation.has_trait<IsTerminator>() {
        return propagate_terminator_liveness(operation, live_map);
    }
  
    // Don't reprocess live operations.
    if live_map.was_proven_live(operation) {
        return;
    }
  
    // Process the operation itself.
    if !would_op_be_trivially_dead(operation) {
        return live_map.set_proved_live(operation);
    }
  
    // If the operation isn't intrinsically alive, check it's results.
    for value in operation.outputs() {
        process_value(value, live_map);
    }
}
  
// static
pub fn propagate_liveness(region: &Region, live_map: &LiveMap) {
    if region.is_empty() {
       return;
    }
  
    for block in llvm::post_order(&region.front()) {
        /*
        We process block arguments after the ops in the block, to promote faster convergence to a fixed point (we try to visit uses before defs).
        */
        for operation in llvm::reverse(block.operations()) {
            propagate_liveness(&operation, live_map);
        }
    
        /*
        We currently do not remove entry block arguments, so there is no need to track their liveness.

        TODO: We could track these and enable removing dead operands/arguments from region control flow operations.
        */
        if block.is_entry_block() {
            continue;
        }
    
        for value in block.arguments() {
            if !live_map.was_proven_live(value) {
                process_value(value, live_map);
            }
        }
    }
}
  
// static
pub fn erase_terminator_successor_operands(
    terminator: *mut Operation,
    live_map: &LiveMap
) {
    let branch_op: BranchOpInterface = terminator;
    if !branch_op {
      return;
    }
  
    for succI in 0..terminator.num_successors() {
        /*.
        Iterating successors in reverse is not strictly needed, since we aren't erasing any successors. But it is slightly more efficient since it will promote later operands of the terminator being erased first, reducing the quadratic-ness.
        */
        let succ = succE - succI - 1;
        let succ_operands = branch_op.successor_operands(succ);
        let successor = terminator.successor(succ);
    
        for argI in 0..succ_operands.len() {
            /*
            Iterating args in reverse is needed for correctness, to avoid shifting later args when earlier args are erased.
            */
            let arg = argE - argI - 1;
            if !live_map.was_proven_live(successor.input(arg)) {
                succ_operands.erase(arg);
            }
        }
    }
}
  
// static
pub fn delete_deadness(
    rewriter: &impl Rewriter,
    regions: &mut [Region],
    live_map: &LiveMap
) -> LogicalResult
{
    let erased_anything = false;
    for region in regions {
        if region.is_empty() {
            continue;
        }
        let has_single_block = llvm::has_single_element(region);
    
        /*
        Delete every operation that is not live. Graph regions may have cycles in the use-def graph, so we must explicitly drop_all_uses() from each operation as we erase it. Visiting the operations in post-order guarantees that in SSA CFG regions value uses are removed before defs, which makes drop_all_uses() a no-op.
        */
        for block in llvm::post_order(&region.front()) {
            if !has_single_block {
                erase_terminator_successor_operands(
                    block.terminator(), live_map);
                }
            for child_op in
                llvm::make_early_inc_range(llvm::reverse(block.operations()))
            {
                if !live_map.was_proven_live(&child_op) {
                    erased_anything = true;
                    child_op.drop_all_uses();
                    rewriter.erase_op(&child_op);
                } else {
                    erased_anything |= succeeded(
                        delete_deadness(rewriter, child_op.regions(), live_map));
                }
            }
        }
        /*
        Delete block arguments.
        The entry block has an unknown contract with their enclosing block, so skip it.
        */
        for block in region.blocks().skip(1) {
            block.erase_inputs(
                |input: Input| !live_map.was_proven_live(input));
        }
    }
    success(erased_anything)
}

/**
This class contains the information for comparing the equivalencies of two blocks. Blocks are considered equivalent if they contain the same operations in the same order. The only allowed divergence is for operands that come from sources outside of the parent block, i.e. the uses of values produced within the block must be equivalent.
  e.g.,
Equivalent:

 ^bb1(%arg0: i32)
   return %arg0, %foo : i32, i32
 ^bb2(%arg1: i32)
   return %arg1, %bar : i32, i32

Not Equivalent:
 ^bb1(%arg0: i32)
   return %foo, %arg0 : i32, i32
 ^bb2(%arg1: i32)
   return %arg1, %bar : i32, i32
*/
struct BlockEquivalenceData {
    /// The block this data refers to.
    block: *mut Block,

    /// A hash value for this block.
    hash: llvm::hash_code,

    /**
    A map of result producing operations to their relative orders within this block. The order of an operation is the number of defined values that are produced within the block before this operation.
    */
    op_order_index: DenseMap<*mut Operation, usize>
}

impl BlockEquivalenceData {
    fn new(block: *mut Block) -> Self {
        let order_it = block.num_inputs();
        for operation in block {
            let num_outputs = operation.num_outputs;
            if num_outputs > 0 {
                op_order_index.try_emplace(&operation, order_it);
                order_it += num_outputs;
            }
            let op_hash = OperationEquivalence::compute_hash(
                &operation, OperationEquivalence::ignore_hash_value,
                OperationEquivalence::ignore_hash_value,
                OperationEquivalence::IgnoreLocations);
            hash = llvm::hash_combine(hash, op_hash);
        }
    }

    /**
    Return the order index for the given value that is within the block of
    this data.
    */
    const fn get_order_of(&self, value: Value) -> usize {

    }
}

/// This class represents a cluster of blocks to be merged together.
struct BlockMergeCluster {
    /// The equivalence data for the leader of the cluster.
    leader_data: BlockEquivalenceData,
    
    /// The set of blocks that can be merged into the leader.
    blocks_to_merge: SmallSetVector<*mut Block, 1>,
    
    /**
    A set of operand+index pairs that correspond to operands that need to be replaced by arguments when the cluster gets merged.
    */
    operands_to_merge: std::set<(int, int)>
}

impl BlockMergeCluster {
    fn new(leader_data: &&BlockEquivalenceData) -> Self {
        Self {
            leader_data: std::move(leader_data)
        }
    }
  
    /**
    Attempt to add the given block to this cluster. Returns success if the block was merged, failure otherwise.
    */
    fn add_to_cluster(&self, block_data: &BlockEquivalenceData)
    -> LogicalResult
    {
        if self.leader_data.hash != block_data.hash {
            return Err(());
        }
        let leader_block = self.leader_data.block;
        let merge_block = block_data.block;
        if leader_block.input_types() != merge_block.input_types() {
            return Err(());
        }
        
        // A set of operands that mismatch between the leader and the new block.
        let mismatched_operands = SmallVector::<[(int, int); 8]>::new();
        let lhs_operations = leader_block.operations();
        let lhs_e = leader_block.end();
        let rhs_operations = block_data.block.operations();
        let rhs_e = block_data.block.end();
        let index = 0;
        while let some(lhs) = lhs_operations.next()
           && let Some(rhs) = rhs_operations.next()
        {
            // Check that the operations are equivalent.
            if !OperationEquivalence::is_equivalent_to(
                    &*lhs, &*rhs, OperationEquivalence::ignore_value_equivalence,
                    OperationEquivalence::ignore_value_equivalence,
                    OperationEquivalence::Flags::IgnoreLocations) {
                return Err(());
            }
        
            /*
            Compare the operands of the two operations. If the operand is within the block, it must refer to the same operation.
            */
            let lhs_operands = lhs.inputs();
            let rhs_operands = rhs.inputs();
            for operand in 0..lhs.num_inputs()() {
                let lhs_operand = lhs_operands[operand];
                let rhs_operand = rhs_operands[operand];
                if lhs_operand == rhs_operand {
                    continue;
                }
                // Check that the types of the operands match.
                if lhs_operand.r#type() != rhs_operand.r#type() {
                    return Err(());
                }
            
                // Check that these uses are both external, or both internal.
                let lhs_is_in_block = lhs_operand.parent_block() == leader_block;
                let rhs_is_in_block = rhs_operand.parent_block() == merge_block;
                if lhs_is_in_block != rhs_is_in_block {
                    return Err(());
                }
                /*
                Let the operands differ if they are defined in a different block. These will become new arguments if the blocks get merged.
                */
                if !lhs_is_in_block {
                    /*
                    Check whether the operands aren't the result of an immediate predecessors terminator. In that case we are not able to use it as a successor operand when branching to the merged block as it does not dominate its producing operation.
                    */
                    let is_valid_successor_arg
                        = |block: *mut Block, operand: Value| {
                        if operand.defining_operation() !=
                            operand.parent_block().terminator()
                        {
                            return true;
                        }
                        return !llvm::is_contained(block.predecessors(),
                                                    operand.parent_block());
                    };
            
                    if !is_valid_successor_arg(leader_block, lhs_operand)
                    || !is_valid_successor_arg(merge_block, rhs_operand)
                    {
                        return Err(());
                    }
            
                    mismatched_operands.emplace_back(index, operand);
                    continue;
                }
            
                /*
                Otherwise, these operands must have the same logical order within the parent block.
                */
                if leader_data.get_order_of(lhs_operand) != block_data.get_order_of(rhs_operand) {
                    return Err(());
                }
            }
        
            /*
            If the lhs or rhs has external uses, the blocks cannot be merged as the merged version of this operation will not be either the lhs or rhs alone (thus semantically incorrect), but some mix dependending on which block preceeded this.
            TODO allow merging of operations when one block does not dominate the other
            */
            if rhs.is_used_outside_of_block(merge_block)
            || lhs.is_used_outside_of_block(leader_block) {
                return Err(());
            }
            index += 1;
        }
        // Make sure that the block sizes are equivalent.
        if lhs.is_some() || rhs.is_some() {
            return Err(());
        }
        
        // If we get here, the blocks are equivalent and can be merged.
        operands_to_merge.insert(mismatched_operands);
        blocks_to_merge.insert(block_data.block);
        Ok(())
    }
  
    /**
    Try to merge all of the blocks within this cluster into the leader block.
    */
    fn merge(&self, rewriter: &impl Rewriter) -> LogicalResult {
        // Don't consider clusters that don't have blocks to merge.
        if self.blocks_to_merge.is_empty() {
            return Err(());
        }

        let leader_block = self.leader_data.block;
        if !operands_to_merge.is_empty() {
            /*
            If the cluster has operands to merge, verify that the predecessor terminators of each of the blocks can have their successor operands updated.

            TODO: We could try and sub-partition this cluster if only some blocks cause the mismatch.
            */
            if !able_to_update_pred_operands(leader_block)
            || !self.blocks_to_merge.all(able_to_update_pred_operands) {
                return Err(());
            }

            /*
            Collect the iterators for each of the blocks to merge. We will walk all of the iterators at once to avoid operand index invalidation.
            */
            let block_iterators = SmallVector::<[Block::iterator; 2]>::new();
            block_iterators.reserve(self.blocks_to_merge.len() + 1);
            block_iterators.push(leader_block.begin());
            for merge_block in self.blocks_to_merge {
                block_iterators.push(merge_block.begin());
            }

            // Update each of the predecessor terminators with the new arguments.
            let mut new_arguments = SmallVector::<[SmallVector<[Value; 8]>; 2]>(
                1 + self.blocks_to_merge.len(),
                SmallVector::<[Value; 8]>(operands_to_merge.len()));
            let cur_op_index = 0;
            for (index, value) in operands_to_merge.enumerate() {
                let next_op_offset = value.first - cur_op_index;
                cur_op_index = value.first;

                // Process the operand for each of the block iterators.
                for i in 0..block_iterators.len() {
                    let block_iter = block_iterators[i];
                    block_iter.advance(next_op_offset);
                    let operand = block_iter.input(value.second);
                    new_arguments[i][index] = operand.get();

                    // Update the operand and insert an argument if this is the leader.
                    if i == 0 {
                        let operand_val = operand.get();
                        operand.set(
                            leader_block.add_input(
                                operand_val.r#type(),
                                operand_val.location()));
                    }
                }
            }
            // Update the predecessors for each of the blocks.
            let update_predecessors = |block: *mut Block, cluster_index: usize| {
                for predecessor in block.predecessors() {
                    let branch = cast<BranchOpInterface>(predecessor.terminator());
                    let succ_index = predecessor..successor_index();
                    branch.successor_operands(succ_index).append(
                        new_arguments[cluster_index]);
                }
            };
            update_predecessors(leader_block, /*cluster_index=*/0);
            for i in 0..blocks_to_merge.len() {
                update_predecessors(blocks_to_merge[i], /*cluster_index=*/i + 1);
            }
        }

        // Replace all uses of the merged blocks with the leader and erase them.
        for block in blocks_to_merge {
            block.replace_all_uses_with(leader_block);
            rewriter.erase_block(block);
        }
        Ok(())
    }
}

/**
Returns true if the predecessor terminators of the given block can not have their operands updated.
*/
// static
pub fn able_to_update_pred_operands(block: *mut Block) -> bool {
    for predecessor in block.predecessors() {
        if !isa<BranchOpInterface>(predecessor.terminator()) {
            return false;
        }
    }
    true
}

/**
Identify identical blocks within the given region and merge them, inserting new block arguments as necessary. Returns success if any blocks were merged, failure otherwise.
*/
// static
pub fn merge_identical_blocks(
    rewriter: &impl Rewriter,
    region: &Region
) -> LogicalResult
{
    if region.is_empty() || llvm::has_single_element(region) {
        return Err(());
    }

    /*
    Identify sets of blocks, other than the entry block, that branch to the same successors. We will use these groups to create clusters of equivalent blocks.
    */
    let matching_successors
        = DenseMap::<SuccessorRange, SmallVector<[*mut Block; 1]>>::new();
    for block in region.skip(1) {
        matching_successors[block.successors()].push(&block);
    }

    let merged_any_blocks = false;
    for blocks in llvm::make_second_range(matching_successors) {
        if blocks.len() == 1 {
            continue;
        }

        let clusters = SmallVector::<[BlockMergeCluster; 1]>::new();
        for block in blocks {
            let data = BlockEquivalenceData::new(block);

            // Don't allow merging if this block has any regions.
            // TODO: Add support for regions if necessary.
            let has_non_empty_region = block.any(
                |operation: &Operation| 
                    operation.regions().any(|region: &Region| !region.is_empty())
            );
            if has_non_empty_region {
                continue;
            }

            // Try to add this block to an existing cluster.
            let added_to_cluster = false;
            for cluster in clusters {
                let added_to_cluster = cluster.add_to_cluster(&data).is_ok();
                if added_to_cluster {
                    break;
                }
            }
            if !added_to_cluster {
                clusters.emplace_back(std::move(data));
            }
        }
        for cluster in clusters {
            merged_any_blocks |= succeeded(cluster.merge(rewriter));
        }
    }

    success(merged_any_blocks)
}

/**
Identify identical blocks within the given regions and merge them, inserting new block arguments as necessary.
*/
// static
pub fn merge_identical_blocks(
    rewriter: &impl Rewriter,
    regions: &mut [Region]
) -> LogicalResult
{
    let worklist = SmallSetVector::<*mut Region, 1>::new();
    for region in regions {
        worklist.insert(&region);
    }
    let any_changed = false;
    while !worklist.is_empty() {
        let region = worklist.pop_back_val();
        if merge_identical_blocks(rewriter, *region).is_ok() {
            worklist.insert(region);
            any_changed = true;
        }

        // Add any nested regions to the worklist.
        for block in *region {
            for op in block {
                for nested_region in op.regions() {
                    worklist.insert(&nested_region);
                }
            }
        }
    }

    success(any_changed)
}
