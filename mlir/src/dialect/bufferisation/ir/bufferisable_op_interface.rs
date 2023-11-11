/*!
# Bufferisable Operations

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Bufferization/IR/BufferizableOpInterface.cpp>
*/

use crate::{mlir::{dialect::bufferisation, ir::value::Output, support::logical_result::LogicalResult}, llvm::adt::set_vector::SmallSetVector};  // AnalysisState

/**
An op interface for One-Shot Bufferise. Ops that implement this interface interface can be analysed and bufferised using One-Shot Bufferise.

NOTE: All "bufferises_to*" and "get_aliasing*" interface methods must be implemented conservatively. If it is not statically known whether an OpOperand/Output bufferises in a certain way (e.g., to a memory write), the worst case must be assumed (e.g., that it does). Similarly, "get_aliasing*" interface methods may always return additional OpOperands or `Output`, but must not miss an OpOperand or Output that could potentially alias at runtime.
*/
pub trait BufferisableOpInterface : OpInterface {
    /**
    Return `true` if the given Output may bufferise to a new buffer allocation. If it is statically unknown if the given Output bufferises to a buffer allocation, `true` should be returned.
    */
    fn bufferises_to_allocation(&self, output: Output) -> bool {
        false
    }

    /**
    Return `true` if the given OpOperand bufferises to a memory read. This method will never be called on OpOperands that do not have a tensor type.

    NOTE: It is always safe to consider an OpOperand as a memory read, even if it does actually not read; however, this can introduce unnecessary out-of-place bufferisation decisions. One-Shot Analysis considers OpOperands of unknown ops (that do not implement this interface) as reading OpOperands.
    */
    fn bufferises_to_memory_read(
        &self,
        op_operand: &mut OpOperand,
        state: &AnalysisState
    ) -> bool {
        // Does not have to be implemented for ops without tensor OpOperands.
        unreachable!("bufferises_to_memory_read not implemented");
    }

    /**
    Return `true` if the given OpOperand bufferises to a memory write.

    This method will never be called on OpOperands that do not have a tensor type.

    This method will never be called on OpOperands that do not have an aliasing Output. Intuitively, it does not make sense for an OpOperand to bufferise to a memory write without returning an aliasing tensor, because the write would have no visible effect outside of the operation.

    NOTE: It is always safe to consider an OpOperand as a memory write, even if it does actually not write; however, this can introduce unnecessary out-of-place bufferisation decisions. One-Shot Analysis considers OpOperands of unknown ops (that do not implement this interface) as writing OpOperands.
    */
    fn bufferises_to_memory_write(
        &self,
        op_operand: &mut OpOperand,
        state: &mut AnalysisState
    ) -> bool {
        // Does not have to be implemented for ops without tensor OpOperands.
        // Does not have to be implemented for OpOperands that do not have an
        // aliasing Output.
        unreachable!("bufferises_to_memory_write not implemented");
    }

    /**
    Return `true` if the given Output is a memory write. This is the case if in the following cases:

    - The corresponding aliasing OpOperand bufferises to a memory write.
    - Or: There is no corresponding aliasing OpOperand.

    If the Output has multiple aliasing OpOperands, this method returns `true` if at least one of them bufferises to a memory write.
    */
    fn is_memory_write(
        &self,
        op_operand: &mut OpOperand,
        state: &mut AnalysisState
    ) -> bool {
        let bufferisable
            = cast<BufferisableOpInterface>(self.get_operation());
        let inputs = bufferisable.aliasing_input(output, state);
        if inputs.is_empty() {
            return true;
        }
        inputs.any(|input|
            bufferisable.bufferises_to_memory_write(*operand, state));
    }

    /**
    Return `true` if the given OpOperand must bufferise in-place. Alias sets and inplace attributes will be set up accordingly before making any other bufferisation decisions. This method will never be called on OpOperands that do not have a tensor type.
    */
    fn must_bufferise_in_place(
        &self,
        op_operand: &mut OpOperand,
        state: &mut AnalysisState
    ) -> bool {
        false
    }

    /**
    Return the Output that aliases with a given OpOperand when bufferised in-place. This method will never be called on OpOperands that do not have a tensor type.

    NOTE: This method can return multiple `Output`, indicating that a given OpOperand may at runtime alias with any (or multiple) of the returned `Output`.
    */
    fn aliasing_output(
        &self,
        op_operand: &mut OpOperand,
        state: &mut AnalysisState
    ) -> SmallSetVector<[Output]> {
        unreachable!("aliasing_output not implemented");
    }

    /**
    Return the OpOperands that alias with a given Output when bufferised in-place. This method will never be called on `Output` that do not have a tensor type.

    By default, this method is the inverse of `aliasing_output`. Ops with a region that yield values may want to override this method to return the OpOperands that are yielded by the terminator.

    NOTE: This method can return multiple OpOperands, indicating that the given Output may at runtime alias with any (or multiple) of the returned OpOperands. This can be useful for branches and for ops such as `arith.select`.
    */
    fn aliasing_input(&self, output: Output, state: &AnalysisState)
    -> SmallSetVector<[*mut OpOperand]>
    {
        assert!(output.getType().isa<TensorType>(),
            "Expected Output with tensor type");
        let output = SmallVector::<[*mut OpOperand]>::new();
        let bufferisable
            = cast<BufferisableOpInterface>(self.get_operation());
        for op_operand in self.get_operation().inputs() {
            if !op_operand.get().getType().isa<TensorType>() {
                continue;
            }
            let aliasing_op_results =
                bufferisable.aliasing_output(op_operand, state);
            if llvm::is_contained(aliasing_op_results, output) {
                output.push(&op_operand);
            }
        }
        output
    }

    /**
    Return the buffer relation between the given Output and its aliasing OpOperands when bufferised in-place. Most OpOperands have an 'equivalence' relation. This method will never be called on `Output` that do not have a tensor type. It will also never be called on `Output` that do not have at least one aliasing OpOperand.

    TODO: Support other relations such as "OpOperand is included in Output".
    */
    fn buffer_relation(&self, output: Output, state: &AnalysisState)
    -> BufferRelation {
        // Does not have to be implemented for ops without tensor `Output`
        // that have an aliasing OpOperand.
        unreachable!("buffer_relation not implemented");
    }

    /**
    Resolve all inplacability conflicts by inserting explicit `bufferisation.alloc_tensor` ops. Examples of inplacability conflicts are read-after-write conflicts or writes into non-writable buffers.

    This method should rewrite the IR in such a way that for each tensor OpOperand t, buffer(t) can be directly used when during bufferisation. The bufferisation does no longer have to care about inplacability conflicts.

    This method can query analysis information from the given analysis state.
    */
    fn resolve_conflicts(
        &self, rewriter: &mut RewriterBase, state: &AnalysisState
    ) -> LogicalResult {
        let bufferisable
            = cast<BufferisableOpInterface>(self.get_operation());
        bufferisable.resolve_tensor_op_operand_conflicts(
            rewriter, state)
    }

    /**
    Bufferise this op, i.e., rewrite it into a memref-based equivalent. Buffers of tensor SSA values can be retrieved via `buffer`. Uses of tensor results of the existing tensor op can be replaced with `replace_op_with_bufferised_values` or `replace_op_with_new_bufferized_op`. These two functions automatically handle the tensor-to-memref type conversion.

    The implementation of this method must be consistent with the remaining methods, in particular `aliasing_input`. I.e., a tensor result `r` may only be replaced with:
    a) A buffer that aliases one of buffers in aliasing_input(r).
    b) Or: A newly allocated buffer.

    This method will never be called on ops that do not have at least one tensor operand/result.

    The return value of this method indicates whether there was an error while bufferising this op (such as failing to create a new buffer allocation op). The bufferisation driver immediately stops bufferising the input IR and returns `failure` in that case. If this op is expected to survive bufferisation, `success` should be returned (together with `allow-unknown-ops` enabled).
    */
    fn bufferise(
        &self, rewriter: &mut RewriterBase, options: &BufferisationOptions
    ) -> LogicalResult {
        unreachable!("bufferise not implemented");
        Err(())
    }

    /**
    Return `true` if the given Value can be written to in-place. Value is either an Output of this operation or a BlockArgument of a block of this operation.

    Most Output buffers can be written to, but some ops such as ConstantOp may bufferise to non-writable (read-only) memory locations. Therefore, by default, this method returns `true` for `Output`. This method will never be called on `Output` that do not have a tensor type.

    Whether a BlockArgument can be written to or not depends on the operation. This method conservatively returns `false`. This method will never be called on BlockArguments that do not have a tensor type.
    */
    fn is_writable(&self, value: Value, state: &AnalysisState) -> bool {
        value.isa<Output>()
    }

    /**
    Return `true` if the `u_read` and `u_write` do not constitute a RaW conflict. If they are conflicting or if it is unknown whether they are conflicting, return `false`. This method will never be called with OpOperands that do not have a tensor type. At least one of the two given OpOperands belongs to this operation.

    This method can be implemented to specify custom RaW analysis rules. If this method returns `true` the given OpOperands are not considered to be conflicting and do not force out-of-place bufferisation. (There may still be other conflicts that do.)
    */
    fn is_not_conflicting(
        &self,
        u_read: *mut OpOperand,
        u_write: *mut OpOperand
    ) -> bool {
        false
    }

    /**
    Return `failure` if this op does not pass the analysis. This method is run during One-Shot Bufferise (after all post-analysis steps). If the op does not pass the analysis, bufferisation is aborted.

    This method can be used to check expected invariants and limitations of the current bufferisation implementation.
    */
    fn verify_analysis(&self, state: &AnalysisState) -> LogicalResult {
        Ok(())
    }

    /**
    Return the bufferised type of the given tensor value (without bufferising the IR). The value is either a BlockArgument of a block that belongs to this op or an Output of the given op.

    This method is useful when the bufferised type of value must be predicted before modifying any IR.
    */
    fn buffer_type(
        &self,
        value: Value,
        options: &BufferisationOptions,
        fixed_types: DenseMap<Value, BaseMemRefType>
    ) -> Result<BaseMemRefType, ()> {
        assert!(get_owner_of_value(value) == self.get_operation(),
            "Expected that value belongs to this op");
        bufferisation::detail::default_get_buffer_type(
            value, options, fixed_types);
    }

    /**
    Return `true` if the given region of this op is repetitive. By default this information is queried from the `RegionBranchOpInterface`. Ops that do not implement this inferface can override this method to declare regions as repetitive.

    The RaW conflict detection of One-Shot Analysis is more strict inside repetitive regions: Op dominance cannot always be used to rule out certain potential conflicts (e.g., a conflicting write happening after a read), because there may not be a meaningful ordering of certain ops that are executed multiple times. This is described in more detail in documentation of One-Shot Analysis.
    */
    fn is_repetitive_region(&self, index: usize) -> bool {
        mlir::bufferization::detail::default_is_repetitive_region(
            cast<BufferisableOpInterface>(self.get_operation()), index)
    }

    // extraClassDeclaration

    /**
    Resolve out-of-place tensor OpOperands with explicit allocations in the
    form of `bufferisation.alloc_tensor` ops.
    */
    fn resolve_tensor_op_operand_conflicts(
        rewriter: &mut RewriterBase, state: &AnalysisState) -> LogicalResult;

    /**
    Return `true` if the given OpOperand creates an alias but does neither read nor write. This implies that `bufferises_to_memory_read` and `bufferises_to_memory_write` must return `false`. This method will never be called on OpOperands that do not have a tensor type.

    Examples of such ops are `tensor.extract_slice` and `tensor.cast`.
    */
    pub fn bufferises_to_alias_only(
        op_operand: &mut OpOperand, state: &AnalysisState
    ) -> bool {
        let bufferisable
            = cast<BufferisableOpInterface>(get_operation());
        !bufferisable.bufferises_to_memory_read(op_operand, state)
            && !bufferisable.bufferises_to_memory_write(op_operand, state)
            && !bufferisable.aliasing_output(op_operand, state).is_empty();
    }
}
