/*!
# Bufferisation Operation Definitions

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Bufferization/IR/BufferizationOps.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Bufferization/IR/BufferizationOps.cpp>
*/

/*
----------------------------------------------------------------------
AllocTensor
----------------------------------------------------------------------
*/

use crate::{mlir::{ir::{
    builtins::types::Tensor,
    value::{Value, Output},
}, interfaces::{copy_op_interface::CopyOpInterface, side_effect_interfaces::MemoryEffectsOpInterface}, support::logical_result::LogicalResult}, llvm::adt::set_vector::SmallSetVector};

use super::{allocation_op_interface::AllocationOpInterface, bufferisable_op_interface::BufferisableOpInterface};

/**
Buffer allocation in tensor land.

`bufferisation.alloc_tensor` materialises an uninitialized tensor with a
given shape (dynamic or static). It always bufferises to a new buffer
allocation of the given shape. The optional `copy` operand specifies the
contents of the tensors. If no `copy` operand is specified, reading from the
result of an `alloc_tensor` op yields an undefined value.

If `copy` is specified, no dynamic sizes should be passed, since they are
the same as the dynamic sizes of the `copy` operand.

`alloc_tensor` is a helper op for bufferisation. The operation is provided as an anchor that marks the beginning of a new tensor SSA use-def chain. It can be used to control in-place bufferisation decisions during One-Shot Bufferise: The bufferised result of a `bufferisation.alloc_tensor` does not alias with any other buffer, so it can be used to resolve read-after-write conflicts that would have been introduced by the in-place bufferisation of another op.

The optional `memory_space` attribute specifies the memory space when
bufferising this op. The memory space is inferred from `copy` if specified.
If neither `copy` nor `memory_space` is specified, the default memory space
is used during bufferisation.

The optional `size_hint` operand specifies the number of non-zero elements
for sparse tensors. The value of `size_hint` should be not less than 1 and
not larger than the linear size of the corresponding dense tensor type. If
this requirement is not met, the behaviour of the operator is undefined.

Both dense and sparse tensor types are supported. The result of a
`bufferisation.alloc_tensor` is a tensor value that can be used like any
other tensor value. In practice, it is often used as the "out" operand of
another op. Sparse tensor allocations should always be used in a local
construction operation and never escape the function boundary directly.

# Examples

```mlir
%c = bufferisation.alloc_tensor(%d1, %d2) : tensor<?x?xf32, #SparseMatrix>
%0 = linalg.matmul
    ins(%a, %b: tensor<?x?xf32, #SparseMatrix>, tensor<?x?xf32, #SparseMatrix>)
    outs(%c: tensor<?x?xf32, #SparseMatrix>) -> tensor<?x?xf32, #SparseMatrix>
return %0 : tensor<?x?xf32, #SparseMatrix>
```

```mlir
%c = bufferisation.alloc_tensor(%d1, %d2) size_hint = %noe
    : tensor<?x?xf32, #SparseMatrix>
```
*/
#[mlir(
    traits = [AttrSizedOperandSegments]
)]
pub struct AllocTensor<T, const N: usize> {
    dynamic_sizes: Variadic<Index>,
    copy: Option<Tensor<T, N>>,
    size_hint: Option<Index>,
    #[attribute]
    memory_space: Option<_>,
    #[output]
    output: Tensor<T, N>  // renamed from `result`
}

//   let builders = [
//     // Build an op without `copy` or `memory_space` or `size_hint`.
//     OpBuilder<(ins "RankedTensorType":$type, "ValueRange":$dynamic_sizes)>,

//     // Build an op without `memory_space` or `size_hint`.
//     OpBuilder<(ins "RankedTensorType":$type, "ValueRange":$dynamic_sizes,
//                    "Value":$copy)>,

//     // Build an op without `size_hint`.
//     OpBuilder<(ins "Tensor":$type, "ValueRange":$dynamic_sizes,
//                    "Value":$copy, "IntegerAttr":$memory_space)>,
//   ];

impl AllocTensor {
    RankedTensorType get_type() {
      return get_result().get_type().cast<RankedTensorType>();
    }

    /// Returns true if the size of the tensor is dynamic at `index`
    bool is_dynamic_dim(index: usize) {
      return get_type().is_dynamic_dim(index);
    }

    /**
    Returns the argument position that contains the dynamic size of the tensor at dimension `index`. Asserts that the shape is dynamic at that `index`.
    */
    usize get_index_of_dynamic_size(index: usize) {
        assert!(!getCopy() && "no dim sizes specified when copying a tensor");
        assert!(is_dynamic_dim(index) && "expected dynamic size");
        ArrayRef<i64> shape = get_type().getShape();
        return std::count_if(
            shape.begin(), shape.begin() + index,
            [&](i64 size) { return ShapedType::is_dynamic(size); });
        }

    /**
    Return the Value of the dynamic size of the tensor at dimension `index`. Asserts that the shape is dynamic at that `index.
    */
    Value get_dynamic_size(b: &mut OpBuilder, index: usize);

    /**
    Assert that the size of the result tensor is static at `index` and return the shape.
    */
    i64 get_static_size(index: usize) {
        assert!(!is_dynamic_dim(index) && "expected static size");
        return get_type().getShape()[index];
    }
}

impl Verify for AllocTensor {

}

impl Canonicalise for AllocTensor {

}

impl AssemblyFormat for AllocTensor {

}

impl BufferisableOpInterface for AllocTensor {
    LogicalResult bufferise(rewriter: &mut RewriterBase,
        options: &BufferisationOptions);

    bool is_memory_write(output: Output, state: &AnalysisState);

    bool bufferises_to_allocation(output: Output) { return true; }

    bool bufferises_to_memory_read(op_operand: &mut OpOperand,
                state: &AnalysisState);

    bool bufferises_to_memory_write(op_operand: &mut OpOperand,
                state: &AnalysisState);

    SmallVector<Output> get_aliasing_output(
    op_operand: &mut OpOperand, state: &AnalysisState);

    FailureOr<BaseMemRefType> get_buffer_type(
    value: Value, options: &BufferisationOptions,
    const DenseMap<Value, BaseMemRefType> &fixedTypes);
}

impl ReifyRankedShapedTypeOpInterface for AllocTensor {

}

/*
----------------------------------------------------------------------
Clone
----------------------------------------------------------------------
*/

/**
Clones the data in the input view into an implicitly defined output view.

# Usage

```mlir
%arg1 = bufferisation.clone %arg0 : memref<?xf32> to memref<?xf32>
```

Valid implementations of this operation may alias the input and output views or create an actual copy. Mutating the source or result of the clone operation after the clone operation thus leads to undefined behaviour.
*/
#[mlir(
    assembly_format = "$input attr-dict `:` type($input) `to` type($output)"
)]
pub struct Clone<> {
    #[input]
    input: AnyRankedOrUnrankedMemRef, // <, "", []>,
    #[output]
    output: AnyRankedOrUnrankedMemRef  // Arg<, "", []>
}

//OpBuilder
// impl Build<(Value)> for Clone {
//     fn build() {
//         build($_builder, $_state, value.get_type(), value)
//     }
// }

impl Clone {
    // Value getSource() { return getInput(); }
    // Value getTarget() { return getOutput(); }
}

impl Fold for Clone {

}

impl Canonicalise for Clone {

}

impl CopyOpInterface for Clone {

}

impl MemoryEffectsOpInterface for Clone {
    
}

impl AllocationOpInterface for Clone {
    // fn build_dealloc
    // fn build_clone
}

/*
----------------------------------------------------------------------
DeallocTensor
----------------------------------------------------------------------
*/

/**
Releases underlying sparse storage format of given tensor.

`bufferisation.dealloc_tensor` is a buffer deallocation in tensor land. This op can be used for manual buffer deallocation. Some bufferisations (such as One-Shot Bufferise) take care of buffer deallocation, in which case this op is usually not needed. Details can be found in the documentation of the respective bufferisation passes.

In case of a dense tensor, this op lowers to a `memref.dealloc` op during bufferisation.

In case of a sparse tensor, this op releases the underlying sparse storage format for a tensor that materialised earlier through a `new` operation, a `convert` operation with annotated destination tensor type (unless the convert is folded away), or a `bufferisation.alloc_tensor` operation. The release operation should only be called once for any materialised tensor.
After this operation, any subsequent `memref` querying operation on the tensor returns undefined results.

# Example

```mlir
bufferisation.dealloc_tensor %tensor : tensor<1024x1024xf64, #CSR>
```
*/
#[mlir(
    assembly_format = "$tensor attr-dict `:` type($tensor)"
)]
pub struct DeallocTensor {
    #[input]
    tensor: Tensor<_, _>
  // output
}

impl DeallocTensor {

}

impl BufferisableOpInterface for DeallocTensor {
    fn bufferises_to_memory_read(
        &self,
        op_operand: &mut OpOperand,
        state: &AnalysisState
    ) -> bool {
            false
    }

    fn bufferises_to_memory_write(
        &self,
        op_operand: &mut OpOperand,
        state: &mut AnalysisState
    ) -> bool {
        false
    }

    fn aliasing_output(
        &self,
        op_operand: &mut OpOperand,
        state: &mut AnalysisState
    ) -> SmallSetVector<[Output]> {
        Default::default()
    }

    fn bufferise(
        &self, rewriter: &mut RewriterBase, options: &BufferisationOptions
    ) -> LogicalResult {
        
    }
}

/*
----------------------------------------------------------------------
ToTensor
----------------------------------------------------------------------
*/

/**
`memref` to tensor operation.

Create a tensor from a `memref`, making an independent copy of the element data. The result value is a tensor whose shape and element type match the memref operand.

The opposite of this op is `to_memref`. Together, these two ops are useful for source/target materialisations when doing type conversions involving tensors and memrefs.

# Examples

Produces a value of tensor<4x?xf32> type:

```mlir
%12 = bufferisation.to_tensor %10 : memref<4x?xf32, #layout, memspace0>
```

If tensor load is used in the bufferisation steps, mutating the source buffer after loading leads to undefined behaviour.
*/
#[mlir(
    traits = [
        // TypesMatchWith<"result type matches tensor equivalent of 'memref'",
        //                "memref", "result",
        //                "memref::getTensorTypeFromMemRefType($_self)">
    ],
    assembly_format = "$memref attr-dict `:` type($memref)"
)]
pub struct ToTensor<T, const N: usize> {
    /// Reference to load from.
    #[input]
    memref: MemRef<T, N>,  // AnyRankedOrUnrankedMemRef,  // Arg<, [MemRead]>,
    #[output]
    output: Tensor<T, N>  // renamed from `result1`
}

impl ToTensor {
    /// The result of a to_tensor is always a tensor.
    Tensor get_type() {
        Type resultType = get_result().get_type();
        if (resultType.isa<Tensor>())
            return resultType.cast<Tensor>();
        return {};
    }

    //===------------------------------------------------------------------===//
    // BufferisableOpInterface implementation
    //===------------------------------------------------------------------===//

    /*
    ToTensor conceptually loads a tensor from a memory location. The
    One-Shot analysis has no information about the memref that is loaded from
    by ToTensor. We have to assume that the loaded tensor may after
    bufferisation potentially alias with any other bufferised tensor. Since
    ToTensor and ToMemref have no aliasing OpOperand/Output pairs, this
    cannot be encoded directly in the analysis. However, declaring ToTensor
    results as not writable enforces a buffer copy and has the same effect.
    */

    LogicalResult bufferise(rewriter: &mut RewriterBase,
                            options: &BufferisationOptions) const {
    /*
    `to_tensor` cannot be bufferised. However, other ops that are using `to_tensor`'s result will eventually be bufferised. At that point, they
    will start using `to_tensor`'s memref operand. Once all users of
    `to_tensor` are bufferised, the op will not have any users anymore and
    DCE away. In case of partial bufferisation, `to_memref(to_tensor(x))`
    constructs may be left over. These are folded by the canonicaliser or
    FinalisingBufferise.
    */
      return success();
    }

    bool is_writable(value: Value, state: &AnalysisState) const {
        // It is unknown whether the memref operand is writable or not.
        return false;
    }

    FailureOr<BaseMemRefType> get_buffer_type(
        value: Value, options: &BufferisationOptions,
        const DenseMap<Value, BaseMemRefType> &fixedTypes) {
      return getMemref().get_type().cast<BaseMemRefType>();
    }
}

impl Fold for ToTensor {

}

impl Canonicalise for ToTensor {

}

impl BufferisableOpInterface for ToTensor {

}


/*
----------------------------------------------------------------------
ToMemref
----------------------------------------------------------------------
*/

/**
tensor to memref cast operation.

Casts a tensor to a memref.

# Examples

Result type is `memref<4x?xf32, #layout, 42>`:

```mlir
%12 = bufferisation.to_memref %10 : memref<4x?xf32, #layout, 42>
```

Note, that mutating the result of the `to_memref` operation leads to undefined behaviour.

This operation is a specialised variant of the built-in `unrealised_conversion_cast` and is intended for use in the context of gradual bufferisation.
*/
#[mlir(
    traits = [
        Pure,
        // TypesMatchWith<"type of 'tensor' is the tensor equivalent of 'memref'",
        //            "memref", "tensor",
        //            "memref::getTensorTypeFromMemRefType($_self)">
    ],
    assembly_format = "$tensor attr-dict `:` type($memref)"
)]
pub struct ToMemref<T, const N: usize> {
    #[input]
    tensor: Tensor<T, N>,
    #[output]
    memref:  MemRef<T, N>  // AnyRankedOrUnrankedMemRef
}

impl ToMemref {
    //===------------------------------------------------------------------===//
    // BufferisableOpInterface implementation
    //===------------------------------------------------------------------===//

    /*
    NOTE: ToMemref / ToTensor are temporary ops that are inserted at the bufferisation boundary. When One-Shot bufferisation is complete, there should be no such ops left over. If `allowUnknownOps` (or after running a partial bufferisation pass), such ops may be part of the resulting IR, but such IR may no longer be analyzable by One-Shot analysis.
    */

    bool bufferises_to_memory_read(op_operand: &mut OpOperand,
                                state: &AnalysisState) const {
      // It is unknown whether the resulting memref will be read or not.
      return true;
    }

    bool bufferises_to_memory_write(op_operand: &mut OpOperand,
                                 state: &AnalysisState) const {
      // It is unknown whether the resulting MemRef will be written or not.
      return true;
    }

    bool must_bufferise_in_place(op_operand: &mut OpOperand,
                              state: &AnalysisState) const {
      // ToMemrefOps always bufferise inplace.
      return true;
    }

    SmallVector<Output> get_aliasing_output(
        op_operand: &mut OpOperand, state: &AnalysisState) const {
      return {};
    }

    LogicalResult bufferise(rewriter: &mut RewriterBase,
                            options: &BufferisationOptions);
}

impl Fold for ToMemref {

}

impl Canonicalise for ToMemref {
    
}

impl BufferisableOpInterface for ToMemref {

}
