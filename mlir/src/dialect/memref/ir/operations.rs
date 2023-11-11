/*!
# MemRef Operaiton Definitions

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/MemRef/IR/MemRefOps.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp>
*/

/*
----------------------------------------------------------------------
AllocLike
----------------------------------------------------------------------
*/

use crate::mlir::{
    interfaces::{
        cast_interfaces::CastOpInterface,
        infer_type_op_interface::InferTypeOpInterface,
        view_like_interface::{
            OffsetSizeAndStrideOpInterface, ViewLikeOpInterface
        }
    },
    ir::{
        builtins::types::{MemRef, IndexType},
        operation::asm_interface::OpAsmOpInterface,
        symbol_interfaces::SymbolUserOpInterface, value::Value
    },
};

/*
Base class for memref allocating ops: alloca and alloc.

```mlit
%0 = alloclike(%m)[%s] : memref<8x?xf32, affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>>
```
*/
#[mlir(
    traits = [AttrSizedOperandSegments],
    assembly_format = "`(`$dynamic_sizes`)` (`` `[` $symbol_operands^ `]`)? attr-dict `:` type($memref)"
)]
pub struct AllocLike {
    dynamic_sizes: [IndexType],
    // The symbolic operands (the ones in square brackets)
    // bind to the symbols of the memref's layout map.
    symbol_operands: [IndexType],
    #[input([IntMinValue<0>])]
    alignment: Option<u64>,
    #[output([MemAlloc<resource>])]
    memref: MemRef<?>

//   let builders = [
//     OpBuilder<(ins memref_type: MemRef,
//                   CArg<"IntegerAttr", "IntegerAttr()">:$alignment), [{
//       return build($_builder, $_state, memref_type, {}, alignment);
//     }]>,
//     OpBuilder<(ins memref_type: MemRef, dynamic_sizes: ValueRange,
//                   CArg<"IntegerAttr", "IntegerAttr()">:$alignment), [{
//       return build($_builder, $_state, memref_type, dynamic_sizes, {}, alignment);
//     }]>,
//     OpBuilder<(ins memref_type: MemRef, dynamic_sizes: ValueRange,
//                   symbol_operands: ValueRange,
//                   CArg<"IntegerAttr", "{}">:$alignment), [{
//       $_state.types.push(memref_type);
//       $_state.add_operands(dynamic_sizes);
//       $_state.add_operands(symbol_operands);
//       $_state.add_attribute(getOperandSegmentSizeAttr(),
//           $_builder.getDenseI32ArrayAttr({
//               static_cast<int32_t>(dynamic_sizes.size()),
//               static_cast<int32_t>(symbol_operands.size())}));
//       if (alignment)
//         $_state.add_attribute(getAlignmentAttrStrName(), alignment);
//     }]>];
}

impl AllocLike {
    static &'static str getAlignmentAttrStrName() { return "alignment"; }

    MemRef get_type() { return self.result.get_type().cast<MemRef>(); }
}

impl Verify for AllocLike {

}

impl Canonicalise for AllocLike {

}

/*
----------------------------------------------------------------------
AssumeAlignment
----------------------------------------------------------------------
*/

/**
Assertion that gives alignment information to the input memref.

The `assume_alignment` operation takes a memref and an integer of alignment value, and internally annotates the buffer with the given alignment. If the buffer isn't aligned to the given alignment, the behaviour is undefined.

This operation doesn't affect the semantics of a correct program. It's for optimisation only, and the optimisation is best-effort.
*/
#[mlir(
    assembly_format = "$memref `,` $alignment attr-dict `:` type($memref)"
)]
pub struct AssumeAlignment<T>  {
    #[input]
    memref: MemRef<T>,
    #[input]
    alignment: ConfinedAttr<I32Attr, [IntPositive]>,
}

impl Verify for AssumeAlignment {

}

/*
----------------------------------------------------------------------
Alloc
----------------------------------------------------------------------
*/

/**
memory allocation operation.

`memref.alloc` operation allocates a region of memory, as specified by its memref type.

# Examples

```mlir
%0 = memref.alloc() : memref<8x64xf32, 1>
```

The optional list of dimension operands are bound to the dynamic dimensions specified in its memref type. In the example below, the ssa value `%d` is bound to the second dimension of the memref (which is dynamic).

```mlir
%0 = memref.alloc(%d) : memref<8x?xf32, 1>
```

The optional list of symbol operands are bound to the symbols of the memrefs affine map. In the example below, the ssa value `%s` is bound to the symbol `s0` in the affine map specified in the allocs memref type.

```mlir
%0 = memref.alloc()[%s]
    : memref<8x64xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
```

This operation returns a single ssa value of memref type, which can be used by subsequent load and store operations.

The optional `alignment` attribute may be specified to ensure that the region of memory that will be indexed is aligned at the specified byte boundary.

```mlir
%0 = memref.alloc()[%s] { alignment = 8 }
    : memref<8x64xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
```
*/
#[mlir]
pub struct Alloc {
}

impl Verify for Alloc {

}

impl OpAsmOpInterface for Alloc {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        
    }
}

// AllocLike<"alloc", DefaultResource, [
// fn asm_result_names"\]>]>

/*
----------------------------------------------------------------------
Realloc
----------------------------------------------------------------------
*/


/**
`memref.realloc` Memory reallocation operation.

The `realloc` operation changes the size of a memory region. The memory region is specified by a 1D source memref and the size of the new memory region is specified by a 1D result memref type and an optional dynamic Value of `IndexType` type. The source and the result memref must be in the same memory space and have the same element type.

The operation may move the memory region to a new location. In this case, the content of the memory block is preserved up to the lesser of the new and old sizes. If the new size if larger, the value of the extended memory is undefined. This is consistent with the ISO C realloc.

The operation returns an SSA value for the memref.

# Examples

```mlir
%0 = memref.realloc %src : memref<64xf32> to memref<124xf32>
```

The source memref may have a dynamic shape, in which case, the compiler will generate code to extract its size from the runtime data structure for the memref.

```mlir
%1 = memref.realloc %src : memref<?xf32> to memref<124xf32>
```

If the result memref has a dynamic shape, a result dimension operand is needed to spefify its dynamic dimension. In the example below, the ssa value `%d` specifies the unknown dimension of the result memref.

```mlir
%2 = memref.realloc %src(%d) : memref<?xf32> to memref<?xf32>
```

An optional `alignment` attribute may be specified to ensure that the region of memory that will be indexed is aligned at the specified byte boundary.  This is consistent with the fact that memref.alloc supports such an optional alignment attribute. Note that in ISO C standard, neither alloc nor realloc supports alignment, though there is aligned_alloc but not `aligned_realloc`.

```mlir
%3 = memref.ralloc %src { alignment = 8 } : memref<64xf32> to memref<124xf32>
```

Referencing the memref through the old SSA value after realloc is undefined behaviour.

```mlir
%new = memref.realloc %old : memref<64xf32> to memref<124xf32>
%4 = memref.load %new[%index]  // Ok
%5 = memref.load %old[%index]  // Undefined behaviour
```
*/
#[mlir(
    assembly_format = "$source (`(` $dynamic_result_size^ `)`)? attr-dict `:` type($source) `to` type(results)"
)]
pub struct Realloc<T: Type> {
    #[input]
    source: MemRef<T, 1>,
    #[input]
    dynamic_result_size: Option<Index>,
    #[attribute([IntMinValue<0>])]
    alignment: Option<u64>,
    #[outout]
    result: MemRef<AnyType, 1>

//   let builders = [
//     OpBuilder<(ins result_type: MemRef,
//                   source: Value,
//                   CArg<Value, "Value()">:$dynamic_result_size), [{
//       return build($_builder, $_state, result_type, source, dynamic_result_size,
//                    IntegerAttr());
//     }]>];
}

impl Realloc {
    /// The result of a realloc is always a memref.
    MemRef get_type() { return self.result.get_type().cast<MemRef>(); }
}

impl Verify for Realloc {
    fn verify(&self) -> LogicalResult {
        let source_type = self.source;
        let result_type = self.result;
      
        // The source memref should have identity layout (or none).
        if !source_type.layout.is_identity() {
            return emit_error(
                "Unsupported layout for source memref type {}", source_type);
        }
      
        // The result memref should have identity layout (or none).
        if !result_type.layout.is_identity() {
            return emit_error(
                "Unsupported layout for result memref type {}", result_type);
        }
      
        // The source memref and the result memref has should be in the same memory space.
        if source_type.memory_space != result_type.memory_space {
            return emit_error(
                "Different memory spaces specified for source memref type {} and result memref type {}",
                source_type,
                result_type
            );
        }
      
        /*
        The source memref and the result memref should have the same element type.
        */
        if source_type.element_type() != result_type.element_type() {
            return emit_error(
                "Different element types specified for source memref type {} and result memref type {}",
                source_type,
                result_type
            );
        }
      
        // Verify that we have the dynamic dimension operand when it is needed.
        if result_type.num_dynamic_dims() && !self.dynamic_output_size() {
            return emit_error("Missing dimension operand for result type ")
                 << result_type;
        }
        if !result_type.num_dynamic_dims() && self.dynamic_output_size() {
            return emit_error(
                "Unnecessary dimension operand for result type {}", result_type);
        }
      
        Ok(())
    }
}

impl Canonicalise for Realloc {
    fn canonicalisation_patterns(
        results: &RewritePatternSet,
        context: *mut MLIRContext
    ) {
        results.add<SimplifyDeadAlloc<ReallocOp>>(context);
    }
}

/*
----------------------------------------------------------------------
Alloca
----------------------------------------------------------------------
*/

/**
`memref.alloca` Stack memory allocation operation.

The `alloca` operation allocates memory on the stack, to be automatically released when control transfers back from the region of its closest surrounding operation with an [`AutomaticAllocationScope`](../Traits.md/#automaticallocationscope) trait. The amount of memory allocated is specified by its memref and additional operands. For example:

# Examples

```mlir
%0 = memref.alloca() : memref<8x64xf32>
```

The optional list of dimension operands are bound to the dynamic dimensions specified in its memref type. In the example below, the SSA value `%d` is bound to the second dimension of the memref (which is dynamic).

```mlir
%0 = memref.alloca(%d) : memref<8x?xf32>
```

The optional list of symbol operands are bound to the symbols of the memref's affine map. In the example below, the SSA value `%s` is bound to the symbol `s0` in the affine map specified in the allocs memref type.

```mlir
%0 = memref.alloca()[%s]
    : memref<8x64xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>>
```

This operation returns a single SSA value of memref type, which can be used by subsequent load and store operations. An optional alignment attribute, if specified, guarantees alignment at least to that boundary. If not specified, an alignment on any convenient boundary compatible with the type will be chosen.
*/
#[mlir]
pub struct Alloca {
}

impl Verify for Alloca {
    fn verify(&self) -> LogicalResult {
        // An alloca op needs to have an ancestor with an allocation scope trait.
        if !self.parent_with_trait<AutomaticAllocationScope>() {
            return emit_op_error(
              "Requires an ancestor op with AutomaticAllocationScope trait");
        }
        verify_alloc_like_op(self)
    }
      
}

impl OpAsmOpInterface for Alloca {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        set_name(self.result, "alloca");
    }
}

// AllocLike<"alloca", AutomaticAllocationScopeResource,[
// fn asm_result_names"\]>]>

/*
----------------------------------------------------------------------
AllocaScope
----------------------------------------------------------------------
*/

/**
`memref.alloca_scope` Explicitly delimited scope for stack allocation.

The `memref.alloca_scope` operation represents an explicitly-delimited scope for the alloca allocations. Any `memref.alloca` operations that are used within this scope are going to be cleaned up automatically once the control-flow exits the nested region. For example:

# Examples

```mlir
memref.alloca_scope {
    %myalloca = memref.alloca() : memref<4x3xf32>
    ...
}
```

Here, `%myalloca` memref is valid within the explicitly delimited scope and is automatically deallocated at the end of the given region. Conceptually, `memref.alloca_scope` is a passthrough operation with `AutomaticAllocationScope` that spans the body of the region within the operation.

`memref.alloca_scope` may also return results that are defined in the nested region. To return a value, one should use `memref.alloca_scope.return` operation:

```mlir
%result = memref.alloca_scope {
    ...
    memref.alloca_scope.return %value
}
```

If `memref.alloca_scope` returns no value, the `memref.alloca_scope.return ` can be left out, and will be inserted implicitly.
*/
#[mlir(
    traits = [
        AutomaticAllocationScope,
        SingleBlockImplicitTerminator<"AllocaScopeReturn">,
        RecursiveMemoryEffects, NoRegionArguments
    ]
)]
pub struct AllocaScope {
    #[output]
    results: [AnyType],
    #[regions]
    body_region: SizedRegion<1>
}

impl Canonicalise for AllocaScope {
    fn canonicalisation_patterns(results: &RewritePatternSet,
                                                    context: *mut MLIRContext) {
    results.add<AllocaScopeInliner, AllocaScopeHoister>(context);
    }
}

impl AssemblyFormat for AllocaScope {
    fn parse(&self, parser: &OpAsmParser, result: &OperationState) -> ParseResult {
        // Create a region for the body.
        result.regions.reserve(1);
        let body_region = result.add_region();
      
        // Parse optional results type list.
        if parser.parse_optional_arrow_type_list(result.types) {
            return Err(());
        }
      
        // Parse the body region.
        if parser.parse_region(*body_region, /*arguments=*/{}) {
            return Err(());
        }
        AllocaScope::ensure_terminator(
            *body_region, parser.get_builder(),
            result.location);
      
        // Parse the optional attribute list.
        if parser.parse_optional_attr_dict(result.attributes) {
            return Err(());
        }
        Ok(())
    }

    fn print(&self, p: &OpAsmPrinter) {
        let print_block_terminators = false;
      
        p << ' ';
        if !self.results.is_empty() {
            p << " -> (" << get_result_types() << ")";
            print_block_terminators = true;
        }
        p << ' ';
        p.print_region(self.body_region,
                      /*printEntryBlockArgs=*/false,
                      /*print_block_terminators=*/print_block_terminators);
        p.print_optional_attr_dict((self).attrs());
    }
}

impl RegionBranchOpInterface for AllocaScope {

}

/*
----------------------------------------------------------------------
AllocaScopeReturn
----------------------------------------------------------------------
*/

/**
Terminator for `alloca_scope` operation.

`memref.alloca_scope.return` operation returns zero or more SSA values from the region within `memref.alloca_scope`. If no values are returned, the return operation may be omitted. Otherwise, it has to be present to indicate which values are going to be returned. For example:

```mlir
memref.alloca_scope.return %value
```
*/
#[mlir(
    traits = [HasParent<"AllocaScope">, Pure, ReturnLike, Terminator],
    assembly_format = "attr-dict ($results^ `:` type($results))?"
)]
pub struct AllocaScopeReturn {
    #[output]
    outputs: [AnyType]  // renamed from `results`
//   let builders = [OpBuilder<(ins), [{ /*nothing to do */ }]>];
}

/*
----------------------------------------------------------------------
Cast
----------------------------------------------------------------------
*/

/**
`memref.cast` operation.

# Syntax

```text
operation ::= ssa-id `=` `memref.cast` ssa-use `:` type `to` type
```

The `memref.cast` operation converts a memref from one type to an equivalent type with a compatible shape. The source and destination types are compatible if:

# a. Both are ranked memref types with the same element type, address space, and rank and:

1. Both have the same layout or both have compatible strided layouts.
2. The individual sizes (resp. offset and strides in the case of strided memrefs) may convert constant dimensions to dynamic dimensions and vice-versa.

If the cast converts any dimensions from an unknown to a known size, then it acts as an assertion that fails at runtime if the dynamic dimensions disagree with resultant destination size.

# Examples

Assert that the input dynamic shape matches the destination static shape:

```mlir
%2 = memref.cast %1 : memref<?x?xf32> to memref<4x4xf32>
```

Erase static shape information, replacing it with dynamic information:

```mlir
%3 = memref.cast %1 : memref<4xf32> to memref<?xf32>
```

The same holds true for offsets and strides.

Assert that the input dynamic shape matches the destination static stride:

```mlir
%4 = memref.cast %1
    : memref<12x4xf32, strided<[?, ?], offset: ?>>
        to memref<12x4xf32, strided<[4, 1], offset: 5>>
```

Erase static offset and stride information, replacing it with dynamic information:

```mlir
%5 = memref.cast %1
    : memref<12x4xf32, strided<[4, 1], offset: 5>>
        to memref<12x4xf32, strided<[?, ?], offset: ?>>
```

# b. Either or both memref types are unranked with the same element type, and
 address space.

# Examples

Cast to concrete shape:

```mlir
%4 = memref.cast %1 : memref<*xf32> to memref<4x?xf32>
```

Erase rank information:

```mlir
%5 = memref.cast %1 : memref<4x?xf32> to memref<*xf32>
```
*/
#[mlir(
    traits = [MemRefsNormalisable, Pure, SameOperandsAndResultShape],
    assembly_format = "$source attr-dict `:` type($source) `to` type($dest)"
)]
pub struct Cast<T, const N: usize> {
    #[input]
    source: AnyTypeOf<UnrankedMemRef<T>, MemRef<T, N>>,
    #[output]
    dest: AnyTypeOf<UnrankedMemRef<U>, MemRef<U, N>>
}

impl Cast {
    /// Fold the given Cast into consumer op.
    static bool can_fold_into_consumer_op(Cast cast_op);

    
}

impl Fold for Cast {
    fn fold(&self) -> FoldResult {
        if fold_mem_ref_cast(self).is_ok() { self.result } else { Value() }
    }
}

impl CastOpInterface for Cast {
    fn are_cast_compatible(
        nputs: TypeRange,
        outputs: TypeRange
    ) -> bool {
        if inputs.size() != 1 || outputs.size() != 1 {
          return false;
        }
        let a = inputs.front();
        let b = outputs.front();
        let a_t = a.dyn_cast<MemRef>();
        let b_t = b.dyn_cast<MemRef>();
      
        let ua_t = a.dyn_cast<UnrankedMemRefType>();
        let ub_t = b.dyn_cast<UnrankedMemRefType>();
      
        if a_t && b_t {
            if a_t.element_type() != b_t.element_type() {
                return false;
            }
            if a_t.layout != b_t.layout {
                i64 a_offset;
                i64 b_offset;
                SmallVector<[i64; 4]> a_strides, b_strides;
                if (failed(self.strides_and_offset(a_t, a_strides, a_offset)) ||
                    failed(self.strides_and_offset(b_t, b_strides, b_offset)) ||
                    a_strides.size() != b_strides.size())
                {
                    return false;
                }
        
                // Strides along a dimension/offset are compatible if the value in the
                // source memref is static and the value in the target memref is the
                // same. They are also compatible if either one is dynamic (see
                // description of MemRefCastOp for details).
                let check_compatible = |i64 a, i64 b| {
                    (ShapedType::is_dynamic(a) ||
                            ShapedType::is_dynamic(b) || a == b)
                };
                if !check_compatible(a_offset, b_offset) {
                    return false;}
                for a_stride in a_strides.enumerate() {
                    if !check_compatible(a_stride.value(), b_strides[a_stride.index()]) {
                        return false;
                    }
                }
            }
            if a_t.memory_space != b_t.memory_space {
                return false;
            }
        
            // They must have the same rank, and any specified dimensions must match.
            if a_t.rank() != b_t.rank() {
                return false;
            }
        
            for i in 0..a_t.rank() {
                let a_dim = a_t.dim_size(i), b_dim = b_t.dim_size(i);
                if !ShapedType::is_dynamic(a_dim)
                && !ShapedType::is_dynamic(b_dim)
                && a_dim != b_dim
                {
                    return false;
                }
            }
            return true;
        } else {
            if !a_t && !ua_t {
                return false;
            }
            if !b_t && !ub_t {
                return false;
            }
            // Unranked to unranked casting is unsupported
            if ua_t && ub_t {
                return false;
            }
        
            let a_elt_type = (a_t) ? a_t.element_type() : ua_t.element_type();
            let b_elt_type = (b_t) ? b_t.element_type() : ub_t.element_type();
            if a_elt_type != b_elt_type {
                return false;
            }
        
            let a_mem_space = (a_t) ? a_t.memory_space : ua_t.memory_space;
            let b_mem_space = (b_t) ? b_t.memory_space : ub_t.memory_space;
            return a_mem_space == b_mem_space;
        }
      
        return false;
    }
}

impl OpAsmOpInterface for Cast {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        set_name(self.result, "cast");
    }
}

impl ViewLikeOpInterface for Cast {
    fn view_source(&self) -> Value {
        self.source
    }
}
/*
----------------------------------------------------------------------
Copy
----------------------------------------------------------------------
*/

/**
`memref.copy` copies the data from the source to the destination memref.

# Usage

```mlir
memref.copy %arg0, %arg1 : memref<?xf32> to memref<?xf32>
```

Source and destination are expected to have the same element type and shape.
Otherwise, the result is undefined. They may have different layouts.
*/
#[mlir(
    traits = [SameOperandsElementType, SameOperandsShape],
    assembly_format = "$source `,` $target attr-dict `:` type($source) `to` type($target)"
)]
pub struct Copy {
    /// memref to copy from.
    #[input([MemRead])]
    source: AnyTypeOf<UnrankedMemRefType<T>, MemRef<T>>,
    /// memref to copy to.
    #[input([MemWrite])]
    target: AnyTypeOf<UnrankedMemRefType<U>, MemRef<U>>
}

impl Fold for Copy {
    fn fold(&self, results: &SmallVector<[FoldResult]>) -> LogicalResult {
        /// copy(memrefcast) -> copy
        let folded = false;
        let op = self;
        for operand in op.inputs() {
            let cast_op = operand.get().defining_op::<Cast>();
            if cast_op && Cast::can_fold_into_consumer_op(cast_op) {
                operand.set(cast_op.get_operand());
                folded = true;
            }
        }
        return success(folded);
    }
}

impl Canonicalise for Copy {
    fn canonicalisation_patterns(
        results: &RewritePatternSet,
        context: *mut MLIRContext
    ) {
        results.add::<FoldCopyOfCast, FoldSelfCopy>(context);
    }
}

impl CopyOpInterface for Copy {

}

/*
----------------------------------------------------------------------
Dealloc
----------------------------------------------------------------------
*/

/**
`memref.dealloc` Memory deallocation operation.

The `dealloc` operation frees the region of memory referenced by a memref which was originally created by the `alloc` operation.
The `dealloc` operation should not be called on memrefs which alias an alloc'd memref (e.g. memrefs returned by `view` operations).

# Example

```mlir
%0 = memref.alloc() : memref<8x64xf32, affine_map<(d0, d1) -> (d0, d1), 1>>
memref.dealloc %0 : memref<8x64xf32,  affine_map<(d0, d1) -> (d0, d1), 1>>
```
*/
#[mlir(
    traits = [MemRefsNormalisable],
    assembly_format = "$memref attr-dict `:` type($memref)"
)]
pub struct Dealloc {
    #[input([MemFree])]
    memref: AnyTypeOf<UnrankedMemRefType<T>, MemRef<T>>
}

impl Fold for Dealloc {
    fn fold(&self, results: &SmallVector<[FoldResult]>) -> LogicalResult {
        /// dealloc(memrefcast) -> dealloc
        fold_mem_ref_cast(self)
    }
}

/*
----------------------------------------------------------------------
Dim
----------------------------------------------------------------------
*/

/**
`memref.dim` dimension index operation.

The `dim` operation takes a memref and a dimension operand of type `index`.
It returns the size of the requested dimension of the given memref.
If the dimension index is out of bounds the behaviour is undefined.

The specified memref type is that of the first operand.

# Examples

Always returns 4, can be constant folded:

```mlir
%c0 = arith.constant 0 : index
%x = memref.dim %A, %c0 : memref<4x?xf32>
```

Returns the dynamic dimension of `%A`:

```mlir
%c1 = arith.constant 1 : index
%y = memref.dim %A, %c1 : memref<4x?xf32>
```

Equivalent generic form:

```mlir
%x = "memref.dim"(%A, %c0) : (memref<4x?xf32>, index) -> index
%y = "memref.dim"(%A, %c1) : (memref<4x?xf32>, index) -> index
```
*/
#[mlir(
    traits = [MemRefsNormalisable, NoMemoryEffect],
    assembly_format = "attr-dict $source `,` $index `:` type($source)"
)]
pub struct Dim {
    source: AnyTypeOf<[UnrankedMemRefType<?>, MemRef<?>],
    index: IndexType,
    #[output]
    result: IndexType

//   let builders = [
//     OpBuilder<(ins source: Value, "i64":$index)>,
//   ];
}

impl Dim {
    /// Helper function to get the index as a simple integer if it is constant.
    Option<i64> constant_index();
}

impl Fold for Dim {
    fn fold(&self) -> FoldResult {
        // All forms of folding require a known index.
        let index = self.index.dyn_cast_or_null<IntegerAttr>();
        if !index {
            return {};
        }
      
        // Folding for unranked types (UnrankedMemRefType) is not supported.
        let memref_type = self.source.get_type().dyn_cast<MemRef>();
        if !memref_type {
            return {};
        }
      
        // Fold if the shape extent along the given index is known.
        if !memref_type.is_dynamic_dim(index.get_int()) {
          let builder = Builder::new(self.context());
          return builder.getIndexAttr(memref_type.shape[index.get_int()]);
        }
      
        // The size at the given index is now known to be a dynamic size.
        let unsigned_index = index.value().z_ext_value();
      
        // Fold dim to the size argument for an `Alloc`, `View`, or `SubView`.
        let defining_op = self.source.defining_op();
      
        if let alloc = dyn_cast_or_null<Alloc>(defining_op) {
            return *(alloc.dynamic_sizes().begin() +
                   memref_type.dynamic_dim_index(unsigned_index));}
      
        if let alloca = dyn_cast_or_null<Alloca>(defining_op) {
            return *(alloca.dynamic_sizes().begin() +
                   memref_type.dynamic_dim_index(unsigned_index));
        }
        if let view = dyn_cast_or_null<View>(defining_op) {
            return *(view.dynamic_sizes().begin() +
                   memref_type.dynamic_dim_index(unsigned_index));
        }
      
        if let subview = dyn_cast_or_null<SubView>(defining_op) {
            let unused_dims = subview.dropped_dims();
            let result_index = 0;
            let source_rank = subview.source_type().rank();
            let source_index = 0;
            for i in 0..source_rank {
                    if unused_dims.test(i) {
                        continue;
                    }
                    if result_index == unsigned_index {
                        source_index = i;
                        break;
                    }
                    result_index++;
            }
            assert!(subview.is_dynamic_size(source_index),
                    "Expected dynamic subview size");
            subview.dynamic_size(source_index)
        }
      
        if (let size_interface =
                dyn_cast_or_null<OffsetSizeAndStrideOpInterface>(defining_op)) {
            assert!(size_interface.is_dynamic_size(unsigned_index),
                    "Expected dynamic subview size");
            return size_interface.dynamic_size(unsigned_index);
        }
      
        // dim(memrefcast) -> dim
        if fold_mem_ref_cast(self).is_ok() {
            return self.result;
        }
      
        return {};
    }
}

impl Verify for Dim {
    fn verify(&self) -> LogicalResult {
        // Assume unknown index to be in range.
        let index = get_constant_index();
        if !index {
            return Ok(());
        }
      
        // Check that constant index is not knowingly out of range.
        let r#type = self.source.get_type();
        if let memref_type = r#type.dyn_cast<MemRef>() {
            if *index >= memref_type.rank() {
                return emit_op_error("Index is out of range");
            }
        } else if r#type.isa<UnrankedMemRefType>() {
            // Assume index to be in range.
        } else {
            unreachable!("Expected operand with memref type");
        }
        Ok(())
    }
}

impl Canonicalise for Dim {
    fn canonicalisation_patterns(
        results: &RewritePatternSet,
        context: *mut MLIRContext
    ) {
        results.add<DimOfMemRefReshape>(context);
    }
}

impl OpAsmOpInterface for Dim {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        set_name(self.result, "dim");
    }
}

impl ShapedDimOpInterface for Dim {
    /// Interface method of ShapedDimOpInterface: Return the source memref.
    Value shaped_value() { self.source }

    /// Interface method of ShapedDimOpInterface: Return the dimension.
    FoldResult dimension() { self.index }
}

impl ConditionallySpeculatable for Dim {
    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability get_speculatability();
}

/*
----------------------------------------------------------------------
DmaStart
----------------------------------------------------------------------
*/

/**
Non-blocking DMA operation that starts a transfer.

DmaStart starts a non-blocking DMA operation that transfers data from a source memref to a destination memref. The source and destination memref need not be of the same dimensionality, but need to have the same elemental type. The operands include the source and destination memref's each followed by its indices, size of the data transfer in terms of the number of elements (of the elemental type of the memref), a tag memref with its indices, and optionally at the end, a stride and a number_of_elements_per_stride arguments. The tag location is used by a DmaWait to check for completion.
The indices of the source memref, destination memref, and the tag memref have the same restrictions as any load/store. The optional stride arguments should be of 'index' type, and specify a stride for the slower memory space (memory space with a lower memory space id), transferring chunks of number_of_elements_per_stride every stride until %num_elements are transferred. Either both or no stride arguments should be specified. If the source and destination locations overlap the behaviour of this operation is not defined.

For example, a DmaStart operation that transfers 256 elements of a memref `%src` in memory space 0 at indices [%i, %j] to memref `%dst` in memory space 1 at indices [%k, %l], would be specified as follows:

```mlir
%num_elements = arith.constant 256
%idx = arith.constant 0 : index
%tag = memref.alloc() : memref<1xi32, affine_map<(d0) -> (d0)>, 4>
dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%idx]
    : memref<40x128xf32>, affine_map<(d0) -> (d0)>, 0>,
      memref<2x1024xf32>, affine_map<(d0) -> (d0)>, 1>,
      memref<1xi32>, affine_map<(d0) -> (d0)>, 2>
```

If `%stride` and `%num_elt_per_stride` are specified, the DMA is expected to transfer `%num_elt_per_stride` elements every `%stride` elements apart from memory space 0 until `%num_elements` are transferred.

```mlir
dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%idx], %stride,
            %num_elt_per_stride :
```

TODO: add additional operands to allow source and destination striding, and
multiple stride levels.

TODO: Consider replacing src/dst memref indices with view memrefs.
*/
pub struct DmaStart {
    src_memref: Value,
    src_indices: ValueRange,
    dest_memref: Value,
    dest_indices: ValueRange,
    num_elements: Value,
    tag_memref: Value,
    tag_indices: ValueRange,
    stride: CArg<Value, "{}">,
    elements_per_stride: CArg<Value, "{}">
    // operands: [AnyType]

//   let builders = [
//     OpBuilder<(
//         src_memref: Value,
//         src_indices: ValueRange,
//         dest_memref: Value,
//         dest_indices: ValueRange,
//         num_elements: Value,
//         tag_memref: Value,
//         tag_indices: ValueRange,
//         stride: CArg<Value, "{}">,
//         elements_per_stride: CArg<Value, "{}">
//     )>
//   ];
}

impl DmaStart {
    /// Returns the rank (number of indices) of the source MemRef.
    usize src_memref.rank() {
      return self.src_memref.get_type().cast<MemRef>().rank();
    }
    // Returns the source memref indices for this DMA operation.
    operand_range src_indices {
      return {self.operand_begin() + 1,
              self.operand_begin() + 1 + self.src_memref.rank()};
    }

    // Returns the destination MemRef for this DMA operations.
    Value dest_memref { return get_operand(1 + self.src_memref.rank()); }
    // Returns the rank (number of indices) of the destination MemRef.
    usize dst_mem_ref_rank() {
      return self.self.dest_memref.get_type().cast<MemRef>().rank();
    }
    usize get_src_memory_space() {
      return self.src_memref.get_type().cast<MemRef>().memory_space_as_int();
    }
    usize get_dst_memory_space() {
      return self.self.dest_memref.get_type().cast<MemRef>().memory_space_as_int();
    }

    // Returns the destination memref indices for this DMA operation.
    operand_range self.dest_indices {
      return {(self)->operand_begin() + 1 + self.src_memref.rank() + 1,
              (self)->operand_begin() + 1 + self.src_memref.rank() + 1 +
                  self.dst_mem_ref_rank()};
    }

    // Returns the number of elements being transferred by this DMA operation.
    Value self.num_elements {
      return get_operand(1 + self.src_memref.rank() + 1 + self.dst_mem_ref_rank());
    }

    // Returns the Tag MemRef for this DMA operation.
    Value self.tag_memref {
      return get_operand(1 + self.src_memref.rank() + 1 + self.dst_mem_ref_rank() + 1);
    }

    /// Returns the rank (number of indices) of the tag MemRef.
    usize self.tag_memref.rank {
      return self.tag_memref.get_type().cast<MemRef>().rank();
    }

    /// Returns the tag memref index for this DMA operation.
    operand_range self.tag_indices {
      usize tag_index_start_pos =
          1 + self.src_memref.rank() + 1 + self.dst_mem_ref_rank() + 1 + 1;
      return {(self)->operand_begin() + tag_index_start_pos,
              (self)->operand_begin() + tag_index_start_pos + self.tag_memref.rank};
    }

    /**
    Returns true if this is a DMA from a faster memory space to a slower one.
    */
    bool is_dest_memory_space_faster() {
      return (get_src_memory_space() < get_dst_memory_space());
    }

    /**
    Returns true if this is a DMA from a slower memory space to a faster one.
    */
    bool is_src_memory_space_faster() {
      // Assumes that a lower number is for a slower memory space.
      return (get_dst_memory_space() < get_src_memory_space());
    }

    /**
    Given a DMA start operation, returns the operand position of either the source or destination memref depending on the one that is at the higher level of the memory hierarchy. Asserts failure if neither is true.
    */
    usize get_faster_mem_pos() {
      assert!(is_src_memory_space_faster() || is_dest_memory_space_faster());
      return is_src_memory_space_faster() ? 0 : self.src_memref.rank() + 1;
    }

    bool is_strided() {
      return get_num_operands() != 1 + self.src_memref.rank() + 1 +
                                 self.dst_mem_ref_rank() + 1 + 1 +
                                 self.tag_memref.rank;
    }

    Value self.stride {
        if !is_strided() {
            return null();
        }
        return get_operand(get_num_operands() - 1 - 1);
    }

    Value get_num_elements_per_stride() {
        if !is_strided() {
            return null();
        }
        return get_operand(get_num_operands() - 1);
    }
}

impl Fold for DmaStart {
    fn fold(&self,
        results: &SmallVector<[FoldResult]>) -> LogicalResult {
    /// dma_start(memrefcast) -> dma_start
    return fold_mem_ref_cast(self);
    }
}

impl Verify for DmaStart {

    fn verify(&self) -> LogicalResult {
        let num_operands = get_num_operands();
      
        /*
        Mandatory non-variadic operands are: src memref, dst memref, tag memref and the number of elements.
        */
        if num_operands < 4 {
            return emit_op_error("Expected at least 4 operands");
        }
      
        /*
        Check types of operands. The order of these calls is important: the later calls rely on some type properties to compute the operand position.
        1. Source memref.
        */
        if !self.src_memref.get_type().isa<MemRef>() {
            return emit_op_error("Expected source to be of memref type");
        }
        if num_operands < self.src_memref.rank() + 4 {
            return emit_op_error() << "Expected at least {} operands." << self.src_memref.rank() + 4;
        }
        if !self.src_indices.is_empty()
        && !self.src_indices.types().all(|r#type| r#type.is_index())
        {
            return emit_op_error("Expected source indices to be of index type");
        }
      
        // 2. Destination memref.
        if !self.dest_memref.get_type().isa<MemRef>() {
            return emit_op_error("Expected destination to be of memref type");
        }
        let num_expected_operands = self.src_memref.rank() + self.dest_memref.rank() + 4;
        if (num_operands < num_expected_operands){
          return emit_op_error() << "Expected at least {} operands.", num_expected_operands;
        }
        if !self.dest_indices.is_empty()
        && !self.dest_indices.types().all(|r#type| r#type.is_index())
        {
            return emit_op_error("Expected destination indices to be of index type");
        }
      
        // 3. Number of elements.
        if !self.num_elements.get_type().is_index() {
            return emit_op_error("Expected num elements to be of index type");
        }
      
        // 4. Tag memref.
        if !self.tag_memref.get_type().isa<MemRef>() {
            return emit_op_error("Expected tag to be of memref type");
        }
        num_expected_operands += self.tag_memref.rank;
        if num_operands < num_expected_operands {
            return emit_op_error() << "Expected at least {} operands." << num_expected_operands;
        }
        if !self.tag_indices.is_empty()
        && self.tag_indices.types().all(|r#type| r#type.is_index())
        {
            return emit_op_error("Expected tag indices to be of index type.");
        }
      
        /*
        Optional stride-related operands must be either both present or both
        absent.
        */
        if num_operands != num_expected_operands
        && num_operands != num_expected_operands + 2 {
            return emit_op_error("Incorrect number of operands");
        }
      
        // 5. Strides.
        if is_strided() {
            if !self.stride.get_type().is_index()
            || !get_num_elements_per_stride().get_type().is_index()
            {
                return emit_op_error(
                    "Expected stride and num elements per stride to be of type index");
            }
        }
      
        Ok(())
    }
}

impl AssemblyFormat for DmaStart {
    fn verify(&self) -> LogicalResult {
        let num_operands = get_num_operands();
      
        /*
        Mandatory non-variadic operands are: src memref, dst memref, tag memref and the number of elements.
        */
        if num_operands < 4 {
            return emit_op_error("Expected at least 4 operands");
        }
      
        /*
        Check types of operands. The order of these calls is important: the later calls rely on some type properties to compute the operand position.
        1. Source memref.
        */
        if !self.src_memref.get_type().isa<MemRef>() {
            return emit_op_error("Expected source to be of memref type");
        }
        if num_operands < self.src_memref.rank() + 4 {
            return emit_op_error(
                "Expected at least {} operands.",
                self.src_memref.rank() + 4
            );
        }
        if !self.src_indices.is_empty() &&
            !llvm::all_of(self.src_indices.types(),
                          |r#type| r#type.is_index())
        {
            return emit_op_error("Expected source indices to be of index type");
        }
      
        // 2. Destination memref.
        if !self.self.dest_memref.get_type().isa<MemRef>() {
          return emit_op_error("Expected destination to be of memref type");
        }
        let num_expected_operands = self.src_memref.rank() + self.dest_memref.rank() + 4;
        if num_operands < num_expected_operands {
            return emit_op_error() << "Expected at least " << num_expected_operands
                               << " operands";
        }
        if (!self.dest_indices.is_empty() &&
            !llvm::all_of(self.dest_indices.types(),
                          |r#type| r#type.is_index()))
          return emit_op_error("Expected destination indices to be of index type");
      
        // 3. Number of elements.
        if !self.num_elements.get_type().is_index() {
            return emit_op_error("Expected num elements to be of index type");
        }
      
        // 4. Tag memref.
        if !self.tag_memref.get_type().isa<MemRef>() {
            return emit_op_error("Expected tag to be of memref type");
        }
        num_expected_operands += self.tag_memref.rank();
        if num_operands < num_expected_operands {
            return emit_op_error() << "Expected at least " << num_expected_operands
                               << " operands";
        }
        if (!self.tag_indices.is_empty() &&
            !llvm::all_of(self.tag_indices.types(),
                          |r#type| r#type.is_index()))
        {
            return emit_op_error("Expected tag indices to be of index type");
        }
      
        // Optional stride-related operands must be either both present or both
        // absent.
        if num_operands != num_expected_operands
        && num_operands != num_expected_operands + 2
        {
            return emit_op_error("Incorrect number of operands");
        }
      
        // 5. Strides.
        if is_strided() {
            if !self.stride.get_type().is_index()
            || !get_num_elements_per_stride().get_type().is_index()
            {
            return emit_op_error(
                "Expected stride and num elements per stride to be of type index");
            }
        }
      
        Ok(())
    }

    fn print(&self, p: &OpAsmPrinter) {
        p << " " << self.src_memref << '[' << self.src_indices << "], "
          << self.self.dest_memref << '[' << self.dest_indices << "], " << self.num_elements
          << ", " << self.tag_memref << '[' << self.tag_indices << ']';
        if (is_strided())
          p << ", " << self.stride << ", " << get_num_elements_per_stride();
      
        p.print_optional_attr_dict((self)->get_attrs());
        p << " : " << self.src_memref.get_type() << ", " << self.self.dest_memref.get_type()
          << ", " << self.tag_memref.get_type();
    }
}

/*
----------------------------------------------------------------------
DmaWait
----------------------------------------------------------------------
*/

/**
Blocking DMA operation that waits for transfer completion.

DmaWait blocks until the completion of a DMA operation associated with the tag element `%tag[%index]`. %tag is a memref, and %index has to be an index with the same restrictions as any load/store index. %num_elements is the number of elements associated with the DMA operation.

# Example

```mlir
dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%index]
    : memref<2048xf32>, affine_map<(d0) -> (d0)>, 0>,
      memref<256xf32>, affine_map<(d0) -> (d0)>, 1>
      memref<1xi32>, affine_map<(d0) -> (d0)>, 2>
...
...
dma_wait %tag[%index], %num_elements
    : memref<1xi32, affine_map<(d0) -> (d0)>, 2>
```
*/
#[mlir(
    assembly_format = "$tag_memref `[` $tag_indices `]` `,` $num_elements attr-dict `:` type($tag_memref)"
)]
pub struct DmaWait {
    tag_memref: MemRef<T>,
    tag_indices: [IndexType],
    num_elements: IndexType,
}

impl Fold for DmaWait {
    fn fold(&self, results: &SmallVector<[FoldResult]>) -> LogicalResult {
        /// dma_wait(memrefcast) -> dma_wait
        fold_mem_ref_cast(self)
    }
}

impl Verify for DmaWait {
    fn verify(&self) -> LogicalResult {
        // Check that the number of tag indices matches the tag_memref rank.
        let num_tag_indices = self.tag_indices.len();
        let tag_mem_ref_rank = self.tag_memref.rank();
        if num_tag_indices != tag_mem_ref_rank {
            return emit_op_error(
                "Expected tag_indices to have the same number of elements as the tag_memref rank, expected {}, but got {}",
                tag_mem_ref_rank,
                num_tag_indices
            );
        }
        Ok(())
    }
}

/*
----------------------------------------------------------------------
ExtractAlignedPointerAsIndex
----------------------------------------------------------------------
*/

/**
`memref.extract_aligned_pointer_as_index` Extracts a memref's underlying aligned pointer as an index.

Extracts the underlying aligned pointer as an index.

This operation is useful for lowering to lower-level dialects while still avoiding the need to define a pointer type in higher-level dialects such as the memref dialect.

This operation is intended solely as step during lowering, it has no side effects. A reverse operation that creates a memref from an index interpreted as a pointer is explicitly discouraged.

# Example

```mlir
%0 = memref.extract_aligned_pointer_as_index %arg : memref<4x4xf32> -> index
%1 = arith.index_cast %0 : index to i64
%2 = llvm.inttoptr %1 : i64 to !llvm.ptr<f32>
call @foo(%2) : (!llvm.ptr<f32>) ->()
```
*/
#[mlir(
    traits = [Pure, SameVariadicResultSize],
    assembly_format = "$source `:` type($source) `->` type(results) attr-dict"
)]
pub struct ExtractAlignedPointerAsIndex {
    #[input([HasStridesPred])]
    source: MemRef<?>
    #[output]
    aligned_pointer: IndexType
}

impl OpAsmOpInterface for ExtractAlignedPointerAsIndex {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        set_name(self.result, "intptr");
    }
}

/*
----------------------------------------------------------------------
ExtractStridedMetadata
----------------------------------------------------------------------
*/

/**
`memref.extract_strided_metadata` Extracts a buffer base with offset and strides.

Extracts a base buffer, offset and strides. This op allows additional layers of transformations and foldings to be added as lowering progresses from higher-level dialect to lower-level dialects such as the LLVM dialect.

The op requires a strided memref source operand. If the source operand is not a strided memref, then verification fails.

This operation is also useful for completeness to the existing memref.dim op.
While accessing strides, offsets and the base pointer independently is not available, this is useful for composing with its natural complement op: `memref.reinterpret_cast`.

Intended Use Cases:

The main use case is to expose the logic for manipulate memref metadata at a higher level than the LLVM dialect.
This makes lowering more progressive and brings the following benefits:

- not all users of MLIR want to lower to LLVM and the information to e.g. lower to library calls – like libxsmm – or to SPIR-V was not available.
- foldings and canonicalizations can happen at a higher level in MLIR: before this op existed, lowering to LLVM would create large amounts of LLVMIR. Even when LLVM does a good job at folding the low-level IR from a performance perspective, it is unnecessarily opaque and inefficient to send unkempt IR to LLVM.

# Examples

```mlir
%base, %offset, %sizes:2, %strides:2
    = memref.extract_strided_metadata %memref
    : memref<10x?xf32>, index, index, index, index, index

// After folding, the type of %m2 can be memref<10x?xf32> and further folded to %memref.
%m2 = memref.reinterpret_cast %base to
    offset: [%offset],
    sizes: [%sizes#0, %sizes#1],
    strides: [%strides#0, %strides#1]
    : memref<f32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
```
*/
#[mlir(
    traits =  [Pure, SameVariadicResultSize],
    assembly_format = "$source `:` type($source) `->` type(results) attr-dict"
)]
pub struct ExtractStridedMetadata {
    #[input([HasStridesPred])]
    source: MemRef<?>
    #[output([HasStridesPred])]
    base_buffer: MemRef<?, [0]>,
    offset: IndexType,
    sizes: [IndexType],
    strides: [IndexType]
}

impl ExtractStridedMetadata {
    /**
    Return a vector of all the static or dynamic sizes of the op, while statically inferring the sizes of the dynamic sizes, when possible.
    This is best effort.
    E.g., if `get_sizes` returns `[%dyn_size0, %dyn_size1]`, but the source memref type is `memref<2x8xi16>`, this method will return `[2, 8]`.
    Similarly if the resulting memref type is `memref<2x?xi16>`, but `%dyn_size1` can statically be pinned to a constant value, this constant value is returned instead of `%dyn_size`.
    */
    SmallVector<[FoldResult]> constified_mixed_sizes();
    /// Similar to `constified_mixed_sizes` but for strides.
    SmallVector<[FoldResult]> constified_mixed_strides();
    /// Similar to `constified_mixed_sizes` but for the offset.
    FoldResult constified_mixed_offset();
}

impl Fold for ExtractStridedMetadata {
    fn fold(&self, results: &SmallVector<[FoldResult]>) -> LogicalResult {
        let builder = Builder::new(self);
        
        let at_least_one_replacement = replace_constant_uses_of(
            builder,
            self.location,
            ArrayRef<TypedValue<IndexType>>(self.offset),
            constified_mixed_offset());
        at_least_one_replacement |= replace_constant_uses_of(
            builder, self.location, self.sizes, constified_mixed_sizes());
        at_least_one_replacement |= replace_constant_uses_of(
            builder, self.location, self.strides, constified_mixed_strides());
        
        return success(at_least_one_replacement)
    }
}

impl OpAsmOpInterface for ExtractStridedMetadata {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        set_name(self.base_buffer, "base_buffer");
        set_name(self.offset, "offset");
        // For multi-result to work properly with pretty names and packed syntax `x:3`
        // we can only give a pretty name to the first value in the pack.
        if !self.sizes.is_empty() {
            set_name(self.sizes.front(), "sizes");
            set_name(self.strides.front(), "strides");
        }
    }
}

impl InferTypeOpInterface for ExtractStridedMetadata {
    /**
    The number and type of the results are inferred from the shape of the source.
    */
    fn infer_return_types(
        context: *mut MLIRContext,
        location: Option<Location>,
        operands: ValueRange,
        attributes: DictionaryAttribute,
        regions: RegionRange,
        inferred_return_types: &SmallVector<[Type]>
    ) -> LogicalResult {
        ExtractStridedMetadataOpAdaptor extract_adaptor(operands, attributes, regions);
        let source_type = extract_adaptor.self.source.get_type().dyn_cast<MemRef>();
        if (!source_type){
            return Err(());}

        let source_rank = source_type.rank();
        let index_type = IndexType::new(context);
        let memref_type =
            MemRef::new({}, source_type.element_type(),
                            MemRefLayoutAttrInterface{}, source_type.memory_space);
        // Base.
        inferred_return_types.push(memref_type);
        // Offset.
        inferred_return_types.push(index_type);
        // Sizes and strides.
        for i in 0..(source_rank * 2) {
            inferred_return_types.push(index_type);
        }
        Ok(())
    }
}

/*
----------------------------------------------------------------------
GenericAtomicRMW
----------------------------------------------------------------------
*/

/**
`memref.generic_atomic_rmw` Atomic read-modify-write operation with a region.

The `memref.generic_atomic_rmw` operation provides a way to perform a read-modify-write sequence that is free from data races. The memref operand represents the buffer that the read and write will be performed against, as accessed by the specified indices. The arity of the indices is the rank of the memref. The result represents the latest value that was stored. The region contains the code for the modification itself. The entry block has a single argument that represents the value stored in `memref[indices]` before the write is performed. No side-effecting ops are allowed in the body of `GenericAtomicRMW`.

# Example

```mlir
%x = memref.generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%current_value : f32):
        %c1 = arith.constant 1.0 : f32
        %inc = arith.addf %c1, %current_value : f32
        memref.atomic_yield %inc : f32
}
```
*/
#[mlir(
    traits = [SingleBlockImplicitTerminator<"AtomicYield">],
)]
pub struct GenericAtomicRMW<T: [AnySignlessInteger, AnyFloat]> {
    memref: MemRef<T>,
    indices: [IndexType],
    #[output]
    result: T
    #[region]
    region: AnyRegion

//   let skipDefaultBuilders = 1;
//   let builders = [OpBuilder<(ins memref: Value, ivs: ValueRange)>];
}

impl GenericAtomicRMW {
    // TODO: remove post migrating callers.
    Region &body() { return self.region; }

    // The value stored in memref[ivs].
    Value current_value() {
      return self.region.input(0);
    }
}

impl Verify for GenericAtomicRMW {
    fn verify(&self) -> LogicalResult {
        let &body = self.region;
        if body.get_num_arguments() != 1 {
            return emit_op_error("Expected single number of entry block arguments");
        }
      
        if self.result.get_type() != body.input(0).get_type() {
          return emit_op_error("Expected block argument of the same type result type");}
      
        let has_side_effects =
            body.walk((Operation *nested_op) {
                  if (is_memory_effect_free(nested_op)){
                    return WalkResult::advance();}
                  nested_op.emit_error(
                      "Body of 'memref.generic_atomic_rmw' should contain "
                      "only operations with no side effects");
                  return WalkResult::interrupt();
                })
                .was_interrupted();
        has_side_effects ? Err(()) : Ok(())
    }
}

impl AssemblyFormat for GenericAtomicRMW {
    fn parse(
        &self,
        OpAsmParser &parser,
        OperationState &result
    ) -> ParseResult {
        OpAsmParser::UnresolvedOperand memref;
        Type memref_type;
        SmallVector<OpAsmParser::UnresolvedOperand, 4> ivs;

        let index_type = parser.get_builder().get_index_type();
        if parser.parse_operand(memref) ||
            parser.parse_operand_list(ivs, OpAsmParser::Delimiter::Square) ||
            parser.parse_colon_type(memref_type) ||
            parser.resolve_operand(memref, memref_type, result.operands) ||
            parser.resolve_operands(ivs, index_type, result.operands)
        {
            return Err(());
        }

        let body = result.add_region();
        if parser.parse_region(*body, {})
        || parser.parse_optional_attr_dict(result.attributes)
        {
            return Err(());
        }
        result.types.push(memref_type.cast<MemRef>().element_type());
        Ok(())
    }

    fn print(&self, p: &OpAsmPrinter) {
        p << ' ' << self.memref << "[" << self.indices
            << "] : " << self.memref.get_type() << ' ';
        p.print_region(self.region);
        p.print_optional_attr_dict(self.get_attrs());
    }
}

/**
`memref.atomic_yield` yield operation for GenericAtomicRMW.

`memref.atomic_yield` yields an SSA value from a GenericAtomicRMW region.
*/
#[mlir(
    traits = [HasParent<"GenericAtomicRMW">, Pure, Terminator],
    assembly_format = "$result attr-dict `:` type($result)"
)]
pub struct AtomicYield<T> {
    result: T
}

impl Verify for AtomicYield {
    fn verify(&self) -> LogicalResult {
        let parent_type = self.parent().output_types().front();
        let result_type = self.result.get_type();
        if parent_type != result_type {
            return emit_op_error(
                "Types mismatch between yield op: {} and its parent: {}",
                result_type,
                parent_type
            );
        }
        Ok(())
    }
}

/*
----------------------------------------------------------------------
GetGlobal
----------------------------------------------------------------------
*/

/**
`memref.get_global` Get the memref pointing to a global variable.

The `memref.get_global` operation retrieves the memref pointing to a named global variable. If the global variable is marked constant, writing to the result memref (such as through a `memref.store` operation) is undefined.

# Example

```mlir
%x = memref.get_global @foo : memref<2xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$name `:` type($result) attr-dict"
)]
pub struct GetGlobal {
    name: FlatSymbolRefAttr,
    #[output]
    result: AnyStaticShapeMemRef
}

impl SymbolUserOpInterface for GetGlobal {
    fn verify_symbol_uses(
        &self,
        symbol_table: SymbolTableCollection
    ) -> LogicalResult {
        /*
        Verify that the result type is same as the type of the referenced `memref.global` operation.
        */
        let global =
            symbol_table.lookup_nearest_symbol_from<Global>(self, get_name_attr());
        if !global {
            return emit_op_error(
                "'{}' does not reference a valid global memref.",
                self.name
            );
        }
        
        let result_type = self.result.get_type();
        if global.get_type() != result_type {
            return emit_op_error(
                "Result type {} does not match type {} of the global memref @{}",
                result_type,
                global.get_type(),
                self.name
            );
        }
        Ok(())
    }
}

/*
----------------------------------------------------------------------
Global
----------------------------------------------------------------------
*/

/**
`memref.global` Declare or define a global memref variable.

The `memref.global` operation declares or defines a named global memref variable. The backing memory for the variable is allocated statically and is described by the type of the variable (which should be a statically shaped memref type). The operation is a declaration if no `initial_value` is specified, else it is a definition. The `initial_value` can either be a unit attribute to represent a definition of an uninitialized global variable, or an elements attribute to represent the definition of a global variable with an initial value. The global variable can also be marked constant using the `constant` unit attribute. Writing to such constant global variables is undefined.

The global variable can be accessed by using the `memref.get_global` to retrieve the memref for the global variable. Note that the memref for such global variable itself is immutable (i.e., memref.get_global for a given global variable will always return the same memref descriptor).

# Examples

Private variable with an initial value:

```mlir
memref.global "private" @x : memref<2xf32> = dense<0.0, 2.0>
```

Private variable with an initial value and an alignment (power of 2):

```mlir
memref.global "private" @x : memref<2xf32> = dense<0.0, 2.0> { alignment = 64 }
```

Declaration of an external variable:

```mlir
memref.global "private" @y : memref<4xi32>
```

Uninitialised externally visible variable:

```mlir
memref.global @z : memref<3xf16> = uninitialised
```

Externally visible constant variable:

```mlir
memref.global constant @c : memref<2xi32> = dense<1, 4>
```
*/
#[mlir(
    traits = [Symbol],
    assembly_format = "($sym_visibility^)?
    (`constant` $constant^)?
    $sym_name `:`
    custom<GlobalMemrefOpTypeAndInitialValue>($type, $initial_value)
    attr-dict"
)]
pub struct Global {
    sym_name: SymbolNameAttribute,
    sym_visibility: OptionalAttr<StringAttribute>,
    r#type: MemRefTypeAttribute,
    initial_value: OptionalAttr<AnyAttribute>,
    constant: UnitAttribute,
    #[attribute]
    alignment: OptionalAttr<I64Attribute>
}

impl Global {
    bool is_external() { return !self.initial_value; }
    bool is_uninitialised() {
        return !is_external() && self.initial_value->isa<UnitAttribute>();
    }
    /**
    Returns the constant initial value if the memref.global is a constant, or null otherwise.
    */
    ElementsAttr get_constant_init_value();
}

impl Verify for Global {
    fn verify(&self) -> LogicalResult {
        let memref_type = get_type().dyn_cast<MemRef>();
        if !memref_type || !memref_type.has_static_shape() {
            return emit_op_error(
                "Type should be static shaped  memref, but got {}",
                get_type()
            );
        }
      
        // Verify that the initial value, if present, is either a unit attribute or
        // an elements attribute.
        if self.initial_value.has_value() {
            Attribute init_value = self.initial_value.value();
            if !init_value.isa<UnitAttr>() && !init_value.isa<ElementsAttr>() {
                return emit_op_error("Initial value should be a unit or elements "
                                "attribute, but got ")
                    << init_value;
            }
        
            // Check that the type of the initial value is compatible with the type of
            // the global variable.
            if let elements_attr = init_value.dyn_cast<ElementsAttr>() {
                let init_type = elements_attr.get_type();
                let tensor_type = get_tensor_type_from_mem_ref_type(memref_type);
                if init_type != tensor_type {
                return emit_op_error("Initial value expected to be of type ")
                        << tensor_type << ", but was of type " << init_type;}
            }
        }
      
        if Option<uint64_t> align_attr = get_alignment() {
            let alignment = *align_attr;
        
            if !llvm::isPowerOf2_64(alignment) {
                return emit_error() << "alignment attribute value " << alignment
                                << " is not a power of 2";
            }
        }
      
        // TODO: verify visibility for declarations.
        Ok(())
    }
}

/*
----------------------------------------------------------------------
Load
----------------------------------------------------------------------
*/

/**
`memref.load` Load operation.

The `load` op reads an element from a memref specified by an index list. The output of load is a new value with the same type as the elements of the memref. The arity of indices is the rank of the memref (i.e., if the memref loaded from is of rank 3, then 3 indices are required for the load following the memref identifier).

In an `affine.if` or `affine.for` body, the indices of a load are restricted to SSA values bound to surrounding loop induction variables, [symbols](Affine.md/#dimensions-and-symbols), results of a constant operations, or the result of an `affine.apply` operation that can in turn take as arguments all of the aforementioned SSA values or the recursively result of such an `affine.apply` operation.

# Examples

```mlir
%1 = affine.apply affine_map<(d0, d1) -> (3 * d0)> (%i, %j)
%2 = affine.apply affine_map<(d0, d1) -> (d1 + 1)> (%i, %j)
%12 = memref.load %A[%1, %2] : memref<8x?xi32, #layout, memspace0>
```

Example of an indirect load (treated as non-affine):

```mlir
%3 = affine.apply affine_map<(d0) -> (2 * d0 + 1)>(%12)
%13 = memref.load %A[%3, %2] : memref<4x?xi32, #layout, memspace0>
```

**Context:** The `load` and `store` operations are specifically crafted to fully resolve a reference to an element of a memref, and (in affine `affine.if` and `affine.for` operations) the compiler can follow use-def chains (e.g. through [`affine.apply`](Affine.md/#affineapply-affineapplyop) operations) to precisely analyze references at compile-time using polyhedral techniques. This is possible because of the [restrictions on dimensions and symbols](Affine.md/#restrictions-on-dimensions-and-symbols) in these contexts.
*/
#[mlir(
    traits = [
        // TypesMatchWith<"result type matches element type of 'memref'",
        // "memref", "result",
        // "$_self.cast<MemRef>().element_type()">,
        MemRefsNormalisable
    ],
    assembly_format = "$memref `[` $indices `]` attr-dict `:` type($memref)"
)]
pub struct Load<T> {
    /// Reference to load from
    #[input([MemRead])]
    memref: MemRef<T>,
    indices: [IndexType],
    #[output]
    result: AnyType
}

impl Load {
    pub fn set_mem_ref(value: Value) { set_operand(0, value); }
}

impl Fold for Load {
    fn fold(&self) -> FoldResult {
        /// load(memrefcast) -> load
        if fold_mem_ref_cast(self).is_ok() {
            return self.result;
        }
        FoldResult()
    }
}

impl Verify for Load {
    fn verify(&self) -> LogicalResult {
        if get_num_operands() != 1 + self.memref.rank() {
            return emit_op_error("Incorrect number of indices for load.");
        }
        Ok(())
    }
}

/*
----------------------------------------------------------------------
Prefetch
----------------------------------------------------------------------
*/

/**
`memref.prefetch` Prefetch operation.

The `prefetch` op prefetches data from a memref location described with subscript indices similar to memref.load, and with three attributes: a read/write specifier, a locality hint, and a cache type specifier as shown
below:

```mlir
memref.prefetch %0[%i, %j], read, locality<3>, data : memref<400x400xi32>
```

The read/write specifier is either 'read' or 'write', the locality hint ranges from locality<0> (no locality) to locality<3> (extremely local keep in cache). The cache type specifier is either 'data' or 'instr' and specifies whether the prefetch is performed on data cache or on instruction cache.
*/
pub struct Prefetch<T> {
    memref: MemRef<T>,
    indices: [IndexType],
    is_write: BoolAttr,
    locality_hint: ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<3>]>,
    is_data_cach: BoolAttr
}

impl Prefetch {
    static &'static str get_locality_hint_attr_str_name() {
        "locality_hint"
    }
    static &'static str get_is_write_attr_str_name() {
        "is_write"
    }
    static &'static str get_is_data_cache_attr_str_name() {
        "isDataCache"
    }
}

impl Fold for Prefetch {
    fn fold(&self,
        results: &SmallVector<[FoldResult]>) -> LogicalResult {
// prefetch(memrefcast) -> prefetch
return fold_mem_ref_cast(self);
}
}

impl Verify for Prefetch {
    fn verify(&self) -> LogicalResult {
        if (get_num_operands() != 1 + self.memref.rank())
          return emit_op_error("too few indices");
      
        Ok(())
    }
}

impl AssemblyFormat for Prefetch {
    ParseResult PrefetchOp::parse(parser: &OpAsmParser, result: &OperationState) {
        OpAsmParser::UnresolvedOperand memrefInfo;
        SmallVector<OpAsmParser::UnresolvedOperand, 4> indexInfo;
        IntegerAttr localityHint;
        MemRef type;
        StringRef read_or_write, cacheType;
      
        let index_ty = parser.get_builder().get_index_type();
        let i32_type = parser.get_builder().get_integer_type(32);
        if (parser.parse_operand(memrefInfo) ||
            parser.parse_operand_list(indexInfo, OpAsmParser::Delimiter::Square) ||
            parser.parse_comma() || parser.parse_keyword(&read_or_write) ||
            parser.parse_comma() || parser.parse_keyword("locality") ||
            parser.parseLess() ||
            parser.parseAttribute(localityHint, i32_type, "localityHint",
                                  result.attributes) ||
            parser.parse_greater() || parser.parse_comma() ||
            parser.parse_keyword(&cacheType) || parser.parse_colon_type(type) ||
            parser.resolve_operand(memrefInfo, type, result.operands) ||
            parser.resolve_operands(indexInfo, index_ty, result.operands))
          return Err(());
      
        if (!read_or_write.equals("read") && !read_or_write.equals("write")){
          return parser.emit_error(parser.name_loc(),
                                  "rw specifier has to be 'read' or 'write'");}
        result.add_attribute(
            PrefetchOp::get_is_write_attr_str_name(),
            parser.get_builder().get_bool_attr(read_or_write.equals("write")));
      
        if !cacheType.equals("data") && !cacheType.equals("instr") {
          return parser.emit_error(parser.name_loc(),
                                  "cache type has to be 'data' or 'instr'");
        }
      
        result.add_attribute(
            PrefetchOp::get_is_data_cache_attr_str_name(),
            parser.get_builder().get_bool_attr(cacheType.equals("data")));
      
        Ok(())
    }

    fn print(&self, p: &OpAsmPrinter) {
        p << " " << self.memref << '[';
        p.print_operands(self.indices);
        p << ']' << ", " << (get_is_write() ? "write" : "read");
        p << ", locality<" << get_locality_hint();
        p << ">, " << (get_is_data_cache() ? "data" : "instr");
        p.print_optional_attr_dict(
            (self).get_attrs(),
            /*elidedAttrs=*/{"localityHint", "isWrite", "isDataCache"});
        p << " : " << self.memref;
    }
      
}

/*
----------------------------------------------------------------------
ReinterpretCast
----------------------------------------------------------------------
*/

/**
`memref.reinterpret_cast` operation.

Modify offset, sizes and strides of an unranked/ranked memref.

# Examples

```mlir
memref.reinterpret_cast %ranked to
    offset: [0],
    sizes: [%size0, 10],
    strides: [1, %stride1]
    : memref<?x?xf32> to memref<?x10xf32, strided<[1, ?], offset: 0>>
```

```mlir
memref.reinterpret_cast %unranked to
    offset: [%offset],
    sizes: [%size0, %size1],
    strides: [%stride0, %stride1]
    : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
```
*/
#[mlir(
    traits = [AttrSizedOperandSegments, MemRefsNormalisable, Pure],
    assembly_format = "$source `to` `offset` `` `:`
    custom<DynamicIndexList>($offsets, $static_offsets)
    `` `,` `sizes` `` `:`
    custom<DynamicIndexList>($sizes, $static_sizes)
    `` `,` `strides` `` `:`
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)"
)]
pub struct ReinterpretCast<T> {
    source: Arg<AnyRankedOrUnrankedMemRef, "", []>,
    offsets: [IndexType],
    sizes: [IndexType],
    strides: [IndexType],
    static_offsets: DenseI64ArrayAttribute,
    static_sizes: DenseI64ArrayAttribute,
    static_strides: DenseI64ArrayAttribute,
    #[output]
    result: MemRef<T>

//   let builders = [
//     // Build a ReinterpretCast with mixed static and dynamic entries.
//     OpBuilder<(ins result_type: MemRef, source: Value,
//       "FoldResult":$offset, "&[FoldResult]":$sizes,
//       "&[FoldResult]":$strides,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
//     // Build a ReinterpretCast with static entries.
//     OpBuilder<(ins result_type: MemRef, source: Value,
//       "i64":$offset, "&[i64]":$sizes,
//       "&[i64]":$strides,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
//     // Build a ReinterpretCast with dynamic entries.
//     OpBuilder<(ins result_type: MemRef, source: Value,
//       offset: Value, sizes: ValueRange,
//       strides: ValueRange,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
//   ];

//   let extraClassDeclaration = extraBaseClassDeclaration # [{
//   }];
}

impl ReinterpretCast {
    // The result of the op is always a ranked memref.
    MemRef get_type() { return self.result.get_type().cast<MemRef>(); }

    /// Return the rank of the source ShapedType.
    usize result_rank() {
        return self.result.get_type().cast<ShapedType>().rank();
    }

    /**
    Return the expected rank of each of the`static_offsets`, `static_sizes` and `static_strides` attributes.
    */
    [usize; 3] array_attr_max_ranks() {
        usize result_rank = self.result.get_type().cast<ShapedType>().rank();
        return {1, result_rank, result_rank};
    }

    /**
    Return the number of leading operands before the `offsets`, `sizes` and `strides` operands.
    */
    static usize offset_size_and_stride_start_operand_index() { return 1; }

    /**
    Return a vector of all the static or dynamic sizes of the op, while
    statically inferring the sizes of the dynamic sizes, when possible.
    This is best effort.
    E.g., if `get_mixed_sizes` returns `[2, %dyn_size]`, but the resulting
    memref type is `memref<2x8xi16>`, this method will return `[2, 8]`.
    Similarly if the resulting memref type is `memref<2x?xi16>`, but
    `%dyn_size` can statically be pinned to a constant value, this
    constant value is returned instead of `%dyn_size`.
    */
    SmallVector<[FoldResult]> constified_mixed_sizes();
    /// Similar to `constified_mixed_sizes` but for strides.
    SmallVector<[FoldResult]> constified_mixed_strides();
    /// Similar to `constified_mixed_sizes` but for the offset.
    FoldResult constified_mixed_offset();
}

impl Fold for ReinterpretCast {
    fn fold(&self) -> FoldResult {
        let src = self.source;
        let get_prev_src = [&]() -> Value {
            // reinterpret_cast(reinterpret_cast(x)) -> reinterpret_cast(x).
            if let prev = src.defining_op::<ReinterpretCastOp>() {
                return prev.self.source;
            }
        
            // reinterpret_cast(cast(x)) -> reinterpret_cast(x).
            if let prev = src.defining_op::<Cast>() {
                return prev.self.source;
            }
        
            /*
            reinterpret_cast(subview(x)) -> reinterpret_cast(x) if subview offsets are 0.
            */
            if let prev = src.defining_op::<SubView>() {
                if prev.mixed_offsets().all(|FoldResult val|
                    is_constant_int_value(val, 0))
                {
                    return prev.self.source;
                }
            }
        
            return null();
        };
      
        if let prev_src = get_prev_src() {
            get_source_mutable().assign(prev_src);
            return self.result;
        }
        return null();
    }
}

impl Verify for ReinterpretCast {
    /*
    TODO: ponder whether we want to allow missing trailing sizes/strides that are completed automatically, like we have for subview and extract_slice.
    */
    fn verify(&self) -> LogicalResult {
        // The source and result memrefs should be in the same memory space.
        let src_type = self.source.get_type().cast<BaseMemRefType>();
        let result_type = get_type().cast<MemRef>();
        if src_type.memory_space != result_type.memory_space {
            return emit_error(
                "Different memory spaces specified for source type {} and result memref type {}",
                src_type,
                result_type
            );
        }
        if src_type.element_type() != result_type.element_type() {
            return emit_error(
                "Different element types specified for source type {} and result memref type {}",
                src_type,
                result_type
            );
        }
    
        // Match sizes in result memref type and in static_sizes attribute.
        for (index, value) in result_type.shape.zip(self.static_sizes()).enumerate {
            let result_size = std::get<0>(value);
            let expected_size = std::get<1>(value);
            if !ShapedType::is_dynamic(result_size) &&
                !ShapedType::is_dynamic(expected_size) && result_size != expected_size
            {
                return emit_error(
                    "Expected result type with size = {} instead of {} in dim = {}",
                    expected_size,
                    result_size,
                    en.index()
                );
            }
        }
    
        /*
        Match offset and strides in static_offset and static_strides attributes. If result memref type has no affine map specified, this will assume an identity layout.
        */
        i64 result_offset;
        let mut result_strides = SmallVector<[i64; 4]>::new();
        if self.strides_and_offset(result_type, result_strides, result_offset).is_err() {
            return emit_error("Expected result type to have strided layout but found {}", result_type);
            }
    
        // Match offset in result memref type and in static_offsets attribute.
        let expected_offset = self.static_offsets().front();
        if !ShapedType::is_dynamic(result_offset)
        && !ShapedType::is_dynamic(expected_offset)
        && result_offset != expected_offset
        {
            return emit_error(
                "Expected result type with offset = {} instead of {}",
                result_offset,
                expected_offset
            );
        }
    
        // Match strides in result memref type and in static_strides attribute.
        for (index, (result_stride, expected_stride))
        in result_strides.zip(self.static_strides()).enumerate()
        {
            if !ShapedType::is_dynamic(result_stride)
            && !ShapedType::is_dynamic(expected_stride)
            && result_stride != expected_stride
            {
                return emit_error(
                    "Expected result type with stride = {} instead of {} in dim = {}",
                    expected_stride,
                    result_stride,
                    index
                );
            }
        }
    
        Ok(())
    }
}

impl Canonicalise for ReinterpretCast {
    fn canonicalisation_patterns(
        results: &RewritePatternSet,
        context: *mut MLIRContext
    ) {
        results.add<ReinterpretCastOpExtractStridedMetadataFolder>(context);
    }
}

impl OpAsmOpInterface for ReinterpretCast {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        set_name(self.result, "reinterpret_cast");
    }
}

impl OffsetSizeAndStrideOpInterface for ReinterpretCast {

}

impl ViewLikeOpInterface for ReinterpretCast {
    fn view_source(&self) -> Value {
        self.source
    }
}

/*
----------------------------------------------------------------------
Rank
----------------------------------------------------------------------
*/

/**
`memref.rank` operation.

The `memref.rank` operation takes a memref operand and returns its rank.

# Example

```mlir
%0 = memref.rank %arg0 : memref<*xf32>
%1 = memref.rank %arg1 : memref<?x?xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$memref attr-dict `:` type($memref)"
)]
pub struct Rank {
    memref: AnyRankedOrUnrankedMemRef,
    #[output]
    result: IndexType
}

impl Fold for Rank {
    fn fold(&self) -> FoldResult {
        // Constant fold rank when the rank of the operand is known.
        let r#type = get_operand().get_type();
        let shaped_type = r#type.dyn_cast<ShapedType>();
        if shaped_type && shaped_type.has_rank() {
            return IntegerAttr::new(IndexType::new(self.context()), shaped_type.rank());
        }
        IntegerAttribute::new()
    }
}

/*
----------------------------------------------------------------------
Reshape
----------------------------------------------------------------------
*/

/**
`memref.reshape` operation.

The `reshape` operation converts a memref from one type to an equivalent type with a provided shape. The data is never copied or modified. The source and destination types are compatible if both have the same element type, same number of elements, address space and identity layout map. The following combinations are possible:

# a. Source type is ranked or unranked. Shape argument has static size.
Result type is ranked.

Reshape statically-shaped memref:

```mlir
%dst = memref.reshape %src(%shape)
    : (memref<4x1xf32>, memref<1xi32>) to memref<4xf32>
%dst0 = memref.reshape %src(%shape0)
    : (memref<4x1xf32>, memref<2xi32>) to memref<2x2xf32>
```

Flatten unranked memref:

```mlir
%dst = memref.reshape %src(%shape)
    : (memref<*xf32>, memref<1xi32>) to memref<?xf32>
```

# b. Source type is ranked or unranked. Shape argument has dynamic size.
Result type is unranked.

Reshape dynamically-shaped 1D memref:

```mlir
%dst = memref.reshape %src(%shape)
    : (memref<?xf32>, memref<?xi32>) to memref<*xf32>
```

Reshape unranked memref:

```mlir
%dst = memref.reshape %src(%shape)
    : (memref<*xf32>, memref<?xi32>) to memref<*xf32>
```
*/
#[mlir(
    traits = [Pure, ViewLikeOpInterface],
    assembly_format = "$source `(` $shape `)` attr-dict `:` functional-type(operands, results)"
)]
pub struct Reshape {
    source: AnyRankedOrUnrankedMemRef,
    shape: MemRefRankOf<[AnySignlessInteger, IndexType], [1]>
    #[output]
    result: AnyRankedOrUnrankedMemRef

//   let builders = [OpBuilder<
//      (ins result_type: MemRef, operand: Value, shape: Value), [{
//        $_state.add_operands(operand);
//        $_state.add_operands(shape);
//        $_state.add_types(result_type);
//      }]>];
}

impl Reshape {
    MemRef get_type() { return self.result.get_type().cast<MemRef>(); }
}

impl Verify for Reshape {
    fn verify(&self) -> LogicalResult {
        let operand_type = self.source.get_type();
        let result_type = self.result.get_type();
      
        let operand_element_type = operand_type.cast<ShapedType>().element_type();
        let result_element_type = result_type.cast<ShapedType>().element_type();
        if operand_element_type != result_element_type {
            return emit_op_error(
                "Element types of source and destination memref types should be the same");
        }
      
        if (let operandMemRefType = operand_type.dyn_cast<MemRef>()){
          if (!operandMemRefType.layout.is_identity()){
            return emit_op_error("source memref type should have identity affine map");}}
      
        let shape_size = self.shape.get_type().cast<MemRef>().dim_size(0);
        let result_mem_ref_type = result_type.dyn_cast<MemRef>();
        if result_mem_ref_type {
            if !result_mem_ref_type.layout.is_identity() {
                return emit_op_error(
                    "Result memref type should have identity affine map");
            }
            if shape_size == ShapedType::kDynamic {
                return emit_op_error(
                    "Cannot use shape operand with dynamic length to reshape to statically-ranked memref type.");
            }
            if shape_size != result_mem_ref_type.rank() {
                return emit_op_error(
                    "Length of shape operand differs from the result's memref rank");
            }
        }
        Ok(())
    }
}

impl OpAsmOpInterface for Reshape {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        set_name(self.result, "reshape");
    }
}

impl ViewLikeOpInterface for Reshape {
    fn view_source(&self) -> Value {
        self.source}
    }
}

/*
----------------------------------------------------------------------
ExpandShape / CollapseShape
----------------------------------------------------------------------
*/

#[mlir(
    traits = [Pure],
    assembly_format = "$src $reassociation attr-dict `:` type($src) `into` type($result)"
)]
pub struct ReassociativeReshape {
    src: AnyStridedMemRef,
    reassociation: IndexListArrayAttr,
    #[output]
    result: AnyStridedMemRef
}

// code commonExtraClassDeclaration
impl ReassociativeReshape {
    // SmallVector<[AffineMap; 4]> self.reassociation_maps();

    // SmallVector<[ReassociationExprs; 4]> self.reassociation_exprs();

    // SmallVector<[ReassociationIndices; 4]> self.reassociation_indices() {
    //   SmallVector<[ReassociationIndices; 4]> reassociation_indices;
    //   for (let attr : getReassociation()){
    //     reassociation_indices.push(llvm::to_vector<2>(
    //         llvm::map_range(attr.cast<ArrayAttr>(), [&](Attribute indexAttr) {
    //           return indexAttr.cast<IntegerAttr>().get_int();
    //         })));}
    //   return reassociation_indices;
    // };

    // MemRef self.src_type() { return self.src.get_type().cast<MemRef>(); }

    // MemRef self.result { return self.result.get_type().cast<MemRef>(); }
}

impl Folder for ReassociativeReshape {
    fn fold(&self) -> FoldResult {
        fold_reshape_op::<CollapseShapeOp, ExpandShapeOp>(
            self, adaptor.get_operands())
    }
}

impl Verify for ReassociativeReshape {
    fn verify(&self) -> LogicalResult {
        let src_type = self.src_type();
        let result_type = self.result;
      
        if src_type.rank() <= result_type.rank() {
            return emit_op_error(
                "Expected rank reduction, but found source rank {} <= result rank {}",
                src_type.rank(),
                result_type.rank()
            );
        }
      
        // Verify result shape.
        if verify_collapsed_shape(
            get_operation(), result_type.shape,
            src_type.shape, self.reassociation_indices(),
            /*allowMultipleDynamicDimsPerGroup=*/true).is_err()
        {
            return Err(());
        }
      
        // Compute expected result type (including layout map).
        let expected_result_type: MemRef;
        if src_type.layout.is_identity() {
            /*
            If the source is contiguous (i.e., no layout map specified), so is the result.
            */
            MemRefLayoutAttrInterface layout;
            expected_result_type =
                MemRef::new(result_type.shape, src_type.element_type(), layout, src_type.memory_space);
        } else {
            /*
            Source may not be fully contiguous. Compute the layout map.
            Note: Dimensions that are collapsed into a single dim are assumed to be contiguous.
            */
            let computed_layout = compute_collapsed_layout_map(
                    src_type, self.reassociation_indices());
            if computed_layout.is_err() {
                return emit_op_error(
                    "Invalid source layout map or collapsing non-contiguous dims");
            }
            expected_result_type =
                MemRef::new(result_type.shape, src_type.element_type(),
                                *computed_layout, src_type.memory_space);
        }
      
        if expected_result_type != result_type {
            return emit_op_error(
                "Expected collapsed type to be {} but found {}."
                expected_result_type, result_type
            );
        }
      
        Ok(())
    }
}

impl Canonicalise for ReassociativeReshape {
    fn canonicalisation_patterns(
        results: &RewritePatternSet,
        context: *mut MLIRContext)
    {
        results.add<ComposeReassociativeReshapeOps<CollapseShapeOp>,
                ComposeCollapseOfExpandOp<CollapseShapeOp, ExpandShapeOp, Cast>,
                CollapseShapeOpMemRefCastFolder>(context);
    }
}

impl ViewLikeOpInterface for ReassociativeReshape {
    fn view_source(&self) -> Value {
        self.src
    }
}

/**
Operation to produce a memref with a higher rank.

The `memref.expand_shape` op produces a new view with a higher rank whose sizes are a reassociation of the original `view`. The operation is limited to such reassociations, where a dimension is expanded into one or multiple contiguous dimensions. Such reassociations never require additional allocs or copies.

A reassociation is defined as a grouping of dimensions and is represented with an array of DenseI64ArrayAttribute attributes.

# Example

```mlir
%r = memref.expand_shape %0 [[0, 1], [2]]
    : memref<?x?xf32> into memref<?x5x?xf32>
```

At most one dimension of a reassociation group (e.g., [0, 1] above) may be dynamic in the result type. Otherwise, the op would be ambiguous, as it would not be clear how the source dimension is extended.

If an op can be statically proven to be invalid (e.g, an expansion from `memref<10xf32>` to `memref<2x6xf32>`), it is rejected by the verifier. If it cannot statically be proven invalid (e.g., the full example above; it is unclear whether the first source dimension is divisible by 5), the op is accepted by the verifier. However, if the op is in fact invalid at runtime, the behaviour is undefined.

The source memref can be zero-ranked. In that case, the reassociation indices must be empty and the result shape may only consist of unit dimensions.

For simplicity, this op may not be used to cast dynamicity of dimension sizes and/or strides. I.e., if and only if a source dimension is dynamic, there must be a dynamic result dimension in the corresponding reassociation group. Same for strides.

NOTE: This op currently assumes that the inner strides are of the source/result layout map are the faster-varying ones.
*/
pub struct ExpandShape {

//   let builders = [
//     // Builders using ReassociationIndices.
//     OpBuilder<(ins "Type":$result_type, src: Value,
//       "ArrayRef<ReassociationIndices>":$reassociation,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
//     [{
//       build($_builder, $_state, result_type, src, attrs);
//       $_state.add_attribute("reassociation",
//                           getReassociationIndicesAttribute($_builder, reassociation));
//     }]>,

//     // Builder using ReassociationExprs.
//     OpBuilder<(ins "Type":$result_type, src: Value,
//       "ArrayRef<ReassociationExprs>":$reassociation,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
//     [{
//       let reassociationMaps =
//           convertReassociationMapsToIndices($_builder, reassociation);
//       build($_builder, $_state, result_type, src, reassociationMaps, attrs);
//     }]>,

//     // Builder that infers the result layout map. The result shape must be
//     // specified. Otherwise, the op may be ambiguous.
//     OpBuilder<(ins "&[i64]":$resultShape, src: Value,
//                "ArrayRef<ReassociationIndices>":$reassociation)>
//   ];

//   let extraClassDeclaration = commonExtraClassDeclaration # [{
//     static FailureOr<MemRef> computeExpandedType(
//         MemRef src_type, &[i64] resultShape,
//         ArrayRef<ReassociationIndices> reassociation);
//   }];

//   let hasVerifier = 1;
}

// ReassociativeReshape<"expand_shape", [
// fn asm_result_names"\]>]>

/**
`memref.collapse_shape` Operation to produce a memref with a smaller rank.

The `memref.collapse_shape` op produces a new view with a smaller rank whose sizes are a reassociation of the original `view`. The operation is limited to such reassociations, where subsequent, contiguous dimensions are collapsed into a single dimension. Such reassociations never require additional allocs or copies.

Collapsing non-contiguous dimensions is undefined behaviour. When a group of dimensions can be statically proven to be non-contiguous, collapses of such groups are rejected in the verifier on a best-effort basis. In the general case, collapses of dynamically-sized dims with dynamic strides cannot be proven to be contiguous or non-contiguous due to limitations in the memref type.

A reassociation is defined as a continuous grouping of dimensions and is represented with an array of DenseI64ArrayAttribute attribute.

NOTE: Only the dimensions within a reassociation group must be contiguous.
The remaining dimensions may be non-contiguous.

The result memref type can be zero-ranked if the source memref type is statically shaped with all dimensions being unit extent. In such a case, the reassociation indices must be empty.

# Examples

```mlir
// Dimension collapse (i, j) -> i' and k -> k'
%1 = memref.collapse_shape %0 [[0, 1], [2]]
    : memref<?x?x?xf32, stride_spec> into memref<?x?xf32, stride_spec_2>
```

For simplicity, this op may not be used to cast dynamicity of dimension sizes and/or strides. I.e., a result dimension must be dynamic if and only if at least one dimension in the corresponding reassociation group is dynamic. Similarly, the stride of a result dimension must be dynamic if and only if the corresponding start dimension in the source type is dynamic.

NOTE: This op currently assumes that the inner strides are of the source/result layout map are the faster-varying ones.
*/
pub struct CollapseShape {

//   let builders = [
//     // Builders for a contracting reshape whose result type is computed from
//     // `src` and `reassociation`.
//     OpBuilder<(ins src: Value,
//       "ArrayRef<ReassociationIndices>":$reassociation,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
//     OpBuilder<(ins src: Value,
//       "ArrayRef<ReassociationExprs>":$reassociation,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
//     [{
//       let reassociationMaps =
//           convertReassociationMapsToIndices($_builder, reassociation);
//       build($_builder, $_state, src, reassociationMaps, attrs);
//     }]>,

//     // Builders for a reshape whose result type is passed explicitly.
//     OpBuilder<(ins "Type":$result_type, src: Value,
//       "ArrayRef<ReassociationIndices>":$reassociation,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
//     [{
//       build($_builder, $_state, result_type, src, attrs);
//       $_state.add_attribute("reassociation",
//                           getReassociationIndicesAttribute($_builder, reassociation));
//     }]>,
//     OpBuilder<(ins "Type":$result_type, src: Value,
//       "ArrayRef<ReassociationExprs>":$reassociation,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
//     [{
//       let reassociationMaps =
//           convertReassociationMapsToIndices($_builder, reassociation);
//       build($_builder, $_state, result_type, src, reassociationMaps, attrs);
//     }]>
//   ];

//   let extraClassDeclaration = commonExtraClassDeclaration # [{
//     /// Return `true` if this source MemRef type is guaranteed to be collapsible
//     /// according to the given reassociation indices. In the presence of dynamic
//     /// strides this is usually not the case.
//     static bool isGuaranteedCollapsible(
//         MemRef src_type, ArrayRef<ReassociationIndices> reassociation);

//     static MemRef computeCollapsedType(
//         MemRef src_type, ArrayRef<ReassociationIndices> reassociation);
//   }];
}

impl Verify for CollapseShape {

}

impl OpAsmOpInterface for CollapseShape {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        // TODO
    }
}

/*
----------------------------------------------------------------------
Store
----------------------------------------------------------------------
*/

/**
`memref.store` operation.

Store a value to a memref location given by indices. The value stored should have the same type as the elemental type of the memref. The number of arguments provided within brackets need to match the rank of the memref.

In an affine context, the indices of a store are restricted to SSA values bound to surrounding loop induction variables, [symbols](Affine.md/#restrictions-on-dimensions-and-symbols), results of a `constant` operation, or the result of an [`affine.apply`](Affine.md/#affineapply-affineapplyop) operation that can in turn take as arguments all of the aforementioned SSA values or the recursively result of such an `affine.apply` operation.

# Example

```mlir
memref.store %100, %A[%1, 1023] : memref<4x?xf32, #layout, memspace0>
```

**Context:** The `load` and `store` operations are specifically crafted to fully resolve a reference to an element of a memref, and (in polyhedral `affine.if` and `affine.for` operations) the compiler can follow use-def chains (e.g. through [`affine.apply`](Affine.md/#affineapply-affineapplyop) operations) to precisely analyze references at compile-time using polyhedral techniques. This is possible because of the [restrictions on dimensions and symbols](Affine.md/#restrictions-on-dimensions-and-symbols) in these contexts.
*/
#[mlir(
    traits = [MemRefsNormalisable],
    assembly_format = "$value `,` $memref `[` $indices `]` attr-dict `:` type($memref)"
)]
pub struct Store<T: AnyType> {
    value: T,
    /// Reference to store to.
    #[input([MemWrite])]
    memref: MemRef<T>,
    indices: [IndexType]

//   let builders = [
//     OpBuilder<(ins valueToStore: Value, memref: Value), [{
//       $_state.add_operands(valueToStore);
//       $_state.add_operands(memref);
//     }]>];
}

impl Store {
    pub fn set_mem_ref(value: Value) { set_operand(1, value); }
}

impl Fold for Store {
    fn fold(&self, results: &SmallVector<[FoldResult]>) -> LogicalResult {
        /// store(memrefcast) -> store
        return fold_mem_ref_cast(self, self.value);
    }
}

impl Verify for Store {
    fn verify(&self) -> LogicalResult {
        if get_num_operands() != 2 + self.memref.rank() {
            return emit_op_error("Store index operand count not equal to memref rank");
        }
        Ok(())
    }
}

/*
----------------------------------------------------------------------
SubView
----------------------------------------------------------------------
*/

/**
`memref.subview` operation.

The `subview` operation converts a memref type to another memref type which represents a reduced-size view of the original memref as specified by the operation's offsets, sizes and strides arguments.

The SubView operation supports the following arguments:

- source: the "base" memref on which to create a "view" memref.
- offsets: memref-rank number of offsets into the "base" memref at which to create the "view" memref.
- sizes: memref-rank number of sizes which specify the sizes of the result "view" memref type.
- strides: memref-rank number of strides that compose multiplicatively with the base memref strides in each dimension.

The representation based on offsets, sizes and strides support a partially-static specification via attributes specified through the `static_offsets`, `static_sizes` and `static_strides` arguments. A special sentinel value ShapedType::kDynamic and ShapedType::kDynamic encodes that the corresponding entry has a dynamic value. 

A subview operation may additionally reduce the rank of the resulting view by removing dimensions that are statically known to be of size 1.

# Examples

Create a sub-view of "base" memref `%0` with offset arguments `%c0`, dynamic sizes for each dimension, and stride arguments `%c1`:

```mlir
%0 = memref.alloc() : memref<64x4xf32, affine_map<(d0, d1) -> (d0 * 4 + d1)>>
%1 = memref.subview %0[%c0, %c0][%size0, %size1][%c1, %c1]
    : memref<64x4xf32, affine_map<(d0, d1) -> (d0 * 4 + d1)>>
    to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + d1 + s0)>>
```

Create a sub-view of "base" memref `%0` with dynamic offsets, sizes, and strides.
Note that dynamic offsets are represented by the linearised dynamic offset symbol `s0` in the subview memref layout map, and that the dynamic strides operands, after being applied to the base memref strides in each dimension, are represented in the view memref layout map as symbols `s1`, `s2` and `s3`:

```mlir
%0 = memref.alloc()
    : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>
%1 = memref.subview %0[%i, %j, %k][%size0, %size1, %size2][%x, %y, %z]
    : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>
    to memref<?x?x?xf32,
        affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + d1 * s2 + d2 * s3 + s0)>>
```

Subview with constant offsets, sizes and strides:

```mlir
%0 = memref.alloc()
    : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>
%1 = memref.subview %0[0, 2, 0][4, 4, 4][1, 1, 1]
    : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>
    to memref<4x4x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2 + 8)>>
```

Subview with constant size, but dynamic offsets and strides. The resulting memref has a static shape, but if the base memref has an affine map to describe the layout, the result memref also uses an affine map to describe the layout. The strides of the result memref is computed as follows:

Let `#map1` represents the layout of the base memref, and `#map2` represents the layout of the result memref. A `#mapsubview` can be constructed to map an index from the result memref to the base memref (note that the description below uses more convenient naming for symbols, while in affine maps, symbols are represented as usize numbers that identify that symbol in the given affine map.

```mlir
#mapsubview = (d0, d1)[o0, o1, t0, t1] -> (d0 * t0 + o0, d1 * t1 + o1)
```

where, `o0`, `o1`, ... are offsets, and `t0`, `t1`, ... are strides. Then,

```mlir
#map2 = #map1.compose(#mapsubview)
```

If the layout map is represented as

```mlir
#map1 = (d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)
```

then,

```mlir
#map2 = (d0, d1)[s0, s1, s2, o0, o1, t0, t1] ->
             (d0 * s1 * t0 + d1 * s2 * t1 + o0 * s1 + o1 * s2 + s0)
```

Representing this canonically

```mlir
#map2 = (d0, d1)[r0, r1, r2] -> (d0 * r1 + d1 * r2 + r0)
```

where, r0 = o0 * s1 + o1 * s2 + s0, r1 = s1 * t0, r2 = s2 * t1.

Note that the subview op does not guarantee that the result memref is "inbounds" w.r.t to base memref. It is upto the client to ensure that the subview is accessed in a manner that is in-bounds:

```mlir
%0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
%1 = memref.subview %0[%i, %j][4, 4][%x, %y]
    : memref<?x?xf32, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)>> to
    memref<4x4xf32, affine_map<(d0, d1)[r0, r1, r2] -> (d0 * r1 + d1 * r2 + r0)>>
```

## Example 5

Rank-reducing subview:

```mlir
%1 = memref.subview %0[0, 0, 0][1, 16, 4][1, 1, 1]
    : memref<8x16x4xf32> to memref<16x4xf32>
```

Original layout: `(d0, d1, d2) -> (64 * d0 + 16 * d1 + d2)`
Subviewed layout:` (d0, d1, d2) -> (64 * (d0 + 3) + 4 * (d1 + 4) + d2 + 2) = (64 * d0 + 4 * d1 + d2 + 210)`
After rank reducing: `(d0, d1) -> (4 * d0 + d1 + 210)`

```mlir
%3 = memref.subview %2[3, 4, 2][1, 6, 3][1, 1, 1]
    : memref<8x16x4xf32> to memref<6x3xf32, strided<[4, 1], offset: 210>>
```
*/
#[mlir(
    traits = [AttrSizedOperandSegments, Pure],
    assembly_format = "$source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)"
)]
pub struct SubView<T> {
    source: MemRef<?>,
    offsets: [IndexType],
    sizes: [IndexType],
    strides: [IndexType],
    static_offsets: DenseI64ArrayAttribute,
    static_sizes: DenseI64ArrayAttribute,
    static_strides: DenseI64ArrayAttribute,
    #[output]
    result: MemRef<T>

//   let builders = [
//     // Build a SubView with mixed static and dynamic entries and custom
//     // result type. If the type passed is nullptr, it is inferred.
//     OpBuilder<(ins source: Value, "&[FoldResult]":$offsets,
//       "&[FoldResult]":$sizes, "&[FoldResult]":$strides,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
//     // Build a SubView with mixed static and dynamic entries and inferred
//     // result type.
//     OpBuilder<(ins result_type: MemRef, source: Value,
//       "&[FoldResult]":$offsets, "&[FoldResult]":$sizes,
//       "&[FoldResult]":$strides,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
//     // Build a SubView with static entries and custom result type. If the
//     // type passed is nullptr, it is inferred.
//     OpBuilder<(ins source: Value, "&[i64]":$offsets,
//       "&[i64]":$sizes, "&[i64]":$strides,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
//     // Build a SubView with static entries and inferred result type.
//     OpBuilder<(ins result_type: MemRef, source: Value,
//       "&[i64]":$offsets, "&[i64]":$sizes,
//       "&[i64]":$strides,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
//     // Build a SubView with dynamic entries and custom result type. If the
//     // type passed is nullptr, it is inferred.
//     OpBuilder<(ins source: Value, offsets: ValueRange,
//       sizes: ValueRange, strides: ValueRange,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
//     // Build a SubView with dynamic entries and inferred result type.
//     OpBuilder<(ins result_type: MemRef, source: Value,
//       offsets: ValueRange, sizes: ValueRange, strides: ValueRange,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
//   ];
}

// let extraClassDeclaration = extraBaseClassDeclaration
impl Subview {
    /// Returns the type of the base memref operand.
    MemRef source_type() {
      return self.source.get_type().cast<MemRef>();
    }

    /// The result of a subview is always a memref.
    MemRef get_type() { return self.result.get_type().cast<MemRef>(); }

    /**
    A subview result type can be fully inferred from the source type and the
    static representation of offsets, sizes and strides. Special sentinels
    encode the dynamic case.
    */
    static Type infer_result_type(MemRef sourceMemRefType,
                                &[i64] staticOffsets,
                                &[i64] staticSizes,
                                &[i64] staticStrides);
    static Type infer_result_type(MemRef sourceMemRefType,
                                &[FoldResult] staticOffsets,
                                &[FoldResult] staticSizes,
                                &[FoldResult] staticStrides);

    /**
    A rank-reducing result type can be inferred from the desired result
    shape. Only the layout map is inferred.

    Note: The result shape cannot be inferred with just the result rank and
    and the desired sizes. In case there are more "ones" among the sizes
    than the difference in source/result rank, it is not clear which dims of
    size one should be dropped.
    */
    static Type inferRankReducedResultType(&[i64] resultShape,
                                           MemRef sourceMemRefType,
                                           &[i64] staticOffsets,
                                           &[i64] staticSizes,
                                           &[i64] staticStrides);
    static Type inferRankReducedResultType(&[i64] resultShape,
                                           MemRef sourceMemRefType,
                                           &[FoldResult] staticOffsets,
                                           &[FoldResult] staticSizes,
                                           &[FoldResult] staticStrides);

    /**
    Return the expected rank of each of the`static_offsets`, `static_sizes`
    and `static_strides` attributes.
    */
    [usize; 3] array_attr_max_ranks() {
      usize rank = source_type().rank();
      return {rank, rank, rank};
    }

    /**
    Return the number of leading operands before the `offsets`, `sizes` and
    and `strides` operands.
    */
    static usize offset_size_and_stride_start_operand_index() { return 1; }

    /**
    Return the dimensions of the source type that are dropped when
    the result is rank-reduced.
    */
    llvm::SmallBitVector dropped_dims();

    /**
    Given a `value`, asserted to be of MemRef, build a SubView that
    results in a rank reduction to the desired memref shape and return the
    new value created.
    If the shape of `value` is already the `desiredShape`, just return
    `value`.
    If the shape of `value` cannot be rank-reduced to `desiredShape`, fail.
    */
    static FailureOr<Value> rankReduceIfNeeded(
      OpBuilder &b, Location loc, value: Value, &[i64] desiredShape);
}

impl Fold for Subview {
    fn fold(&self) -> FoldResult {
        let result_shaped_type = self.result.cast<ShapedType>();
        let source_shaped_type = self.source.cast<ShapedType>();
      
        if (result_shaped_type.has_static_shape() &&
            result_shaped_type == source_shaped_type) {
            return get_view_source();
        }
      
        // Fold subview(subview(x)), where both subviews have the same size and the
        // second subview's offsets are all zero. (I.e., the second subview is a
        // no-op.)
        if let srcSubview = get_view_source().defining_op::<SubView>() {
          let src_sizes = srcSubview.self.mixed_sizes();
          let sizes = self.mixed_sizes();
          let offsets = self.mixed_offsets();
          let all_offsets_zero = llvm::all_of(
              offsets, [](FoldResult ofr) { return is_constant_int_value(ofr, 0); });
          let strides = self.mixed_strides();
          let all_strides_one = llvm::all_of(
              strides, [](FoldResult ofr) { return is_constant_int_value(ofr, 1); });
          let all_sizes_same = llvm::equal(sizes, src_sizes);
          if (all_offsets_zero && all_strides_one && all_sizes_same &&
              result_shaped_type == source_shaped_type)
            return get_view_source();
        }
      
        return {};
    }
}

impl Verify for Subview {
    /// Verifier for SubView.
    fn verify(&self) -> LogicalResult {
        let base_type = source_type();
        let sub_view_type = get_type();

        // The base memref and the view memref should be in the same memory space.
        if base_type.memory_space != sub_view_type.memory_space {
            return emit_error(
                "Ifferent memory spaces specified for base memref type {} and subview memref type {}",
                base_type, sub_view_type
            );
        }

        // Verify that the base memref type has a strided layout map.
        if !is_strided(base_type) {
            return emit_error("Base type {} is not strided", base_type);
        }

        // Verify result type against inferred type.
        let expected_type = SubView::infer_result_type(
            base_type, self.static_offsets(), self.static_sizes(), self.static_strides());

        let result = is_rank_reduced_mem_ref_type(
            expected_type.cast<MemRef>(),
            sub_view_type, self.mixed_sizes());
        return produce_sub_view_error_msg(result, self, expected_type);
    }
}

impl Canonicalise for Subview {
    fn canonicalisation_patterns(results: &RewritePatternSet,
                                                context: *mut MLIRContext) {
    results
        .add<OpWithOffsetSizesAndStridesConstantArgumentFolder<
                SubView, SubViewReturnTypeCanonicalizer, SubViewCanonicalizer>,
            SubViewOpMemRefCastFolder, TrivialSubViewOpFolder>(context);
    }
}

impl OpAsmOpInterface for Subview {
    void SubView::asm_output_names(
        function_ref<void(Value, StringRef)> set_name) {
      set_name(self.result, "subview");
    }
}

impl ViewLikeOpInterface for Subview {
    /// For ViewLikeOpInterface.
    fn view_source(&self) -> Value {
        self.source
    }
}

impl OffsetSizeAndStrideOpInterface for Subview {

}

/*
----------------------------------------------------------------------
TensorStore
----------------------------------------------------------------------
*/

/**
`memref.tensor_store` Tensor store operation.

Stores the contents of a tensor into a memref. The first operand is a value of tensor type, the second operand is a value of memref type. The shapes and element types of these must match, and are specified by the memref type.

# Example

```mlir
%9 = dim %8, 1 : tensor<4x?xf32>
%10 = memref.alloc(%9) : memref<4x?xf32, #layout, memspace0>
memref.tensor_store %8, %10 : memref<4x?xf32, #layout, memspace0>
```
*/
#[mlir(
    traits = [SameOperandsShape, SameOperandsElementType,
    // TypesMatchWith<"type of 'value' matches tensor equivalent of 'memref'",
    //                "memref", "tensor",
    //                "get_tensor_type_from_mem_ref_type($_self)">
    ],
    assembly_format = "$tensor `,` $memref attr-dict `:` type($memref)"
)]
pub struct TensorStore {
    tensor: Tensor<?>,
    /// Reference to store to.
    #[input([MemWrite])]
    memref: AnyTypeOf<[UnrankedMemRefType<?>, MemRef<?>]>
}

/*
----------------------------------------------------------------------
Transpose
----------------------------------------------------------------------
*/

/**
`memref.transpose` produces a new strided memref (metadata-only).

The `transpose` op produces a strided memref whose sizes and strides are a permutation of the original `in` memref. This is purely a metadata transformation.

# Example

```mlir
%1 = memref.transpose %0 (i, j) -> (j, i)
    : memref<?x?xf32>
    to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d1 * s0 + d0)>>
```
*/
#[mlir(
    traits = [Pure]
)]
pub struct Transpose {
    #[input([HasStridesPred])]
    r#in: MemRef<?>,
    permutation: AffineMapAttr,
    #[output([HasStridesPred])]
    result: MemRef<?>

//   let builders = [
//     OpBuilder<(ins in: Value, AffineMapAttr:$permutation,
//       CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];
}

impl Transpose {
    static &'static str permutation_attr_str_name() { return "permutation"; }
    ShapedType get_shaped_type() { return self.r#in.get_type().cast<ShapedType>(); }
}

impl Fold for Transpose {
    fn fold(&self) -> FoldResult {
        if (succeeded(fold_mem_ref_cast(self))) {
            return self.result;
        }
        return {};
    }
}

impl Verify for Transpose {

}

impl AssemblyFormat for Transpose {
    fn parse(
        parser: &OpAsmParser,
        result: &OperationState
    ) -> ParseResult  {
        OpAsmParser::UnresolvedOperand in;
        AffineMap permutation;
        MemRef src_type, dst_type;
        if (parser.parse_operand(in) || parser.parse_affine_map(permutation) ||
            parser.parse_optional_attr_dict(result.attributes) ||
            parser.parse_colon_type(src_type) ||
            parser.resolve_operand(in, src_type, result.operands) ||
            parser.parse_keyword_type("to", dst_type) ||
            parser.add_type_to_list(dst_type, result.types))
          return Err(());
      
        result.add_attribute(Transpose::permutation_attr_str_name(),
                            AffineMapAttr::new(permutation));
        Ok(())
    }
      
      fn verify(&self) -> LogicalResult {
        if (!self.permutation.is_permutation()){
          return emit_op_error("Expected a permutation map");}
        if (self.permutation.num_dims() != get_shaped_type().rank()){
          return emit_op_error("Expected a permutation map of same rank as the input");}
      
        let src_type = self.r#in.get_type().cast<MemRef>();
        let dst_type = get_type().cast<MemRef>();
        let transposed_type = infer_transpose_result_type(src_type, self.permutation);
        if (dst_type != transposed_type) {
            return emit_op_error(
                "Output type {} does not match transposed input type {}, {}",
                dst_type,
                src_type,
                transposed_type
            );
        }
        Ok(())
    }

    // transpose $in $permutation attr-dict : type($in) `to` type(results)
    fn print(&self, p: &OpAsmPrinter) {
        p << " " << self.r#in << " " << self.permutation;
        p.print_optional_attr_dict((self).get_attrs(), {permutation_attr_str_name()});
        p << " : " << self.r#in.get_type() << " to " << get_type();
    }
}

impl OpAsmOpInterface for Transpose {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        set_name(self.result, "transpose");
    }
}

/*
----------------------------------------------------------------------
View
----------------------------------------------------------------------
*/

/**
`memref.view` operation.

The `view` operation extracts an N-D contiguous memref with empty layout map with arbitrary element type from a 1-D contiguous memref with empty layout map of i8 element  type. The View supports the following arguments:

- A single dynamic byte-shift operand must be specified which represents a shift of the base 1-D memref pointer from which to create the resulting contiguous memref view with identity layout.
- A dynamic size operand that must be specified for each dynamic dimension in the resulting view memref type.

The `view` operation gives a structured indexing form to a flat 1-D buffer.
Unlike `subview` it can perform a type change. The type change behaviour requires the op to have special semantics because, e.g. a byte shift of 3 cannot be represented as an offset on f64. For now, a `view` op:

1. Only takes a contiguous source memref with 0 offset and empty layout.
2. Must specify a byte_shift operand (in the future, a special integer attribute may be added to support the folded case).
3. Returns a contiguous memref with 0 offset and empty layout.

# Examples

Allocate a flat 1D/i8 memref:

```mlir
%0 = memref.alloc() : memref<2048xi8>
```

View with dynamic offset and static sizes:

```
%1 = memref.view %0[%offset_1024][] : memref<2048xi8> to memref<64x4xf32>
```

View with dynamic offset and two dynamic size:

```mlir
%2 = memref.view %0[%offset_1024][%size0, %size1]
    : memref<2048xi8> to memref<?x4x?xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$source `[` $byte_shift `]` `` `[` $sizes `]` attr-dict
    `:` type($source) `to` type(results)"
)]
pub struct View<T> {
    #[input]
    source: MemRef<I8, [1]>,
    byte_shift: IndexType,
    sizes: [IndexType],
    #[output]
    result: MemRef<T>

}

impl View {
    /**
    Returns the dynamic sizes for this view operation. This is redundant with `sizes` but needed in template implementations. More specifically:

    ```text
    template <typename AnyMemRefDefOp>
    bool isMemRefSizeValidSymbol(AnyMemRefDefOp memrefDefOp, usize index,
                                 Region *region)
    ```
    */
    operand_range get_dynamic_sizes() {
        return {self.sizes.begin(), self.sizes.end()};
    }
}

impl Verify for View {
    fn verify(&self) -> LogicalResult {
        let base_type = self.source;
        let view_type = self.result;
      
        // The base memref should have identity layout map (or none).
        if !base_type.layout.is_identity() {
            return emit_error(
                "Unsupported map for base memref type {}", base_type);
        }
      
        // The result memref should have identity layout map (or none).
        if !view_type.layout.is_identity() {
            return emit_error(
                "Unsupported map for result memref type {}", view_type);
        }
      
        // The base memref and the view memref should be in the same memory space.
        if base_type.memory_space != view_type.memory_space {
            return emit_error(
"Different memory spaces specified for base memref type {} and view memref type {}.",
                base_type,
                view_type
            );
        }
      
        // Verify that we have the correct number of sizes for the result type.
        let num_dynamic_dims = view_type.num_dynamic_dims();
        if self.sizes.len() != num_dynamic_dims {
            return emit_error("Incorrect number of size operands for type {}", view_type);
        }
      
        Ok(())
    }
}

impl Canonicalise for View {
    fn get_canonicalization_patterns(
        results: &RewritePatternSet,
        context: *mut MLIRContext)
    {
        results.add<ViewOpShapeFolder, ViewOpMemrefCastFolder>(context);
    }
}

impl OpAsmOpInterface for View {
    fn asm_result_names(&self, set_name: fn(Value, &str)) {
        set_name(self.result, "view");
    }
}

impl ViewLikeOpInterface for View {
    fn view_source(&self) -> Value {
        self.source
    }
}

/*
----------------------------------------------------------------------
AtomicRMW
----------------------------------------------------------------------
*/

/**
`memref.atomic_rmw` atomic read-modify-write operation.

The `memref.atomic_rmw` operation provides a way to perform a read-modify-write sequence that is free from data races. The kind enumeration specifies the modification to perform. The value operand represents the new value to be applied during the modification. The memref operand represents the buffer that the read and write will be performed against, as accessed by the specified indices. The arity of the indices is the rank of the memref. The result represents the latest value that was stored.

# Example

```mlir
%x = memref.atomic_rmw "addf" %value, %I[%i] : (f32, memref<10xf32>) -> f32
```
*/
#[mlir(
    traits = [],
    assembly_format = "$kind $value `,` $memref `[` $indices `]` attr-dict `:` `(` type($value) `,`
    type($memref) `)` `->` type($result)"
)]
pub struct AtomicRMW<T: AnyTypeOf<[AnySignlessInteger, AnyFloat]>> {
    kind: AtomicRMWKindAttr,
    value: T,
    memref: MemRef<T>,
    indices: [IndexType]
    #[output]
    result: T
}

impl Fold for AtomicRMW {
    fn fold(&self) -> FoldResult {
        /// atomicrmw(memrefcast) -> atomicrmw
        if fold_mem_ref_cast(self, self.value).is_ok() {
            return self.result;
        }
        FoldResult()
    }
}

impl Verify for AtomicRMW {
    fn verify(&self) -> LogicalResult {
        if (self.memref.rank() != get_num_operands() - 2) {
            return emit_op_error(
              "Expects the number of subscripts to be equal to memref rank");
        }
        match self.kind {
            AtomicRMWKind::addf:
            AtomicRMWKind::maxf:
            AtomicRMWKind::minf:
            AtomicRMWKind::mulf:
          if (!value().get_type().isa<FloatType>()){
            return emit_op_error() << "with kind '"
                                 << arith::stringifyAtomicRMWKind(self.kind)
                                 << "' expects a floating-point type";
          break;}
            AtomicRMWKind::addi:
            AtomicRMWKind::maxs:
            AtomicRMWKind::maxu:
            AtomicRMWKind::mins:
            AtomicRMWKind::minu:
            AtomicRMWKind::muli:
            AtomicRMWKind::ori:
            AtomicRMWKind::andi:
          if (!value().get_type().isa<IntegerType>()){
            return emit_op_error() << "with kind '"
                                 << arith::stringifyAtomicRMWKind(self.kind)
                                 << "' expects an integer type";
          break;}
        default:
          break;
        }
        Ok(())
    }
}
