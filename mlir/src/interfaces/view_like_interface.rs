/*!
# View-Like Operations Interface

This file implements the operation interface for view-like operations.

Defines the interface for view-like operations.


- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/ViewLikeInterface.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/ViewLikeInterface.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Interfaces/ViewLikeInterface.cpp>
*/

use crate::{
    mlir::{
        dialect::utils::static_value_utils,
        ir::{
            builders,
            builtins::{
                attributes,
                types
            },
            operation::{
                asm_interface::OpAsmOpInterface,
                base::OpInterface,
                implementation, definition::FoldResult
            },
            value::Value,
            value_range::InputRange
        }
    },
    llvm::adt::set_vector::SmallSetVector
};

/**
A view-like operation 'views' a buffer in a potentially different way. It takes in a (view of) buffer (and potentially some other operands) and returns another view of buffer.
*/
pub trait ViewLikeOpInterface: OpInterface {
    /// Returns the source buffer from which the view is created.
    fn view_source(&self) -> Value;
}

/**
Common interface for ops that allow specifying mixed dynamic and static offsets, sizes and strides variadic operands.
Ops that implement this interface need to expose the following methods:

1. `array_attr_max_ranks` to specify the length of static integer attributes.
2. `offsets`, `sizes` and `strides` variadic operands.
3. `static_offsets`, resp. `static_sizes` and `static_strides` integer array attributes.
4. `offset_size_and_stride_start_operand_index` method that specifies the starting index of the OffsetSizeAndStrideOpInterface operands

The invariants of this interface are:

1. `static_offsets`, `static_sizes` and `static_strides` have length exactly `array_attr_max_ranks()`[0] (resp. [1], [2]).
2. `offsets`, `sizes` and `strides` have each length at most `array_attr_max_ranks()`[0] (resp. [1], [2]).
3. if an entry of `static_offsets` (resp. `static_sizes`, `static_strides`) is equal to a special sentinel value, namely `ShapedType::kDynamic` (resp. `ShapedType::kDynamic`, `ShapedType::kDynamic`), then the corresponding entry is a dynamic offset (resp. size, stride).
4. a variadic `offset` (resp. `sizes`, `strides`) operand  must be present for each dynamic offset (resp. size, stride).
5. `offsets`, `sizes` and `strides` operands are specified in this order at operand index starting at `offset_size_and_stride_start_operand_index`.

This interface is useful to factor out common behaviour and provide support for carrying or injecting static behaviour through the use of the static attributes.
*/
pub trait OffsetSizeAndStrideOpInterface: OpInterface {
    /**
    Returns the number of leading operands before the `offsets`, `sizes` and and `strides` operands.
    */
    fn offset_size_and_stride_start_operand_index() -> usize;

    /**
    Returns the expected rank of each of the`static_offsets`, `static_sizes` and `static_strides` attributes.
    */
    fn array_attr_max_ranks(&self) -> [usize; 3];

    /// Returns the dynamic offset operands.
    fn offsets(&self) -> InputRange;

    /// Returns the dynamic size operands.
    fn sizes(&self) -> InputRange;

    /// Returns the dynamic stride operands.
    fn strides(&self) -> InputRange;

    /// Returns the static offset attributes.
    fn static_offsets(&self) -> [i64];

    /// Returns the static size attributes.
    fn static_sizes(&self) -> [i64];

    /// Returns the dynamic stride attributes.
    fn static_strides(&self) -> [i64];

    /// Returns a vector of all the static or dynamic sizes of the op.
    fn mixed_offsets(&self) -> SmallSetVector<[FoldResult; 4]> {
        let builder = Builder::new(self.context());
        mixed_values(self.static_offsets(), self.offsets(), builder);
    }

    /// Returns a vector of all the static or dynamic sizes of the operaiton.
    fn mixed_sizes(&self) -> SmallVector<[FoldResult; 4]> {
        let builder = Builder::new(self.context());
        mixed_values(self.static_sizes(), self.sizes(), builder)
    }

    /// Returns a vector of all the static or dynamic strides of the operation.
    fn mixed_strides(&self) -> SmallVector<[FoldResult; 4]> {
        let builder = Builder::new(self.context());
        mixed_values(self.static_strides(), self.strides(), builder);
    }

    /// Returns true if the offset `index` is dynamic.
    fn is_dynamic_offset(&self, index: usize) -> bool {
        ShapedType::is_dynamic(self.static_offsets()[index])
    }

    /// Returns true if the size `index` is dynamic.
    fn is_dynamic_size(&self, index: usize) -> bool {
        ShapedType::is_dynamic(self.static_sizes()[index])
    }

    /// Returns true if the stride `index` is dynamic.
    fn is_dynamic_stride(&self, index: usize) -> bool {
        ShapedType::is_dynamic(self.static_strides()[index])
    }

    /// Assert the offset `index` is a static constant and return its value.
    fn static_offset(&self, index: usize) -> i64 {
        assert!(!self.is_dynamic_offset(index), "Expected static offset");
        self.static_offsets()[index]
    }

    /// Assert the size `index` is a static constant and return its value.
    fn static_size(&self, index: usize) -> i64 {
        assert!(!self.is_dynamic_size(index), "Expected static size");
        self.static_sizes()[index]
    }

    /// Assert the stride `index` is a static constant and return its value.
    fn static_stride(&self, index: usize) -> i64 {
        assert!(!self.is_dynamic_stride(index), "Expected static stride");
        self.static_strides()[index]
    }

    /**
    Assert the offset `index` is dynamic and return the position of the corresponding operand.
    */
    fn index_of_dynamic_offset(&self, index: usize) -> usize {
        assert!(self.is_dynamic_offset(index), "Expected dynamic offset");
        let num_dynamic = num_dynamic_entries_up_to_idx(
            static_offsets(),
            ShapedType::is_dynamic,
            index);
        self.offset_size_and_stride_start_operand_index() + num_dynamic
    }

    /**
    Assert the size `index` is dynamic and return the position of the corresponding operand.
    */
    fn index_of_dynamic_size(&self, index: usize) -> usize {
        assert!(self.is_dynamic_size(index), "Expected dynamic size");
        let num_dynamic = num_dynamic_entries_up_to_idx(
            static_sizes(), ShapedType::is_dynamic, index);
        self.offset_size_and_stride_start_operand_index() +
          offsets().size() + num_dynamic
    }

    /**
    Assert the stride `index` is dynamic and return the position of the corresponding operand.
    */
    fn index_of_dynamic_stride(&self, index: usize) -> usize {
        assert!(self.is_dynamic_stride(index), "Expected dynamic stride");
        let num_dynamic = num_dynamic_entries_up_to_idx(
            self.static_strides(),
            ShapedType::is_dynamic,
            index);
        self.offset_size_and_stride_start_operand_index() +
            self.offsets().size() + self.sizes().size() + num_dynamic
    }

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Helper method to compute the number of dynamic entries of `static_vals`, up to
        `index` using `is_dynamic` to determine whether an entry is dynamic.
      }],
      /*retTy=*/"usize",
      /*methodName=*/"num_dynamic_entries_up_to_idx",
      /*args=*/(ins "ArrayRef<i64>":$static_vals,
                    "function_ref<bool(i64)>":$is_dynamic,
                    index: usize),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return std::count_if(
          static_vals.begin(), static_vals.begin() + index,
          (i64 val) {
            return is_dynamic(val);
          });
      }]
    >,

    InterfaceMethod<
      /*desc=*/[{
        Assert the offset `index` is dynamic and return its value.
      }],
      /*retTy=*/Value,
      /*methodName=*/"dynamic_offset",
      /*args=*/(ins index: usize),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return self.input((index));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Assert the size `index` is dynamic and return its value.
      }],
      /*retTy=*/Value,
      /*methodName=*/"dynamic_size",
      /*args=*/(ins index: usize),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return self.input(index_of_dynamic_size(index));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Assert the stride `index` is dynamic and return its value.
      }],
      /*retTy=*/Value,
      /*methodName=*/"dynamic_stride",
      /*args=*/(ins index: usize),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return self.input(index_of_dynamic_stride(index));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if all `other`'s offsets, sizes and strides are the same.
        Takes a custom `cmp` comparison function on FoldResult to avoid taking
        a dialect dependence.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"is_same_as",
      /*args=*/(ins "OffsetSizeAndStrideOpInterface":$other,
                   "function_ref<bool(FoldResult, FoldResult)>":$cmp),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return detail::same_offsets_sizes_and_strides(
          cast<OffsetSizeAndStrideOpInterface>(
            self.get_operation()), other, cmp);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{ Return true if all strides are guaranteed to be 1. }],
      /*retTy=*/"bool",
      /*methodName=*/"has_unit_stride",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return all_of(mixed_strides(), [](FoldResult ofr) {
          return constant_int_value(ofr) == static_cast<i64>(1);
        });
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{ Return true if all offsets are guaranteed to be 0. }],
      /*retTy=*/"bool",
      /*methodName=*/"has_zero_offset",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return all_of(mixed_offsets(), [](FoldResult ofr) {
          return constant_int_value(ofr) == static_cast<i64>(0);
        });
      }]
    >,
  ];

  let extraClassDeclaration = [{
    static usize offset_operand_group_position() { return 0; }
    static usize size_operand_group_position() { return 1; }
    static usize stride_operand_group_position() { return 2; }
    static StringRef static_offsets_attr_name() {
      return "static_offsets";
    }
    static StringRef static_sizes_attr_name() {
      return "static_sizes";
    }
    static StringRef static_strides_attr_name() {
      return "static_strides";
    }
    static ArrayRef<StringRef> special_attr_names() {
      static SmallVector<StringRef, 4> names{
        OffsetSizeAndStrideOpInterface::static_offsets_attr_name(),
        OffsetSizeAndStrideOpInterface::static_sizes_attr_name(),
        OffsetSizeAndStrideOpInterface::static_strides_attr_name(),
        OpTrait::AttrSizedOperandSegments<void>::operand_segment_size_attr()};
      return names;
    }
  }];

  let verify = [{
    return detail::verifyOffsetSizeAndStrideOp(
        cast<OffsetSizeAndStrideOpInterface>(self));
  }];
}
