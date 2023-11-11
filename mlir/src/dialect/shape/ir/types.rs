/*!
- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Shape/IR/ShapeBase.td>
*/


/**
`shape.shape` represents either an unranked shape, a ranked shape with possibly unknown dimensions or an invalid shape. The rank is of type `shape.size` and, if rank is known, the extent is a 1D tensor of type `shape.size`.

Shape is printed:

- `[*]` if it is an unranked shape
- `[?, 2]` if a rank 2 tensor with one unknown dimension
- `[3, 4]` is a rank 2 static tensor
- `[]` is a scalar
- `[1]` is a rank 1 tensor with 1 element
- `[invalid]` for an invalid shape
*/
pub struct ShapeType {
}

/**
`shape.size` represents a non-negative integer with support for being unknown and invalid.

Operations on `shape.size` types are specialized to handle unknown/dynamic value. So, for example, `<unknown> + x == <unknown>` for all non-error `x : !shape.size` (e.g., an unknown value does not become known due to addition).
*/
pub struct SizeType {
}

/**
`shape.value_shape` represents the value produced by an operation (this corresponds to `Value` in the compiler) and a shape. Conceptually this is a tuple of a value (potentially unknown) and `shape.shape`. The value and shape can either or both be unknown. If both the `value` and `shape` are known, then the shape of `value` is conformant with `shape`. That is, the shape of the value conforms to the shape of the ValueShape, so that if we have ValueShape `(value, shape)` then `join(shape_of(value), shape)` would be error free and in particular it means that if both are statically known, then they are equal.
*/
pub struct ValueShapeType {
}

/**
The extent tensor is a tensor of rank one with arbitrarily many index elements (tensor<?xindex>). Like `!shape.shape`, it is used to represent shapes with the difference that it is guaranteed to be error-free.
*/
// 1DTensorOf<[Index]>,
//     BuildableType<"::mlir::RankedTensorType::get({ShapedType::kDynamic}, "
//                   "$_builder.getType<::mlir::IndexType>())">
pub struct ExtentTensorType {
}

def ShapeOrSizeType : AnyTypeOf<[SizeType, ShapeType], "shape or size">;

def ShapeOrExtentTensorType : AnyTypeOf<[ShapeType,
                                               ExtentTensorType],
                                              "shape or extent tensor">;

def SizeOrIndexType : AnyTypeOf<[SizeType, Index], "size or index">;

// Any type representing a shape or size/dim.
def AnyShapeOrSizeType : AnyTypeOf<
    [SizeOrIndexType, ShapeOrExtentTensorType],
  "any shape or size">;

/**
A witness is a structural device in the compiler to maintain ordering of code relying on information obtained from passing assertions. Witnesses do not represent any physical data.

"cstr_" operations will return witnesses and be lowered into assertion logic when not resolvable at compile time.

"assuming_" operations will take witnesses as input and represent only information to the compiler, so they do not exist in executing code. Code that is dependent on "assuming_" operations can assume all cstr operations transitively before are honored as true.

These abstractions are intended to allow the compiler more freedom with assertions by merely showing the assertion through dataflow at this time rather than a side effecting operation that acts as a barrier. This can be viewed similarly to a compiler representation of promises from asynchronous, possibly crashing assertions. Reliant code will not be reordered to before the code and non-reliant code can be reordered freely, and there are no guarantees on the final ordering of the assertions or their related code.
*/
pub struct WitnessType {
}
