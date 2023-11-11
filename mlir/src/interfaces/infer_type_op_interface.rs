/*!
# Infer Type Interfaces

This file contains the definitions of the infer op interfaces defined in `InferTypeOpInterface.td`.

This file contains a set of interfaces that can be used to define information related to type inference.

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/InferTypeOpInterface.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/InferTypeOpInterface.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Interfaces/InferTypeOpInterface.cpp>
*/

/*
OpInterface to compute the return type of an operation. The arguments match those in Operation::create with the exception that the location is optional (if no location is provided, then the method will not emit an error on mismatch).
*/
/**
Interface to infer the return types for an operation that could be used during op construction, verification or type inference.
*/
pub trait InferTypeOpInterface: OpInterface {
    /**
    Infer the return types that an op would generate.

    The method takes an optional location which, if set, will be used to report errors on. The operands and attributes correspond to those with which an Operation would be created (e.g., as used in Operation::create) and the regions of the op. Be aware that this method is supposed to be called with valid arguments, e.g., operands are verified, or it may result in an undefined behaviour.
    */
    fn infer_return_types(
        &self,
        context: *mut MLIRContext,
        location: Option<Location>,
        // operands: ValueRange,
        // attributes: DictionaryAttribute,
        // regions: RegionRange,
        inferred_return_types: &mut SmallVector<[Type]>
    ) -> LogicalResult;

    /**
    Refine the return types that an op would generate.

    This method computes the return types as `infer_return_types` does but additionally takes the existing result types as input. The existing result types can be checked as part of inference to provide more op-specific error messages as well as part of inference to merge additional information, attributes, during inference. It is called during verification for ops implementing this trait with default behaviour reporting mismatch with current and inferred types printed.

    The operands and attributes correspond to those with which an Operation would be created (e.g., as used in Operation::create) and the regions of the op. The method takes an optional location which, if set, will be used to report errors on.

    The return types may be elided or specific elements be null for elements that should just be returned but not verified.

    Be aware that this method is supposed to be called with valid arguments, e.g., operands are verified, or it may result in an undefined behaviour.
    */
    fn refine_return_types(
        context: *mut MLIRContext,
        location: Option<Location>,
        operands: ValueRange,
        attributes: DictionaryAttribute,
        regions: RegionRange,
        return_types: &SmallVector<[Type]>
    ) -> LogicalResult {
        let mut inferred_return_types = SmallVector<[Type; 4]>::new();
        if (failed(ConcreteOp::infer_return_types(context, location, operands,
                                                attributes, regions,
                                                inferred_return_types))){
            return failure();}
        if (!ConcreteOp::is_compatible_return_types(inferred_return_types,
                                                return_types)) {
            return emitOptionalError(
            location, "'", ConcreteOp::getOperationName(),
            "' op inferred type(s) ", inferred_return_types,
            " are incompatible with return type(s) of operation ",
            return_types);
        }
        Ok(())
    }

    /**
    Returns whether two array of types are compatible result types for an operation.
    */
    fn is_compatible_return_types(
        lhs: TypeRange,
        rhs: TypeRange
    ) -> bool {
        // Returns whether two arrays are equal as strongest check for
        /// compatibility by default.
        lhs == rhs
    }

//   // Inferring result types may need to access the region operations.
//   let verifyWithRegions = 1;
//   let verify = [{
//     return detail::verifyInferredResultTypes($_op);
//   }];
}

/**
Interface to infer the components of a ShapedType returned by an operation that could be used during op construction, verification or shape inference.

The components consists of element type, shape and raw attribute.
*/
pub trait InferShapedTypeOpInterface: OpInterface {
    /**
    Infer the components of return type of shape containter.

    The method takes an optional location which, if set, will be used to report errors on. The operands and attributes correspond to those with which an Operation would be created (e.g., as used in Operation::create) and the regions of the op.

    Unknown (e.g., unranked) shape and nullptrs for element type and attribute may be returned by this function while returning success. E.g., partial population of components is not error condition.
    */
    fn infer_return_type_components(
        context: *mut MLIRContext,
        location: Option<Location>,
        operands: ValueShapeRange,
        attributes: DictionaryAttribute,
        regions: RegionRange,
        inferred_return_shapes: &SmallVectorImpl<ShapedTypeComponents>
    ) -> LogicalResult {
        Err(())
    }

    /**
    Reify the shape computation for the operation.

    Insert operations using the given OpBuilder that computes the result shape. This interface is supposed to be workable during dialect conversion (e.g. convert from tensor world to buffer world), where `getOperand` may be invalid. For example, some ops (e.g. dynamic_reshape(input, target_shape)) may depend on their operands to calculate the result shape. When the `matchAndRewrite ` method of a conversion pattern is called, the operands of the op to convert may have been converted into other types, which makes it invalid to call the `getOperand` method of such op directly inside the conversion pattern.  To solve this problem, this interface follows the design of the conversion pattern, that is, accepting passed in operands to avoid calling `getOperand` directly inside the interface implementation.
    */
    fn reify_return_type_shapes(
        &self,
        builder: Builder,
        operands: ValueRange,
        reified_return_shapes: &SmallVectorImpl<Value>
    ) -> LogicalResult {
        Err(())
    }
}

// // Convenience class grouping together type and shaped type op interfaces for
// // ops that have tensor return types.
// class InferTensorTypeBase<list<string> overridenMethods = []> : TraitList<
//   [
//     // Op implements infer type op interface.
//     InferTypeOpInterface,
//     // The op will have methods implementing the ShapedType type inference
//     // interface.
//     DeclareOpInterfaceMethods<InferShapedTypeOpInterface, overridenMethods>,
//     // The op produces tensors and will use the ShapedType type infer interface
//     // along with knowledge that it is producing Tensors to infer the type.
//     NativeOpTrait<"InferTensorType">
//   ]>;

// def InferTensorType : InferTensorTypeBase<["infer_return_type_components"]>;
// def InferTensorTypeWithReify: InferTensorTypeBase<[
//     "infer_return_type_components", "reify_return_type_shapes"]>;

// def ReifyRankedShapedTypeOpInterface :
//     OpInterface<"ReifyRankedShapedTypeOpInterface"> {
//   let description = [{
//     Interface to compute the shape of the result of an operation when
//     the result is a ranked shape type, i.e. `RankedTensorType` or
//     `MemRef`.
//   }];
//   let cppNamespace = "::mlir";

//   let methods = [
//     InterfaceMethod<
//       /*desc=*/[{
//         Reify the shape of the result of an operation (typically in
//         terms of shape of its operands)

//         Insert operations using the given `OpBuilder` that computes
//         the result shape. The `reified_return_shapes` is expected to be
//         populated with as many vectors as the number of results of the
//         op. Each of these vectors is expected to be of size equal to
//         rank of the corresponding result. If the shape of a particular
//         result cannot be computed it must be empty.
//       }],
//       /*retTy=*/"::mlir::LogicalResult",
//       /*methodName=*/"reifyResultShapes",
//       /*args=*/(ins "::mlir::OpBuilder &":$builder,
//         "::mlir::ReifiedRankedShapedTypeDims &":$reified_return_shapes)
//     >
//   ];
// }

// // Op has the same operand and result type.
// // TODO: Change from hard coded to utilizing type inference trait.
// def SameOperandsAndResultType : NativeOpTrait<"SameOperandsAndResultType">;

/**
Adaptor class to abstract the differences between whether value is from a ShapedType or ShapedTypeComponents or DenseIntElementsAttribute.
*/
pub struct ShapeAdaptor {
}

/**
ShapedTypeComponents that represents the components of a ShapedType.
The components consist of

- A ranked or unranked shape with the dimension specification match those
   of ShapeType's getShape() (e.g., dynamic dimension represented using
   ShapedType::kDynamic)
- A element type, may be unset (nullptr)
- A attribute, may be unset (nullptr)
Used by ShapedType type inferences.
*/
pub struct ShapedTypeComponents {
}

/**
Range of values and shapes (corresponding effectively to Shapes dialect's ValueShape type concept).
*/
/*
Currently this exposes the Value (of operands) and Type of the Value. This is not ideal as then one can accidentally reference an out of date shape. This is done to both enable gradual switch and also as OpAdaptor doesn't currently allow returning anything other than Value.
*/
pub struct ValueShapeRange {

}
