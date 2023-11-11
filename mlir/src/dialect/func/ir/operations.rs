/*!
# Func Dialect Operations

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Func/IR/FuncOps.td>
*/

use crate::mlir::{ir::{operation::asm_interface::OpAsmOpInterface, function::interfaces::FunctionOpInterface}, interfaces::call_interfaces::CallableOpInterface};

/**
`func.call` operation.

The `func.call` operation represents a direct call to a function that is within the same symbol scope as the call. The operands and result types of the call must match the specified function type. The callee is encoded as a symbol reference attribute named `callee`.

# Example

```mlir
%2 = func.call @my_add(%0, %1) : (f32, f32) -> f32
```
*/
pub struct Call {
    callee: FlatSymbolRefAttr,
    operands: Variadic<AnyType>,
    // output
    result: Variadic<AnyType>

//   let builders = [
//     OpBuilder<(ins "Func":$callee, CArg<"ValueRange", "{}">:$operands), [{
//       $_state.addOperands(operands);
//       $_state.addAttribute("callee", SymbolRefAttr::get(callee));
//       $_state.addTypes(callee.self.function_type().outputs());
//     }]>,
//     OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
//       CArg<"ValueRange", "{}">:$operands), [{
//       $_state.addOperands(operands);
//       $_state.addAttribute("callee", callee);
//       $_state.addTypes(results);
//     }]>,
//     OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
//       CArg<"ValueRange", "{}">:$operands), [{
//       build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
//     }]>,
//     OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
//       CArg<"ValueRange", "{}">:$operands), [{
//       build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
//             results, operands);
//     }]>];

//   let assemblyFormat = [{
//     $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
//   }];
}

impl Call {
    FunctionType get_callee_type();

    /// Get the argument operands to the called function.
    operand_range get_arg_operands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable get_callable_for_callee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
}

// Func_Op<"call",
//     [CallOpInterface, MemRefsNormalisable,
//      DeclareOpInterfaceMethods<SymbolUserOpInterface>]>

/**
`func.call_indirect` indirect call operation

The `func.call_indirect` operation represents an indirect call to a value of function type. The operands and result types of the call must match the specified function type.

Function values can be created with the [`func.constant` operation](#funcconstant-constantop).

# Example

```mlir
%func = func.constant @my_func
    : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
%result = func.call_indirect %func(%0, %1)
    : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
```
*/
#[mlir(
    traits = [],
    assembly_format = "$callee `(` $callee_operands `)` attr-dict `:` type($callee)"
)]
// TypesMatchWith<"callee input types match argument types",
//                      "callee", "callee_operands",
//                      "$_self.cast<FunctionType>().inputs()">,
//       TypesMatchWith<"callee result types match result types",
//                      "callee", "results",
//                      "$_self.cast<FunctionType>().outputs()">
pub struct CallIndirect {
    callee: FunctionType,
    callee_operands: [AnyType],
    #[output]
    results: [AnyType]

//   let builders = [
//     OpBuilder<(ins "Value":$callee, CArg<"ValueRange", "{}">:$operands), [{
//       $_state.operands.push_back(callee);
//       $_state.addOperands(operands);
//       $_state.addTypes(callee.getType().cast<FunctionType>().outputs());
//     }]>];

//   let hasCanonicaliseMethod = 1;
}

impl CallIndirect {
    // TODO: Remove once migrated callers.
    ValueRange operands() { return getCalleeOperands(); }

    /// Get the argument operands to the called function.
    operand_range get_arg_operands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable get_callable_for_callee() { return getCallee(); }
}

impl CallOpInterface for CallIndirect {

}

/**
constant

The `func.constant` operation produces an SSA value from a symbol reference to a `func.func` operation.

# Examples

Reference to function `@myfn`:

```mlir
%2 = func.constant @myfn : (tensor<16xf32>, f32) -> tensor<16xf32>
```

Equivalent generic forms:

```mlir
%2 = "func.constant"() { value = @myfn } : () -> ((tensor<16xf32>, f32) -> tensor<16xf32>)
```

MLIR does not allow direct references to functions in SSA operands because the compiler is multithreaded, and disallowing SSA values to directly reference a function simplifies this ([rationale](../Rationale/Rationale.md#multithreading-the-compiler)).
*/
#[mlir(
    traits = [ConstantLike, Pure],
    assembly_format = "attr-dict $value `:` type(results)"
)]
pub struct Constant {
    value: FlatSymbolRefAttr,
    #[output]
    result: AnyType
}

impl Constant {
    /**
    Returns true if a constant operation can be built with the given value and result type.
    */
    fn is_buildable_with(value: Attribute, r#type: Type) -> bool;
}

impl Fold for Constant {

}

impl Verify for Constant {

}

impl OpAsmOpInterface for Constant {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        
    }
}

/**
An operation with a name containing a single `SSACFG` region.

Operations within the function cannot implicitly capture values defined
outside of the function, i.e. Functions are `IsolatedFromAbove`. All external references must use function arguments or attributes that establish a symbolic connection (e.g. symbols referenced by name via a string attribute like SymbolRefAttr). An external function declaration (used when referring to a function declared in some other module) has no body. While the MLIR textual form provides a nice inline syntax for function arguments, they are internally represented as “block arguments” to the first block in the region.

Only dialect attribute names may be specified in the attribute dictionaries for function arguments, results, or the function itself.

# Example

```mlir
// External function definitions.
func.func @abort()
func.func @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64

// A function that returns its argument twice:
func.func @count(%x: i64) -> (i64, i64)
    attributes { fruit = "banana" } {
    return %x, %x: i64, i64
}

// A function with an argument attribute
func.func @example_fn_arg(%x: i32 { swift.self = unit })

// A function with a result attribute
func.func @example_fn_result() -> (f64 { dialect_name.attr_name = 0 : i64 })

// A function with an attribute
func.func @example_fn_attr() attributes { dialect_name.attr_name = false }
```
*/
#[mlir(
    traits = [
        AffineScope, AutomaticAllocationScope, IsolatedFromAbove, , Symbol
    ],
)]
pub struct Func {
    sym_name: SymbolNameAttr,
    function_type: TypeAttrOf<FunctionType>,
    sym_visibility: OptionalAttr<StringAttr>,
    arg_attrs: OptionalAttr<DictArrayAttr>,
    res_attrs: OptionalAttr<DictArrayAttr>,
    #[region]
    body: AnyRegion

//   let builders = [OpBuilder<(ins
//     "StringRef":$name, "FunctionType":$type,
//     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
//     CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
//   >];
}

impl Func {
    static Func create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static Func create(Location location, StringRef name, FunctionType type,
                         Operation::dialect_attr_range attrs);
    static Func create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);

    /// Create a deep copy of this function and all of its blocks, remapping any
    /// operands that use values outside of the function using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for function arguments, these arguments are not
    /// included in the new function. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    Func clone(BlockAndValueMapping &mapper);
    Func clone();

    /// Clone the internal blocks and attributes from this function into dest.
    /// Any cloned blocks are appended to the back of dest. This function
    /// asserts that the attributes of the current function and dest are
    /// compatible.
    void cloneInto(Func dest, BlockAndValueMapping &mapper);

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *self.callable_region() { return self.is_external() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> self.callable_results() { return self.function_type().outputs(); }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> self.argument_types() { return self.function_type().inputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> self.result_types() { return self.function_type().outputs(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "func"; }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return self.is_external(); }
}

impl AssemblyFormat for Func {

}

impl CallableOpInterface for Func {

}

impl FunctionOpInterface for Func {

}

impl OpAsmOpInterface for Func {

}

/**
Function return operation.

The `func.return` operation represents a return operation within a function.
The operation takes variable number of operands and produces no results.
The operand number and types must match the signature of the function that contains the operation.

# Example

```mlir
func.func @foo() : (i32, f8) {
    ...
    return %0, %1 : i32, f8
}
```
*/
#[mlir(
    traits = [
        Pure, HasParent<"Func">, MemRefsNormalisable, ReturnLike, Terminator
    ],
    assembly_format = "attr-dict ($operands^ `:` type($operands))?"
)]
pub struct Return {
    operands: [AnyType]

//   let builders = [OpBuilder<(ins), [{
//     build($_builder, $_state, std::nullopt);
//   }]>];
}

impl Verify for Return {

}
