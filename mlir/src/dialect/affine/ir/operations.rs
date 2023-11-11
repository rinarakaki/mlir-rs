//! # MLIR Affine Operations
//!
//! - include
//!   - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Affine/IR/AffineOps.h>
//!   - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Affine/IR/AffineOps.td>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Affine/IR/AffineOps.cpp>

use crate::mlir::ir::{
    affine_map::AffineMap,
    builtins::{
        types::{MemRef, Vector},
        attributes::StringAttribute
    },
    integer_set::IntegerSet,
    value_range::ValueRange
};

/**
Affine apply operation.

The affine.apply operation applies an [affine mapping](#affine-expressions) to a list of SSA values, yielding a single SSA value. The number of dimension and symbol arguments to `affine.apply` must be equal to the respective number of dimensional and symbolic inputs to the affine mapping; the affine mapping has to be one-dimensional, and so the `affine.apply` operation always returns one value. The input operands and result must all have ‘index’ type.

# Examples

```mlir
#map10 = affine_map<(d0, d1) -> (d0 floordiv 8 + d1 floordiv 128)>
...
%1 = affine.apply #map10 (%s, %t)
```

Inline example:

```mlir
%2 = affine.apply affine_map<(i)[s0] -> (i + s0)> (%42)[%n]
```
*/
#[mlir(
  traits = [Pure]
)]
pub struct Apply<const D: usize, const S: usize> {
    #[attribute]
    map: AffineMap<D, S, 1>,
    #[input]
    inputs: [usize; D + S],  // renamed from `map_operands`
    #[output]
    output: usize  // renamed from `result`
}

// TODO: The auto-generated builders should check to see if the return type
// has a constant builder. That way we wouldn't need to explicitly specify the
// result types here.
// let builders = [
// Builder<(ins AffineMap:$map, ValueRange:$map_operands),
// [{
//     build($_builder, $_state, $_builder.getIndexType(), map, map_operands);
// }]>,
// Builder<(ins "ArrayRef<AffineExpr> ":$exprList,ValueRange:$map_operands),
// [{
//     build($_builder, $_state, $_builder.getIndexType(),
//         AffineMap::inferFromExprList(exprList).front(), map_operands);
// }]>
// ];

impl Apply {
    /// Returns the affine map to be applied by this operation.
    AffineMap get_affine_map() { self.map }

    /// Returns the affine value map computed from this operation.
    pub fn affine_value_map(&self) -> AffineValueMap {
        AffineValueMap(self.map, self.inputs, self.output)
    }

    /// Returns true if the result of this operation can be used as dimension id
    /// in the region of the closest surrounding op with trait AffineScope.
    bool is_valid_dim();

    /// Returns true if the result of this operation can be used as dimension id
    /// within 'region', i.e., for all its uses with `region`.
    bool is_valid_dim(Region *region);

    /// Returns true if the result of this operation is a symbol in the region
    /// of the closest surrounding op that has the trait AffineScope.
    bool is_valid_symbol();

    /// Returns true if the result of this operation is a symbol for all its
    /// uses in `region`.
    bool is_valid_symbol(Region *region);
}

impl Fold for Apply {
  
}
impl Canonicalise for Apply {
  
}
impl AssemblyFormat for Apply {
    fn parse(parser: &OpAsmParser, result: &OperationState) -> ParseResult {
        let builder = parser.builder();
        let index_type = builder.get_index_type();
    
        AffineMapAttr map_attr;
        let num_dims: usize;
        if parser.parse_attribute(map_attr, "map", result.attributes)
        || parse_dim_and_symbol_list(parser, result.operands, num_dims)
        || parser.parse_optional_attr_dict(result.attributes)
        {
            return Err(());
        }
        let map = map_attr.get_value();
    
        if map.num_dims() != num_dims
        || num_dims + map.num_symbols() != result.operands.len() {
            return parser.emit_error(parser.name_loc(),
                                    "Dimension or symbol index mismatch");
        }
    
        result.types.append(map.num_outputs(), index_type);
        Ok(())
    }
    
    fn print(&self, p: &OpAsmPrinter) {
        p << " " << self.map;
        print_dim_and_symbol_list(
            self.inputs, self.map.num_dims(), p);
        p.print_optional_attr_dict((self).attrs(), /*elidedAttrs=*/{"map"});
    }
}

/**
`for` operation.

# Syntax

```text
operation   ::= `affine.for` ssa-id `=` lower-bound `to` upper-bound
                (`step` integer-literal)? `{` op* `}`

lower-bound ::= `max`? affine-map-attribute dim-and-symbol-use-list | shorthand-bound
upper-bound ::= `min`? affine-map-attribute dim-and-symbol-use-list | shorthand-bound
shorthand-bound ::= ssa-id | `-`? integer-literal
```

The `affine.for` operation represents an affine loop nest. It has one region containing its body. This region must contain one block that terminates with [`affine.yield`](#affineyield-affineyieldop). *Note:* when `affine.for` is printed in custom format, the terminator is omitted. The block has one argument of [`index`](Builtin.md/#indextype) type that represents the induction variable of the loop.

The `affine.for` operation executes its body a number of times iterating from a lower bound to an upper bound by a stride. The stride, represented by `step`, is a positive constant integer which defaults to `1` if not present. The lower and upper bounds specify a half-open range: the range includes the lower bound but does not include the upper bound.

The lower and upper bounds of a `affine.for` operation are represented as an application of an affine mapping to a list of SSA values passed to the map.
The [same restrictions](#restrictions-on-dimensions-and-symbols) hold for these SSA values as for all bindings of SSA values to dimensions and symbols.

The affine mappings for the bounds may return multiple results, in which case the `max`/`min` keywords are required (for the lower/upper bound respectively), and the bound is the maximum/minimum of the returned values.
There is no semantic ambiguity, but MLIR syntax requires the use of these keywords to make things more obvious to human readers.

Many upper and lower bounds are simple, so MLIR accepts two custom form syntaxes: the form that accepts a single 'ssa-id' (e.g. `%N`) is shorthand for applying that SSA value to a function that maps a single symbol to itself, e.g., `()[s]->(s)()[%N]`. The integer literal form (e.g. `-42`) is shorthand for a nullary mapping function that returns the constant value (e.g. `().(-42)()`).

Example showing reverse iteration of the inner loop:

```mlir
#map57 = affine_map<(d0)[s0] -> (s0 - d0 - 1)>

func.func @simple_example(%A: memref<?x?xf32>, %B: memref<?x?xf32>) {
    %N = dim %A, 0 : memref<?x?xf32>
    affine.for %i = 0 to %N step 1 {
        affine.for %j = 0 to %N {   // implicitly steps by 1
            %0 = affine.apply #map57(%j)[%N]
            %tmp = call @F1(%A, %i, %0)
                : (memref<?x?xf32>, index, index) -> (f32)
            call @F2(%tmp, %B, %i, %0)
                : (f32, memref<?x?xf32>, index, index) -> ()
        }
    }
    return
}
```

`affine.for` can also operate on loop-carried variables (`iter_args`) and return the final values after loop termination. The initial values of the variables are passed as additional SSA operands to the `affine.for` following the operands for the loop's lower and upper bounds. The operation's region has equivalent arguments for each variable representing the value of the variable at the current iteration.

The region must terminate with an `affine.yield` that passes all the current iteration variables to the next iteration, or to the `affine.for`'s results if at the last iteration. For `affine.for`'s that execute zero iterations, the initial values of the loop-carried variables (corresponding to the SSA operands) will be the op's results.

For example, to sum-reduce a memref:

```mlir
func.func @reduce(%buffer: memref<1024xf32>) -> (f32) {
    // Initial sum set to 0.
    %sum_0 = arith.constant 0.0 : f32
    // iter_args binds initial values to the loop's region arguments.
    %sum = affine.for %i = 0 to 10 step 2
        iter_args(%sum_iter = %sum_0) -> (f32) {
        %t = affine.load %buffer[%i] : memref<1024xf32>
        %sum_next = arith.addf %sum_iter, %t : f32
        // Yield current iteration sum to next iteration %sum_iter or to %sum
        // if final iteration.
        affine.yield %sum_next : f32
    }
    return %sum : f32
}
```

```mlir
%res:2 = affine.for %i = 0 to 128 iter_args(%arg0 = %init0, %arg1 = %init1)
            -> (index, index) {
    %y0 = arith.addi %arg0, %c1 : index
    %y1 = arith.addi %arg1, %c2 : index
    affine.yield %y0, %y1 : index, index
}
```

If the `affine.for` defines any values, a yield terminator must be explicitly present. The number and types of the `affine.for` results must match the initial values in the `iter_args` binding and the yield operands.
*/
#[mlir(
    traits = [
        AutomaticAllocationScope, ImplicitAffineTerminator,
        ConditionallySpeculatable, RecursiveMemoryEffects
    ]
)]
pub struct For<> {
    #[input]
    inputs: [_; _],  // renamed from `input`
    #[output]
    outputs: [_; _],  // renamed from `results`
    #[region]
    region: SizedRegion<1>
}

// let skipDefaultBuilders = 1;

// impl Build<
//     (
//         i64, i64, i64 /* 1 */, ValueRange /* Nonew */,
//         fn(&Builder, Location, Value, ValueRange) /* null */
//     )
// > for For {
//     fn build(
//         builder: &Builder,
//         (
//             lower_bound,
//             upper_bound,
//             step,
//             iter_args,
//             body_builder
//         ),
//         output: &mut 
//     ) {
        
//     }
// }

// impl Build<(
//     ValueRange, AffineMap, ValueRange, AffineMap, i64 /* 1 */,
//     ValueRange, /* Nonew */
//     fn(&Builder, Location, Value, ValueRange) /* null */
// )> for For {
//     fn build(
//         builder: &Builder,
//         (
//             lower_bound_inputs,
//             lower_bound_map,
//             upper_bound_inputs,
//             upper_bound_map,
//             step,
//             iter_args,
//             body_builder
//         ): (

//         ),
//         output: &mut 
//     ) {

//     }
// }

impl For {
    /// Defining the function type we use for building the body of affine.for.
    using BodyBuilderFn =
        function_ref<void(Builder &, Location, Value, ValueRange)>;

    BlockArgument get_induction_var() { return get_body().input(0); }
    Block::BlockArgListType getRegionIterArgs() {
      return get_body().inputs().drop_front();
    }
    Operation::operand_range get_iter_operands() {
        return self.inputs.drop_front(get_num_control_operands());
    }

    // TODO: provide iterators for the lower and upper bound operands
    // if the current access via get_lower_bound(), get_upper_bound() is too slow.

    /// Returns operands for the lower bound map.
    operand_range get_lower_bound_operands();

    /// Returns operands for the upper bound map.
    operand_range get_upper_bound_operands();

    /// Returns operands for the lower and upper bound maps with the operands
    /// for the lower bound map in front of those for the upper bound map.
    operand_range get_control_operands();

    /// Returns information about the lower bound as a single object.
    AffineBound get_lower_bound();

    /// Returns information about the upper bound as a single object.
    AffineBound get_upper_bound();

    /// Returns loop step.
    pub fn step(&self) -> i64 {
        self.get_attr("step").cast<IntegerAttribute>().get_int()
    }

    /// Returns affine map for the lower bound.
    pub fn lower_bound_map(&self) -> AffineMap {
        self.get_attr("lower_bound").cast<AffineMapAttr>().get_value()
    }

    /// Returns affine map for the upper bound. The upper bound is exclusive.
    pub fn upper_bound_map(&self) -> AffineMap {
        self.get_attr("upper_bound").cast<AffineMapAttr>().get_value()
    }

    /// Set lower bound. The new bound must have the same number of operands as
    /// the current bound map. Otherwise, 'replaceForLowerBound' should be used.
    pub fn set_lower_bound(operands: ValueRange, map: AffineMap);

    /// Set upper bound. The new bound must not have more operands than the
    /// current bound map. Otherwise, 'replaceForUpperBound' should be used.
    pub fn set_upper_bound(operands: ValueRange, map: AffineMap);

    /// Set the lower bound map without changing operands.
    pub fn set_lower_bound_map(map: AffineMap);

    /// Set the upper bound map without changing operands.
    pub fn set_upper_bound_map(map: AffineMap);

    /// Set loop step.
    pub fn set_step(&mut self, step: i64) {
        assert!(step > 0, "step has to be a positive integer constant");
        let context = self.lower_bound_map().context();
        self.set_attr(StringAttribute::get(context, "step"),
                        IntegerAttribute::get(IndexType::get(context), step));
    }

    /// Returns number of region arguments for loop-carried values.
    usize get_num_region_iter_args() {
        return get_body().getNumArguments() - 1;
    }

    /// Number of operands controlling the loop: lb and ub.
    usize get_num_control_operands() {
        get_operation().num_inputs() - get_num_iter_operands()
    }

    /// Get the number of loop-carried values.
    usize get_num_iter_operands();

    /// Returns true if the lower bound is constant.
    bool has_constant_lower_bound();
    /// Returns true if the upper bound is constant.
    bool has_constant_upper_bound();
    /// Returns true if both bounds are constant.
    bool has_constant_bounds() {
        return has_constant_lower_bound() && has_constant_upper_bound();
    }
    /// Returns the value of the constant lower bound.
    /// Fails assertion if the bound is non-constant.
    i64 get_constant_lower_bound();

    /// Returns the value of the constant upper bound. The upper bound is
    /// exclusive. Fails assertion if the bound is non-constant.
    i64 get_constant_upper_bound();

    /// Sets the lower bound to the given constant value.
    void set_constant_lower_bound(i64 value);
    
    /// Sets the upper bound to the given constant value.
    void set_constant_upper_bound(i64 value);

    /// Returns true if both the lower and upper bound have the same operand
    /// lists (same operands in the same order).
    bool matching_bound_operand_list();

    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability get_speculatability();
}

impl RegionVerify for For {

}

impl Fold for For {

}

impl Canonicalise for For {

}

impl AssemblyFormat for For {

}

impl LoopLikeOpInterface for For {
    fn single_induction_var() {

    }

    fn single_lower_bound() {

    }

    fn single_step() {

    }

    fn single_upper_bound() {

    }
}

impl RegionBranchOpInterface for For {
    fn successor_entry_operands() {

    }
}

/**
if-then-else operation.

# Syntax

```
operation  ::= `affine.if` if-op-cond `{` op* `}` (`else` `{` op* `}`)?
if-op-cond ::= integer-set-attr dim-and-symbol-use-list
```

The `affine.if` operation restricts execution to a subset of the loop iteration space defined by an integer set (a conjunction of affine constraints). A single `affine.if` may end with an optional `else` clause.

The condition of the `affine.if` is represented by an [integer set](#integer-sets) (a conjunction of affine constraints), and the SSA values bound to the dimensions and symbols in the integer set. The [same restrictions](#restrictions-on-dimensions-and-symbols) hold for these SSA values as for all bindings of SSA values to dimensions and symbols.

The `affine.if` operation contains two regions for the 'then' and 'else' clauses. `affine.if` may return results that are defined in its regions. The values defined are determined by which execution path is taken. Each region of the `affine.if` must contain a single block with no arguments, and be terminated by `affine.yield`. If `affine.if` defines no values, the `affine.yield` can be left out, and will be inserted implicitly. Otherwise, it must be explicit. If no values are defined, the else block may be empty (i.e. contain no blocks).

# Examples

```mlir
#set = affine_set<(d0, d1)[s0]
    : (d0 - 10 >= 0, s0 - d0 - 9 >= 0, d1 - 10 >= 0, s0 - d1 - 9 >= 0)>
func.func @reduced_domain_example(%A, %X, %N) : (memref<10xi32>, i32, i32) {
    affine.for %i = 0 to %N {
        affine.for %j = 0 to %N {
            %0 = affine.apply #map42(%j)
            %tmp = call @S1(%X, %i, %0)
            affine.if #set(%i, %j)[%N] {
                %1 = affine.apply #map43(%i, %j)
                call @S2(%tmp, %A, %i, %1)
            }
        }
    }
    return
}
```

With an explicit yield (initialisation with edge padding):

```mlir
#interior = affine_set<(i, j)
    : (i - 1 >= 0, j - 1 >= 0,  10 - i >= 0, 10 - j >= 0)> (%i, %j)
func.func @pad_edges(%I : memref<10x10xf32>) -> memref<12x12xf32> {
    %O = alloc memref<12x12xf32>
    affine.parallel (%i, %j) = (0, 0) to (12, 12) {
        %1 = affine.if #interior (%i, %j) {
            %2 = load %I[%i - 1, %j - 1] : memref<10x10xf32>
            affine.yield %2
        } else {
            %2 = arith.constant 0.0 : f32
            affine.yield %2 : f32
        }
        affine.store %1, %O[%i, %j] : memref<12x12xf32>
    }
    return %O
}
```
*/
#[mlir(
    traits = [
        ImplicitAffineTerminator, RecursivelySpeculatable,
        RecursiveMemoryEffects, NoRegionArguments
   ]
)]
pub struct If<const I: usize, const O: usize> {
    #[input]
    inputs: [_; I],
    #[output]
    outputs: [_; O],  // renamed from `results`
    #[region]
    then: SizedRegion<1>,  // renamed from `then_region`
    #[region]
    r#else: AnyRegion  // renamed from `else_region`
}

// let skipDefaultBuilders = 1;

// impl Build<(IntegerSet, ValueRange, bool)> for If {
//     fn build(
//         builder,
//         (set, inputs, with_else_region),
//         output
//     ) {

//     }
// }

// impl Build<(TypeRange, IntegerSet, ValueRange, bool)> for If {
//     fn build(
//         builder,
//         (
//             output_types,
//             set,
//             inputs,
//             with_else_region
//         ),
//         output
//     ) {

//     }
// }

impl If {
    IntegerSet get_integer_set();
    void set_integer_set(IntegerSet newSet);

    /// Sets the integer set with its operands.
    void set_conditional(IntegerSet set, ValueRange operands);

    /// Returns true if an else block exists.
    bool has_else() { return !self.r#else.is_empty(); }

    Block *get_then_block() {
        assert!(!self.then.is_empty(), "Unexpected empty 'then' region.");
        return &self.then.front();
    }

    Block *get_else_block() {
        assert!(has_else(), "Empty 'else' region.");
        return &self.r#else.front();
        }

    Builder get_then_body_builder() {
        assert!(!self.then.is_empty(), "Unexpected empty 'then' region.");
        Block &body = self.then.front();
        return Builder(&body, std::prev(body.end()));
    }
    Builder get_else_body_builder() {
        assert!(has_else(), "No 'else' block");
        let body = self.r#else.front();
        return Builder(&body, std::prev(body.end()));
    }
}

impl Verify for If {
    fn verify(&self) -> LogicalResult {
        // Verify that we have a condition attribute.
        // FIXME: This should be specified in the arguments list in ODS.
        let condition_attr =
            self.get_attr_of_type<IntegerSetAttr>("condition");
        if (!condition_attr){
          return emit_op_error("requires an integer set attribute named 'condition'");}
      
        // Verify that there are enough operands for the condition.
        let condition = condition_attr.get_value();
        if I != condition.num_inputs() {
            return emit_op_error(
                "Operand count and condition integer set dimension and symbol count must match");
        }
      
        // Verify that the operands are valid dimension/symbols.
        if failed(verify_dim_and_symbol_identifiers(
            self, self.inputs, condition.num_dims()))
        {
            return Err(());
        }
      
        return Ok(());
      }
}

impl Fold for If {

}

impl Canonicalise for If {

}

impl AssemblyFormat for If {

}

impl RegionBranchOpInterface for If {

}

/**
`affine.load` operation.

`affine.load` operation reads an element from a memref, where the index for each memref dimension is an affine expression of loop induction variables and symbols. The output of 'affine.load' is a new value with the same type as the elements of the memref. An affine expression of loop IVs and symbols must be specified for each dimension of the memref. The keyword `symbol` can be used to indicate SSA identifiers which are symbolic.

# Examples

```mlir
%1 = affine.load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
```

Uses `symbol` keyword for symbols `%n` and `%m`:

```mlir
%1 = affine.load %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>
```
*/
#[mlir(
    traits = [MemRefsNormalisable]
)]
pub struct Load<T, const N: usize, const > {
    /// Reference to load from.
    #[input(traits = [MemRead])]
    memref: MemRef<T, N>,
    #[input]
    indices: [usize; _],
    #[output]
    output: T  // renamd from `result`
}

/// Builds an affine load op with the specified map and operands.
impl Build<(AffineMap<D, S, O>, [usize; 1 + D + S])> for Load<T, N> {
    fn build(
        builder: &Builder,
        (map, inputs): (AffineMap<D, S, O>, [usize; 1 + D + S]),
        output: &mut OperationState
    ) {
        assert!(
            inputs.len() == 1 + D + S,
            "Inconsistent operands");
        output.add_inputs(inputs);
        if (map) {
            output.add_attribute("map", AffineMap);
        }
        output.types.push(T);
    }
}

/// Builds an affine load op with an identity map and operands.
impl Build<(MemRef<T, N>, [usize; _])> for Load {
    fn build(
        builder: &Builder,
        (memref, indices): (MemRef<T, N>, [usize; _]),
        output: &mut OperationState
    ) {
        // Create identity map for memrefs with at least one dimension or `() -> ()` for zero-dimensional memrefs.
        let map = if N > 0 {
            builder.get_multi_dim_identity_map(N)  // N, 0, N
        } else {
            builder.get_empty_affine_map()
        };
        Self::build(builder, (memref, map, indices), output);
    }
}

/// Builds an affine load op with the specified map and its operands.
impl Build<(MemRef<T, N>, AffineMap<D, S, O>, [usize; D + S])> for Load {
    fn build(
        builder: &Builder,
        (memref, map, inputs): (
            MemRef<T, N>, AffineMap<D, S, O>, [usize; D + S]
        ),
        output: &mut OperationState
    ) {
        output.add_operands(memref);
        output.add_operands(inputs);
        output.add_attribute("map", map);
        output.types.push(T);
    }
}

impl Load {
    pub fn set_memref(&mut self, value: Value) {
        self.memref = value;
    }

    /// Returns the affine map used to index the memref for this operation.
    pub fn affine_map_attr(&self) -> AffineMapAttr {
        self.get_attr("map").cast<AffineMapAttr>()
    }
}

impl Verify for Load {
    fn verify(&self) -> LogicalResult {
        if verify_memory_op_indexing(
            self,
            self.get_attr_of_type<AffineMapAttr>("map"),
            map_operands(),
            self.memref,
            self.num_operands() - 1).is_err()
        {
            return Err(());
        }
    
        Ok(())
    }
}

impl Fold for Load {

}

impl Canonicalise for Load {
    fn canonicalisation_patterns(
        results: &RewritePatternSet, context: *mut MLIRContext*context
    ) {
        results.add<SimplifyAffine<Load>>(context);
    }
    
}

impl AssemblyFormat for Load {

}

impl AffineReadOpInterface for Load {

}

impl AffineMapAccessInterface for Load {

}

class AffineMinMaxOpBase<string mnemonic, list<Trait> traits = []> :
    Op<Affine_Dialect, mnemonic, traits> {



  let extraClassDeclaration = [{
    AffineMap getAffineMap() { return self.map; }
    ValueRange get_map_operands() { return self.inputs; }
    ValueRange getDimOperands() {
      return OperandRange{self.inputs.begin(),
                          self.inputs.begin() + self.map.num_dims()};
    }
    ValueRange getSymbolOperands() {
      return OperandRange{self.inputs.begin() + self.map.num_dims(),
                          self.inputs.end()};
    }
  }];
}

/**
`affine.min` operation.

# Syntax

```text
operation ::= ssa-id `=` `affine.min` affine-map-attribute dim-and-symbol-use-list
```

The `affine.min` operation applies an [affine mapping](#affine-expressions) to a list of SSA values, and returns the minimum value of all result expressions. The number of dimension and symbol arguments to `affine.min` must be equal to the respective number of dimensional and symbolic inputs to the affine mapping; the `affine.min` operation always returns one value. The input operands and result must all have `index` type.

# Example

```mlir
%0 = affine.min affine_map<(d0)[s0] -> (1000, d0 + 512, s0)> (%arg0)[%arg1]
```
*/
#[mlir(
    traits = [Pure]
)]
pub struct Min<const D: usize, const S: usize, O> {
    #[attribute]
    map: AffineMap<D, S, O>,
    #[input]
    inputs: [usize; D + S],  // renamed from operands
    #[output]
    output: usize
}

// impl Build<(AffineMap<D, S, O>, [usize; D + S])> for Min {
//     fn build(
//         builder,
//         (map, inputs): (AffineMap<D, S, O>, [usize; D + S]),
//         output
//     ) {
//         build($_builder, $_state, $_builder.getIndexType(), map, inputs);
//     }
// }

impl Fold for Min {

}

impl Canonicalise for Min {

}

impl AssemblyFormat for Min {

}

/**
`affine.max` operation.

`affine.max` operation computes the maximum value result from a multi-result
affine map.

# Example

```mlir
%0 = affine.max (d0) -> (1000, d0 + 512) (%i0) : index
```
*/
#[mlir(
    traits = [Pure]
)]
pub struct Max<const D: usize, const S: usize, O> {
    #[attribute]
    map: AffineMap<D, S, O>,
    #[input]
    inputs: [usize; D + S],  // renamed from operands
    #[output]
    output: usize
}

// let builders = [
//     Builder<(ins "AffineMap":$affineMap, "ValueRange":$map_operands),
//     [{
//       build($_builder, $_state, $_builder.getIndexType(), affineMap, map_operands);
//     }]>
//   ];

impl Fold for Max {

}

impl Canonicalise for Max {

}

impl AssemblyFormat for Max {
    
}

/**
Multi-index parallel band operation.

`affine.parallel` operation represents a hyper-rectangular affine parallel band, defining zero or more SSA values for its induction variables. It has one region capturing the parallel band body. The induction variables are represented as arguments of this region. These SSA values always have type index, which is the size of the machine word. The strides, represented by steps, are positive constant integers which defaults to `1` if not present. The lower and upper bounds specify a half-open range: the range includes the lower bound but does not include the upper bound. The body region must contain exactly one block that terminates with `affine.yield`.

The lower and upper bounds of a parallel operation are represented as an application of an affine mapping to a list of SSA values passed to the map. The same restrictions hold for these SSA values as for all bindings of SSA values to dimensions and symbols. The list of expressions in each map is interpreted according to the respective bounds group attribute. If a single expression belongs to the group, then the result of this expression is taken as a lower(upper) bound of the corresponding loop induction variable. If multiple expressions belong to the group, then the lower(upper) bound is the max(min) of these values obtained from these expressions. The loop band has as many loops as elements in the group bounds attributes.

Each value yielded by affine.yield will be accumulated/reduced via one of the reduction methods defined in the AtomicRMWKind enum. The order of reduction is unspecified, and lowering may produce any valid ordering. Loops with a 0 trip count will produce as a result the identity value associated with each reduction (i.e. 0.0 for addf, 1.0 for mulf). Assign reductions for loops with a trip count != 1 produces undefined results.

NOTE: Calling Parallel::build will create the required region and block, and insert the required terminator if it is trivial (i.e. no values are yielded). Parsing will also create the required region, block, and terminator, even when they are missing from the textual representation.

# Examples

3x3 valid convolution:

```mlir
func.func @conv_2d(%D : memref<100x100xf32>, %K : memref<3x3xf32>) -> (memref<98x98xf32>) {
    %O = alloc memref<98x98xf32>
    affine.parallel (%x, %y) = (0, 0) to (98, 98) {
    %0 = affine.parallel (%kx, %ky) = (0, 0) to (2, 2) reduce ("addf") {
        %1 = affine.load %D[%x + %kx, %y + %ky] : memref<100x100xf32>
        %2 = affine.load %K[%kx, %ky] : memref<3x3xf32>
        %3 = arith.mulf %1, %2 : f32
        affine.yield %3 : f32
    }
    affine.store %0, O[%x, %y] : memref<98x98xf32>
    }
    return %O
}
```

Tiling by potentially imperfectly dividing sizes:

```mlir
affine.parallel (%ii, %jj) = (0, 0) to (%N, %M) step (32, 32) {
    affine.parallel (%i, %j) = (%ii, %jj)
                            to (min(%ii + 32, %N), min(%jj + 32, %M)) {
    call @f(%i, %j) : (index, index) -> ()
    }
}
```
*/
#[mlir(
    traits = [
        AutomaticAllocationScope, ImplicitAffineTerminator,
        RecursivelySpeculatable, RecursiveMemoryEffects, DeclareOpInterfaceMethods<>,
        MemRefsNormalisable]
)]
pub struct Parallel {
    #[attribute]
    reductions: TypedArrayAttrBase<AtomicRMWKindAttr, "Reduction ops">,
    #[attribute]
    lower_bounds_map: AffineMap,
    #[attribute]
    lower_bounds_groups: I32ElementsAttr,
    #[attribute]
    upper_bounds_map: AffineMap,
    #[attribute]
    upper_bounds_groups: I32ElementsAttr,
    #[attribiute]
    steps: I64SmallVectorArrayAttr,
    #[input]
    inputs: [_; _]  // renamed from `map_operands`
    #[output]
    outputs: [_; _]  // renamed from `results`
    #[region]
    region: SizedRegion<1>
}

// impl Build<(TypeRange, &[AtomicRMWKind], &[i64])> for Parallel {

// }

// let builders = [
// Builder<(ins "":$resultTypes,
//     "ArrayRef<>":$reductions, "ArrayRef<int64_t>":$ranges)>,
// Builder<(ins "TypeRange":$resultTypes,
//     "ArrayRef<arith::AtomicRMWKind>":$reductions, "ArrayRef<AffineMap>":$lbMaps,
//     "ValueRange":$lbArgs, "ArrayRef<AffineMap>":$ubMaps, "ValueRange":$ubArgs,
//     "ArrayRef<int64_t>":$steps)>
// ];

impl Parallel {
    /// Get the number of dimensions.
    usize num_dims();

    /// Get ranges as constants, may fail in dynamic case.
    std::optional<SmallVector<int64_t, 8>> getConstantRanges();

    Block *getBody();
    Builder getBodyBuilder();
    MutableArrayRef<BlockArgument> getIVs() {
      return getBody()->getArguments();
    }

    /// Returns elements of the loop lower bound.
    AffineMap getLowerBoundMap(usize pos);
    operand_range get_lower_bounds_operands();
    AffineValueMap get_lower_bounds_value_map();

    /// Sets elements of the loop lower bound.
    void set_lower_bounds(ValueRange operands, AffineMap map);

    /// Returns elements of the loop upper bound.
    AffineMap get_upper_bound_map(usize pos);
    operand_range get_upper_bounds_operands();
    AffineValueMap get_upper_bounds_value_map();

    /// Sets elements fo the loop upper bound.
    void set_upper_bounds(ValueRange operands, AffineMap map);

    void setSteps(ArrayRef<int64_t> newSteps);

    /// Returns attribute names to use in op construction. Not expected to be
    /// used directly.
    static &'static str get_reductions_attr_str_name() { return "reductions"; }
    static &'static str get_lower_bounds_map_attr_str_name() { return "lower_bounds_map"; }
    static &'static str get_lower_bounds_groups_attr_str_name() {
      return "lower_bounds_groups";
    }
    static &'static str get_upper_bounds_map_attr_str_name() { return "upper_bounds_map"; }
    static &'static str get_upper_bounds_groups_attr_str_name() {
      return "upper_bounds_groups";
    }
    static &'static str get_steps_attr_str_name() { return "steps"; }

    /// Returns `true` if the loop bounds have min/max expressions.
    bool hasMinMaxBounds() {
      return get_lower_bounds_map().num_outputs() != num_dims() ||
             get_upper_bounds_map().num_outputs() != num_dims();
    }
}

impl Verify for Parallel {

}

impl Fold for Parallel {

}

impl AssemblyFormat for Parallel {

}

impl LoopLikeOpInterface for Parallel {

}

/**
`affine.prefetch` operation.

`affine.prefetch` op prefetches data from a memref location described with an affine subscript similar to affine.load, and has three attributes: a read/write specifier, a locality hint, and a cache type specifier as shown below:

```mlir
affine.prefetch %0[%i, %j + 5], read, locality<3>, data : memref<400x400xi32>
```

The read/write specifier is either `read` or `write`, the locality hint specifier ranges from `locality<0>` (no locality) to `locality<3>` (extremely local keep in cache). The cache type specifier is either `data` or `instr` and specifies whether the prefetch is performed on data cache or on instruction cache.
*/
#[mlir(
)]
pub struct Prefetch {
    #[input]
    memref: MemRef<_, _>,
    #[input]
    indices: [usize; _],
    #[attribute]
    is_write: bool,
    #[attribuet]
    locality_hint: ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<3>]>,
    #[attribute]
    is_data_cache: bool
}

// impl Build<(
//     MemRef<_, _>, AffineMap<D, S, O>, [Value; D + S], bool, usize, bool
// )> for Prefetch {
//     fn build(
//         builder,
//         (
//             memref,
//             map,
//             inputs,
//             is_write,
//             locality_hint,
//             is_data_cache
//         ),
//         output
//     ) {
//         let locality_hint_attr = builder.get_i32_integer_attr(locality_hint);
//         let is_write_attr = builder.get_bool_attr(is_write);
//         let is_data_cache_attr = builder.get_bool_attr(is_data_cache);
//         output.add_operands::<>(memref);
//         output.add_attribute::<AffineMap>("map", map);
//         output.add_operands::<>(inputs);
//         output.add_attribute("locality_hint"(), locality_hint_attr);
//         output.add_attribute("is_write", is_write_attr);
//         output.add_attribute("is_data_cache", is_data_cache_attr);
//     }
// }

impl Prefetch {
    /// Returns the affine map used to index the memref for this operation.
    AffineMap getAffineMap() { return affine_map_attr().getValue(); }
    AffineMapAttr affine_map_attr() {
        return self.get_attr("map").cast<AffineMapAttr>();
    }

    /// Get affine map operands.
    operand_range get_map_operands() {
      return {operand_begin() + 1, operand_end()};
    }

    static &'static str get_is_data_cache_attr_str_name() { return "is_data_cache"; }
}

impl Verify for Prefetch {

}

impl Fold for Prefetch {

}

impl Canonicalise for Prefetch {

}

impl AssemblyFormat for Prefetch {

}

impl AffineMapAccessInterface for Prefetch {
    /// Returns the AffineMapAttr associated with `memref`.
    fn affine_map_attr_for_mem_ref(&self, memref: Value) -> NamedAttribute {
        assert!(memref == self.memref,
               "Expected memref argument to match memref operand");
        return {StringAttribute::new(self.context(), "map"),
          affine_map_attr()};
      }
}

/**
`affine.store` operation.

The `affine.store` op writes an element to a memref, where the index
for each memref dimension is an affine expression of loop induction
variables and symbols. The 'affine.store' op stores a new value which is the
same type as the elements of the memref. An affine expression of loop IVs
and symbols must be specified for each dimension of the memref. The keyword
`symbol` can be used to indicate SSA identifiers which are symbolic.

# Examples

```mlir
affine.store %v0, %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
```

Uses `symbol` keyword for symbols `%n` and `%m`:

```mlir
affine.store %v0, %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>
```
*/
#[mlir(
   traits = [MemRefsNormalisable]
)]
pub struct Store {
    #[input]
    value: _,
    /// Reference to store to.
    #[input(traits = [MemWrite])]
    memref: MemRef<_, _>,
    #[input]
    indices: [usize; _]
}

// let skipDefaultBuilders = 1;
// let builders = [
//   Builder<(ins Value:$value_to_store, Value:$memref,
//     "ValueRange":$indices)>,
//   Builder<(ins Value:$value_to_store, Value:$memref, "AffineMap":$map,
//     "ValueRange":$map_operands)>
// ];

impl Store {
    pub fn set_memref(&mut self, value: Value) {
        self.memref = value;
    }

    /// Returns the affine map used to index the memref for this operation.
    pub fn affine_map_attr(&self) -> AffineMapAttr {
        self.get_attr("map").cast<AffineMapAttr>()
    }
}

impl Verify for Store {

}

impl Fold for Store {

}

impl Canonicalise for Store {

}

impl AssemblyFormat for Store {

}

impl AffineWriteOpInterface for Store {

}

impl AffineMapAccessInterface for Store {

}

/**
Yield values to parent operation.

`affine.yield` yields zero or more SSA values from an affine op region and
terminates the region. The semantics of how the values yielded are used
is defined by the parent operation.
If `affine.yield` has any operands, the operands must match the parent
operation's results.
If the parent operation defines no values, then the "affine.yield" may be
left out in the custom syntax and the builders will insert one implicitly.
Otherwise, it has to be present in the syntax to indicate which values are
yielded.
*/
#[mlir(
    traits = [Pure, Terminator, ReturnLike, MemRefsNormalisable],
    assembly_format = "attr-dict ($inputs^ `:` type($inputs))?"
)]
pub struct Yield {
    #[input]
    inputs: [_; _]  // renamed from `operands`
}

impl From<()> for Yield {
    fn from(value: ()) -> Self {
        build($_builder, $_state, None)
    }
}

impl Verify for Yield {

}

/**
`affine.vector_load` operation.

The `affine.vector_load` is the vector counterpart of [affine.load](#affineload-affineloadop). It reads a slice from a [MemRef](Builtin.md/#memreftype), supplied as its first operand, into a [vector](Builtin.md/#vectortype) of the same base elemental type.
The index for each memref dimension is an affine expression of loop induction variables and symbols. These indices determine the start position of the read within the memref. The shape of the return vector type determines the shape of the slice read from the memref. This slice is contiguous along the respective dimensions of the shape. Strided vector loads will be supported in the future.
An affine expression of loop IVs and symbols must be specified for each dimension of the memref. The keyword `symbol` can be used to indicate SSA identifiers which are symbolic.

Examples

8-wide f32 vector load:

```mlir
%1 = affine.vector_load %0[%i0 + 3, %i1 + 7]
    : memref<100x100xf32>, vector<8xf32>
```

4-wide f32 vector load. Uses `symbol` keyword for symbols `%n` and `%m`:

```mlir
%1 = affine.vector_load %0[%i0 + symbol(%n), %i1 + symbol(%m)]
    : memref<100x100xf32>, vector<4xf32>
```

2-dim f32 vector load:

```mlir
%1 = affine.vector_load %0[%i0, %i1] : memref<100x100xf32>, vector<2x8xf32>
```

TODO:
- Add support for strided vector loads.
- Consider adding a permutation map to permute the slice that is read from memory
(see [vector.transfer_read](../Vector/#vectortransfer_read-vectortransferreadop)).
*/
#[mlir(
    traits = [MemRefsNormalisable]
)]
pub struct VectorLoad<T, const N: usize> {
    /// Reference to load from.
    #[input(traits = [MemRead])]
    memref: MemRef<_, _>,
    #[input]
    indices: [usize; _]
    #[output]
    output: Vector<_, _> // renamed from `result`
}

/// Builds an affine vector load op with the specified map and operands.
impl From<(Vector<_, _>, AffineMap, ValueRange)> for VectorLoad {

}

/// Builds an affine vector load op with an identity map and operands.
impl From<(Vector<_, _>, Value,  CArg<"ValueRange", "{}">)> for VectorLoad {

}

/// Builds an affine vector load op with the specified map and its operands.
impl From<(Vector<_, _>, Value, AffineMap, ValueRange)> for VectorLoad {

}

//   let builders = [
    
//     Builder<(ins "VectorType":$resultType, "AffineMap":$map,
//       "ValueRange":$operands)>,
    
//     Builder<(ins "VectorType":$resultType, Value:$memref,
//      :$indices)>,
    
//     Builder<(ins "VectorType":$resultType, Value:$memref,
//       "AffineMap":$map, "ValueRange":$map_operands)>
//   ];

impl VectorLoad {
    // extraClassDeclarationBase #
    pub fn set_memref(&mut self, value: Value) {
        self.memref = value;
    }

    /// Returns the affine map used to index the memref for this operation.
    pub fn affine_map_attr(&self) -> AffineMapAttr {
        self.get_attr("map").cast<AffineMapAttr>()
    }
}

impl Verify for VectorLoad {

}

impl Canonicalise for VectorLoad {
    
}

impl AssemblyFormat for VectorLoad {
    
}

impl AffineReadOpInterface for VectorLoad {
    
}

impl AffineMapAccessInterface for VectorLoad {

}

/**
`affine.vector_store` operation.

The `affine.vector_store` is the vector counterpart of [affine.store](#affinestore-affinestoreop). It writes a [vector](Builtin.md/#vectortype), supplied as its first operand, into a slice within a [MemRef](Builtin.md/#memreftype) of the same base elemental type, supplied as its second operand.
The index for each memref dimension is an affine expression of loop induction variables and symbols. These indices determine the start position of the write within the memref. The shape of th input vector determines the shape of the slice written to the memref. This slice is contiguous along the respective dimensions of the shape. Strided vector stores will be supported in the future.
An affine expression of loop IVs and symbols must be specified for each dimension of the memref. The keyword `symbol` can be used to indicate SSA identifiers which are symbolic.

# Examples

8-wide f32 vector store:

```mlir
affine.vector_store %v0, %0[%i0 + 3, %i1 + 7]
    : memref<100x100xf32>, vector<8xf32>
```

4-wide f32 vector store. Uses `symbol` keyword for symbols `%n` and `%m`:

```mlir
affine.vector_store %v0, %0[%i0 + symbol(%n), %i1 + symbol(%m)]
    : memref<100x100xf32>, vector<4xf32>
```

2-dim f32 vector store:

```mlir
affine.vector_store %v0, %0[%i0, %i1] : memref<100x100xf32>, vector<2x8xf32>
```

TODO:

- Add support for strided vector stores.
- Consider adding a permutation map to permute the slice that is written to memory
(see [vector.transfer_write](../Vector/#vectortransfer_write-vectortransferwriteop)).
*/
#[mlir(
    traits = [MemRefsNormalisable]
)]
pub struct VectorStore {
    #[input]
    value: Vector<_, _>,
    /// Reference to store to.
    #[input(traits = [MemWrite])]
    memref: MemRef<_, _>,
    #[input]
    indices: [usize: _],
}

//   let skipDefaultBuilders = 1;

impl From<(Value, Value, ValueRange)> for VectorStore {
    fn from((value, memref, indices): (Value, Value, ValueRange)) -> Self {
        
    }
}

impl From<(Value, Value, AffineMap, ValueRange)> for VectorStore {
    fn from((value, memref, map, map_inputs): (Value, Value, ValueRange))
    -> Self {
        
    }
}

impl VectorStore {
    // extraClassDeclarationBase #
    pub fn set_memref(&mut self, value: Value) {
        self.memref = value;
    }

    /// Returns the affine map used to index the memref for this operation.
    pub fn affine_map_attr(&self) -> AffineMapAttr {
        self.get_attr("map").cast<AffineMapAttr>()
    }
}

impl Verify for VectorStore {

}

impl Canonicalise for VectorStore {

}

impl AssemblyFormat for VectorStore {

}

impl AffineWriteOpInterface for VectorStore {

}

impl AffineMapAccessInterface for VectorStore {

}


//===----------------------------------------------------------------------===//
// DelineariseIndex
//===----------------------------------------------------------------------===//

/**
# Delinearise an index

The `affine.delinearise_index` operation takes a single index value and
calculates the multi-index according to the given basis.

# Examples

```mlir
%indices:3 = affine.delinearise_index %linear_index into (%c16, %c224, %c224)
    : index, index, index
```

In the above example, `%indices:3` conceptually holds the following:

```mlir
#map0 = affine_map<()[s0] -> (s0 floordiv 50176)>
#map1 = affine_map<()[s0] -> ((s0 mod 50176) floordiv 224)>
#map2 = affine_map<()[s0] -> (s0 mod 224)>
%indices_0 = affine.apply #map0()[%linear_index]
%indices_1 = affine.apply #map1()[%linear_index]
%indices_2 = affine.apply #map2()[%linear_index]
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$linear_index `into` ` ` `(` $basis `)` attr-dict `:` type($output)"
)]
pub struct DelineariseIndex {
    #[input]
    linear_index: usize,
    #[input]
    basis: [usize; _],
    #[output]
    output: [usize: _]  // renamed from `multi_index`
}

impl From<(Value, &[FoldResult])> for DelineariseIndex {
    fn from((linear_index, basis): (Value, &[FoldResult])) -> Self {
        
    }
}

impl Verify for DelineariseIndex {

}
