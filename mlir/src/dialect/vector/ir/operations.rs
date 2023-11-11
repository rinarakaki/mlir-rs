/*!
# MLIR Vector Dialect Operations

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Vector/IR/VectorOps.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Vector/IR/VectorOps.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Vector/IR/VectorOps.cpp>
*/

use crate::{
    mlir::{
        dialect::{
            arith::ir::operations::ConstantIndex,
            vector::interfaces::maskable_op_interface::MaskableOpInterface
        },
        interfaces::{
            infer_type_op_interface::InferTypeOpInterface,
            vector_interfaces::{
                VectorUnrollOpInterface, VectorTransferOpInterface
            },
            side_effect_interfaces::MemoryEffectsOpInterface,
            destination_style_op_interface::DestinationStyleOpInterface,
            view_like_interface::ViewLikeOpInterface
        },
        ir::{
            affine_map::AffineMap,
            builtins::{
                attributes::ArrayAttribute,
                types::{Vector, MemRef}
            },
            operation::{
                base::AnyType,
                definition::FoldResult
            }
        },
    },
    llvm::adt::small_vector::SmallVector
};

// pub struct Vector_Dialect : Dialect {
//     let name = "vector";
//     let cppNamespace = "::mlir::vector";
  
//     let useDefaultAttributePrinterParser = 1;
//     let hasConstantMaterializer = 1;
//     let dependentDialects = ["arith::ArithDialect"];
// }

// // Base class for Vector dialect ops.
// class Vector_Op<string mnemonic, list<Trait> traits = []> :
//     Op<Vector_Dialect, mnemonic, traits>;

#[repr(u32)]
#[derive(Debug, Default)]
pub enum CombiningKind {
    #[default]
    Add,
    Mul,
    Minui,
    Minsi,
    Minf,
    Maxui,
    Maxsi,
    Maxf,
    And,
    Or,
    Xor
}

// Helper for verifying combining kinds in contractions and reductions.
pub fn is_supported_combining_kind(
    combining_kind: CombiningKind, element_type: dyn Type
) -> bool {
    use CombiningKind::*;
    match combining_kind {
        Add | Mul =>
            element_type.is_int_or_index_or_float(),
        Minui |
        Minsi |
        Maxui |
        Maxsi |
        And |
        Or |
        Xor =>
            element_type.is_int_or_index(),
        Minf| Maxf =>
            element_type.isa<FloatType>()
    }
}

#[repr(u32)]
pub enum IteratorType {
    Parallel,
    Reduction
}

// pub type IteratorTypeEnum
//     : EnumAttr<Vector_Dialect, Vector_IteratorType, "iterator_type"> {
//     let assembly_format = "`<` $value `>`";
// }

// pub type IteratorTypeArrayAttr
//     : TypedArrayAttrBase<IteratorTypeEnum,
//                         "Iterator type should be an enum.">;

/**
`vector.contract` operation.

Computes the sum of products of vector elements along contracting dimension pairs from 2 vectors of rank M and N respectively, adds this intermediate result to the accumulator argument of rank K, and returns a vector result of rank K (where K = num_lhs_free_dims + num_rhs_free_dims + num_batch_dims (see dimension type descriptions below)). For K = 0 (no free or batch dimensions), the accumulator and output are a scalar.

Optional vector mask arguments (produced by CreateMaskOp or ConstantMaskOp) specify the dynamic dimension sizes of valid data within the lhs/rhs vector arguments.

An iterator type attribute list must be specified, where each element of the list represents an iterator with one of the following types:

- `reduction`: reduction dimensions are present in the lhs and rhs arguments but not in the output (and accumulator argument). These are the dimensions along which the vector contraction op computes the sum of products, and contracting dimension pair dimension sizes must match between lhs/rhs.

- `parallel`: Batch dimensions are iterator type "parallel", and are non-contracting dimensions present in the lhs, rhs and output. The lhs/rhs co-iterate along the batch dimensions, which should be expressed in their indexing maps.

    Free dimensions are iterator type "parallel", and are non-contraction, non-batch dimensions accessed by either the lhs or rhs (but not both). The lhs and rhs free dimensions are unrelated to each other and do not co-iterate, which should be expressed in their indexing maps.

An indexing map attribute list must be specified with an entry for lhs, rhs and acc arguments. An indexing map attribute specifies a mapping from each iterator in the iterator type list, to each dimension of an N-D vector.

An optional kind attribute may be used to specify the combining function between the intermediate result and accumulator argument of rank K. This attribute can take the values add/mul/min/max for int/fp, and/or/xor for int only. The default is "add".

# Examples

Simple DOT product (K = 0):

```mlir 
#contraction_accesses = [
    affine_map<(i) -> (i)>,
    affine_map<(i) -> (i)>,
    affine_map<(i) -> ()>
]
#contraction_trait = {
    indexing_maps = #contraction_accesses,
    iterator_types = ["reduction"]
}
%3 = vector.contract #contraction_trait %0, %1, %2
    : vector<10xf32>, vector<10xf32> into f32
```

2D vector contraction with one contracting dimension (matmul, K = 2):

```mlir
#contraction_accesses = [
    affine_map<(i, j, k) -> (i, k)>,
    affine_map<(i, j, k) -> (k, j)>,
    affine_map<(i, j, k) -> (i, j)>
]
#contraction_trait = {
    indexing_maps = #contraction_accesses,
    iterator_types = ["parallel", "parallel", "reduction"]
}
%3 = vector.contract #contraction_trait %0, %1, %2
    : vector<4x3xf32>, vector<3x7xf32> into vector<4x7xf32>
```

4D to 3D vector contraction with two contracting dimensions and one batch dimension (K = 3):

```mlir
#contraction_accesses = [
    affine_map<(b0, f0, f1, c0, c1) -> (c0, b0, c1, f0)>,
    affine_map<(b0, f0, f1, c0, c1) -> (b0, c1, c0, f1)>,
    affine_map<(b0, f0, f1, c0, c1) -> (b0, f0, f1)>
]
#contraction_trait = {
    indexing_maps = #contraction_accesses,
    iterator_types = ["parallel", "parallel", "parallel",
                      "reduction", "reduction"]
}
%4 = vector.contract #contraction_trait %0, %1, %2
    : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
```

4D vector contraction with two contracting dimensions and optional vector mask arguments:

```mlir
%lhs_mask = vector.constant_mask [7, 8, 16, 15] : vector<7x8x16x15xi1>
%rhs_mask = vector.constant_mask [8, 16, 7, 5] : vector<8x16x7x5xi1>
%5 = vector.contract #contraction_trait %0, %1, %2, %lhs_mask, %rhs_mask
    : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x8x5xf32>
```

Vector contraction with mixed typed. lhs/rhs have different element types than accumulator/result:

```mlir
%6 = vector.contract #contraction_trait %0, %1, %2
    : vector<10xf16>, vector<10xf16> into f32
```

Contracts with max (K = 0):

```mlir
#contraction_accesses = [
    affine_map<(i) -> (i)>,
    affine_map<(i) -> (i)>,
    affine_map<(i) -> ()>
]
#contraction_trait = {
    indexing_maps = #contraction_accesses,
    iterator_types = ["reduction"],
    kind = #vector.kind<max>
}
%7 = vector.contract #contraction_trait %0, %1, %2
    : vector<10xf32>, vector<10xf32> into f32
```
*/
// TODO: Add an attribute to specify a different algebra with operators other than the current set: {*, +}.
#[mlir(
    traits = [Pure]
)]
pub struct Contract<T, U> {
    #[input]
    lhs: Vector<T, >,
    #[input]
    rhs: Vector<T, >,
    #[input]
    acc: U,
    #[input]
    masks: [Vector<bool, _>],
    #[attribute]
    indexing_maps: [AffineMap; 3],
    #[attribute]
    iterator_types: [IteratorType],
    #[attribute(default = CombiningKind::Add)]
    kind: CombiningKindAttr,
    #[output]
    output: U  // renamed from `result`
}

//   let builders = [
//     Builder<(ins Value:$lhs, Value:$rhs, Value:$acc,
//       "ArrayAttribute":$indexingMaps, "ArrayAttribute":$iterator_types)>,
//     Builder<(ins Value:$lhs, Value:$rhs, Value:$acc,
//       "ArrayRef<[&AffineExpr]>":$indexingExprs,
//       "[&IteratorType]":$iterator_types)>,
//     Builder<(ins Value:$lhs, Value:$rhs, Value:$acc,
//       "ArrayAttribute":$indexingMaps, "ArrayAttribute":$iterator_types,
//       CombiningKind:$kind)>
//   ];


impl Contract {
    Vector lhs_vector_mask_type() {
        if (llvm::size(self.masks) != 2) return Vector();
        self.acc
    }
    Vector rhs_vector_mask_type() {
        if (llvm::size(self.masks) != 2) return Vector();
        self.mask
    }

    // Returns the bounds of each dimension in the iteration space spanned
    // by the iterator types of this operation.
    pub fn iteration_bounds(&self, iteration_bounds: SmallVector<i64>) {

    }

    /**
    Returns a list of index maps, where there is a list entry for each
    op indexing map attribute (i.e. one for each input and output, with
    the output listed last). Each index map, maps from this operations
    iteration space, to vector dimensions of the maps input/output.
    */
    pub fn iteration_index_map(
        &self, iteration_index_map: &Vec<DenseMap<i64, i64>>
    ) {

    } 

    Vec<(i64, i64)> contracting_dim_map();
    Vec<(i64, i64)> batch_dim_map();

    SmallVector<IteratorType> iterator_types_array() {
        let range =
            self.iterator_types
                .template get_as_value_range<IteratorTypeAttr, IteratorType>();
        return {range.begin(), range.end()};
    }
}

impl Verify for Contract<L, R, A> {
    fn verify(&self) -> LogicalResult {
        /*
        Verify that each index map has 'num_iterators' inputs, no symbols, and that the number of map outputs equals the rank of its associated vector operand.
        */
        let num_iterators = self.iterator_types.len();
        for (index, map) in self.indexing_maps.enumerate() {
            if map.num_symbols != 0 {
                return emit_op_error(
                    "Expected indexing map {} to have no symbols.", index);
            }
            let vector_type = get_operand(index).dyn_cast<Vector>();
            let rank = if vector_type { vector_type.shape.len() } else { 0 };
            /*
            Verify that the map has the right number of inputs, outputs, and indices.
            This also correctly accounts for (..) -> () for rank-0 results.
            */
            if map.num_dims != num_iterators {
                return emit_op_error(
                    "Expected indexing map {} to have {} number of inputs.",
                    index,
                    num_iterators
                );
            }
            if map.num_outputs != rank {
                return emit_op_error(
                    "Expected indexing map {} to have {} number of outputs.",
                    index,
                    rank
                );
            }
            if !map.is_projected_permutation() {
                return emit_op_error(
                    "Expected indexing map {} to be a projected permutation of its inputs.",
                    index
                );
            }
        }
      
        let contracting_dim_map = contracting_dim_map();
        let batch_dim_map = batch_dim_map();
      
        // Verify at least one contracting dimension pair was specified.
        if contracting_dim_map.is_empty() {
            return emit_op_error("Expected at least one contracting dimension pair.");
        }
      
        // Verify contracting dimension map was properly constructed.
        if !verify_dim_map(self.lhs, self.rhs, contracting_dim_map) {
            return emit_op_error("Invalid contracting dimension map");
        }
      
        // Verify batch dimension map was properly constructed.
        if !verify_dim_map(self.lhs, self.rhs, batch_dim_map) {
            return emit_op_error("Invalid batch dimension map");
        }
      
        // Verify `acc` and `result` shape.
        if verify_output_shape(self, self.lhs, self.rhs, self.acc, self.result,
            contracting_dim_map, batch_dim_map).is_err()
        {
            return Err(());
        }
      
        // Verify that either two vector masks are set or none are set.
        let lhs_mask_type = lhs_vector_mask_type();
        let rhs_mask_type = rhs_vector_mask_type();
        if (lhs_mask_type && !rhs_mask_type) || (!lhs_mask_type && rhs_mask_type) {
              return emit_op_error("Invalid number of vector masks specified");
        }
        if lhs_mask_type && rhs_mask_type {
          // Verify mask rank == argument rank.
            if (lhs_mask_type.shape.len() != self.lhs.shape.len() ||
                rhs_mask_type.shape.len() != self.rhs.shape.len())
            {
                return emit_op_error("Invalid vector mask rank");
            }
        }
      
        // Verify supported combining kind.
        let vector_type = self.result.dyn_cast<Vector>();
        let element_type = if vector_type {
            vector_type.element_type
        } else { self.result };
        if !is_supported_combining_kind(self.kind, element_type) {
            return emit_op_error("Unsupported contraction type.");
        }
      
        Ok(())
    }
}

pub fn verify_dim_map(
    lhs: Vector, rhs: Vector, map: &Vec<(i64, i64)>) -> bool {
    for (l, r) in map.iter() {
        if l < 0
        || l >= lhs.shape.len()
        || r < 0
        || r >= rhs.shape.len()
        || lhs.shape[l] != rhs.shape[r]
        {
            return false;
        }
    }
    true
}

pub fn verify_output_shape<L, R>(
    op: Contraction,
    lhs: Vector,
    rhs: Vector,
    acc: Type,
    output: Type,
    contracting_dim_map: &Vec<(i64, i64)>,
    batch_dim_map: &Vec<(i64, i64)>
) -> LogicalResult {
    DenseSet<i64> lhs_contracting_dim_set;
    DenseSet<i64> rhs_contracting_dim_set;
    for (lhs_dim, rhs_dim) in contracting_dim_map {
        lhs_contracting_dim_set.insert(dim_pair.first);
        rhs_contracting_dim_set.insert(dim_pair.second);
    }
    DenseSet<i64> rhs_batch_dim_set;
    for dim_pair in batch_dim_map {
        rhs_batch_dim_set.insert(dim_pair.second);
    }

    // Add free and batch dimensions from 'lhs' to 'expected_result_dims'.
    SmallVector<[i64; 4]> expected_result_dims;
    for i in 0..L {
        if lhs_contracting_dim_set.count(i) > 0 {
            continue;
        }
        expected_result_dims.push(lhs.dim_size(i));
    }

    // Add free dimensions from 'rhs' to 'expected_result_dims'.
    for i in 0..R {
        if (rhs_contracting_dim_set.count(i) > 0 || rhs_batch_dim_set.count(i) > 0) {
            continue;
        }
        expected_result_dims.push(rhs.dim_size(i));
    }

    // Verify 'expected_result_dims'.
    if expected_result_dims.is_empty() {
        // No batch or free dimension implies a scalar result.
        if output.isa<Vector>() || acc.isa<Vector>() {
            return op.emit_op_error("Invalid accumulator/result vector shape");
        }
    } else {
        // At least one batch or free dimension implies a vector result.
        let output = output.dyn_cast<Vector>();
        let acc = acc.dyn_cast<Vector>();
        if !output || !acc {
            return op.emit_op_error("Invalid accumulator/result vector shape");
        }

        // Infer expected result vector type. Lhs + rhs map and lhs + rhs vector
        // types fully define the result vector type. This assumes the affine maps
        // are well-formed, which must have been verified already.
        let ctx = op.context();
        let lhs_map = op.indexing_maps[0];
        let rhs_map = op.indexing_maps[1];
        if unused_dims_bit_vector([lhs_map, rhs_map]).any() {
            return op.emit_op_error(
            "Expected all dimensions to be either a LHS or a RHS dimension");
        }
        SmallVector<[AffineExpr, 4]> extents(lhs_map.num_inputs());
        for pair in {(lhs, lhs_map), (rhs, rhs_map)} {
            let v = pair.first;
            let map = pair.second;
            for index in 0..v.rank() {
                let pos = map.dim_position(index);
                if !extents[pos] {
                    extents[pos] = get_affine_constant_expr(v.shape[index], ctx);
                }
            }
        }
        if !extents.all(|e| { return e; }) {
            return op.emit_op_error("Expected all dimensions to get an extent as either a LHS or a RHS dimension");
        }

        let res_map = op.indexing_maps[2];
        let extents_map = AffineMap::new(/*dimCount=*/extents.size(),
                                        /*symCount=*/0, extents, ctx);
        // Compose the res_map with the extents_map, which is a constant map.
        let expected_map = simplify_affine_map(res_map.compose(extents_map));
        assert!(expected_map.outputs().all(
                |e| { return e.isa<AffineConstantExpr>(); }),
            "Expected constant extent along all dimensions.");
        // Extract the expected shape and build the type.
        let expected_shape = llvm::to_vector<4>(
            expected_map.outputs().map(|e| {
                return e.cast<AffineConstantExpr>().value();
            }));
        let expected =
            Vector::new(expected_shape, output.element_type);
        if output != expected || acc != expected {
            return op.emit_op_error(
                    "Invalid accumulator/result vector shape, expected: {}", expected);
        }
    }
    Ok(())
}

impl Canonicalise for Contract {
    fn canonicalisation_patterns(
        results: &RewritePatternSet,
        context: *mut MLIRContext
    ) {
        results.add::<
            ExtractOpSplatConstantFolder,
            ExtractOpNonSplatConstantFolder,
            ExtractOpFromBroadcast>(context);
    }
}

impl AssemblyFormat for Contract {
    fn parse(OpAsmParser &parser, OperationState &result) -> ParseResult  {
        UnresolvedOperand lhs_info;
        UnresolvedOperand rhs_info;
        UnresolvedOperand accInfo;
        SmallVector<UnresolvedOperand, 2> masks_info;
        SmallVector<Type, 2> types;
        Type result_type;
        auto loc = parser.getCurrentLocation();
        DictionaryAttr dict_attr;
        // TODO: Unify linalg op attribute parsing.
        if (parser.parse_attribute(dict_attr, "_", result.attributes) ||
            parser.parse_operand(lhs_info) || parser.parse_comma() ||
            parser.parse_operand(rhs_info) || parser.parse_comma() ||
            parser.parse_operand(accInfo) ||
            parser.parse_trailing_operand_list(masks_info) ||
            parser.parse_optional_attr_dict(result.attributes) ||
            parser.parse_colon_type_list(types) ||
            parser.parse_keyword_type("into", result_type) ||
            parser.resolve_operand(lhs_info, types[0], result.operands) ||
            parser.resolve_operand(rhs_info, types[1], result.operands) ||
            parser.resolve_operand(accInfo, result_type, result.operands) ||
            parser.add_type_to_list(result_type, result.types))
          return Err(());
        result.attributes.assign(dict_attr.value().begin(),
                                 dict_attr.value().end());
      
        /*
        Convert array of string into an array of IteratyType enums. This is needed,
        because tests still use the old format when 'iterator_types' attribute is
        represented as an array of strings.
        TODO: Remove this conversion once tests are fixed.
        */
        ArrayAttr iterator_types =
            result.attributes.get(get_iterator_types_attr_name(result.name))
                .cast<ArrayAttr>();
      
        SmallVector<Attribute> iterator_type_attrs;
      
        for s in iterator_types.as_value_range<StringAttr>() {
            let maybe_iterator_type = symbolise_iterator_type(s);
            if !maybe_iterator_type.has_value() {
                return parser.emit_error(loc) << "unexpected iterator_type (" << s << ")";
            }
        
            iterator_type_attrs.push(
                IteratorTypeAttr::new(parser.context(), maybe_iterator_type.value()));
        }
        result.attributes.set(get_iterator_types_attr_name(result.name),
                              parser.builder().getArrayAttr(iterator_type_attrs));
      
        if (!result.attributes.get(get_kind_attr_name(result.name))) {
          result.addAttribute(
              get_kind_attr_name(result.name),
              CombiningKindAttr::get(result.context(),
                                     Contraction::get_default_kind()));
        }
        if (masks_info.empty()) {
          return Ok(());}
        if (masks_info.size() != 2){
          return parser.emit_error(parser.get_name_loc(),
                                  "Expected zero or exactly 2 vector mask operands");}
        let lhs = types[0].cast<Vector>();
        let rhs = types[1].cast<Vector>();
        let mask_element_type = parser.builder().get_i1_type();
        [Type; 2] mask_types = {
            Vector::Builder(lhs).set_element_type(mask_element_type),
            Vector::Builder(rhs).set_element_type(mask_element_type)};
        if (parser.resolve_operands(masks_info, mask_types, loc, result.operands)){
          return Err(());}
        return Ok(());
    }
      
    fn print(&self, p: &OpAsmPrinter) {
        // TODO: Unify printing code with linalg ops.
        let attrNames = getTraitAttrNames();
        llvm::StringSet<> traitAttrsSet;
        traitAttrsSet.insert(attrNames.begin(), attrNames.end());
        SmallVector<NamedAttribute, 8> attrs;
        for (let attr : (*this)->get_attrs()) {
          if (attr.getName() == get_iterator_types_attr_name()) {
            let iterator_types =
                attr.value()
                    .cast<ArrayAttr>()
                    .as_value_range<IteratorTypeAttr, IteratorType>();
            // Convert IteratorType enums into the string representation. This is
            // needed, because tests still use the old format when 'iterator_types'
            // attribute is represented as an array of strings.
            // TODO: Remove this conversion once tests are fixed.
            SmallVector<Attribute> iteratorTypeNames = llvm::to_vector(
                llvm::map_range(iterator_types, [&](IteratorType t) -> Attribute {
                  return StringAttr::get(context(), stringifyIteratorType(t));
                }));
      
            attrs.emplace_back(get_iterator_types_attr_name(),
                               ArrayAttr::get(context(), iteratorTypeNames));
          } else if (traitAttrsSet.count(attr.getName().strref()) > 0)
            attrs.push(attr);
        }
      
        auto dict_attr = DictionaryAttr::get(context(), attrs);
        p << " " << dict_attr << " " << get_lhs() << ", ";
        p << get_rhs() << ", " << get_acc();is_
        if (getMasks().size() == 2)
          p << ", " << getMasks();
      
        p.printOptionalAttrDict((*this)->get_attrs(), attrNames);
        p << " : " << get_lhs().getType() << ", " << get_rhs().getType() << " into "
          << get_result_type();
    }
}

impl VectorUnrollOpInterface for Contract {
    fn shape_for_unroll(&self) -> Option<SmallVector<[i64; 4]>> {
        let mut shape = SmallVector<[i64; 4]>::new();
        self.iteration_bounds(shape);
        shape
    }
}

/**
`vector.reduce` operation.

Reduces an 1-D vector 'horizontally' into a scalar using the given operation (add/mul/min/max for int/fp and and/or/xor for int only).
Reductions also allow an optional fused accumulator.

Note that these operations are restricted to 1-D vectors to remain close to the corresponding LLVM intrinsics: <http://llvm.org/docs/LangRef.html#vector-reduction-intrinsics>.

# Examples

```mlir
%1 = vector.reduce <add>, %0 : vector<16xf32> into f32
```

```mlir
%3 = vector.reduce <xor>, %2 : vector<4xi32> into i32
```

```mlir
%4 = vector.reduce <mul>, %0, %1 : vector<16xf32> into f32
```
*/
#[mlir(
    traits = [Pure]
)]
pub struct Reduce<T, const I: usize> {
    #[attribute]
    kind: CombiningKind,
    #[input]
    input: Vector<T, I>,  // renamed from `vector`
    #[input]
    acc: Option<_>,
    #[output]
    output: T  // renamed from `dest`
}
//   let builders = [
//     // Builder that infers the type of `dest`.
//     Builder<(ins CombiningKind:$kind, Value:$vector, Value:$acc)>,
//     // Builder that infers the type of `dest` and has no accumulator.
//     Builder<(ins CombiningKind:$kind, Value:$vector)>
//   ];

impl Verify for Reduce {
    fn verify(&self) -> LogicalResult {
        // Verify for 0-D and 1-D vector.
        if I > 1 {
            return emit_op_error("Unsupported reduction rank: {}", rank);
        }
        // Verify supported reduction kind.
        if !is_supported_combining_kind(self.kind, self.output) {
            return emit_op_error(
                "Unsupported reduction type '{}' for kind '{:?}'.",
                self.output,
                self.kind
            );
        }

        Ok(())
    }
}

impl Canonicalise for Reduce {
    fn canonicalisation_patterns() {
        results.add<ElideSingleElementReduction>(context)
    }
}

// TODO: Migrate to assembly_format once `AllTypesMatch` supports optional
// operands.
impl AssemblyFormat for Reduce {
    fn parse(parser: &OpAsmParser, result: &OperationState) -> ParseResult {
        let mut operands_info = SmallVector::<[UnresolvedOperand; 2]>::new();
        let red_type = Type;
        let output = Type;
        let kind_attr = CombiningKindAttr;
        if parser.parse_custom_attribute_with_fallback(
            kind_attr, Type{}, "kind", result.attributes)
        || parser.parse_comma()
        || parser.parse_operand_list(operands_info)
        || parser.parse_colon_type(red_type)
        || parser.parse_keyword_type("into", output)
        || (!operands_info.is_empty()
            && parser.resolve_operand(
                operands_info[0], red_type, result.operands))
        || (operands_info.len() > 1
            && parser.resolve_operand(
                operands_info[1], output, result.operands))
        || parser.add_type_to_list(output, result.types)
        {
            return Err(());
        }
        if operands_info.is_empty() || operands_info.len() > 2 {
            return parser.emit_error(parser.get_name_loc(),
                            "Unsupported number of operands");
        }
        Ok(())
    }

    fn print(&self, p: &OpAsmPrinter) {
        p << (
            " {:?}, {}",
            self.kind,
            self.input,
        );
        if (self.acc) {
          p << (", {}", self.acc.unwrap());
        }
        (" : {} into {}", self.input, self.output);
      }
}

impl MaskableOpInterface for Reduce {
    fn expected_mask_type(&self) -> Type {
        self.input.clone_with(
            None, IntegerType::new(self.input.context(), 1));
    }
}

impl VectorUnrollOpInterface for Reduce {
    fn shape_for_unroll(&self) -> Option<SmallVector<[i64; 4]>> {
        llvm::to_vector<4>(self.input.shape)
    }
}

/**
`vector.multi_reduce` Multi-dimensional reduction operation.

Reduces an n-D vector into an (n-k)-D vector (or a scalar when k == n) using the given operation (add/mul/min/max for int/fp and and/or/xor for int only).
Takes an initial accumulator operand.

# Examples

```mlir
%1 = vector.multi_reduce <add>, %0, %acc0 [1, 3]
    : vector<4x8x16x32xf32> into vector<4x16xf32>
```

```mlir
%2 = vector.multi_reduce <add>, %1, %acc1 [0, 1]
    : vector<4x16xf32> into f32
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$kind `,` $source `,` $acc attr-dict $reduction_dims `:` type($source) `to` type($output)"
)]
pub struct MultiDimReduce<
    T, const I: usize, const O: usize
> {
    #[attribute]
    kind: CombiningKindAttr,
    #[input]
    input: Vector<T, I>,  // renamed from `source`
    #[input]
    acc: T,
    #[attribute]
    reduction_dims: [u64; _],
    #[output]
    output: Vector<T, O>  // renamed from `dest`
//   let builders = [
//     Builder<(ins Value:$source, Value:$acc,
//                    "&[bool]":$reduction_mask, CombiningKind:$kind)>
//   ];
}

impl MultiDimReduce {
    pub fn is_reduced_dim(&self, d: i64) -> bool {
        assert!(
            d >= 0
            && d < static_cast<i64>(self.reduction_mask().size()),
            "d overflows the number of dims");
        self.reduction_mask()[d]
    }

    pub fn reduction_mask(&self) -> SmallVector<[bool]> {
        let mut output = SmallVector::<[bool]>::new(self.source.rank(), false);
        for dim in self.reduction_dims {
            output[dim] = true;
        }
        output
    }

    pub fn reduction_mask(reduction_dims: &[u64], source_rank: usize)
    -> SmallVector<[bool]> {
        let mut output = SmallVector<[bool]>::new(source_rank, false);
        for dim in reduction_dims {
            output[dim] = true;
        }
        output
    }
}

impl Verify for MultiDimReduce {
    fn verify(&self) -> LogicalResult {
        let mut target_shape = SmallVector::<i64>::new();
        let inferred_return_type: Type;
        for (index, value) in self.input.shape.enumerate() {
            if !self.reduction_dims.any(|dim| dim == index) {
                target_shape.push(value);
            }
        }
        // TODO: update to also allow 0-d vectors when available.
        if target_shape.is_empty() {
            inferred_return_type = self.input.element_type;
        } else {
            inferred_return_type =
              Vector::new(target_shape, self.input.element_type);
        }
        if self.output != inferred_return_type {
            return emit_op_error(
                "Output type {} is incompatible with input type {}.",
                self.output,
                self.input
            );
        }

        Ok(())
    }
}

impl Fold for MultiDimReduce<T, I, O> {
    fn fold(&self) -> FoldResult {
        // Single parallel dim, this is a noop.
        if I == 1 && !self.is_reduced_dim(0) {
            return self.source;
        }
        return {};
    }
}

impl Canonicalise for MultiDimReduce {
    fn canonicalisation_patterns(
        RewritePatternSet &results,
        MLIRContext *context
    ) {
        results.add<ElideUnitDimsInMultiDimReduction>(context);
    }
    
}

impl InferTypeOpInterface for MultiDimReduce {
    
}

impl VectorUnrollOpInterface for MultiDimReduce {
    fn shape_for_unroll(&self) -> Option<SmallVector<[i64; 4]>> {
        llvm::to_vector<4>(self.input.shape)
    }
}

/**
`vector.broadcast` operation.

Broadcasts the scalar or k-D vector value in the source operand to a n-D result vector such that the broadcast makes sense, i.e., the source operand is duplicated to match the given rank and sizes in the result vector. The legality rules are:

- the source operand must have the same element type as the result type
- a k-D vector <s_1 x .. x s_k x type> can be broadcast to a n-D vector <t_1 x .. x t_n x type> if
    - k <= n, and
    - the sizes in the trailing dimensions n-k < i <= n with j=i+k-n
        match exactly as s_j = t_i or s_j = 1:
    ```text
        t_1 x   ..  t_n-k x t_n-k+1 x .. x t_i x .. x t_n
                            s_1     x .. x s_j x .. x s_k
            <duplication>         <potential stretch>
    ```

The source operand is duplicated over all the missing leading dimensions and stretched over the trailing dimensions where the source has a non-equal dimension of 1. These rules imply that any scalar broadcast (k=0) to any shaped vector with the same element type is always legal.

# Example

```mlir
%0 = arith.constant 0.0 : f32
%1 = vector.broadcast %0 : f32 to vector<16xf32>
%2 = vector.broadcast %1 : vector<16xf32> to vector<4x16xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$input attr-dict `:` type($input) `to` type($output)"
)]
pub struct Broadcast<T, const O: usize> {
    #[input]
    input: T,  // renamed from `source`
    #[output]
    output: Vector<T, O>  // renamed from `vector`
}

impl Broadcast {
    /**
    Return the dimensions of the result vector that were formerly ones in the source tensor and thus correspond to "dim-1" broadcasting.
    */
    pub fn compute_broadcasted_unit_dims(&self) -> SetVector<i64> {

    }

    /**
    Broadcast `value` to a vector of `dst_shape`, knowing that exactly the `broadcasted_dims` dimensions in the dst_shape are broadcasted.
    This requires (and asserts) that the broadcast is free of dim-1 broadcasting.
    Since vector.broadcast only allows expanding leading dimensions, an extra vector.transpose may be inserted to make the broadcast possible. `value`, `dst_shape` and `broadcasted_dims` must be properly specified or the helper will assert. This means:

    1. `dst_shape` must not be empty.
    2. `broadcasted_dims` must be confined to [0 .. rank(value.get_vector_type)]
    2. `dst_shape` trimmed of the dimensions specified in `broadcasted_dims`
    */
    //       must match the `value` shape.
    pub fn create_or_fold_broadcast_op(
        builder: &Builder,
        value: Value,
        dst_shape: &[u64],
        broadcasted_dims: &SetVector<i64>
    ) -> Value {

    }
}

impl Verify for Broadcast {
    fn verify(&self) -> LogicalResult {
        let mismatching_dims: (int, int);
        match is_broadcastable_to(self.input, self.output, &mismatching_dims) {
            BroadcastableToResult::Success => return Ok(()),
            BroadcastableToResult::SourceRankHigher
                => return emit_op_error("Source rank higher than destination rank"),
            BroadcastableToResult::DimensionMismatch
                => return emit_op_error(
                    "Dimension mismatch ({} vs. {}).",
                    mismatching_dims.0,
                    mismatching_dims.1
                ),
            BroadcastableToResult::SourceTypeNotAVector
                => return emit_op_error("Source type is not a vector")
        }
    }
}

impl Fold for Broadcast {
    fn fold(&self) -> FoldResult {
        if self.input == self.output {
            return self.input;
        }
        if !self.input {
            return {};
        }
        if self.input.isa<IntegerAttribute, FloatAttr>() {
            return DenseElementsAttribute::new(self.output, self.input);
        }
        if let attr = self.input.dyn_cast<SplatElementsAttribute>() {
            return DenseElementsAttribute::new(self.output, attr.get_splat_value<Attribute>());
        }
        return {};
    }
}

impl Canonicalise for Broadcast {
    fn canonicalisation_patterns(
        results: &RewritePatternSet, context: *mut MLIRContext
    ) {
        /*
        BroadcastToShapeCast is not a default canonicalisation, it is opt-in by calling `populateCastAwayVectorLeadingOneDimPatterns`
        */
        results.add::<BroadcastFolder>(context);
    }
}

/**
`vector.shuffle` operation.

The shuffle operation constructs a permutation (or duplication) of elements from two input vectors, returning a vector with the same element type as the input and a length that is the same as the shuffle mask. The two input vectors must have the same element type, same rank , and trailing dimension sizes and shuffles their values in the leading dimension (which may differ in size) according to the given mask.
The legality rules are:

- the two operands must have the same element type as the result
    - Either, the two operands and the result must have the same rank and trailing dimension sizes, viz. given two k-D operands
            lhs : <s_1 x s_2 x .. x s_k x type> and
            rhs : <t_1 x t_2 x .. x t_k x type>
    we have s_i = t_i for all 1 < i <= k
    - Or, the two operands must be 0-D vectors and the result is a 1-D vector.
- the mask length equals the leading dimension size of the result
- numbering the input vector indices left to right across the operands, all mask values must be within range, viz. given two k-D operands lhs and rhs above, all mask values are in the range [0, s_1 + t_1)

# Example

```mlir
%0 = vector.shuffle %a, %b[0, 3]
    : vector<2xf32>, vector<2xf32>       ; yields vector<2xf32>
%1 = vector.shuffle %c, %b[0, 1, 2]
    : vector<2x16xf32>, vector<1x16xf32> ; yields vector<3x16xf32>
%2 = vector.shuffle %a, %b[3, 2, 1, 0]
    : vector<2xf32>, vector<2xf32>       ; yields vector<4xf32>
%3 = vector.shuffle %a, %b[0, 1]
    : vector<f32>, vector<f32>           ; yields vector<2xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "operands $mask attr-dict `:` type(operands)"
)]
pub struct Shuffle<
    T, const L: usize, const R: usize, const M: usize, const O: usize
> {
    #[input]
    lhs: Vector<T, L>,  // renamed from `v1`
    #[input]
    rhs: Vector<T, R>,  // renamed from `v2`
    #[attribute]
    mask: [u64; M],
    #[output]
    output: Vector<T, O>  // renamed from `vector`
}

//   let builders = [
//     Builder<(ins Value:$lhs, Value:$rhs, "[&i64]")>
//   ];

impl Verify for Shuffle {
    fn verify(&self) -> LogicalResult {
        // Verify ranks.
        let well_formed_0d_case = L == 0 && R == 0 && O == 1;
        let well_formed_nd_case = L == O && R == O;
        if !well_formed_0d_case && !well_formed_nd_case {
            return emit_op_error("Rank mismatch.");
        }
      
        // Verify all but leading dimension sizes.
        for r in 1..L {
            let res_dim = self.output.dim_size(r);
            let lhs_dim = self.lhs.dim_size(r);
            let rhs_dim = self.rhs.dim_size(r);
            if res_dim != lhs_dim || lhs_dim != rhs_dim {
                return emit_op_error("Dimension mismatch.");
            }
        }
        // Verify mask length.
        if M <= 0 {
            return emit_op_error("Invalid mask length");
        }
        if M != self.output.dim_size(0) {
            return emit_op_error("Mask length mismatch");
        }
        // Verify all indices.
        let index_size =
            if L == 0 { 1 } else { self.lhs.dim_size(0) }
            + if R == 0 { 1 } else { self.rhs.dim_size(0) };
        for (index, value) in self.mask.enumerate() {
            if value >= index_size {
                return emit_op_error(
                    "Mask index #{} out of range.",
                    index + 1
                );
            }
        }
        Ok(())
    }
}

impl Fold for Shuffle {
    fn fold(&self) -> FoldResult {
        /*
        For consistency: 0-D shuffle return type is 1-D, this cannot be a folding but must be a canonicalisation into a vector.broadcast.
        */
        if L == 0 {
              return {};
        }
      
        // fold shuffle V1, V2, [0, 1, 2, 3] : <4xi32>, <2xi32> -> V1
        if !self.lhs.is_scalable()
        && is_step_index_array(self.mask, 0, self.lhs.dim_size(0))
        {
              return self.lhs;
        }
        // fold shuffle V1, V2, [4, 5] : <4xi32>, <2xi32> -> V2
        if !self.lhs.is_scalable()
        && !self.rhs.is_scalable()
        && is_step_index_array(self.mask, self.lhs.dim_size(0),
                             self.rhs.dim_size(0))
        {
            return self.rhs;
        }
      
        if !self.lhs || !self.rhs {
            return {};
        }
      
        let lhs = self.lhs.cast<DenseElementsAttribute>().cast<Vector>();
        /*
        Only support 1-D for now to avoid complicated n-D DenseElementsAttribute
        manipulation.
        */
        if L != 1 {
              return {};
        }
        let lhs_size = lhs.dim_size(0);
      
        let mut results = SmallVector::<[Attribute]>::new();
        let lhs_elements = lhs.cast<DenseElementsAttribute>().values::<Attribute>();
        let rhs_elements = self.rhs.cast<DenseElementsAttribute>().values::<Attribute>();
        for index in self.mask.get_as_value_range<IntegerAttribute>() {
            let i = index.get_z_ext_value();
            if i >= lhs_size {
                results.push(rhs_elements[i - lhs_size]);
            } else {
                results.push(lhs_elements[i]);
            }
        }
      
        DenseElementsAttribute::new(get_vector_type(), results)
    }
}

impl Canonicalise for Shuffle {
    fn canonicalisation_patterns(
        results: &RewritePatternSet, context: *mut MLIRContext
    ) {
        results.add<ShuffleSplat, Canonicalize0DShuffle>(context);
    }
}

impl InferTypeOpInterface for Shuffle<T, L, R, M, O> {
    fn infer_return_types(
        &self,
        _: *mut MLIRContext,
        _: Option<Location>,
        inferred_return_types: &mut SmallVector<Type>
    ) -> LogicalResult {
        // Construct resulting type: leading dimension matches mask
        // length, all trailing dimensions match the operands.
        let mut shape = SmallVector<[i64; 4]>::new();
        shape.reserve(L);
        shape.push(1.max(self.mask.len()));
        // In the 0-D case there is no trailing shape to append.
        if L > 0 {
            append_range(shape, self.lhs.shape.drop_front());
        }
        inferred_return_types.push(
            Vector::new(shape, self.lhs.element_type));
        Ok(())
    }
}

/**
`vector.extract_element` operation.

Takes a 0-D or 1-D vector and a optional dynamic index position and extracts the scalar at that position.

Note that this instruction resembles vector.extract, but is restricted to 0-D and 1-D vectors and relaxed to dynamic indices.
If the vector is 0-D, the position must be std::nullopt.

It is meant to be closer to LLVM's version: <https://llvm.org/docs/LangRef.html#extract_element-instruction>

# Example

```mlir
%c = arith.constant 15 : i32
%1 = vector.extract_element %0[%c : i32]: vector<16xf32>
%2 = vector.extract_element %z[]: vector<f32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$vector `[` ($index^ `:` type($index))? `]` attr-dict `:` type($vector)"
)]
pub struct ExtractElement<T, const N: usize> {
    #[input]
    vector: Vector<T, N>,
    #[input]
    index: Option<AnySignlessIntegerOrIndex>,
    #[output]
    output: T  // renamed from `result`
}

//   let builders = [
//     // 0-D builder.
//     Builder<(ins Value:$source)>,
//     // 1-D + position builder.
//     Builder<(ins Value:$source, Value:$index)>,
//   ];

impl Verfy for ExtractElement {
    fn verify(&self) -> LogicalResult {
        if N == 0 {
            if self.index.is_some() {
                return emit_op_error(
                    "Expected index to be empty with 0-D vector");
            }
            return Ok(());
        }
        if N != 1 {
            return emit_op_error("Unexpected >1 vector rank");
        }
        if self.index.is_none() {
            return emit_op_error("Expected index for 1-D vector");
        }
        Ok(())
    }
}

impl Fold for ExtractElement {
    fn fold(&self) -> FoldResult {
        // Skip the 0-D vector here now.
        let index = match self.index {
            None => return {},
            Some(index) => index
        };
      
        // Fold extract_element (splat X) -> X.
        if let splat = self.vector.defining_op<Splat>() {
            return splat.get_input();
        }
      
        // Fold extract_element(broadcast(X)) -> X.
        if let broadcast = self.vector.defining_op<Broadcast>() {
            if !broadcast.source.isa<Vector>() {
                return broadcast.source;
            }
        }
      
        if !index || !self.vector {
              return {};
        }
      
        let src_elements = self.vector.cast<DenseElementsAttribute>().values::<Attribute>();
      
        let attr = index.dyn_cast<IntegerAttribute>();
        let pos_idx = attr.get_int();
      
        src_elements[pos_idx]
    }
}

/**
`vector.extract` operation.

Takes an n-D vector and a k-D position and extracts the (n-k)-D vector at the proper position. Degenerates to an element type in the 0-D case.

# Example

```mlir
%1 = vector.extract %0[3] : vector<4x8x16xf32>
%2 = vector.extract %0[3, 3, 3] : vector<4x8x16xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$vector `` $indices attr-dict `:` type($vector)"
)]
pub struct Extract<T, const N: usize, const M: usize> {
    #[input]
    input: Vector<T, N>, // renamed from `vector`
    #[attribute]
    indices: [u64; M],
    #[output]
    output: T  // renamed from `result`
}

//   let builders = [
//     Builder<(ins Value:$source, "&[u64]":$indices)>,
//     // Convenience builder which assumes the values in `indices` are defined by
//     // ConstantIndex.
//     Builder<(ins Value:$source, ValueRange:$indices)>
//   ];

impl Extract {
    fn is_compatible_return_types(TypeRange l, TypeRange r) -> bool {

    }
}

impl Verify for Extract {
    fn verify(&self) -> LogicalResult {
        if M > N {
            return emit_op_error(
                "Expected indices of rank smaller than vector rank");
        }
        for (index, value) in self.indices.enumerate() {
            if value >= self.vector.dim_size(index)
            {
                return emit_op_error(
    "Expected indices #{} to be smaller than the corresponding vector dimension.",
                    index + 1
                );
            }
        }
        Ok(())
    }
}

impl Fold for Extract {
    fn fold(&self) -> FoldResult {
        if self.indices.is_empty() {
            return self.vector;
        }
        if fold_extract_op_from_extract_chain(self).is_ok() {
            return self.output;
        }
        if let res = ExtractFromInsertTransposeChainState(self).fold() {
            return res;
        }
        if let res = fold_extract_from_broadcast(self) {
            return res;
        }
        if let res = fold_extract_from_shape_cast(self) {
            return res;
        }
        if let val = fold_extract_from_extract_strided(self) {
            return val;
        }
        if let val = fold_extract_strided_op_from_insert_chain(self) {
            return val;
        }
        FoldResult::new()
    }
}

impl Canonicalis for Extract {

}

impl InferTypeOpInterface for Extract<T, N, M> {
    fn infer_return_types(
        &self,
        context: *mut MLIRContext,
        location: Option<Location>,
        inferred_return_types: &mut SmallVector<[Type]>
    ) -> LogicalResult {
        if self.index.len() == N {
            inferred_return_types.push(self.input.element_type);
        } else {
          let n = self.index.len().min(N - 1);
          inferred_return_types.push(
            Vector::new(
                self.input.shape.drop_front(n), self.input.element_type));
        }
        Ok(())
    }
}

/**
`vector.fma` Vector fused multiply-add.

Multiply-add expressions operate on n-D vectors and compute a fused pointwise multiply-and-accumulate: `$result = `$lhs * $rhs + $acc`.
All operands and result have the same vector type. The semantics of the operation correspond to those of the `llvm.fma` [intrinsic](https://llvm.org/docs/LangRef.html#int-fma). In the particular case of lowering to LLVM, this is guaranteed to lower to the `llvm.fma.*` intrinsic.

# Example

```mlir
%3 = vector.fma %0, %1, %2: vector<8x16xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs)"
)]
pub struct FMA<T, const N: usize> {
    #[input]
    lhs: Vector<T, N>,
    #[input]
    rhs: Vector<T, N>,
    #[input]
    acc: Vector<T, N>,
    #[output]
    output: Vector<T, N>  // renamed from `result`
}

impl VectorUnrollOpInterface for FMA {
    fn shape_for_unroll(&self) -> Option<SmallVector<[i64; 4]>> {
        to_vector<4>(self.input.shape)
    }
}

// Op<Vector_Dialect, "fma",  # ElementwiseMappable.traits>,

/**
Insertelement operation.

Takes a scalar source, a 0-D or 1-D destination vector and a dynamic index position and inserts the source into the destination at the proper position.

Note that this instruction resembles vector.insert, but is restricted to 0-D and 1-D vectors and relaxed to dynamic indices.

It is meant to be closer to LLVM's version:
https://llvm.org/docs/LangRef.html#insertelement-instruction

# Example

```mlir
%c = arith.constant 15 : i32
%f = arith.constant 0.0f : f32
%1 = vector.insert_element %f, %0[%c : i32]: vector<16xf32>
%2 = vector.insert_element %f, %z[]: vector<f32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$source `,` $dest `[` ($index^ `:` type($index))? `]`  attr-dict `:` type($output)"
)]
pub struct InsertElement<T, const N: usize> {
    #[input]
    source: T,
    #[input]
    dest: Vector<T, N>,
    #[input]
    index: Option<AnySignlessIntegerOrIndex>,
    #[output]
    output: Vector<T, >  // renamed from `result`
}

//   let builders = [
//     // 0-D builder.
//     Builder<(ins Value:$source, Value:$dest)>,
//   ];

impl Verify for InsertElement {
    fn verify(&self) -> LogicalResult {
        if N == 0 {
            if self.index.is_some() {
                return emit_op_error(
                    "Expected index to be empty with 0-D vector");
            }
            return Ok(());
        }
        if N != 1 {
            return emit_op_error("Unexpected >1 vector rank");
        }
        if self.index.is_none() {
            return emit_op_error("Expected index for 1-D vector");
        }
        Ok(())
    }
}

impl Fold for InsertElement {
    fn fold(&self, operands: [&impl Attribute]) -> FoldResult {
        let index = match self.index {
            // Skip the 0-D vector here.
            None => return {},
            Some(index) => index
        };
        
        if !source || !dest || !index {
            return {};
        }
      
        let dst_elements = self.dest.cast<DenseElementsAttribute>().values::<Attribute>();
      
        SmallVector<[Attribute]> results(dst_elements);
      
        let attr = index.dyn_cast<IntegerAttribute>();
        let pos_idx = attr.get_int();
      
        results[pos_idx] = self.source;
      
        DenseElementsAttribute::new(self.dest, results)
    }
}

/**
`vector.insert` operation.

Takes an n-D source vector, an (n + k)-D destination vector and a k-D indices and inserts the n-D source into the (n + k)-D destination at the proper indices. Degenerates to a scalar source type when n = 0.

# Examples

```mlir
%2 = vector.insert %0, %1[3] : vector<8x16xf32> into vector<4x8x16xf32>
```

```mlir
%5 = vector.insert %3, %4[3, 3, 3] : f32 into vector<4x8x16xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$source `,` $dest $indices attr-dict `:` type($source) `into` type($dest)"
)]
pub struct Insert<T, const N: usize, const O: usize> {
    #[input]
    source: T,
    #[input]
    dest: Vector<T, _>,
    #[attribute]
    indices: [u64; N],
    #[output]
    output: Vector<T, O>  // renamed from `result`
}

//   let builders = [
//     Builder<(ins Value:$source, Value:$dest,
//       "&[u64]":$indices)>,
//     // Convenience builder which assumes all values are constant indices.
//     Builder<(ins Value:$source, Value:$dest, ValueRange:$indices)>
//   ];

impl Verify for Insert {
    fn verify(&self) -> LogicalResult {
        if N > O {
            return emit_op_error(
                "Expected indices of rank smaller than dest vector rank");
        }
        let src_vector_type = self.source.dyn_cast<Vector>();
        if src_vector_type
        && src_vector_type.rank() + N != O
        {
            return emit_op_error(
"Expected indices rank + source rank to match dest vector rank");
        }
        if !src_vector_type
        && N != O
        {
          return emit_op_error(
              "Expected indices rank to match the dest vector rank");
        }
        for (index, value) in self.indices.enumerate() {
            if value >= self.dest.dim_size(index)
            {
                return emit_op_error(
"Expected indices #{} to be smaller than the corresponding dest vector dimension.", index + 1);
            }
        }
        Ok(())
    }
}

impl Fold for Insert {
    /**
    Eliminates insert operations that produce values identical to their source value. This happens when the source and destination vectors have identical sizes.
    */
    fn fold(&self) -> FoldResult {
        if self.indices.is_empty() {
            return input;  // TODO ?
        }
        return {};
    }
}

impl Canonicalise for Insert {
    fn canonicalisation_patterns(
        results: &RewritePatternSet,
        context: *mut MLIRContext) {
        results.add::<
            InsertToBroadcast, BroadcastFolder, InsertSplatToSplat,
            InsertOpConstantFolder>(context);
    }
}

/**
Insert subvector into scalable vector operation.

NOTE: This operation is designed to map to `llvm.vector.insert`, and its documentation should be kept aligned with LLVM IR: <https://llvm.org/docs/LangRef.html#llvm-vector-insert-intrinsic>

This operations takes a rank-1 fixed-length or scalable subvector and inserts it within the destination scalable vector starting from the position specificed by `index`. If the source vector is scalable, the insertion position will be scaled by the runtime scaling factor of the source subvector.

The insertion position must be a multiple of the minimum size of the source vector. For the operation to be well defined, the source vector must fit in the destination vector from the specified position. Since the destination vector is scalable and its runtime length is unknown, the validity of the operation can't be verified nor guaranteed at compile time.

# Examples

```mlir
%2 = vector.scalable.insert %0, %1[8] : vector<4xf32> into vector<[16]xf32>
```

```mlir
%5 = vector.scalable.insert %3, %4[0] : vector<8xf32> into vector<[4]xf32>
```

```mlir
%8 = vector.scalable.insert %6, %7[0] : vector<[4]xf32> into vector<[8]xf32>
```

# Invalid example

```mlir
%2 = vector.scalable.insert %0, %1[5] : vector<4xf32> into vector<[16]xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$source `,` $dest `[` $index `]` attr-dict `:` type($source) `into` type($dest)"
)]
pub struct ScalableInsert<T> {
    #[input]
    source: Vector<T, 1>,
    #[input("ScalableVectorOfRank<[1]>")]
    dest: Vector<T, 1>,
    #[attribute(|index| index % self.source.num_elements == 0)]
    index: u64,
    #[output("ScalableVectorOfRank<[1]>")]
    output: Vector<T, 1>  // renamed from `result`
}

/**
Extract subvector from scalable vector operation.

Takes rank-1 source vector and a position `index` within the source vector, and extracts a subvector starting from that position.

The extraction position must be a multiple of the minimum size of the result vector. For the operation to be well defined, the destination vector must fit within the source vector from the specified position. Since the source vector is scalable and its runtime length is unknown, the validity of the operation can't be verified nor guaranteed at compile time.

# Examples

```mlir
%1 = vector.scalable.extract %0[8] : vector<4xf32> from vector<[8]xf32>
```

```mlir
%3 = vector.scalable.extract %2[0] : vector<[4]xf32> from vector<[8]xf32>
```

# Invalid Example

```mlir
%1 = vector.scalable.extract %0[5] : vector<4xf32> from vector<[16]xf32>
```

NOTE: This operation is designed to map to `llvm.vector.extract`, and its documentation should be kept aligned with LLVM IR: <https://llvm.org/docs/LangRef.html#llvm-vector-extract-intrinsic>.
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$source `[` $index `]` attr-dict `:` type($res) `from` type($source)"
)]
pub struct ScalableExtract<T> {
    #[input]
    source: Vector<T, 1>,  // ScalableVectorOfRank<[1]>,
    #[attribute]
    index: u64,  // |index| index % output.num_element == 0
    #[output]
    output: Vector<T, 1>  // renamed from `result`
}

/**
`vector.insert_strided_slice` operation.

Takes a k-D source vector, an n-D destination vector (n >= k), n-sized `offsets` integer array attribute, a k-sized `strides` integer array attribute and inserts the k-D source vector as a strided subvector at the proper offset into the n-D destination vector.

At the moment strides must contain only 1s.

Returns an n-D vector that is a copy of the n-D destination vector in which the last k-D dimensions contain the k-D source vector elements strided at the proper location as specified by the offsets.

# Example

```mlir
%2 = vector.insert_strided_slice %0, %1
    { offsets = [0, 0, 2], strides = [1, 1] }
    : vector<2x4xf32> into vector<16x4x8xf32>
```
*/
#[mlir(
    traits = [Pure]
    assembly_format = "$source `,` $dest attr-dict `:` type($source) `into` type($dest)"
)]
pub struct InsertStridedSlice<
    T, const N: usize, const M: usize
> {
    #[input]
    source: Vector<T, N>,
    #[input]
    dest: Vector<T, M>,
    #[attribute]
    offsets: [u64; M],
    #[attribute]
    strides: [u64; N],
    #[output]
    result: Vector<T, M>  // renamed from `result`

//   let builders = [
//     Builder<(ins Value:$source, Value:$dest,
//       "[&i64]":$offsets, "[&i64]":$strides)>
//   ];
}

impl InsertStridedSlice {
    pub fn has_non_unit_strides(&self) -> bool {
        self.strides.any(|stride| stride != 1)
    }
}

impl Verify for InsertStridedSlice {
    fn verify(&self) -> LogicalResult {
        if N > M {
            return emit_op_error(
              "Expected source rank to be smaller than destination rank");
        }
      
        let source_shape = self.source.shape;
        let dest_shape = self.dest.shape;
        let mut source_shape_as_dest_shape
            = SmallVector::<[i64; 4]>::new(N - M, 0);
        source_shape_as_dest_shape.append(source_shape);
        if failed(
            is_integer_array_attr_confined_to_shape(
                self, self.offsets, dest_shape, "offsets"))
        || failed(is_integer_array_attr_confined_to_range(
            self, self.strides, 1, 1, "strides", /*halfOpen=*/false))
        || failed(is_sum_of_integer_array_attr_confined_to_shape(
                self, self.offsets,
                make_i64_array_attr(source_shape_as_dest_shape, self.context()), dest_shape,
                "offsets", "source vector shape",
                /*halfOpen=*/false, /*min=*/1))
        {
            return Err(());
        }
      
        Ok(())
    }
}

impl Fold for InsertStridedSlice {
    fn fold(&self) -> FoldResult {
        if self.source == self.dest {
            return self.source;
        }
        return {};
    }
}

impl Canonicalis for InsertStridedSlice {
    fn canonicalisation_patterns(
        results: &RewritePatternSet, context: *mut MLIRContext
    ) {
        results.add::<
            FoldInsertStridedSliceSplat, FoldInsertStridedSliceOfExtract,
                  InsertStridedSliceConstantFolder>(context);
    }
}

/**
Vector outer_product with optional fused add.

Takes 2 1-D vectors and returns the 2-D vector containing the outer-product, as illustrated below:

```
outer |   [c, d]
------+------------
[a, | [ [a*c, a*d],
b] |   [b*c, b*d] ]
```

This operation also accepts a 1-D vector lhs and a scalar rhs. In this case a simple AXPY operation is performed, which returns a 1-D vector.

```text
[a, b] * c = [a * c, b * c]
```

An optional extra vector argument with the same shape as the output vector may be specified in which case the operation returns the sum of the outer-product and the extra vector. In this multiply-accumulate scenario for floating-point arguments, the rounding mode is enforced by guaranteeing that a fused-multiply add operation is emitted. When lowered to the LLVMIR dialect, this form emits `llvm.intr.fma`, which is guaranteed to lower to actual `fma` instructions on x86.

An optional kind attribute may be specified to be add/mul/min/max for int/fp, and and/or/xor for int only. The default is "add", in which case the operation returns a fused multiply-add. In other cases it returns a multiply followed by the appropriate operation (for example, a compare and select for "max").

# Examples

```mlir
%2 = vector.outer_product %0, %1 : vector<4xf32>, vector<8xf32>
return %2: vector<4x8xf32>
```

```mlir
%3 = vector.outer_product %0, %1, %2
    : vector<4xf32>, vector<8xf32>, vector<4x8xf32>
return %3: vector<4x8xf32>
```

```mlir
%4 = vector.outer_product %0, %1, %2
    { kind = #vector.kind<max> }
    : vector<4xf32>, vector<8xf32>, vector<4x8xf32>
return %3: vector<4x8xf32>
```

```mlir
%6 = vector.outer_product %4, %5 : vector<10xf32>, f32
return %6: vector<10xf32>
```
*/
#[mlir(
    traits = [Pure],
)]
pub struct OuterProduct<T> {
    #[input]
    lhs: Vector<T, 1>,
    #[input]
    rhs: Vector<T, 1>,  // TODO Vector<T, 1> or scalar
    #[input]
    acc: Option<Vector<T, 2>>,  // TODO 2 or 1
    #[arribute(default = CombiningKind::Add)]
    kind: CombiningKindAttr,
    #[output]
    result: Vector<T, 2>  // TODO 2 or 1

//   let builders = [
//     // Build an op without mask, use the type of `acc` as the return type.
//     Builder<(ins Value:$lhs, Value:$rhs, Value:$acc)>
//   ];
}

impl Verify for OuterProduct {
    fn verify(&self) -> LogicalResult {
        let v_lhs = self.lhs;
        let v_rhs = self.rhs.dyn_cast<Vector>();
        let v_res = self.result;
      
        if self.rhs {
            // Proper OUTER operation.
            if self.R != 1 {
                return emit_op_error("Expected 1-d vector for operand #2.");
            }
            if v_res.rank() != 2 {
                return emit_op_error("Expected 2-d vector result.");
            }
            if v_lhs.dim_size(0) != v_res.dim_size(0) {
                return emit_op_error(
                    "Expected #1 operand dim to match result dim #1.");
            }
            if self.rhs.dim_size(0) != v_res.dim_size(1) {
                return emit_op_error("Expected #2 operand dim to match result dim #2.");
            }
        } else {
            // An AXPY operation.
            if v_res.rank() != 1 {
                return emit_op_error("Expected 1-d vector result");
            }
            if v_lhs.dim_size(0) != v_res.dim_size(0) {
                return emit_op_error("Expected #1 operand dim to match result dim #1");
            }
        }
      
        // Verify supported combining kind.
        if !is_supported_combining_kind(self.kind, v_res.element_type) {
            return emit_op_error("Unsupported outer_product type");
        }
      
        Ok(())
    }
}

impl AssemblyFormat for OuterProduct {
    fn print(&self, p: &OpAsmPrinter) {
        p << " " << get_lhs() << ", " << get_rhs();
        if (!get_acc().is_empty()) {
          p << ", " << get_acc();
          p.print_optional_attr_dict((*this).get_attrs());
        }
        p << " : " << get_lhs().get_type() << ", " << get_rhs().get_type();
      }
      
    fn parse(OpAsmParser &parser, result: &OperationState) -> ParseResult {
        SmallVector<UnresolvedOperand, 3> operands_info;
        Type t_lhs, t_rhs;
        if (parser.parse_operand_list(operands_info) ||
            parser.parse_optional_attr_dict(result.attributes) ||
            parser.parse_colon_type(t_lhs) || parser.parseComma() ||
            parser.parse_type(t_rhs))
          return Err(());
        if (operands_info.len() < 2)
          return parser.emit_error(parser.name_loc(),
                                  "Expected at least 2 operands");
        let v_lhs = t_lhs.dyn_cast<Vector>();
        let v_rhs = t_rhs.dyn_cast<Vector>();
        if (!v_lhs)
          return parser.emit_error(parser.name_loc(),
                                  "Expected vector type for operand #1");
      
        let num_scalable_dims = v_lhs.num_scalable_dims();
        Vector res_type;
        if (v_rhs) {
          num_scalable_dims += v_rhs.num_scalable_dims();
          res_type = Vector::new({v_lhs.dim_size(0), v_rhs.dim_size(0)},
                                    v_lhs.element_type, num_scalable_dims);
        } else {
          // Scalar RHS operand
          res_type = Vector::new({v_lhs.dim_size(0)}, v_lhs.element_type,
                                    num_scalable_dims);
        }
      
        if (!result.attributes.get("kind")) {
          result.attributes.append(
              "kind",
              CombiningKindAttr::get(result.self.context(),
                                     OuterProduct::get_default_kind()));
        }
      
        return failure(
            parser.resolve_operand(operands_info[0], t_lhs, result.operands) ||
            parser.resolve_operand(operands_info[1], t_rhs, result.operands) ||
            (operands_info.len() > 2 &&
             parser.resolve_operand(operands_info[2], res_type, result.operands)) ||
            parser.add_type_to_list(res_type, result.types));
    }
}

/**
`vector.reshape` Vector reshape operation.

Reshapes its vector operand from `input_shape` to `output_shape` maintaining fixed vector dimension `fixed_vector_sizes` on the innermost vector dimensions.

The parameters `input_shape` and `output_shape` represent valid data shapes across fixed vector shapes. For example, if a vector has a valid data shape [6] with fixed vector size [8], then the valid data elements are assumed to be stored at the beginning of the vector with the remaining vector elements undefined.

In the examples below, valid data elements are represented by an alphabetic character, and undefined data elements are represented by '-'.

# Example

vector<1x8xf32> with valid data shape [6], fixed vector sizes [8]

input: [a, b, c, d, e, f]

layout map: (d0) -> (d0 floordiv 8, d0 mod 8)

vector layout: [a, b, c, d, e, f, -, -]

# Example

vector<2x8xf32> with valid data shape [10], fixed vector sizes [8]

input: [a, b, c, d, e, f, g, h, i, j]

layout map: (d0) -> (d0 floordiv 8, d0 mod 8)

vector layout: [[a, b, c, d, e, f, g, h],
                [i, j, -, -, -, -, -, -]]

# Example

vector<2x2x2x3xf32> with valid data shape [3, 5], fixed vector sizes
[2, 3]

input: [[a, b, c, d, e],
        [f, g, h, i, j],
        [k, l, m, n, o]]

layout map: (d0, d1) -> (d0 floordiv 3, d1 floordiv 5,
                        d0 mod 3, d1 mod 5)

vector layout: [[[[a, b, c],
                    [f, g, h]]
                    [[d, e, -],
                    [i, j, -]]],
                [[[k, l, m],
                    [-, -, -]]
                    [[n, o, -],
                    [-, -, -]]]]

# Example

```mlir
%1 = vector.reshape %0, [%c3, %c6], [%c2, %c9], [4]
    : vector<3x2x4xf32> to vector<2x3x4xf32>
```

input: [[a, b, c, d, e, f],
        [g, h, i, j, k, l],
        [m, n, o, p, q, r]]

layout map: (d0, d1) -> (d0, d1 floordiv 4, d1 mod 4)


Input vector:  [[[a, b, c, d],
                [e, f, -, -]],
                [[g, h, i, j],
                [k, l, -, -]],
                [[m, n, o, p],
                [q, r, -, -]]]

Output vector:  [[[a, b, c, d],
                [e, f, g, h],
                [i, -, -, -]],
                [[j, k, l, m],
                [n, o, p, q],
                [r, -, -, -]]]

TODO: Add transformation which decomposes Reshape into an optimised sequence of vector rotate/shuffle/select operations.
*/
#[mlir(
    traits = [AttrSizedOperandSegments, Pure]
    assembly_format = "$input `,` `[` $input_shape `]` `,` `[` $output_shape `]` `,` $fixed_vector_sizes attr-dict `:` type($vector) `to` type($result)"
)]
pub struct Reshape<
    T, const I: usize, const O: usize, const F: usize
> {
    #[input]
    input: Vector<T, I + F>,
    #[input]
    input_shape: [usize; I],
    #[input]
    output_shape: [usize; O],
    #[attribute]
    fixed_vector_sizes: [u64; F],
    #[output]
    output: Vector<T, O + F>  // renamed from `result`
}

impl Reshape {
    pub fn get_fixed_vector_sizes(&self, results: SmallVector<[i64]>) {

    }
}

impl Verify for Reshape {
    fn verify(&self) -> LogicalResult {
        let mut fixed_vector_sizes = SmallVector<[i64; 4]>::new();
        get_fixed_vector_sizes(fixed_vector_sizes);
      
        /*
        Verify that the 'fixed_vector_sizes' match an input/output vector shape
        suffix.
        */
        for i in 0..F {
            let index = I - i;
            if fixed_vector_sizes[i] != self.input.shape[index] {
                return emit_error(
                    "Fixed vector size must match input vector for dim {}",
                    i
                );
            }
        }
      
        for i in 0..F {
            let index = O - i;
            if fixed_vector_sizes[i] != self.output.shape[index] {
                return emit_error(
                    "Fixed vector size must match output vector for dim {}",
                    i
                );
            }
        }
      
        /*
        If all shape operands are produced by constant ops, verify that product of dimensions for input/output shape match.
        */
        let is_def_by_constant = |operand: Value|
            isa_and_nonnull<ConstantIndex>(operand.defining_op());
        if self.input_shape.all(is_def_by_constant)
        && self.output_shape.all(is_def_by_constant) {
            let mut num_input_elements = 1;
            for operand in self.input_shape {
                num_input_elements *=
                    cast<ConstantIndex>(operand.defining_op()).value();
            }
            let mut num_output_elements = 1;
            for operand in self.output_shape {
                num_output_elements *=
                    cast<ConstantIndex>(operand.defining_op()).value();
            }
            if num_input_elements != num_output_elements {
                return emit_error(
                    "Product of input and output shape sizes must match");
            }
        }
        Ok(())
    }
}

/**
`vector.extract_strided_slice` operation.

Takes an n-D vector, k-D `offsets` integer array attribute, a k-sized `sizes` integer array attribute, a k-sized `strides` integer array attribute and extracts the n-D subvector at the proper offset.

At the moment strides must contain only 1s.
// TODO: support non-1 strides.

Returns an n-D vector where the first k-D dimensions match the `sizes` attribute. The returned subvector contains the elements starting at offset `offsets` and ending at `offsets + sizes`.

# Examples

```mlir
%1 = vector.extract_strided_slice %0
    { offsets = [0, 2], sizes = [2, 4], strides = [1, 1] }
    : vector<4x8x16xf32> to vector<2x4x16xf32>
```

TODO: Evolve to a range form syntax similar to:

```mlir
%1 = vector.extract_strided_slice %0[0:2:1][2:4:1]
    vector<4x8x16xf32> to vector<2x4x16xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$input attr-dict `:` type($vector) `to` type(output)"
)]
pub struct ExtractStridedSlice<T, const N: usize> {
    #[input]
    input: Vector<T, _>,  // renamed from `vector`
    #[attribute]
    offsets: [u64; N],
    #[attribute]
    sizes: [u64; N],
    #[attribute]
    strides: [u64; N],
    #[output]
    output: Vector<T, _>  // renamed from `result`
}

//   let builders = [
//     Builder<(ins Value:$input, "[&i64]":$offsets,
//       &[u64]:$sizes, &[u64]:$strides)>
//   ];

impl ExtractStridedSlice {
    pub fn get_offsets(&self, results: &SmallVector<[i64]>) {

    }

    pub fn has_non_unit_strides(&self) -> bool {
      return self.strides.any(|attr: Attribute| {
            return attr.cast<IntegerAttribute>().get_int() != 1;
      });
    }
}

impl Verify for ExtractStridedSlice {
    fn verify(&self) -> LogicalResult {
        let shape = self.input.shape;
        if failed(
                is_integer_array_attr_smaller_than_shape(
                    self, self.offsets, shape, "offsets"))
        ||
            failed(
                is_integer_array_attr_smaller_than_shape(self, self.sizes, shape, "sizes"))
        ||
            failed(is_integer_array_attr_smaller_than_shape(
                self, self.strides, shape, strides_name))
        ||
            failed(
                is_integer_array_attr_confined_to_shape(
                    self, self.offsets, shape, "offsets"))
        ||
            failed(is_integer_array_attr_confined_to_shape(
                self, self.sizes, shape, "sizes", /*halfOpen=*/false,
                                                     /*min=*/1))
        ||
            failed(is_integer_array_attr_confined_to_range(
                self, self.strides, 1, 1, strides_name,
                                                     /*halfOpen=*/false))
        ||
            failed(is_sum_of_integer_array_attr_confined_to_shape(
                self, self.offsets, self.sizes, shape, "offsets", "sizes",
                                                          /*halfOpen=*/false))
        {
            return Err(());
        }
      
        let result_type
            = infer_strided_slice_op_result_type(self.input, self.offsets, self.sizes, self.strides);
        if self.result != result_type {
            return emit_op_error("Expected result type to be {}", result_type);
        }
      
        Ok(())
    }
}

impl Fold for ExtractStridedSlice {
    fn fold(&self) -> FoldResult {
        if self.input == self.result {
            return self.input;
        }
        if fold_extract_strided_op_from_insert_chain(self).is_ok() {
            return self.result;
        }
        return {};
    }
}

impl Canonicalise for ExtractStridedSlice {
    fn canonicalisation_patterns(
        results: &RewritePatternSet, context: *mut MLIRContext
    ) {
      // Pattern to rewrite a ExtractStridedSlice(ConstantMask) ->
      // ConstantMask and ExtractStridedSlice(ConstantOp) -> ConstantOp.
      results.add::<
        StridedSliceConstantMaskFolder, StridedSliceSplatConstantFolder,
                  StridedSliceNonSplatConstantFolder, StridedSliceBroadcast,
                  StridedSliceSplat>(context);
    }
}

/**
Reads a supervector from memory into an SSA vector value.

The `vector.transfer_read` op performs a read from a slice within a [MemRef](../LangRef.md#memref-type) or a Ranked [Tensor](../LangRef.md#tensor-type) supplied as its first operand into a [vector](../LangRef.md#vector-type) of the same base elemental type.

A memref/tensor operand with vector element type, must have its vector element type match a suffix (shape and element type) of the vector (e.g. memref<3x2x6x4x3xf32>, vector<1x1x4x3xf32>).

The slice is further defined by a full-rank index within the MemRef/Tensor, supplied as the operands `[1 .. 1 + rank(memref/tensor))` that defines the starting point of the transfer (e.g. `%A[%i0, %i1, %i2]`).

The permutation_map [attribute](../LangRef.md#attributes) is an [affine-map](Affine.md#affine-maps) which specifies the transposition on the slice to match the vector shape. The permutation map may be implicit and omitted from parsing and printing if it is the canonical minor identity map (i.e. if it does not permute or broadcast any dimension).

The size of the slice is specified by the size of the vector, given as the return type.

An SSA value `padding` of the same elemental type as the MemRef/Tensor is provided to specify a fallback value in the case of out-of-bounds accesses and/or masking.

An optional SSA value `mask` may be specified to mask out elements read from the MemRef/Tensor. The `mask` type is an `i1` vector with a shape that matches how elements are read from the MemRef/Tensor, *before* any permutation or broadcasting. Elements whose corresponding mask element is `0` are masked out and replaced with `padding`.

An optional boolean array attribute `in_bounds` specifies for every vector dimension if the transfer is guaranteed to be within the source bounds.
While the starting point of the transfer has to be in-bounds, accesses may run out-of-bounds as indices increase. Broadcast dimensions must always be in-bounds. If specified, the `in_bounds` array length has to be equal to the vector rank. In absence of the attribute, accesses along all dimensions (except for broadcasts) may run out-of-bounds. A `vector.transfer_read` can be lowered to a simple load if all dimensions are specified to be within bounds and no `mask` was specified.

This operation is called 'read' by opposition to 'load' because the super-vector granularity is generally not representable with a single hardware register. A `vector.transfer_read` is thus a mid-level abstraction that supports super-vectorisation with non-effecting padding for full-tile only operations.

More precisely, let's dive deeper into the permutation_map for the following MLIR:

```mlir
vector.transfer_read %A[%expr1, %expr2, %expr3, %expr4]
    { permutation_map = (d0, d1, d2, d3) -> (d2, 0, d0) }
    : memref<?x?x?x?xf32>, vector<3x4x5xf32>
```

This operation always reads a slice starting at `%A[%expr1, %expr2, %expr3, %expr4]`. The size of the slice is 3 along d2 and 5 along d0, so the slice is: `%A[%expr1 : %expr1 + 5, %expr2, %expr3:%expr3 + 3, %expr4]`

That slice needs to be read into a `vector<3x4x5xf32>`. Since the permutation map is not full rank, there must be a broadcast along vector dimension `1`.

A notional lowering of vector.transfer_read could generate code resembling:

```mlir
// %expr1, %expr2, %expr3, %expr4 defined before this point
%tmp = alloc() : vector<3x4x5xf32>
%view_in_tmp = "element_type_cast"(%tmp) : memref<1xvector<3x4x5xf32>>
for %i = 0 to 3 {
    affine.for %j = 0 to 4 {
        affine.for %k = 0 to 5 {
            %a = load %A[%expr1 + %k, %expr2, %expr3 + %i, %expr4]
                : memref<?x?x?x?xf32>
            store %tmp[%i, %j, %k] : vector<3x4x5xf32>
        }
    }
}
%c0 = arith.constant 0 : index
%vec = load %view_in_tmp[%c0] : vector<3x4x5xf32>
```

On a GPU one could then map `i`, `j`, `k` to blocks and threads. Notice that the temporary storage footprint is `3 * 5` values but `3 * 4 * 5` values are actually transferred between `%A` and `%tmp`.

Alternatively, if a notional vector broadcast operation were available, the lowered code would resemble:

```mlir
// %expr1, %expr2, %expr3, %expr4 defined before this point
%tmp = alloc() : vector<3x4x5xf32>
%view_in_tmp = "element_type_cast"(%tmp) : memref<1xvector<3x4x5xf32>>
for %i = 0 to 3 {
    affine.for %k = 0 to 5 {
    %a = load %A[%expr1 + %k, %expr2, %expr3 + %i, %expr4]
        : memref<?x?x?x?xf32>
    store %tmp[%i, 0, %k] : vector<3x4x5xf32>
}}
%c0 = arith.constant 0 : index
%tmpvec = load %view_in_tmp[%c0] : vector<3x4x5xf32>
%vec = broadcast %tmpvec, 1 : vector<3x4x5xf32>
```

where `broadcast` broadcasts from element 0 to all others along the specified dimension. This time, the temporary storage footprint is `3 * 5` values which is the same amount of data as the `3 * 5` values transferred.
An additional `1` broadcast is required. On a GPU this broadcast could be implemented using a warp-shuffle if loop `j` were mapped to `threadIdx.x`.

# Syntax

```text
operation ::= ssa-id `=` `vector.transfer_read` ssa-use-list
    `{` attribute-entry `}` `:` memref-type `,` vector-type
```

# Examples

Read the slice `%A[%i0, %i1:%i1+256, %i2:%i2+32]` into vector<32x256xf32> and pad with `%f0` to handle the boundary case:

```mlir
%f0 = arith.constant 0.0f : f32
for %i0 = 0 to %0 {
    affine.for %i1 = 0 to %1 step 256 {
    affine.for %i2 = 0 to %2 step 32 {
        %v = vector.transfer_read %A[%i0, %i1, %i2], (%f0)
            { permutation_map = (d0, d1, d2) -> (d2, d1) }
            : memref<?x?x?xf32>, vector<32x256xf32>
}}}
```

Read the slice `%A[%i0, %i1]` (i.e. the element `%A[%i0, %i1]`) into vector<128xf32>. The underlying implementation will require a 1-D vector broadcast:

```mlir
for %i0 = 0 to %0 {
    affine.for %i1 = 0 to %1 {
    %3 = vector.transfer_read %A[%i0, %i1]
        { permutation_map = (d0, d1) -> (0) }
        : memref<?x?xf32>, vector<128xf32>
    }
}
```

Reads from a memref with vector element type:

```mlir
%4 = vector.transfer_read %arg1[%c3, %c3], %vf0
    { permutation_map = (d0, d1) -> (d0, d1) }
    : memref<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
```

Reads from a tensor with vector element type:

```mlir
%4 = vector.transfer_read %arg1[%c3, %c3], %vf0
    { permutation_map = (d0, d1) -> (d0, d1) }
    : tensor<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
```

Special encoding for 0-d transfer with 0-d tensor/memref, vector shape {1} and permutation_map `() -> (0)`:

```mlir
%0 = vector.transfer_read %arg0[], %f0
    { permutation_map = affine_map<() -> (0)> }
    : tensor<f32>, vector<1xf32>
```
*/
/*
TODO: Tighten semantics so that masks and inbounds can't be used simultaneously within the same transfer op.
*/
#[mlir(
    traits = [AttrSizedOperandSegments],
)]
pub struct TransferRead<T, const N: usize> {
    #[input]
    source: ShapedType<_, N>,
    #[input]
    indices: [usize; N],
    #[attribute]
    permutation_map: AffineMap,
    #[input]
    padding: _,
    #[input]
    mask: Option<Vector<bool, _>>,
    #[attribute]
    in_bounds: Option<BoolArrayAttr>,
    #[output]
    vector: Vector<_, _>
}

//   let builders = [
//     /// 1. Builder that sets padding to zero and an empty mask (variant with attrs).
//     Builder<(ins "Vector":$vector_type,
//                    Value:$source,
//                    ValueRange:$indices,
//                    "AffineMapAttr":$permutationMapAttr,
//                    "ArrayAttribute":$inBoundsAttr)>,
//     /// 2. Builder that sets padding to zero and an empty mask (variant without attrs).
//     Builder<(ins "Vector":$vector_type,
//                    Value:$source,
//                    ValueRange:$indices,
//                    "AffineMap":$permutation_map,
//                    CArg<"Option<[&bool]>", "None">:$in_bounds)>,
//     /// 3. Builder that sets permutation map to 'minor_identity_map'.
//     Builder<(ins "Vector":$vector_type,
//                    Value:$source,
//                    ValueRange:$indices,
//                    Value:$padding,
//                    CArg<"Option<[&bool]>", "None">:$in_bounds)>,
//     /// 4. Builder that sets padding to zero and permutation map to
//     /// 'minor_identity_map'.
//     Builder<(ins "Vector":$vector_type,
//                    Value:$source,
//                    ValueRange:$indices,
//                    CArg<"Option<[&bool]>", "None">:$in_bounds)>,
//   ];

impl TransferRead {
    // MaskableOpInterface methods.
    bool supports_passthru() { return true; }
}

impl Verify for TransferRead {
    fn verify(&self) -> LogicalResult {
        // Consistency of elemental types in source and vector.
        let shaped_type = get_shaped_type();
        let mask_type = get_mask_type();
        let padding_type = self.padding;
        let permutation_map = self.permutation_map;
        let inferred_mask_type = if mask_type {
            infer_transfer_read_mask_type(self.vector, permutation_map)
        } else {
            Vector()
        };
        let source_element_type = shaped_type.element_type;
      
        if (failed(verify_transfer_op(
            cast<VectorTransferOpInterface>(get_operation()),
            shaped_type, self.vector, mask_type,
            inferred_mask_type, permutation_map,
            get_in_bounds() ? *get_in_bounds() : ArrayAttribute::new())))
        {
            return Err(());
        }
      
        if let source_vector_element_type = source_element_type.dyn_cast<Vector>() {
            /*
            Source has vector element type.
            Check that 'source_vector_element_type' and 'padding_type' types match.
            */
            if source_vector_element_type != padding_type {
                return emit_op_error(
                    "Requires source element type and padding type to match.");
                }
      
        } else {
            // Check that 'padding_type' is valid to store in a vector type.
            if !Vector::is_valid_element_type(padding_type){
                return emit_op_error("Requires valid padding vector elemental type");
            }
        
            // Check that padding type and vector element types match.
            if padding_type != source_element_type {
                return emit_op_error(
                    "Requires formal padding and source of the same elemental type");
            }
        }
      
        verify_permutation_map(
            permutation_map,
            |t: Twine| emit_op_error(t))
    }
}

impl Fold for TransferRead {
    fn fold(&self) -> FoldResult {
        if Value vec = fold_raw(self) {
            return vec;
        }
        /// transfer_read(memrefcast) -> transfer_read
        if fold_transfer_in_bounds_attribute(self).is_ok() {
            return self.result;
        }
        if memref::fold_mem_ref_cast(self).is_ok() {
            return self.result;
        }
        if tensor::fold_tensor_cast(self).is_ok() {
            return self.result;
        }
        return FoldResult();
    }
}

impl Canonicalise for TransferRead {
    fn canonicalisation_patterns(
        results: &RewritePatternSet,
        context: *mut MLIRContext
    ) {
        results.add::<
            FoldExtractSliceIntoTransferRead,
            TransferReadAfterWriteToBroadcast>(context);
    }
}

impl AssemblyFormat for TransferRead {
    fn parse(parser: &OpAsmParser, result: &OperationState) -> ParseResult {
        let &builder = parser.builder();
        SMLoc types_loc;
        OpAsmParserUnresolvedOperand source_info;
        SmallVector<OpAsmParserUnresolvedOperand, 8> index_info;
        OpAsmParserUnresolvedOperand padding_info;
        SmallVector<[Type, 2]> types;
        OpAsmParserUnresolvedOperand mask_info;
        // Parsing with support for paddingValue.
        if parser.parse_operand(source_info)
        || parser.parse_operand_list(index_info, OpAsmParser::Delimiter::Square)
        || parser.parse_comma()
        || parser.parse_operand(padding_info) {
            return Err(());
        }
        let has_mask = parser.parse_optional_comma();
        if has_mask.succeeded() {
            if (parser.parse_operand(mask_info)) {
                return Err(());
            }
        }
        if parser.parse_optional_attr_dict(result.attributes)
        || parser.get_current_location(&types_loc)
        || parser.parse_colon_type_list(types) {
            return Err(());
        }
        if types.len() != 2 {
            return parser.emit_error(types_loc, "Requires two types");
        }
        let index_type = builder.get_index_type();
        let shaped_type = types[0].dyn_cast<ShapedType>();
        if (!shaped_type || !shaped_type.isa<MemRef, RankedTensorType>()) {
            return parser.emit_error(
                types_loc, "Requires memref or ranked tensor type");
        }
        let vector_type = types[1].dyn_cast<Vector>();
        if (!vector_type) {
            return parser.emit_error(types_loc, "Requires vector type");
        }
        let perm_map_attr_name = "permutation_map";
        let perm_map_attr = result.attributes.get(perm_map_attr_name);
        perm_map: AffineMap;
        if !perm_map_attr {
            perm_map = get_transfer_minor_identity_map(shaped_type, vector_type);
            result.attributes.set(perm_map_attr_name, AffineMapAttr::get(perm_map));
        } else {
            perm_map = perm_map_attr.cast<AffineMapAttr>().value();
        }
        if parser.resolve_operand(source_info, shaped_type, result.operands)
        || parser.resolve_operands(index_info, index_type, result.operands)
        || parser.resolve_operand(padding_info, shaped_type.element_type,
                                    result.operands))
        {
            return Err(());
        }
        if has_mask.succeeded() {
            if (shaped_type.element_type.dyn_cast<Vector>()) {
                return parser.emit_error(
                    mask_info.location, "Does not support masks with vector element type");
            }
            /*
            Instead of adding the mask type as an op type, compute it based on the vector type and the permutation map (to keep the type signature small).
            */
            let mask_type = infer_transfer_read_mask_type(vector_type, perm_map);
            if (parser.resolve_operand(mask_info, mask_type, result.operands)) {
                return Err(());
            }
        }
        result.add_attribute(
            TransferRead::get_operand_segment_size_attr(),
            builder.get_dense_i32_array_attr(
                {1, static_cast<i32>(index_info.len()), 1,
                static_cast<i32>(has_mask.succeeded())}));
        return parser.add_type_to_list(vector_type, result.types);
    }

    fn print(&self, p: &OpAsmPrinter) {
        p << " " << input << "[" << self.indices << "], " << self.padding;
        if (self.mask)
            p << ", " << self.mask;
        print_transfer_attrs(p, *this);
        p << " : " << get_shaped_type() << ", " << self.input;
    }
}

impl VectorTransferOpInterface for TransferRead {

}

impl VectorUnrollOpInterface for TransferRead {
    fn shape_for_unroll(&self) -> Option<SmallVector<[i64; 4]>> {
        to_vector<4>(self.input.shape)
    }
}

impl MaskableOpInterface for TransferRead {

}

impl MemoryEffectsOpInterface for TransferRead {
    
}

/**
The vector.transfer_write op writes a supervector to memory.

The `vector.transfer_write` op performs a write from a [vector](../LangRef.md#vector-type), supplied as its first operand, into a slice within a [MemRef](../LangRef.md#memref-type) or a Ranked [Tensor](../LangRef.md#tensor-type) of the same base elemental type, supplied as its second operand.

A vector memref/tensor operand must have its vector element type match a suffix (shape and element type) of the vector (e.g. memref<3x2x6x4x3xf32>, vector<1x1x4x3xf32>). If the operand is a tensor, the operation returns a new tensor of the same type.

The slice is further defined by a full-rank index within the MemRef/Tensor, supplied as the operands `[2 .. 2 + rank(memref/tensor))` that defines the starting point of the transfer (e.g. `%A[%i0, %i1, %i2, %i3]`).

The permutation_map [attribute](../LangRef.md#attributes) is an [affine-map](Affine.md#affine-maps) which specifies the transposition on the slice to match the vector shape. The permutation map may be implicit and omitted from parsing and printing if it is the canonical minor identity map (i.e. if it does not permute any dimension). In contrast to `transfer_read`, write ops cannot have broadcast dimensions.

The size of the slice is specified by the size of the vector.

An optional SSA value `mask` may be specified to mask out elements written to the MemRef/Tensor. The `mask` type is an `i1` vector with a shape that matches how elements are written into the MemRef/Tensor, *after* applying any permutation. Elements whose corresponding mask element is `0` are masked out.

An optional SSA value `mask` of the same shape as the vector type may be specified to mask out elements. Elements whose corresponding mask element is `0` are masked out.

An optional boolean array attribute `in_bounds` specifies for every vector dimension if the transfer is guaranteed to be within the source bounds.
While the starting point of the transfer has to be in-bounds, accesses may run out-of-bounds as indices increase. If specified, the `in_bounds` array length has to be equal to the vector rank. In absence of the attribute, accesses along all dimensions may run out-of-bounds. A `vector.transfer_write` can be lowered to a simple store if all dimensions are specified to be within bounds and no `mask` was specified.

This operation is called 'write' by opposition to 'store' because the super-vector granularity is generally not representable with a single hardware register. A `vector.transfer_write` is thus a mid-level abstraction that supports super-vectorisation with non-effecting padding for full-tile-only code. It is the responsibility of `vector.transfer_write`'s implementation to ensure the memory writes are valid. Different lowerings may be pertinent depending on the hardware support.

# Examples

Writes `vector<16x32x64xf32>` into the slice `%A[%i0, %i1:%i1+32, %i2:%i2+64, %i3:%i3+16]`:

```mlir
for %i0 = 0 to %0 {
    affine.for %i1 = 0 to %1 step 32 {
    affine.for %i2 = 0 to %2 step 64 {
        affine.for %i3 = 0 to %3 step 16 {
        %val = `ssa-value` : vector<16x32x64xf32>
        vector.transfer_write %val, %A[%i0, %i1, %i2, %i3]
            { permutation_map: (d0, d1, d2, d3) -> (d3, d1, d2) }
            : vector<16x32x64xf32>, memref<?x?x?x?xf32>
}}}}
```

Writes to a memref with vector element type:

```mlir
vector.transfer_write %4, %arg1[%c3, %c3]
    { permutation_map = (d0, d1) -> (d0, d1) }
    : vector<1x1x4x3xf32>, memref<?x?xvector<4x3xf32>>
```

Returns a tensor where the vector is inserted into the source tensor:

```mlir
%5 = vector.transfer_write %4, %arg1[%c3, %c3]
    { permutation_map = (d0, d1) -> (d0, d1) }
    : vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>
```

Special encoding for 0-d transfer with 0-d tensor/memref, vector shape {1} and permutation_map `() -> (0)`:

```mlir
%1 = vector.transfer_write %0, %arg0[]
    { permutation_map = affine_map<()->(0)> }
    : vector<1xf32>, tensor<f32>
```
*/
/*
TODO: Tighten semantics so that masks and inbounds can't be used simultaneously within the same transfer op.
*/
#[mlir(
    traits = [AttrSizedOperandSegments]
)]
pub struct TransferWrite<T, const N: usize> {
    #[input]
    vector: Vector<, >,
    #[input]
    source: ShapedType<, N>,
    #[input]
    indices: [usize; N],
    #[attribute]
    permutation_map: AffineMap,
    #[input]
    mask: Option<Vector<bool, _>>,
    #[attribute]
    in_bounds: Optional<BoolArrayAttr>,
    #[output]
    result: Option<RankedTensor<T, _>>
}

//   let builders = [
//     /// 1. Builder with type inference.
//     Builder<(ins Value:$vector,
//                    Value:$dest,
//                    ValueRange:$indices,
//                    "AffineMapAttr":$permutationMapAttr,
//                    Value:$mask,
//                    "ArrayAttribute":$inBoundsAttr)>,
//     /// 2. Builder with type inference that sets an empty mask (variant with attrs).
//     Builder<(ins Value:$vector,
//                    Value:$dest,
//                    ValueRange:$indices,
//                    "AffineMapAttr":$permutationMapAttr,
//                    "ArrayAttribute":$inBoundsAttr)>,
//     /// 3. Builder with type inference that sets an empty mask (variant without attrs).
//     Builder<(ins Value:$vector,
//                    Value:$dest,
//                    ValueRange:$indices,
//                    "AffineMap":$permutation_map,
//                    CArg<"Option<[&bool]>", "None">:$in_bounds)>,
//     /// 4. Builder with type inference that sets an empty mask and sets permutation
//     /// map to 'minor_identity_map'.
//     Builder<(ins Value:$vector,
//                    Value:$dest,
//                    ValueRange:$indices,
//                    CArg<"Option<[&bool]>", "None">:$in_bounds)>,
//   ];

impl TransfarWrite {
    /**
    This method is added to maintain uniformity with load/store ops of other dialects.
    */
    Value value() { return self.vector; }

    (i64, i64) get_dps_inits_position_range() {
        (1, 2)  // `source` operand
    }
}

impl Verify for TransferWrite {
    fn verify(&self) -> LogicalResult {
        // Consistency of elemental types in shape and vector.
        let shaped_type = get_shaped_type();
        let mask_type = self.mask;
        let permutation_map = self.permutation_map;
        let inferred_mask_type =
            self.mask ? infer_transfer_write_mask_type(self.vector, permutation_map)
                     : Vector();
      
        /*
        We do not allow broadcast dimensions on TransferWriteOps for the moment, as the semantics is unclear. This can be revisited later if necessary.
        */
        if has_broadcast_dim() {
            return emit_op_error("Should not have broadcast dimensions");
        }
      
        if (failed(verify_transfer_op(
            cast<VectorTransferOpInterface>(
                get_operation()),
                shaped_type, self.vector, mask_type,
                inferred_mask_type, permutation_map,
                get_in_bounds() ? *get_in_bounds() : ArrayAttribute())))
        {
            return Err(());
        }
      
        verify_permutation_map(
            permutation_map,
            |t: Twine| emit_op_error(t))
    }
}

impl Fold for TransferWrite {
    fn fold(
        &self,
        operands: [&impl Attribute],
        results: &SmallVector<FoldResult>) -> LogicalResult
    {
        if fold_read_init_write(self, operands, results).is_ok() {
            return Ok(());
        }
        if fold_war(self, results).is_ok() {
            return Ok(());
        }
        if fold_transfer_in_bounds_attribute(self).is_ok() {
            return Ok(());
        }
        memref::fold_mem_ref_cast(self)
    }
}

impl Canonicalise for TransferWrite {
    fn canonicalisation_patterns(
        results: &RewritePatternSet, context: *mut MLIRContext
    ) {
        results.add<FoldWaw, FoldInsertSliceIntoTransferWrite,
                SwapExtractSliceOfTransferWrite>(context);
    }
}

impl AssemblyFormat for TransferWrite {
    fn parse(parser: &OpAsmParser, result: &OperationState) -> ParseResult {
        let &builder = parser.builder();
        SMLoc types_loc;
        OpAsmParserUnresolvedOperand vector_info, source_info;
        SmallVector<OpAsmParserUnresolvedOperand, 8> index_info;
        SmallVector<[Type, 2]> types;
        OpAsmParserUnresolvedOperand mask_info;
        if parser.parse_operand(vector_info)
        || parser.parse_comma()
        || parser.parse_operand(source_info)
        || parser.parse_operand_list(index_info, OpAsmParser::Delimiter::Square) {
            return Err(());
        }
        let has_mask = parser.parse_optional_comma();
        if (has_mask.succeeded() && parser.parse_operand(mask_info)) {
            return Err(());
        }
        if parser.parse_optional_attr_dict(result.attributes)
        || parser.get_current_location(&types_loc)
        || parser.parse_colon_type_list(types) {
            return Err(());
        }
        if (types.len() != 2) {
            return parser.emit_error(types_loc, "Requires two types");
        }
        let index_type = builder.get_index_type();
        vector_type: Vector = types[0].dyn_cast<Vector>();
        if (!vector_type) {
            return parser.emit_error(types_loc, "Requires vector type");
        }
        ShapedType shaped_type = types[1].dyn_cast<ShapedType>();
        if (!shaped_type || !shaped_type.isa<MemRef, RankedTensorType>()) {
            return parser.emit_error(types_loc, "Requires memref or ranked tensor type");
        }
        let perm_map_attr_name = "permutation_map";
        let perm_map_attr = result.attributes.get(perm_map_attr_name);
        perm_map: AffineMap;
        if !perm_map_attr {
            perm_map = get_transfer_minor_identity_map(shaped_type, vector_type);
            result.attributes.set(perm_map_attr_name, AffineMapAttr::get(perm_map));
        } else {
            perm_map = perm_map_attr.cast<AffineMapAttr>().value();
        }
        if (parser.resolve_operand(vector_info, vector_type, result.operands) ||
            parser.resolve_operand(source_info, shaped_type, result.operands) ||
            parser.resolve_operands(index_info, index_type, result.operands))
            return Err(());
        if has_mask.succeeded() {
            if (shaped_type.element_type.dyn_cast<Vector>()) {
                return parser.emit_error(
                mask_info.location, "Does not support masks with vector element type");
            }
            let mask_type = infer_transfer_write_mask_type(vector_type, perm_map);
            if (parser.resolve_operand(mask_info, mask_type, result.operands)) {
                return Err(());
            }
        }
        result.add_attribute(TransferWrite::get_operand_segment_size_attr(),
                            builder.get_dense_i32_array_attr(
                                {1, 1, static_cast<i32>(index_info.len()),
                                static_cast<i32>(has_mask.succeeded())}));
        failure(shaped_type.isa<RankedTensorType>() &&
              parser.add_type_to_list(shaped_type, result.types));
    }

    fn print(&self, p: &OpAsmPrinter) {
        p << " " << input << ", " << input << "[" << self.indices << "]";
        if (self.mask)
            p << ", " << self.mask;
        print_transfer_attrs(p, *this);
        p << " : " << self.input << ", " << get_shaped_type();
    }
}

impl VectorTransferOpInterface for TransferWrite {

}

impl VectorUnrollOpInterface for TransferWrite {
    fn shape_for_unroll(&self) -> Option<SmallVector<[i64; 4]>> {
        to_vector<4>(self.input.shape)
    }
}

impl MaskableOpInterface for TransferWrite {

}

impl MemoryEffectsOpInterface for TransferWrite {

}

impl DestinationStyleOpInterface for TransferWrite {
    
}

/**
Reads an n-D slice of memory into an n-D vector.

The `vector.load` operation reads an n-D slice of memory into an n-D vector. It takes a 'base' memref, an index for each memref dimension and a result vector type as arguments. It returns a value of the result vector type. The 'base' memref and indices determine the start memory address from which to read. Each index provides an offset for each memref dimension based on the element type of the memref. The shape of the result vector type determines the shape of the slice read from the start memory address.
The elements along each dimension of the slice are strided by the memref strides. Only unit strides are allowed along the most minor memref dimension. These constraints guarantee that elements read along the first dimension of the slice are contiguous in memory.

The memref element type can be a scalar or a vector type. If the memref element type is a scalar, it should match the element type of the result vector. If the memref element type is vector, it should match the result vector type.

# Examples

Example 1: 1-D vector load on a scalar memref.

```mlir
%result = vector.load %base[%i, %j] : memref<100x100xf32>, vector<8xf32>
```

Example 2: 1-D vector load on a vector memref.

```mlir
%result = vector.load %memref[%i, %j]
    : memref<200x100xvector<8xf32>>, vector<8xf32>
```

Example 3:  2-D vector load on a scalar memref.

```mlir
%result = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<4x8xf32>
```

Example 4:  2-D vector load on a vector memref.

```mlir
%result = vector.load %memref[%i, %j]
    : memref<200x100xvector<4x8xf32>>, vector<4x8xf32>
```

Representation-wise, the `vector.load` operation permits out-of-bounds reads. Support and implementation of out-of-bounds vector loads is target-specific. No assumptions should be made on the value of elements loaded out of bounds. Not all targets may support out-of-bounds vector loads.

Example 5:  Potential out-of-bound vector load.

```mlir
%result = vector.load %memref[%index] : memref<?xf32>, vector<8xf32>
```

Example 6:  Explicit out-of-bound vector load.

```mlir
%result = vector.load %memref[%c0] : memref<7xf32>, vector<8xf32>
```
*/
#[mlir(
    assembly_format = "$memref `[` $indices `]` attr-dict `:` type($memref) `,` type($result)"
)]
pub struct Load<T, const N: usize> {
    /// Reference to load from.
    #[input(traits = [MemRead])]
    memref: MemRef<T, N>,  // renamed from `base`
    #[input]
    indices: [Index; N],
    #[output]
    output: Vector<T, _>  // renamed from `result`
}

impl Verify for Load {
    fn verify(&self) -> LogicalResult {
        if failed(verify_load_store_mem_ref_layout(self, self.memref)) {
            return Err(());
        }
      
        // Checks for vector memrefs.
        let mut mem_elem_ty = self.memref.element_type;
        if let mem_vec_ty = mem_elem_ty.dyn_cast<Vector>() {
            if mem_vec_ty != self.output {
                return emit_op_error(
                    "Base memref and output vector types should match");
            }
            mem_elem_ty = mem_vec_ty.element_type;
        }
      
        if self.output.element_type != mem_elem_ty {
            return emit_op_error("Base and result element types should match");
        }
        Ok(())
    }
}

impl Verify for Load {
    fn fold(&self) -> FoldResult {
        if succeeded(fold_mem_ref_cast(self)) {
            return self.output;
        }
        FoldResult {}
    }
}


/**
Writes an n-D vector to an n-D slice of memory.

The `vector.store` operation writes an n-D vector to an n-D slice of memory.
It takes the vector value to be stored, a 'base' memref and an index for each memref dimension. The 'base' memref and indices determine the start memory address from which to write. Each index provides an offset for each memref dimension based on the element type of the memref. The shape of the vector value to store determines the shape of the slice written from the start memory address. The elements along each dimension of the slice are strided by the memref strides. Only unit strides are allowed along the most minor memref dimension. These constraints guarantee that elements written along the first dimension of the slice are contiguous in memory.

The memref element type can be a scalar or a vector type. If the memref element type is a scalar, it should match the element type of the value to store. If the memref element type is vector, it should match the type of the value to store.

Example 1: 1-D vector store on a scalar memref.

```mlir
vector.store %value, %memref[%i, %j] : memref<200x100xf32>, vector<8xf32>
```

Example 2: 1-D vector store on a vector memref.

```mlir
vector.store %value, %memref[%i, %j] : memref<200x100xvector<8xf32>>, vector<8xf32>
```

Example 3:  2-D vector store on a scalar memref.

```mlir
vector.store %value, %memref[%i, %j] : memref<200x100xf32>, vector<4x8xf32>
```

Example 4:  2-D vector store on a vector memref.

```mlir
vector.store %value, %memref[%i, %j] : memref<200x100xvector<4x8xf32>>, vector<4x8xf32>
```

Representation-wise, the `vector.store` operation permits out-of-bounds writes. Support and implementation of out-of-bounds vector stores are target-specific. No assumptions should be made on the memory written out of bounds. Not all targets may support out-of-bounds vector stores.

Example 5:  Potential out-of-bounds vector store.

```mlir
vector.store %value, %memref[%index] : memref<?xf32>, vector<8xf32>
```

Example 6:  Explicit out-of-bounds vector store.

```mlir
vector.store %value, %memref[%c0] : memref<7xf32>, vector<8xf32>
```
*/
#[mlir(
    assembly_format = "$value `,` $memref `[` $indices `]` attr-dict `:` type($memref) `,` type($value)"
)]
pub struct Store<T, const N: usize> {
    #[input]
    value: Vector<T, _>,  // renamed from `value_to_store`
    /// Reference to store to.
    #[input(traits = [MemWrite])]
    memref: MemRef<T, N>,  // renamed from `base`
    #[input]
    indices: [usize; N]
}

impl Verify for Store {
    fn verify(&self) -> LogicalResult {
        if failed(verify_load_store_mem_ref_layout(self, self.memref)) {
            return Err(());
        }
      
        // Checks for vector memrefs.
        let mem_elem_ty = self.memref.element_type;
        if let mem_vec_ty = mem_elem_ty.dyn_cast<Vector>() {
            if mem_vec_ty != self.value {
                return emit_op_error(
                    "Base memref and valueToStore vector types should match");
            }
            mem_elem_ty = mem_vec_ty.element_type;
        }
      
        Ok(())
    }
}

impl Fold for Store {
    fn fold(&self, results: &SmallVector<FoldResult>) -> LogicalResult {
        fold_mem_ref_cast(self)
    }
}

/**
Loads elements from memory into a vector as defined by a mask vector.

The masked load reads elements from memory into a 1-D vector as defined by a base with indices and a 1-D mask vector. When the mask is set, the element is read from memory. Otherwise, the corresponding element is taken from a 1-D pass-through vector. Informally the semantics are:

```tet
result[0] := mask[0] ? base[i+0] : pass_thru[0]
result[1] := mask[1] ? base[i+1] : pass_thru[1]
etc.
```

The masked load can be used directly where applicable, or can be used during progressively lowering to bring other memory operations closer to hardware ISA support for a masked load. The semantics of the operation closely correspond to those of the `llvm.masked.load` [intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-load-intrinsics).

# Examples

```mlir
%0 = vector.masked_load %memref[%i], %mask, %pass_thru
    : memref<?xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
```

```mlir
%1 = vector.masked_load %memref[%i, %j], %mask, %pass_thru
    : memref<?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
```
*/
#[mlir(
    assembly_format = "$memref `[` $indices `]` `,` $mask `,` $pass_thru attr-dict `:` type($memref) `,` type($mask) `,` type($pass_thru) `into` type($output)"
)]
pub struct MaskedLoad<T, const N: usize> {
    #[input(traits = [MemRead])]
    memref: MemRef<T, N>,  // renamed from `base`
    #[input]
    indices: [Index; N],
    #[input]
    mask: Vector<bool, 1>,
    #[input]
    pass_thru: Vector<T, 1>,
    #[output]
    output: Vector<T, 1>  // renamed from `result`
}

impl Verify for MaskedLoad {
    fn verify(&self) -> LogicalResult {
        if self.output.dim_size(0) != self.mask.dim_size(0) {
            return emit_op_error("Expected result dim to match mask dim");
        }
        Ok(())
    }
}

impl Fold for MaskedLoad {
    fn fold(&self) -> FoldResult {
        if fold_mem_ref_cast(self).is_ok() {
            return self.output;
        }
        FoldResult {}
    }
}

impl Canonicalise for MaskedLoad {
    fn canonicalisation_patterns(
        results: &RewritePatternSet, context: *mut MLIRContext
    ) {
        results.add<MaskedLoadFolder>(context);
    }
}

/**
Stores elements from a vector into memory as defined by a mask vector.

The masked store operation writes elements from a 1-D vector into memory as defined by a base with indices and a 1-D mask vector. When the mask is set, the corresponding element from the vector is written to memory. Otherwise, no action is taken for the element. Informally the semantics are:

```text
if (mask[0]) base[i+0] = value[0]
if (mask[1]) base[i+1] = value[1]
etc.
```

The masked store can be used directly where applicable, or can be used during progressively lowering to bring other memory operations closer to hardware ISA support for a masked store. The semantics of the operation closely correspond to those of the `llvm.masked.store` [intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-store-intrinsics).

# Examples

```mlir
vector.masked_store %memref[%i], %mask, %value
    : memref<?xf32>, vector<8xi1>, vector<8xf32>
```

```mlir
vector.masked_store %memref[%i, %j], %mask, %value
    : memref<?x?xf32>, vector<16xi1>, vector<16xf32>
```
*/
#[mlir(
    assembly_format = "$memref `[` $indices `]` `,` $mask `,` $value attr-dict `:` type($memref) `,` type($mask) `,` type($value)"
)]
pub struct MaskedStore<T, const N: usize> {
    #[input(traits = [MemWrite])]
    memref: MemRef<T, N>,  // renamed from `base`
    #[input]
    indices: [Index; N],
    #[input]
    mask: Vector<bool, 1>,
    #[input]
    value: Vector<T, 1>  // renamed from `value_to_store`
}

impl Verify for MaskedStore {
    fn verify(&self) -> LogicalResult {
        if self.value.dim_size(0) != self.mask.dim_size(0) {
            return emit_op_error("Expected valueToStore dim to match mask dim");
        }
        Ok(())
    }
}

impl Fold for MaskedStore {
    fn fold(&self, results: &SmallVector<FoldResult>) -> LogicalResult {
        fold_mem_ref_cast(self)
    }
}

impl Canonicalis for MaskedStore {
    fn canonicalisation_patterns(
        results: &RewritePatternSet, context: *mut MLIRContext
    ) {
        results.add<MaskedStoreFolder>(context);
    }
}

/**
Gathers elements from memory or ranked tensor into a vector as defined by an index vector and mask.

The gather operation gathers elements from memory or ranked tensor into a n-D vector as defined by a base with indices and an additional n-D index vector (each index is a 1-D offset on the base), but only if the corresponding bit is set in a n-D mask vector. Otherwise, the element is taken from a n-D pass-through vector. Informally the semantics are:

```text
result[0] := mask[0] ? base[index[0]] : pass_thru[0]
result[1] := mask[1] ? base[index[1]] : pass_thru[1]
etc.
```

The vector dialect leaves out-of-bounds behaviour undefined.

The gather operation can be used directly where applicable, or can be used during progressively lowering to bring other memory operations closer to hardware ISA support for a gather.

# Examples

```mlir
%0 = vector.gather %memref[%c0][%v], %mask, %pass_thru
    : memref<?xf32>, vector<2x16xi32>, vector<2x16xi1>, vector<2x16xf32> into vector<2x16xf32>
```

```mlir
%1 = vector.gather %memref[%i, %j][%v], %mask, %pass_thru
    : memref<16x16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
```
*/
#[mlir(
    assembly_format = "$memref `[` $indices `]` `[` $index_vec `]` `,` $mask `,` $pass_thru attr-dict `:` type($memref) `,` type($index_vec)  `,` type($mask) `,` type($pass_thru) `into` type($result)"
)]
pub struct Gather<T, const N: usize> {
    #[input(traits = [MemRead])]
    memref: dyn ShapedType<T, N>,  // renamed from `base`
    #[input]
    indices: [usize; N],
    #[input]
    index_vector: Vector<[AnyInteger, Index], _>,  // renamed from `index_vec`
    #[input]
    mask: Vector<bool, _>,
    #[input]
    pass_thru: Vector<_, _>,
    #[output]
    output: Vector<T, _>  // renamed from `result`
}

impl Verify for Gather {
    fn verify(&self) -> LogicalResult {
        if !self.memref.isa<MemRef, RankedTensorType>() {
            return emit_op_error(
                "Requires base to be a memref or ranked tensor type");
        }
      
        if self.output.shape != self.index_vector.shape {
            return emit_op_error("Expected result dim to match indices dim");
        }
        if self.output.shape != self.mask.shape {
            return emit_op_error("Expected result dim to match mask dim");
        }
        if self.output != self.pass_thru {
            return emit_op_error("Expected pass_thru of same type as result type");
        }
        Ok(())
    }
}

impl Canonicalise for Gather {
    fn canonicalisation_patterns(
        results: &RewritePatternSet, context: *mut MLIRContext
    ) {
        results.add<GatherFolder>(context);
    }
}

/**
Scatters elements from a vector into memory as defined by an index vector and mask.

The scatter operation scatters elements from a 1-D vector into memory as defined by a base with indices and an additional 1-D index vector, but only if the corresponding bit in a 1-D mask vector is set. Otherwise, no action is taken for that element. Informally the semantics are:

```
if (mask[0]) base[index[0]] = value[0]
if (mask[1]) base[index[1]] = value[1]
etc.
```

The vector dialect leaves out-of-bounds and repeated index behaviour undefined. Underlying implementations may enforce strict sequential semantics for the latter, though.
TODO: enforce the latter always?

The scatter operation can be used directly where applicable, or can be used during progressively lowering to bring other memory operations closer to hardware ISA support for a scatter. The semantics of the operation closely correspond to those of the `llvm.masked.scatter` [intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-scatter-intrinsics).

# Examples

```mlir
vector.scatter %memref[%c0][%v], %mask, %value
    : memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>

vector.scatter %memref[%i, %j][%v], %mask, %value
    : memref<16x16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
```
*/
#[mlir(
    assembly_format = "$memref `[` $indices `]` `[` $index_vec `]` `,` $mask `,` $value attr-dict `:` type($memref) `,` type($index_vec)  `,` type($mask) `,` type($value)"
)]
pub struct Scatter<T, const N: usize> {
    #[input(traits = [MemWrite])]
    memref: MemRef<T, N>,  // renamed from `base`
    #[input]
    indices: [usize; N],
    #[input]
    index_vector: Vector<[AnyInteger, Index], 1>, // renamed from `index_vec`
    #[input]
    mask: Vector<bool, 1>,
    #[input]
    value: Vector<T, 1>,  // renamed from `value_to_store`
}

impl Verify for Scatter {
    fn verify(&self) -> LogicalResult {
        if self.value.dim_size(0) != self.index_vector.dim_size(0) {
            return emit_op_error(
                "Expected valueToStore dim to match indices dim");
        }
        if self.value.dim_size(0) != self.mask.dim_size(0) {
            return emit_op_error("Expected valueToStore dim to match mask dim");
        }
        Ok(())
    }
}

impl Canonicalise for Scatter {
    fn canonicalisation_patterns(
        results: &RewritePatternSet, context: *mut MLIRContext
    ) {
        results.add<ScatterFolder>(context);
    }
}

/**
Reads elements from memory and spreads them into a vector as defined by a mask.

The expand load reads elements from memory into a 1-D vector as defined
by a base with indices and a 1-D mask vector. When the mask is set, the
next element is read from memory. Otherwise, the corresponding element
is taken from a 1-D pass-through vector. Informally the semantics are:

```text
index = i
result[0] := mask[0] ? base[index++] : pass_thru[0]
result[1] := mask[1] ? base[index++] : pass_thru[1]
etc.
```

Note that the index increment is done conditionally.

The expand load can be used directly where applicable, or can be used during progressively lowering to bring other memory operations closer to hardware ISA support for an expand. The semantics of the operation closely correspond to those of the `llvm.masked.expand_load` [intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-expand_load-intrinsics).

# Examples

```mlir
%0 = vector.expand_load %memref[%i], %mask, %pass_thru
    : memref<?xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
```

```mlir
%1 = vector.expand_load %memref[%i, %j], %mask, %pass_thru
    : memref<?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
```
*/
#[mlir(
    assembly_format = "$memref `[` $indices `]` `,` $mask `,` $pass_thru attr-dict `:` type($memref) `,` type($mask) `,` type($pass_thru) `into` type($result)"
)]
pub struct ExpandLoad<T, const N; usize> {
    #[input(traits = [MemRead])]
    memref: MemRef<T, N>,  // renamed from `base
    #[input]
    indices: [usize; N],
    mask: Vector<bool, 1>,
    pass_thru: Vector<T, 1>,
    #[output]
    output: Vector<T, 1>  // renamed from `result`
}

impl Verify for ExpandLoad {
    fn verify(&self) -> LogicalResult {
        if self.output.dim_size(0) != self.mask.dim_size(0) {
          return emit_op_error("Expected result dim to match mask dim");
        }
        Ok(())
    }
}

impl Canonicalise for ExpandedLoad {
    fn canonicalisation_patterns(
        results: &RewritePatternSet, context: *mut MLIRContext
    ) {
        results.add<ExpandLoadFolder>(context);
    }
}

/**
Writes elements selectively from a vector as defined by a mask.

The compress store operation writes elements from a 1-D vector into memory as defined by a base with indices and a 1-D mask vector. When the mask is set, the corresponding element from the vector is written next to memory.
Otherwise, no action is taken for the element. Informally the semantics are:

```text
index = i
if (mask[0]) base[index++] = value[0]
if (mask[1]) base[index++] = value[1]
etc.
```

Note that the index increment is done conditionally.

The compress store can be used directly where applicable, or can be used during progressively lowering to bring other memory operations closer to hardware ISA support for a compress. The semantics of the operation closely correspond to those of the `llvm.masked.compress_store` [intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-compress_store-intrinsics).

# Examples

```mlir
vector.compress_store %memref[%i], %mask, %value
    : memref<?xf32>, vector<8xi1>, vector<8xf32>
```

```mlir
vector.compress_store %memref[%i, %j], %mask, %value
    : memref<?x?xf32>, vector<16xi1>, vector<16xf32>
```
*/
#[mlir(
    assembly_format = "$memref `[` $indices `]` `,` $mask `,` $value attr-dict `:` type($memref) `,` type($mask) `,` type($value)"
)]
pub struct CompressStore<T, const N: usize> {
    #[input(traits = [MemWrite])]
    memref: MemRef<T, N>,  // renamed from `base`
    #[input]
    indices: [usize; N],
    #[input]
    mask: Vector<bool, 1>,
    #[input]
    value: Vector<T, 1>  // renamed from `value_to_store`
}

impl Verify for CompressStore {
    fn verify(&self) -> LogicalResult {
        if self.value.dim_size(0) != self.mask.dim_size(0) {
            return emit_op_error("Expected value dim to match mask dim");
        }
        Ok(())
    }
}

impl Canonicalise for CompressStore {
    fn canonicalisation_patterns(
        results: &RewritePatternSet, context: *mut MLIRContext
    ) {
        results.add<CompressStoreFolder>(context);
    }
}

/**
`vector.shape_cast` casts between vector shapes.

The `vector.shape_cast` operation casts between an n-D source vector shape and a k-D result vector shape (the element type remains the same).

If reducing rank (n > k), result dimension sizes must be a product of contiguous source dimension sizes.
If expanding rank (n < k), source dimensions must factor into a contiguous sequence of destination dimension sizes.
Each source dim is expanded (or contiguous sequence of source dims combined) in source dimension list order (i.e. 0 <= i < n), to produce a contiguous sequence of result dims (or a single result dim), in result dimension list order (i.e. 0 <= j < k). The product of all source dimension sizes and all result dimension sizes must match.

It is currently assumed that this operation does not require moving data, and that it will be folded away before lowering vector operations.

There is an exception to the folding expectation when targeting llvm.intr.matrix operations. We need a type conversion back and forth from a 2-D MLIR vector to a 1-D flattened LLVM vector.shape_cast lowering to LLVM is supported in that particular case, for now.

# Examples

Casting to a lower vector rank:

```mlir
%1 = vector.shape_cast %0 : vector<5x1x4x3xf32> to vector<20x3xf32>
```

Casting to a higher vector rank:

```mlir
%3 = vector.shape_cast %2 : vector<10x12x8xf32> to vector<5x2x3x4x8xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$source attr-dict `:` type($source) `to` type($result)"
)]
pub struct ShapeCast<T, const I: usize, const O: usize> {
    #[input]
    input: Vector<T, I>,  // renamed from `source`
    #[output]
    output: Vector<T, O>  // renamed from `result`
}

impl Verify for ShapeCast {
    fn verify(&self) -> LogicalResult {
        // Check if source/result are of vector type.
        if self.input && self.output {
            return verify_vector_shape_cast(self, self.input, self.output);
        }
        Ok(())
    }
}

impl Fold for ShapeCast {
    fn fold(&self) -> FoldResult {
        // No-op shape cast.
        if self.input == self.output {
              return self.input;
        }
      
        // Cancelling shape casts.
        if let other_op = self.input.defining_op<ShapeCast>() {
            if self.output == other_op.input {
                return other_op.input;
            }
        
            // Only allows valid transitive folding.
            let src_type = other_op.input.cast<Vector<_, _>>();
            if src_type.rank() < self.output.rank() {
                if !is_valid_shape_cast(src_type.shape, self.output.shape) {
                    return {};
                }
            } else if src_type.rank() > self.output.rank() {
                if !is_valid_shape_cast(self.output.shape, src_type.shape) {
                    return {};
                }
            } else {
                return {};
            }
        
            set_operand(other_op.input);
            return self.output;
        }
      
        // Cancelling broadcast and shape cast ops.
        if let bcast = self.input.defining_op<Broadcast>() {
            if bcast.input == get_type() {
                return bcast_op.input;
            }
        }
      
        return {};
    }
}

impl Canonicalis for ShapeCast {
    fn canonicalisation_patterns(
        results: &RewritePatternSet, context: *mut MLIRContext
    ) {
        results.add<ShapeCastConstantFolder, ShapeCastBroadcastFolder>(context);
    }
}

/**
`vector.bitcast` Bitcast casts between vectors.

The bitcast operation casts between vectors of the same rank, the minor 1-D vector size is casted to a vector with a different element type but same bitwidth. In case of 0-D vectors, the bitwidth of element types must be equal.

# Examples

Casts to a smaller element type:

```mlir
%1 = vector.bitcast %0 : vector<5x1x4x3xf32> to vector<5x1x4x6xi16>
```

Casts to a bigger element type:

```mlir
%3 = vector.bitcast %2 : vector<10x12x8xi8> to vector<10x12x2xi32>
```

Casts to an element type of the same size:

```mlir
%5 = vector.bitcast %4 : vector<5x1x4x3xf32> to vector<5x1x4x3xi32>
```

Casts of 0-D vectors:

```mlir
%7 = vector.bitcast %6 : vector<f32> to vector<i32>
```
*/
#[mlir(
    traits = [Pure],  // AllRanksMatch<["source", "result"]>
    assembly_format = "$input attr-dict `:` type($input) `to` type($output)"
)]
pub struct BitCast<T, const N: usize> {
    #[input]
    input: Vector<T, N>,  // renamed from `source`
    #[output]
    output: Vector<T, N>  // renamed from `result`
}

impl Verify for BitCast {
    fn verify(&self) -> LogicalResult {
        for i in 0..(N - 1) {
            if self.source.dim_size(i) != self.result.dim_size(i) {
                return emit_op_error("Dimension size mismatch at: {}", i);
            }
        }
      
        let data_layout = DataLayout::closest(self);
        let source_element_bits =
            data_layout.get_type_size_in_bits(self.source.element_type);
        let result_element_bits =
            data_layout.get_type_size_in_bits(self.result.element_type);
      
        if N == 0 {
            if source_element_bits != result_element_bits {
                return emit_op_error(
                    "Source/result bitwidth of the 0-D vector element types must be equal.");
            }
        } else if source_element_bits * self.input.shape.back() !=
                   result_element_bits * self.result.shape.back()
        {
            return emit_op_error(
                "Source/result bitwidth of the minor 1-D vectors must be equal.");
        }
      
        Ok(())
    }
}

impl Fold for BitCast {
    fn fold(&self) -> FoldResult {
        // Nop cast.
        if self.input == self.result {
            return self.input;
        }
      
        // Canceling bitcasts.
        if let other_op = self.input.defining_op<BitCastOp>() {
            if self.result == other_op.input {
                return other_op.input;
            }
        
            set_operand(other_op.input);
            return self.result;
        }
      
        let source_constant = self.input;
        if !source_constant {
            return {};
        }
      
        let src_elem_type = self.input.element_type;
        let dst_elem_type = self.result.element_type;
      
        if let float_pack = source_constant.dyn_cast<DenseFPElementsAttr>() {
            if float_pack.is_splat() {
                let splat = float_pack.get_splat_value<FloatAttr>();
        
                // Casting fp16 into fp32.
                if src_elem_type.is_f16() && dst_elem_type.is_f32() {
                    let bits = static_cast<u32>(
                        splat.value().bitcast_to_ap_int().get_z_ext_value());
                    // Duplicate the 16-bit pattern.
                    bits = (bits << 16) | (bits & 0xffff);
                    APInt int_bits(32, bits);
                    APFloat float_bits(llvm::APFloat::IEEEsingle(), int_bits);
                    return DenseElementsAttribute::get(self.result, float_bits);
                }
            }
        }
      
        return {};
    }
}

/**
`vector.type_cast` operation converts a scalar memref to a vector memref.

Performs a conversion from a memref with scalar element to a memref with a *single* vector element, copying the shape of the memref to the vector. This is the minimal viable operation that is required to makeke super-vectorisation operational. It can be seen as a special case of the `view` operation but scoped in the super-vectorisation context.

# Syntax

```text
operation ::= `vector.type_cast` ssa-use : memref-type to memref-type
```

# Example

```mlir
%A  = memref.alloc() : memref<5x4x3xf32>
%VA = vector.type_cast %A : memref<5x4x3xf32> to memref<vector<5x4x3xf32>>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$memref attr-dict `:` type($memref) `to` type($output)"
)]
pub struct TypeCast {
    #[input]
    memref: StaticShapeMemRefOf<[AnyType]>,
    #[output]
    output: MemRef<T, N>  // renamed from `result`
}

//   /// Build the canonical memRefType with a single vector.
//   /// E.g. memref<4 x 5 x vector<6 x f32>> -> memref<vector<4 x 5 x 6 x f32>>.
//   let builders = [Builder<(ins Value:$source)>];


impl Verify for TypeCast {
    fn verify(&self) -> LogicalResult {
        let canonical_type = canonicalise_strided_layout(self.memref);
        if !canonical_type.layout.is_identity() {
            return emit_op_error(
                "Expects operand to be a memref with identity layout.");
        }
        if !self.result.layout.is_identity() {
            return emit_op_error(
                "Expects result to be a memref with identity layout.");
        }
        if self.result.memory_space !=
            self.memref.memory_space
        {
            return emit_op_error("Expects result in same memory space");
        }
      
        if get_element_type_or_self(get_element_type_or_self(self.memref)) !=
            get_element_type_or_self(get_element_type_or_self(self.result))
        {
            return emit_op_error(
            "Expects result and operand with same underlying scalar type: {}",
            self.result
            );
        }
        if extract_shape(self.memref) != extract_shape(self.result) {
            return emit_op_error(
"Expects concatenated result and operand shapes to be equal: {}", self.result);
        }
        Ok(())
    }
}

impl ViewLikeOpInterface for TypeCast {
    fn view_source(&self) -> Value {
        self.memref
    }
}

/**
Ceates a constant vector mask.

Creates and returns a vector mask where elements of the result vector are set to '0' or '1', based on whether the element indices are contained within a hyper-rectangular region specified by the `mask_dim_sizes` array attribute argument. Each element of the 'mask_dim_sizes' array, specifies an exclusive upper bound [0, mask-dim-size-element-value) for a unique dimension in the vector result. The conjunction of the ranges define a hyper-rectangular region within which elements values are set to 1 (otherwise element values are set to 0). Each value of `mask_dim_sizes` must be non-negative and not greater than the size of the corresponding vector dimension (as opposed to vector.create_mask which allows this).

# Example

Creates a constant vector mask of size 4x3xi1 with elements in range 0 <= row <= 2 and 0 <= col <= 1 are set to 1 (others to 0).

```mlir
%1 = vector.constant_mask [3, 2] : vector<4x3xi1>

print %1
```

```text
                columns
            0    1    2
            |------------
        0 | 1    1    0
    rows  1 | 1    1    0
        2 | 1    1    0
        3 | 0    0    0
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$mask_dim_sizes attr-dict `:` type(results)"
)]
pub struct ConstantMask<const M: usize, const N: usize> {
    #[attribute]
    mask_dim_sizes: [u64; M],
    #[output]
    output: Vector<bool, N>  // renamed from `result`
}

impl Verify for ConstantMask {
    fn  verify(&self) -> LogicalResult {
        // Check the corner case of 0-D vectors first.
        if N == 0 {
            if M != 1 {
                return emit_error(
                    "Srray attr must have length 1 for 0-D vectors");
            }
            let dim = self.mask_dim_sizes[0].cast<IntegerAttribute>().get_int();
            if dim != 0 && dim != 1 {
                return emit_error(
                    "Mask dim size must be either 0 or 1 for 0-D vectors");
            }
            return Ok(())
        }
      
        // Verify that array attr size matches the rank of the vector result.
        if static_cast<i64>(M) != N {
            return emit_op_error(
                "must specify array attr of size equal vector result rank");
        }
        /*
        Verify that each array attr element is in bounds of corresponding vector result dimension size.
        */
        let result_shape = self.output.shape;
        let mut mask_dim_sizes = SmallVector::<[i64; 4]>::new();
        for (index, value) in self.mask_dim_sizes.enumerate() {
            let attr_value = value.cast<IntegerAttribute>().get_int();
            if value > result_shape[index] {
                return emit_op_error(
        "Array attr of size out of bounds of vector result dimension size");
            }
            mask_dim_sizes.push(attr_value);
        }
        /*
        Verify that if one mask dim size is zero, they all should be zero (because the mask region is a conjunction of each mask dimension interval).
        */
        let any_zeros = llvm::is_contained(mask_dim_sizes, 0);
        let all_zeros = mask_dim_sizes.all(|size| s == 0);
        if any_zeros && !all_zeros {
            return emit_op_error(
"Expected all mask dim sizes to be zeros, as a result of conjunction with zero mask dim");
        }
        /*
        Verify that if the mask type is scalable, dimensions should be zero because constant scalable masks can only be defined for the "none set" or "all set" cases, and there is no VLA way to define an "all set" case for `vector.constant_mask`. In the future, a convention could be established to decide if a specific dimension value could be considered as "all set".
        */
        if self.output.is_scalable()
        && self.mask_dim_sizes[0].cast<IntegerAttribute>().get_int() != 0
        {
            return emit_op_error(
                "Expected mask dim sizes for scalable masks to be 0.");
        }
        Ok(())
    }
}

/**
`vector.vector_mask` operation.

Creates and returns a vector mask where elements of the result vector are set to '0' or '1', based on whether the element indices are contained within a hyper-rectangular region specified by the operands. Specifically, each operand specifies a range [0, operand-value) for a unique dimension in the vector result. The conjunction of the operand ranges define a hyper-rectangular region within which elements values are set to 1 (otherwise element values are set to 0). If operand-value is negative, it is treated as if it were zero, and if it is greater than the corresponding dimension size, it is treated as if it were equal to the dimension size.

# Example

Creates a vector mask of size 4x3xi1 where elements in range 0 <= row <= 2 and 0 <= col <= 1 are set to 1 (others to 0).

```mlir
%1 = vector.create_mask %c3, %c2 : vector<4x3xi1>

print %1
```

```text
           columns
          0    1    2
        |------------
      0 | 1    1    0
rows  1 | 1    1    0
      2 | 1    1    0
      3 | 0    0    0
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$operands attr-dict `:` type(output)"
)]
pub struct CreateMask<const N: usize> {
    #[input]
    operands: [usize; _],
    #[output]
    output: Vector<bool, N>  // renamed from `result`
}

impl Verify for CreateMask {
    fn verify(&self) -> LogicalResult {
        /*
        Verify that an operand was specified for each result vector each dimension.
        */
        if N == 0 {
            if get_num_operands() != 1 {
                return emit_op_error(
                    "Must specify exactly one operand for 0-D create_mask");
            }
        } else if get_num_operands() != N {
          return emit_op_error(
              "Must specify an operand for each result vector dimension");
        }
        Ok(())
    }
}

impl Canonicalise for CreateMask {
    fn canonicalisation_patterns(
        results: &RewritePatternSet, context: *mut MLIRContext
    ) {
        results.add<CreateMaskFolder>(context);
    }
}

/**
Predicates a maskable vector operation.

The `vector.mask` is a `MaskingOpInterface` operation that predicates the execution of another operation. It takes an `i1` vector mask and an optional passthru vector as arguments.
A `vector.yield`-terminated region encloses the operation to be masked.
Values used within the region are captured from above. Only one *maskable* operation can be masked with a `vector.mask` operation at a time. An operation is *maskable* if it implements the `MaskableOpInterface`.

The vector mask argument holds a bit for each vector lane and determines which vector lanes should execute the maskable operation and which ones should not. The `vector.mask` operation returns the value produced by the masked execution of the nested operation, if any. The masked-off lanes in the result vector are taken from the corresponding lanes of the pass-thru argument, if provided, or left unmodified, otherwise.

The `vector.mask` operation does not prescribe how a maskable operation should be masked or how a masked operation should be lowered. Masking constraints and some semantic details are provided by each maskable operation through the `MaskableOpInterface`. Lowering of masked operations is implementation defined. For instance, scalarising the masked operation or executing the operation for the masked-off lanes are valid lowerings as long as the execution of masked-off lanes does not change the observable behaviour of the program.

# Examples

```mlir
%0 = vector.mask %mask {
    vector.reduction <add>, %a : vector<8xi32> into i32
} : vector<8xi1> -> i32
```

```mlir
%0 = vector.mask %mask, %passthru {
    arith.divsi %a, %b : vector<8xi32>
} : vector<8xi1> -> vector<8xi32>
```

```mlir
vector.mask %mask {
    vector.transfer_write %val, %t0[%index] : vector<16xf32>, memref<?xf32>
} : vector<16xi1>
```
*/
#[mlir(
    traits = [
        SingleBlockImplicitTerminator<"Yield">,
        RecursiveMemoryEffects, NoRegionArguments
    ]
)]
pub struct Mask<const N: usize> {
    // TODO: Support multiple results and passthru values.
    #[input]
    mask: Vector<bool, N>,
    #[input]
    passthru: Option<_>,
    #[output]
    results: Option<_>,  // renamed from `results`
    #[region]
    mask_region: SizedRegion<1>
}

//   let skipDefaultBuilders = 1;
//   let builders = [
//     Builder<(ins Value:$mask,
//                    CArg<"function_ref<void(Builder &, Location)>",
//                         "buildTerminatedBody">:$mask_region)>,
//     Builder<(ins "Type":$result_type, Value:$mask,
//                    CArg<"function_ref<void(Builder &, Location)>",
//                         "buildTerminatedBody">:$mask_region)>,
//     Builder<(ins "Type":$result_type, Value:$mask,
//                    Value:$passthru,
//                    CArg<"function_ref<void(Builder &, Location)>",
//                         "buildTerminatedBody">:$mask_region)>
//   ];

impl Mast {
    fn ensure_terminator(
        Region &region, Builder &builder, Location loc
    ) {

    }
}

impl Verify for Mask {
    fn verify(&self) -> LogicalResult {
        // Structural checks.
        let block = self.mask_region.blocks()[0];
        if block.operations().len() < 2 {
            return emit_op_error("Expects an operation to mask");
        }
        if block.operations().len() > 2 {
            return emit_op_error("Expects only one operation to mask");
        }
      
        let maskable_op = dyn_cast<MaskableOpInterface>(block[0]);
        if !maskable_op {
            return emit_op_error("Expects a maskable operation.");
        }
      
        // Result checks.
        if maskable_op.num_outputs != get_num_results() {
            return emit_op_error("Expects number of results to match maskable operation number of results");
        }
      
        if !llvm::equal(maskable_op.result_types(),  self.result_types()) {
            return emit_op_error(
                "Expects result type to match maskable operation result type");
        }
      
        // Mask checks.
        let expected_mask_type = maskable_op.get_expected_mask_type();
        if self.mast != expected_mask_type {
            return emit_op_error(
                "Expects a {} mask for the maskable operation",
                expected_mask_type
            );
        }
      
        // Passthru checks.
        let passthru = self.passthru;
        if passthru {
            if !maskable_op.supports_passthru() {
                return emit_op_error(
                    "Doesn't expect a passthru argument for this maskable operation");
            }
        
            if maskable_op.num_outputs != 1 {
                return emit_op_error("Expects result when passthru argument is provided");
            }
        
            if passthru != maskable_op.result_types()[0] {
                return emit_op_error("Expects passthru type to match result type");
            }
        }
      
        Ok(())
    }
}

impl AssemblyFormat for Mask {
    fn parse(parser: &OpAsmParser, result: &OperationState) -> ParseResult {
        // Create the op region.
        result.regions.reserve(1);
        let mask_region = *result.add_region();
    
        let &builder = parser.builder();
    
        // Parse all the operands.
        OpAsmParserUnresolvedOperand mask;
        if (parser.parse_operand(mask)){
            return Err(());}
    
        // Optional passthru operand.
        OpAsmParserUnresolvedOperand passthru;
        ParseResult parse_passthru = parser.parse_optional_comma();
        if (parse_passthru.succeeded() && parser.parse_operand(passthru)){
            return Err(());}
    
        // Parse op region.
        if (parser.parseRegion(mask_region, /*arguments=*/{}, /*argTypes=*/{})){
            return Err(());}
    
        Mask::ensure_terminator(mask_region, builder, result.location);
    
        // Parse the optional attribute list.
        if (parser.parse_optional_attr_dict(result.attributes)){
            return Err(());}
    
        // Parse all the types.
        Type mask_type;
        if (parser.parse_colon_type(mask_type)){
            return Err(());}
    
        SmallVector<Type> resultTypes;
        if (parser.parseOptionalArrowTypeList(resultTypes)){
            return Err(());}
        result.types.append(resultTypes);
    
        // Resolve operands.
        if (parser.resolve_operand(mask, mask_type, result.operands)){
            return Err(());}
    
        if (parse_passthru.succeeded()) {
            if (parser.resolve_operand(passthru, resultTypes[0], result.operands)) {
                return Err(());
            }
        }
    
        Ok(())
    }
    
    fn print(&self, p: &OpAsmPrinter) {
        p << " " << self.mask;
        if (self.passthru)
            p << ", " << self.passthru;
    
        // Print single masked operation and skip terminator.
        p << " { ";
        let single_block = &self.mask_region.blocks().front();
        if (single_block && single_block.operations().len() > 1)
            p.print_custom_or_generic_op(&single_block.front());
        p << " }";
    
        p.print_optional_attr_dict(get_operation().get_attrs());
    
        p << " : " << self.mask.get_type();
        if (num_output() > 0)
            p << " -> " << get_result_types();
    }
}

impl MaskingOpInterface for Mask {

}

/**
`vector.transpose` operation.

Takes a n-D vector and returns the transposed n-D vector defined by the permutation of ranks in the n-sized integer array attribute (in case of 0-D vectors the array attribute must be empty).
In the operation

```mlir
%1 = vector.transpose %0, [i_1, .., i_n]
    : vector<d_1 x .. x d_n x f32>
    to vector<d_trans[0] x .. x d_trans[n-1] x f32>
```

the transp array [i_1, .., i_n] must be a permutation of [0, .., n-1].

# Example

```mlir
%1 = vector.transpose %0, [1, 0] : vector<2x3xf32> to vector<3x2xf32>
```

```text
[[a, b, c],      [[a, d],
 [d, e, f]]  ->   [b, e],
                  [c, f]]
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$vector `,` $transp attr-dict `:` type($vector) `to` type($output)"
)]
pub struct Transpose<T, const N: usize> {
    #[input]
    vector: Vector<T, N>,
    #[attribute]
    transp: [u64; N],
    #[output]
    output: Vector<T, N>  // renamed from `result`
}

//   let builders = [
//     Builder<(ins Value:$vector, "[&i64]":$transp)>
//   ];


impl Transpose {
    fn get_transp(&self, results: &SmallVector<[i64]>) {
        populate_from_int64_attr_array(self.transp, results);
    }
}

impl Verify for Transpose {
    fn verify(&self) -> LogicalResult {
        let rank = N;
        let mut seen = SmallVector::<[bool; 8]>::new(rank, false);
        for (index, value) in self.transp.enumerate() {
            if value >= rank {
                return emit_op_error("Transposition index out of range: {}", value);
            }
            if seen[value] {
                return emit_op_error("Duplicate position index: {}", value);
            }
            seen[value] = true;
            if self.output.dim_size(index) != self.vector.dim_size(value) {
                return emit_op_error("Dimension size mismatch at: {}", value);
            }
        }
        Ok(())
    }
}

impl Fold for Transpose {
    fn fold(&self) -> FoldResult {
        // Eliminate splat constant transpose ops.
        if let attr = self.vactor.dyn_cast_or_null<DenseElementsAttribute>() {
            if attr.is_splat() {
                return attr.reshape(self.result);
            }
        }
      
        /*
        Eliminate identity transpose ops. This happens when the dimensions of the input vector remain in their original order after the transpose operation.
        */
        let mut transp = SmallVector<[i64; 4]>::new();
        get_transp(transp);
      
        /*
        Check if the permutation of the dimensions contains sequential values: {0, 1, 2, ...}.
        */
        for i in 0..transp.len() {
            if transp[i] != i {
                return {};
            }
        }
      
        self.vector
    }
}

impl Canonicalise for Transpose {
    fn canonicalisation_patterns(
        results: &RewritePatternSet, context: *mut MLIRContext) {
      results
          .add<FoldTransposedScalarBroadcast, TransposeFolder, FoldTransposeSplat>(
              context);
    }
}

impl VectorUnrollOpInterface for Transpose {
    fn shape_for_unroll(&self) -> Option<SmallVector<[i64; 4]>> {
        to_vector<4>(self.output.shape)
    }
}

/**
`vector.print` operation (for testing and debugging).

Prints the source vector (or scalar) to stdout in human readable format (for testing and debugging). No return value.

# Example

```mlir
%0 = arith.constant 0.0 : f32
%1 = vector.broadcast %0 : f32 to vector<4xf32>
vector.print %1 : vector<4xf32>
```

when lowered to LLVM, the vector print is unrolled into elementary printing method calls that at runtime will yield

( 0.0, 0.0, 0.0, 0.0 )

on stdout when linked with a small runtime support library, which only needs to provide a few printing methods (single value for all data types, opening/closing bracket, comma, newline).
*/
#[mlir(
    assembly_format = "$source attr-dict `:` type($source)"
)]
pub struct Print<T> {
    #[input]
    input: T  // renamed from `source`
}

impl Print {
    Type get_print_type() {
        return self.input;
    }
}

/*
----------------------------------------------------------------------
Ops used for supporting progressive lowering and conversion type changes.
The Ops are typically not used directly by higher level dialects, but are used by intra-dialect rewriting rules to bring vector operations closer to the hardware ISA.
----------------------------------------------------------------------
*/

/**
`vector.matrix_multiply` operation that operates on flattened 1-D MLIR vectors.

This is the counterpart of `llvm.matrix.multiply` in MLIR.
This may seem redundant with vector.contract but it serves the purposes of more progressive lowering and localized type conversion on the path: `vector<...x...xf32> -> vector<...xf32> -> !llvm<... x float>`.

This is the counterpart of llvm.matrix.multiply in MLIR. It serves the purposes of more progressive lowering and localized type conversion.
Higher levels typically lower matrix multiplications into 'vector.contract' operations. Subsequent rewriting rule progressively lower these operations into `vector.matrix_multiply` operations to bring the operations closer to the hardware ISA.

The `vector.matrix_multiply` op treats `lhs` as matrix with <lhs_rows> rows and <lhs_columns> columns, `rhs` as matrix with <lhs_columns> rows and <rhs_columns> and multiplies them. The result matrix is returned embedded in the result vector.

Also see:

<http://llvm.org/docs/LangRef.html#llvm-matrix-multiply-intrinsic>

# Example

```mlir
%C = vector.matrix_multiply %A, %B
    { lhs_rows = 4: i32, lhs_columns = 16: i32 , rhs_columns = 3: i32 }
    : (vector<64xf64>, vector<48xf64>) -> vector<12xf64>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res)"
)]
// TODO: tighten vector element types that make sense.
pub struct Matmul<
    T: [AnySignlessInteger, AnySignedInteger, Index, AnyFloat]
> {
    #[input]
    lhs: Vector<T, 1>,
    #[input]
    rhs: Vector<T, 1>,
    #[attribute]
    lhs_rows: u32,
    #[attribute]
    lhs_columns: u32,
    #[attribute]
    rhs_columns: u32,
    #[output]
    result: Vector<T, 1>
}

//   let builders = [
//    Builder<(ins Value:$lhs, Value:$rhs, "usize":$lhsRows,
//      "usize":$lhsColumns, "usize":$rhsColumns),
//    [{
//      $_state.addOperands({lhs, rhs});
//      $_state.addAttribute("lhs_rows",$_builder.getI32IntegerAttr(lhsRows));
//      $_state.addAttribute("lhs_columns",$_builder.getI32IntegerAttr(lhsColumns));
//      $_state.addAttribute("rhs_columns",$_builder.getI32IntegerAttr(rhsColumns));
//      $_state.addTypes(Vector::get(lhsRows * rhsColumns,
//        lhs.cast<Vector>().element_type));
//    }]>,
//   ];


/**
Vector dialect matrix tranposition op that operates on flattened 1-D MLIR vectors.

This is the counterpart of `llvm.matrix.transpose` in MLIR.
This may seem redundant with vector.transpose but it serves the purposes of more progressive lowering and localized type conversion on the path: `vector<...x...xf32> -> vector<...xf32> -> !llvm<... x float>`.

This is the counterpart of llvm.matrix.transpose in MLIR. It serves the purposes of more progressive lowering and localized type conversion.
Higher levels typically lower matrix tranpositions into 'vector.transpose' operations. Subsequent rewriting rule progressively lower these operations into `vector.flat_transpose` operations to bring the operations closer to the hardware ISA.

The `vector.flat_transpose` op treats the 1-D input `matrix` as a 2-D matrix with <rows> rows and <columns> columns, and returns the transposed matrix in flattened form in 'res'.

Also see:

<http://llvm.org/docs/LangRef.html#llvm-matrix-transpose-intrinsic>

# Example

```mlir
%1 = vector.flat_transpose %0 { rows = 4: i32, columns = 4: i32 }
    : (vector<16xf32>) -> vector<16xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$matrix attr-dict `:` type($matrix) `->` type($res)"
)]
// TODO: tighten vector element types that make sense.
pub struct FlatTranspose<
    T: [AnySignlessInteger, AnySignedInteger, Index, AnyFloat]
> {
    #[input]
    matrix: Vector<T, 1>,
    #[attribute]
    rows: u32,
    #[attribute]
    columns: u32,
    #[output]
    result: Vector<T, 1>
}

/*
----------------------------------------------------------------------
Splat
----------------------------------------------------------------------
*/

/**
Vector splat or broadcast operation.

Broadcast the operand to all elements of the result vector. The operand is required to be of integer/index/float type.

# Example

```mlir
%s = arith.constant 10.1 : f32
%t = vector.splat %s : vector<8x16xi32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$input attr-dict `:` type($output)"
)]
pub struct Splat<
    T: AnyTypeOf<[AnySignlessInteger, Index, AnyFloat],
    const N: usize
> {
    /// integer/index/float type
    #[input]
    input: T,
    #[output]
    output: Vector<T, N>  // renamed from `aggregate`
}

//   let builders = [
//     Builder<(ins Value:$element, "Type":$aggregateType),
//     [{ build($_builder, $_state, aggregateType, element); }]>];

impl Fold for Splat {
    fn fold(&self) -> FoldResult {
        if !self.input.isa_and_nonnull<IntegerAttribute, FloatAttr>() {
            return {};
        }
      
        // SplatElementsAttribute::new treats single value for second arg as being a splat.
        SplatElementsAttribute::new(get_type(), {self.input})
    }
}

/*
----------------------------------------------------------------------
VectorScale
----------------------------------------------------------------------
*/

/**
Load vector scale size.

The `vscale` op returns the scale of the scalable vectors, a positive integer value that is constant at runtime but unknown at compile-time.
The scale of the vector indicates the multiplicity of the vectors and vector operations. For example, a `vector<[4]xi32>` is equivalent to `vscale` consecutive `vector<4xi32>`; and an operation on a `vector<[4]xi32>` is equivalent to performing that operation `vscale` times, once on each `<4xi32>` segment of the scalable vector. The `vscale` op can be used to calculate the step in vector-length agnostic (VLA) loops.
Right now we only support one contiguous set of scalable dimensions, all of them grouped and scaled with the value returned by `vscale`.

TODO: In the future, we might want to have scalable vectors with different scales for different dimensions. E.g.: `vector<[16]x[16]xf32>`, in which case we might need to add an index to 'vscale' to select one of them. In order to support GPUs, we might also want to differentiate between a 'global' scale, a scale that's fixed throughout the execution, and a 'local' scale that is fixed but might vary with each call to the function. For that, it might be useful to have a 'vector.scale.global' and a 'vector.scale.local' operation.
*/
#[mlir(
    traits = [Pure],
    assembly_format = "attr-dict"
)]
pub struct VectorScale {
    #[output]
    output: usize  // renamed from `result`
}

/*
----------------------------------------------------------------------
VectorScan
----------------------------------------------------------------------
*/

/**
`vector.scan` operation.

Performs an inclusive/exclusive scan on an n-D vector along a single dimension returning an n-D result vector using the given operation (add/mul/min/max for int/fp and and/or/xor for int only) and a specified value for the initial value. The operator returns the result of scan as well as the result of the last reduction in the scan.

# Example

```mlir
%1:2 = vector.scan <add>, %0, %acc
    { inclusive = false, reduction_dim = 1 : i64 }
    : vector<4x8x16x32xf32>, vector<4x16x32xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$kind `,` $source `,` $initial_value attr-dict `:` type($source) `,` type($initial_value)"
)]
pub struct Scan<T, const N: usize, U> {
    #[attribute]
    kind: CombiningKind,
    #[input]
    source: Vector<T, N>,
    #[input]
    initial_value: Vector<U, N - 1>,
    #[attribute]
    reduction_dim: u64,
    #[attribute]
    inclusive: bool,
    #[outputs]
    dest: Vector<T, N>,
    #[outputs]
    accumulated_value: Vector<U, N - 1>
}

//   let builders = [
//     Builder<(ins Value:$source, Value:$initial_value,
//                    CombiningKind:$kind,
//                    CArg<"i64", "0">:$reduction_dim,
//                    CArg<"bool", "true">:$inclusive)>
//   ];

impl Scan {
    Vector get_accumulator_type() {
        return self.accumulated_value.cast<Vector>();
    }
}

impl Verify for Scan {
    fn verify(&self) -> LogicalResult {
        // Check reduction dimension < rank.
        if self.reduction_dim >= N {
            return emit_op_error(
                "Reduction dimension {} has to be less than {}",
                self.reduction_dim,
                N
            );
        }
      
        // Check shapes of initial value and src.
        let src_shape = self.source.shape;
        let initial_value_shapes = self.initial_value.shape;
        let expected_shape = SmallVector::<[i64]>::new();
        for i in 0..N {
            if i != self.reduction_dim {
                expected_shape.push(src_shape[i]);
            }
        }
        if !llvm::equal(initial_value_shapes, expected_shape) {
            return emit_op_error("Incompatible input/initial value shapes");
        }
      
        // Verify supported reduction kind.
        let elt_type = self.dest.element_type;
        if !is_supported_combining_kind(self.kind, elt_type) {
            return emit_op_error(
                "Unsupported reduction type {} for kind '{}'",
                elt_type,
                stringify_combining_kind(self.kind)
            );
        }
      
        Ok(())
    }
}

/**
Terminates and yields values from vector regions.

`vector.yield` yields an SSA value from the Vector dialect op region and terminates the regions. The semantics of how the values are yielded is defined by the parent operation.
If `vector.yield` has any operands, the operands must correspond to the parent operation's results.
If the parent operation defines no value the vector.yield may be omitted when printing the region.
*/
#[mlir(
    traits = [Pure, ReturnLike, Terminator],
    assembly_format = "attr-dict ($operands^ `:` type($operands))?"
)]
pub struct Yield {
    #[input]
    operands: Variadic<AnyType>
}

//   let builders = [
//     Builder<(ins), [{ /* nothing to do */ }]>,
//   ];


/**
Executes operations in the associated region on thread #0 of a SPMD program.

`warp_execute_on_lane_0` is an operation used to bridge the gap between vector programming and SPMD programming model like GPU SIMT. It allows to trivially convert a region of vector code meant to run on a multiple threads into a valid SPMD region and then allows incremental transformation to distribute vector operations on the threads.

Any code present in the region would only be executed on first thread/lane based on the `laneid` operand. The `laneid` operand is an integer ID between [0, `warp_size`). The `warp_size` attribute indicates the number of lanes in a warp.

Operands are vector values distributed on all lanes that may be used by the single lane execution. The matching region argument is a vector of all the values of those lanes available to the single active lane. The distributed dimension is implicit based on the shape of the operand and argument. the properties of the distribution may be described by extra attributes (e.g. affine map).

Return values are distributed on all lanes using laneId as index. The vector is distributed based on the shape ratio between the vector type of the yield and the result type.
If the shapes are the same this means the value is broadcasted to all lanes.
In the future the distribution can be made more explicit using affine_maps and will support having multiple Ids.

Therefore the `warp_execute_on_lane_0` operations allow to implicitly copy between lane0 and the lanes of the warp. When distributing a vector from lane0 to all the lanes, the data are distributed in a block cyclic way.
For exemple `vector<64xf32>` gets distributed on 32 threads and map to `vector<2xf32>` where thread 0 contains vector[0] and vector[1].

During lowering values passed as operands and return value need to be visible to different lanes within the warp. This would usually be done by going through memory.

The region is *not* isolated from above. For values coming from the parent region not going through operands only the lane 0 value will be accesible so it generally only make sense for uniform values.

# Examples

```mlir
// Execute in parallel on all threads/lanes.
vector.warp_execute_on_lane_0 (%laneid)[32] {
    // Serial code running only on thread/lane 0.
    ...
}
// Execute in parallel on all threads/lanes.
```

This may be lowered to an scf.if region as below:

```mlir
// Execute in parallel on all threads/lanes.
%cnd = arith.cmpi eq, %laneid, %c0 : index
scf.if %cnd {
// Serial code running only on thread/lane 0.
...
}
// Execute in parallel on all threads/lanes.
```

When the region has operands and/or return values:

```mlir
// Execute in parallel on all threads/lanes.
%0 = vector.warp_execute_on_lane_0(%laneid)[32]
args(%v0 : vector<4xi32>) -> (vector<1xf32>) {
^bb0(%arg0 : vector<128xi32>) :
    // Serial code running only on thread/lane 0.
    ...
    vector.yield %1 : vector<32xf32>
}
// Execute in parallel on all threads/lanes.
```

values at the region boundary would go through memory:

```mlir
// Execute in parallel on all threads/lanes.
...
// Store the data from each thread into memory and Synchronization.
%tmp0 = memreg.alloc() : memref<128xf32>
%tmp1 = memreg.alloc() : memref<32xf32>
%cnd = arith.cmpi eq, %laneid, %c0 : index
vector.store %v0, %tmp0[%laneid] : memref<128xf32>, vector<4xf32>
some_synchronisation_primitive
scf.if %cnd {
    // Serialised code running only on thread 0.
    // Load the data from all the threads into a register from thread 0. This
    // allow threads 0 to access data from all the threads.
    %arg0 = vector.load %tmp0[%c0] : memref<128xf32>, vector<128xf32>
    ...
    // Store the data from thread 0 into memory.
    vector.store %1, %tmp1[%c0] : memref<32xf32>, vector<32xf32>
}
// Synchronisation and load the data in a block cyclic way so that the
// vector is distributed on all threads.
some_synchronisation_primitive
%0 = vector.load %tmp1[%laneid] : memref<32xf32>, vector<32xf32>
// Execute in parallel on all threads/lanes.
```
*/
#[mlit(
    traits = [SingleBlockImplicitTerminator<"Yield">, RecursiveMemoryEffects]
)]
pub struct WarpExecuteOnLane0 {
    laneid: Index,
    warp_size: I64Attr,
    args: Variadic<AnyType>,
    #[output]
    results: Variadic<AnyType>,
    #[region]
    warp_region: SizedRegion<1>
}

//   let skipDefaultBuilders = 1;
//   let builders = [
//     Builder<(ins Value:$laneid, "i64":$warpSize)>,
//     Builder<(ins "TypeRange":$resultTypes, Value:$laneid,
//                    "i64":$warpSize)>,
//     // `blockArgTypes` are different than `args` types as they are they
//     // represent all the `args` instances visibile to lane 0. Therefore we need
//     // to explicit pass the type.
//     Builder<(ins "TypeRange":$resultTypes, Value:$laneid,
//                    "i64":$warpSize, ValueRange:$args,
//                    "TypeRange":$blockArgTypes)>
//   ];


impl WarpExecuteOnLane0 {
    bool is_defined_outside_of_region(Value value) {
        !get_region().is_ancestor(value.get_parent_region())
    }
}

impl Verify for WarpExecuteOnLane0 {

}

impl AssemblyFormat for WarpExecuteOnLane0 {

}

impl RegionBranchOpInterface for WarpExecuteOnLane0 {
    // fn are_types_compatible
}

