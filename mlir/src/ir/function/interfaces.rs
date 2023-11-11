/*!
# Utility Types for Function-Like Operations

This file defines support types for Operations that represent function-like constructs to use.

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/FunctionInterfaces.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/FunctionInterfaces.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/FunctionInterfaces.cpp>
*/

use crate::{
    mlir::ir::{
        builders,
        builtins::{
            attributes::{TypeAttribute, DictionaryAttribute},
            types,
        },
        operation::definition,
        symbol_table,
        r#type::Type,
    },
    llvm::adt::{
        bit_vector::BitVector,
        small_string
    }
};

/**
This interfaces provides support for interacting with operations that behave like functions. In particular, these operations:
  
- must be symbols, i.e. have the `Symbol` trait.
- must have a single region, that may be comprised with multiple blocks, that corresponds to the function body.
    - when this region is is_empty, the operation corresponds to an external function.
    - leading arguments of the first block of the region are treated as function arguments.
  
The function, aside from implementing the various interface methods, should have the following ODS arguments:
  
- `function_type` (required)
    - A TypeAttribute that holds the signature type of the function.
  
- `input_attrs` (optional)
    - An ArrayAttribute of DictionaryAttribute that contains attribute dictionaries for each of the function arguments.
  
- `res_attrs` (optional)
    - An ArrayAttribute of DictionaryAttribute that contains attribute dictionaries for each of the function outputs.
*/
pub trait FunctionOpInterface: OpInterface {
    /// Returns the type of the function.
    fn function_type(&self) -> Type;

    /**
    Set the type of the function. This method should perform an unsafe modification to the function type; it should not update argument or result attributes.
    */
    fn set_function_type_attr(&mut self, r#type: TypeAttribute);

    /**
    Get the array of input attribute dictionaries. The method should return
    an array attribute containing only dictionary attributes equal in number
    to the number of function inputs. Alternatively, the method can return
    null to indicate that the function has no input attributes.
    */
    fn input_attrs_attr(&self) -> ArrayAttribute;

    /**
    Get the array of output attribute dictionaries. The method should return an array attribute containing only dictionary attributes equal in number to the number of function outputs. Alternatively, the method can return null to indicate that the function has no output attributes.
    */
    fn output_attrs_attr(&self) -> ArrayAttribute;

    /**
    Set the array of input attribute dictionaries.
    */
    fn set_input_attrs_attr(&mut self, attrs: ArrayAttribute);

    /// Set the array of output attribute dictionaries.
    fn set_output_attrs_attr(&mut self, attrs: ArrayAttribute);

    /**
    Remove the array of input attribute dictionaries. This is the same as setting all input attributes to an is_empty dictionary. The method should return the removed attribute.
    */
    fn remove_input_attrs_attr(&mut self) -> Attribute;

    /**
    Remove the array of output attribute dictionaries. This is the same as setting all output attributes to an is_empty dictionary. The method should return the removed attribute.
    */
    fn remove_output_attrs_attr(&mut self) -> Attribute;

    /**
    Returns the function input types based exclusively on the type (to allow for this method may be called on function declarations).
    */
    fn input_types(&self) -> [Type];

    /**
    Returns the function output types based exclusively on the type (to allow for this method may be called on function declarations).
    */
    fn output_types(&self) -> [Type];

    /**
    Returns a clone of the function type with the given argument and result types.
  
    NOTE: The default implementation assumes the function type has an appropriate clone method:

    `fn clone(inputs: [Type] , outputs: [Type]) -> Type`
    */
    fn clone_type_with(
        &self, inputs: TypeRange, outputs: TypeRange
    ) -> Type {
        self.function_type().clone(inputs, outputs);
    }

    /**
    Verify the contents of the body of this function.
  
    NOTE: The default implementation merely checks that if the entry block exists, it has the same number and type of arguments as the function type.
    */
    fn verify_body(&self) -> LogicalResult {
        if self.is_external() {
            return Ok(());
        }
        let input_types = self.input_types();
        /*
        NOTE: This should just be self.front() but access generically because the interface methods defined here may be shadowed in arbitrary ways. <https://github.com/llvm/llvm-project/issues/54807>
        */
        let entry_block = self.region(0).front();

        let num_inputs = input_types.len();
        if entry_block.num_inputs() != num_inputs {
            return self.emit_op_error(
                "Entry block must have {} arguments to match function signature.",
                num_inputs
            );
        }

        for i in 0..input_types.len() {
            let input_type = entry_block.input(i).get_type();
            if input_types[i] != input_type {
                return self.emit_op_error(
                    "Type of entry block argument #{}({})) must match the type of the corresponding argument in {}function signature({}).",
                    i, input_type, input_types[i]
                );
            }
        }

        Ok(())
    }

    /**
    Verify the type attribute of the function for derived op-specific invariants.
    */
    fn verify_type(&self) -> LogicalResult {
        Ok(())
    }
  
    //===------------------------------------------------------------------===//
    // Name
    //===------------------------------------------------------------------===//
  
    /// Return the name of the function.
    pub fn name(&self) -> &'static str { SymbolTable::symbol_name(self) }

    // extraTraitClassDeclaration

    //===------------------------------------------------------------------===//
    // Builders
    //===------------------------------------------------------------------===//
  
    /**
    Build the function with the given name, attributes, and type. This builder also inserts an entry block into the function body with the given argument types.
    */
    // static
    fn build_with_entry_block(
        builder: &Builder,
        state: &OperationState,
        name: &str,
        r#type:Type,
        attrs: &[NamedAttribute],
        input_types: TypeRange
    ) {
        state.add_attribute(SymbolTable::symbol_attr_name(),
                            builder.get_string_attr(name));
        state.add_attribute(ConcreteOp::function_type_attr_name(state.name),
                            TypeAttribute::new(r#type));
        state.attributes.append(attrs);

        // Add the function body.
        let body_region = state.add_region();
        let body = Block::new();
        body_region.push(body);
        for input in input_types {
            body.add_argument(input, state.location);
        }
    }

    // extraSharedClassDeclaration
    
    // /// Block list iterator types.
    // using BlockListType = Region::BlockListType;
    // using iterator = BlockListType::iterator;
    // using reverse_iterator = BlockListType::reverse_iterator;

    // /// Block argument iterator types.
    // using BlockArgListType = Region::BlockArgListType;
    // using args_iterator = BlockArgListType::iterator;

    //===------------------------------------------------------------------===//
    // Body Handling
    //===------------------------------------------------------------------===//

    /// Returns true if this function is external, i.e. it has no body.
    pub fn is_external(&self) -> bool { self.is_empty() }

    /// Return the region containing the body of this function.
    pub fn function_body(&self) -> &Region { self.region(0) }

    /// Delete all blocks from this function.
    pub fn erase_body(&mut self) {
        self.function_body().drop_all_references();
        self.function_body().blocks().clear();
    }

    /// Return the list of blocks within the function body.
    pub fn blocks(&self) -> &BlockListType {
        self.function_body().blocks()
    }

    // iterator begin() { return self.function_body().begin(); }
    // iterator end() { return self.function_body().end(); }
    // reverse_iterator rbegin() { return self.function_body().rbegin(); }
    // reverse_iterator rend() { return self.function_body().rend(); }

    /// Returns true if this function has no blocks within the body.
    pub fn is_empty(&self) -> bool {
        self.function_body().is_empty()
    }

    /// Push a new block to the back of the body region.
    pub fn push(&mut self, block: *mut Block) {
        self.function_body().push(block)
    }
  
    /// Push a new block to the front of the body region.
    pub fn push_front(block: *mut Block) { self.function_body().push_front(block); }

    /// Return the last block in the body region.
    pub fn back(&self) -> &Block {
        self.function_body().back()
    }

    /// Return the first block in the body region.
    pub fn front(&self) -> &Block {
        self.function_body().front()
    }

    /**
    Add an entry block to an is_empty function, and set up the block arguments to match the signature of the function. The newly inserted entry block is returned.
    */
    fn add_entry_block(&mut self) -> *mut Block {
        assert!(self.is_empty(), "Function already has an entry block");
        let entry = Block::new();
        self.push(entry);

        /*
        FIXME: Allow for passing in locations for these arguments instead of using the operations location.
        */
        let input_types = self.input_types();
        let locations = SmallVector<[Location]>::new(
            input_types.len(), self.get_operation().location);
        entry.add_arguments(input_types, locations);
        entry
    }

    /**
    Add a normal block to the end of the function's block list. The function should at least already have an entry block.
    */
    fn add_block(&mut self) -> *mut Block {
        assert!(
            !self.is_empty(),
            "Function should at least have an entry block");
        self.push(Block::new());
        return &self.back();
    }
  
    //===------------------------------------------------------------------===//
    // Type Attribute Handling
    //===------------------------------------------------------------------===//

    /**
    Change the type of this function in place. This is an extremely dangerous operation and it is up to the caller to ensure that this is legal for this function, and to restore invariants:

    - the entry block args must be updated to match the function params.
    - the argument/result attributes may need an update: if the new type has less parameters we drop the extra attributes, if there are more parameters they won't have any attributes.
    */
    pub fn set_type(&mut self, new_type: Type) {
        function_interface_impl::set_function_type(self.get_operation(), new_type);
    }

    //===------------------------------------------------------------------===//
    // Argument and Result Handling
    //===------------------------------------------------------------------===//

    /// Returns the number of function arguments.
    pub fn num_inputs(&self) -> usize {
        self.input_types().len()
    }

    /// Returns the number of function outputs.
    pub fn num_outputs(&self) -> usize {
        self.output_types().len()
    }

    /// Returns the entry block function argument at the given index.
    pub fn input(&self, index: usize) -> BlockArgument {
        self.function_body().input(index)
    }

    /// Support argument iteration.
    // args_iterator args_begin() { return self.function_body().args_begin(); }
    // args_iterator args_end() { return self.function_body().args_end(); }
    pub fn inputs(&self) -> BlockArgListType {
        self.function_body().inputs()
    }

    /**
    Insert a single argument of type `input_type` with attributes `input_attrs` and location `input_location` at `input_index`.
    */
    pub fn insert_argument(
        &mut self,
        input_index: usize,
        input_type: Type,
        input_attrs: DictionaryAttribute,
        input_location: Location
    ) {
        self.insert_arguments({input_index}, {input_type}, {input_attrs}, {input_location});
    }

    /**
    Inserts arguments with the listed types, attributes, and locations at the listed indices. `input_indices` must be sorted. Arguments are inserted in the order they are listed, such that arguments with identical index will appear in the same order that they were listed here.
    */
    pub fn insert_arguments(
        &mut self,
        input_indices: &[usize],
        input_types: TypeRange,
        input_attrs: &[DictionaryAttribute],
        input_locations: &[Location]
    ) {
        let original_num_inputs = self.num_inputs();
        let new_type = self.type_with_inputs_and_outputs(
            input_indices, input_types, /*output_indices=*/{}, /*output_types=*/{});
        function_interface_impl::insert_function_inputs(
            self.get_operation(), input_indices, input_types, input_attrs, input_locations,
            original_num_inputs, new_type);
    }

    /// Insert a single result of type `output_type` at `output_index`.
    pub fn insert_output(
        output_index: usize,
        output_type: Type,
        output_attrs: DictionaryAttribute
    ) {
        self.insert_outputs({output_index}, {output_type}, {output_attrs});
    }
  
    /**
    Inserts outputs with the listed types at the listed indices.
    `output_indices` must be sorted. Results are inserted in the order they are listed, such that outputs with identical index will appear in the same order that they were listed here.
    */
    pub fn insert_outputs(
        &mut self,
        output_indices: &[usize],
        output_types: TypeRange,
        output_attrs: &[DictionaryAttribute]
    ) {
        let original_num_outputs = self.num_outputs();
        let new_type = self.type_with_inputs_and_outputs(
            /*input_indices=*/{}, /*input_types=*/{}, output_indices, output_types);
        function_interface_impl::insert_function_outputs(
            self.get_operation(), output_indices, output_types, output_attrs,
            original_num_outputs, new_type);
    }

    /// Erase a single argument at `input_index`.
    pub fn erase_input(&mut self, input_index: usize) {
        let inputs_to_erase = BitVector::new(self.num_inputs());
        inputs_to_erase.set(input_index);
        erase_inputs(inputs_to_erase);
    }

    /// Erases the arguments listed in `input_indices`.
    pub fn erase_inputs(&mut self, input_indices: &BitVector) {
        let new_type = self.type_without_inputs(input_indices);
        function_interface_impl::erase_function_inputs(
            self.get_operation(), input_indices, new_type);
    }

    /// Erase a single result at `output_index`.
    pub fn erase_output(&mut self, output_index: usize) {
        let outputs_to_erase = BitVector::new(self.num_outputs());
        outputs_to_erase.set(output_index);
        self.erase_outputs(outputs_to_erase);
    }

    /// Erases the results listed in `output_indices`.
    pub fn erase_outputs(&mut self, output_indices: &BitVector) {
        let new_type = self.type_without_outputs(output_indices);
        function_interface_impl::erase_function_outputs(
            self.get_operation(), output_indices, new_type);
    }
  
    /**
    Return the type of this function with the specified arguments and results inserted. This is used to update the function's signature in the `insert_arguments` and `self.insert_outputs` methods. The arrays must be sorted by increasing index.
    */
    pub fn type_with_inputs_and_outputs(
        &self,
        input_indices: &[usize],
        input_types: TypeRange,
        output_indices: &[usize],
        output_types: TypeRange
    ) -> Type {
        let input_storage = SmallVector::<[Type]>::new();
        let output_storage = SmallVector::<[Type]>::new();
        let new_input_types = function_interface_impl::insert_types_into(
            self.input_types(), input_indices, input_types, input_storage);
        let new_output_types = function_interface_impl::insert_types_into(
            self.output_types(), output_indices, output_types, output_storage);
        self.clone_type_with(new_input_types, new_output_types)
    }

    /**
    Return the type of this function without the specified arguments and results. This is used to update the function's signature in the `erase_inputs` and `self.erase_outputs` methods.
    */
    pub fn type_without_inputs_and_outputs(
        &self,
        input_indices: &BitVector,
        output_indices: &BitVector
    ) -> Type {
        let input_storage = SmallVector::<[Type]>::new();
        let output_storage = SmallVector::<[Type]>::new();
        let new_input_types = function_interface_impl::filter_types_out(
            self.input_types(), input_indices, input_storage);
        let new_output_types = function_interface_impl::filter_types_out(
            self.output_types(), output_indices, output_storage);
        return self.clone_type_with(new_input_types, new_output_types);
    }

    pub fn type_without_inputs(&self, input_indices: &BitVector) -> Type {
        let mut input_storage = SmallVector::<[Type]>::new();
        let new_input_types = function_interface_impl::filter_types_out(
            self.input_types(), input_indices, input_storage);
        self.clone_type_with(new_input_types, self.output_types())
    }

    pub fn type_without_outputs(&self, output_indices: &BitVector) -> Type {
        let mut output_storage = SmallVector::<[Type]>::new();
        let new_output_types = function_interface_impl::filter_types_out(
            self.output_types(), output_indices, output_storage);
        self.clone_type_with(self.input_types(), new_output_types)
    }

    //===------------------------------------------------------------------===//
    // Argument Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the argument at 'index'.
    pub fn input_attrs(&self, index: usize) -> ArrayRef<NamedAttribute> {
        function_interface_impl::input_attrs(self.get_operation(), index)
    }

    /**
    Return an ArrayAttribute containing all argument attribute dictionaries of this function, or null() if no arguments have attributes.
    */
    pub fn all_input_attrs(&self) -> ArrayAttribute { self.arg_attrs_attr() }
  
    /// Return all argument attributes of this function.
    pub fn all_input_attrs(
        &self,
        result: &SmallVector<[DictionaryAttribute]>
    ) {
        if let input_attrs = self.all_input_attrs() {
            let input_attr_range = input_attrs. get_as_range::<DictionaryAttribute>();
            result.append(input_attr_range);
        } else {
            result.append(
                self.num_inputs(),
                DictionaryAttribute::new(self.get_operation().context()));
        }
    }

    /**
    Return the specified attribute, if present, for the argument at 'index', null otherwise.
    */
    pub fn input_attr(&self, index: usize, name: StringAttribute) -> Attribute {
        let input_dict = self.input_attr_dict(index);
        if input_dict { input_dict.get(name) } else { null() }
    }
    pub fn input_attr(&self, index: usize, name: &str) -> Attribute {
        let input_dict = self.input_attr_dict(index);
        if input_dict { input_dict.get(name) } else { null() }
    }

    pub fn input_attr_of_type<AttrClass>(
        &self,
        index: usize,
        name: StringAttribute
    ) -> AttrClass {
        self.input_attr(index, name).dyn_cast_or_null<AttrClass>()
    }

    pub fn input_attr_of_type<AttrClass>(
        &self,
        index: usize,
        name: &str
    ) -> AttrClass {
        self.input_attr(index, name).dyn_cast_or_null<AttrClass>()
    }

    /// Set the attributes held by the argument at 'index'.
    pub fn set_input_attrs(
        &mut self, index: usize, attributes: &[NamedAttribute]
    ) {
        function_interface_impl::set_input_attrs(self, index, attributes);
    }
  
    /**
    Set the attributes held by the argument at 'index'. `attributes` may be null, in which case any existing argument attributes are removed.
    */
    pub fn set_input_attrs(
        &mut self,
        index: usize,
        attributes: DictionaryAttribute
    ) {
        function_interface_impl::set_input_attrs(self, index, attributes);
    }

    pub fn set_all_input_attrs(
        &mut self, attributes: &[DictionaryAttribute]
    ) {
        assert!(attributes.len() == self.num_inputs());
        function_interface_impl::set_all_input_attr_dicts(self.get_operation(), attributes);
    }

    pub fn set_all_input_attrs(attributes: &[Attribute]) {
        assert!(attributes.len() == self.num_inputs());
        function_interface_impl::set_all_input_attr_dicts(self.get_operation(), attributes);
    }

    pub fn set_all_input_attrs(&mut self, attributes: ArrayAttribute) {
        assert!(attributes.len() == self.num_inputs());
        self.set_arg_attrs_attr(attributes);
    }

    /**
    If the an attribute exists with the specified name, change it to the new value. Otherwise, add a new attribute with the specified name/value.
    */
    pub fn set_input_attr(index: usize, name: StringAttribute, value: Attribute) {
        function_interface_impl::set_input_attr(self, index, name, value);
    }
    pub fn set_input_attr(index: usize, name: &str, value: Attribute) {
        set_input_attr(index,
                StringAttribute::new(self.get_operation().context(), name),
                value);
    }

    /**
    Remove the attribute 'name' from the argument at 'index'. Return the attribute that was erased, or null() if there was no attribute with such name.
    */
    pub fn remove_input_attr(
        &mut self,
        index: usize, name: StringAttribute
    ) -> Attribute {
        function_interface_impl::remove_input_attr(self, index, name)
    }
    pub fn remove_input_attr(
        &mut self, index: usize, name: &str
    ) -> Attribute {
        remove_input_attr(
            index, StringAttribute::new(self.get_operation().context(), name))
    }
  
    //===------------------------------------------------------------------===//
    // Result Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the result at 'index'.
    pub fn output_attrs(index: usize) -> ArrayRef<NamedAttribute> {
        function_interface_impl::output_attrs(self.get_operation(), index)
    }
  
    /**
    Return an ArrayAttribute containing all result attribute dictionaries of this function, or null() if no result have attributes.
    */
    pub fn all_output_attrs(&self) -> ArrayAttribute {
        self.output_attrs_attr()
    }
  
    /// Return all result attributes of this function.
    pub fn all_output_attrs(
        &self, result: &SmallVector<[DictionaryAttribute]>
    ) {
        if let input_attrs = self.all_output_attrs() {
            let input_attr_range = input_attrs. get_as_range::<DictionaryAttribute>();
            result.append(input_attr_range);
        } else {
            result.append(
                self.num_outputs(),
                DictionaryAttribute::new(self.get_operation().context()));
        }
    }
  
    /**
    Return the specified attribute, if present, for the result at 'index', null otherwise.
    */
    pub fn output_attr(index: usize, name: StringAttribute) -> Attribute {
        let input_dict = self.output_attr_dict(index);
        if input_dict { input_dict.get(name) } else { null() }
    }

    pub fn output_attr(index: usize, name: &str) -> Attribute {
        let input_dict = self.output_attr_dict(index);
        if input_dict { input_dict.get(name) } else { null() }
    }

    pub fn output_attr_of_type<AttrClass>(
        index: usize, name: StringAttribute
    ) -> AttrClass {
        output_attr(index, name).dyn_cast_or_null<AttrClass>()
    }

    pub fn output_attr_of_type<AttrClass>(
        &self, index: usize, name: &str
    ) -> AttrClass {
        output_attr(index, name).dyn_cast_or_null<AttrClass>()
    }
  
    /// Set the attributes held by the result at 'index'.
    pub fn set_result_attrs(index: usize, attributes: &[NamedAttribute]) {
        function_interface_impl::set_result_attrs(self, index, attributes);
    }

    /**
    Set the attributes held by the result at 'index'. `attributes` may be
    null, in which case any existing argument attributes are removed.
    */
    pub fn set_result_attrs(index: usize, attributes: DictionaryAttribute) {
        function_interface_impl::set_result_attrs(self, index, attributes);
    }

    pub fn set_all_result_attrs(attributes: &[DictionaryAttribute]) {
        assert!(attributes.len() == self.num_outputs());
        function_interface_impl::set_all_result_attr_dicts(
            self.get_operation(), attributes);
    }

    pub fn set_all_result_attrs(attributes: &[Attribute]) {
        assert!(attributes.len() == self.num_outputs());
        function_interface_impl::set_all_result_attr_dicts(
            self.get_operation(), attributes);
    }

    pub fn set_all_result_attrs(attributes: ArrayAttribute) {
        assert!(attributes.len() == self.num_outputs());
        self.set_res_attrs_attr(attributes);
    }

    /**
    If the an attribute exists with the specified name, change it to the new value. Otherwise, add a new attribute with the specified name/value.
    */
    pub fn set_output_attr(index: usize, name: StringAttribute, value: Attribute) {
        function_interface_impl::set_output_attr(self, index, name, value);
    }

    pub fn set_output_attr(index: usize, name: &str, value: Attribute) {
        set_output_attr(
            index,
            StringAttribute::new(self.get_operation().context(), name),
            value);
    }
  
    /**
    Remove the attribute 'name' from the result at 'index'. Return the attribute that was erased, or null() if there was no attribute with such name.
    */
    pub fn remove_result_attr(
        &mut self, index: usize, name: StringAttribute
    ) -> Attribute {
        function_interface_impl::remove_result_attr(self, index, name)
    }

    /**
    Returns the dictionary attribute corresponding to the argument at 'index'. If there are no argument attributes at 'index', a null attribute is returned.
    */
    pub fn input_attr_dict(&self, index: usize) -> DictionaryAttribute {
        assert!(index < self.num_inputs(), "Invalid argument number");
        function_interface_impl::input_attr_dict(self.get_operation(), index)
    }
  
    /**
    Returns the dictionary attribute corresponding to the result at 'index'.
    If there are no result attributes at 'index', a null attribute is
    returned.
    */
    pub fn output_attr_dict(&self, index: usize) -> DictionaryAttribute {
        assert!(index < self.num_outputs(), "Invalid result number");
        function_interface_impl::output_attr_dict(self.get_operation(), index)
    }
  
    // let verify = "return function_interface_impl::verifyTrait(cast<ConcreteOp>(self));";
}
