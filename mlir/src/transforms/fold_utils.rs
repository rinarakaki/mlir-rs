/*!
# Operation Fold Utilities

This header file declares various operation folding utilities. These utilities are intended to be used by passes to unify and simply their logic.

This file defines various operation fold utilities. These utilities are intended to be used by passes to unify and simply their logic.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Transforms/FoldUtils.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Transforms/Utils/FoldUtils.cpp>
*/

use crate::{
    mlir::{
        interfaces::fold_interfaces::DialectFoldInterface,
        ir::{
            attribute::Attribute,
            block::Block,
            builders::Builder,
            matchers::{m_constant, match_pattern},
            dialect::{
                Dialect,
                interface::DialectInterfaceCollection,
            },
            location::Location,
            mlir_context::MLIRContext,
            operation::{
                Operation,
                definition::{IsIsolatedFromAbove, FoldResult}
            },
            region::Region,
            r#type::Type,
            value::Value
        },
        support::logical_result::LogicalResult
    },
    llvm::adt::{
        dense_map::DenseMap,
        small_vector::SmallVector
    }
};

/**
A utility class for folding operations, and unifying duplicated constants generated along the way.
*/
pub struct OperationFolder {
    /**
    A mapping between an insertion region and the constants that have been created within it.
    */
    fold_scopes: DenseMap<*mut Region, ConstantMap>,

    /**
    This map tracks all of the dialects that an operation is referenced by; given that many dialects may generate the same constant.
    */
    referenced_dialects:
        DenseMap<*mut Operation, SmallVector<[*mut Dialect; 2]>>,

    /// A collection of dialect folder interfaces.
    interfaces: DialectInterfaceCollection<dyn DialectFoldInterface>
}

impl OperationFolder {
    pub fn new(context: *mut MLIRContext) -> Self {
        Self {
            interfaces: context,
            ..Default::default()
        }
    }

    /**
    Tries to perform folding on the given `operation`, including unifying deduplicated constants. If successful, replaces `operation`'s uses with folded results, and returns success. `pre_replace_action` is invoked on `operation` before it is replaced. 'process_generated_constants' is invoked for any new operations generated when folding. If the operation was completely folded it is erased. If it is just updated in place, `in_place_update` is set to true.
    */
    pub fn try_to_fold(
        &self,
        operation: *mut Operation,
        process_generated_constants: fn(*mut Operation),  // function_ref, // = nullptr,
        pre_replace_action: fn(*mut Operation),  // function_ref, // = nullptr,
        in_place_update: *mut bool  // = nullptr
    ) -> LogicalResult
    {
        if !in_place_update.is_null() {
            unsafe {
                *in_place_update = false;
            }
        }
            
        /*
        If this is a unique'd constant, return failure as we know that it has already been folded.
        */
        if self.is_folder_owned_constant(operation) {
            /*
            Check to see if we should rehoist, i.e. if a non-constant operation was inserted before this one.
            */
            let block = operation.block();
            if
                !block.is_null()
                && block.front() != operation
                && !self.is_folder_owned_constant(operation.prev_node())
            {
                operation.move_before(&block.front());
            }
            return Err(());
        }

        // Try to fold the operation.
        let results = SmallVector::<[Value; 8]>::new();
        let builder = Builder::new(operation);
        self.try_to_fold(
            builder, operation, results, process_generated_constants)?;

        // Check to see if the operation was just updated in place.
        if results.is_empty() {
            if !in_place_update.is_null() {
                unsafe {
                    *in_place_update = true;
                }
            }
            
            return Ok(());
        }

        /*
        Constant folding succeeded. We will start replacing this operation's uses and erase this operation. Invoke the callback provided by the caller to perform any pre-replacement action.
        */
        if !pre_replace_action.is_null() {
            (*pre_replace_action)(operation);
        }
            

        // Replace all of the result values and erase the operation.
        for i in 0..results.len() {
            operation.output(i).replace_all_uses_with(results[i]);
        }
            
        operation.erase();
        Ok(())
    }

    /**
    Tries to fold a pre-existing constant operation. `const_value` represents the value of the constant, and can be optionally passed if the value is already known (e.g. if the constant was discovered by m_constant). This is purely an optimisation opportunity for callers that already know the value of the constant. Returns false if an existing constant for `operation` already exists in the folder, in which case `operation` is replaced and erased.
    Otherwise, returns true and `operation` is inserted into the folder (and hoisted if necessary).
    */
    pub fn insert_known_constant(
        &self,
        operation: *mut Operation,
        const_value: impl Attribute  // = {}
    ) -> bool
    {
        let block = operation.block();

        /*
        If this is a constant we unique'd, we don't need to insert, but we can check to see if we should rehoist it.
        */
        if self.is_folder_owned_constant(operation) {
            if
                &block.first() != operation
                && !self.is_folder_owned_constant(operation.prev_node())
            {
                operation.move_before(&block.first());
            }
            return true;
        }

        // Get the constant value of the operation if necessary.
        if !const_value {
            match_pattern(&operation, m_constant(&const_value));
            assert!(const_value, "Expected `operation` to be a constant");
        } else {
            // TODO
        }

        // Check for an existing constant operation for the attribute value.
        let insert_region = insertion_region(&self.interfaces, &block);
        let uniqued_constants = self.fold_scopes[insert_region];
        let folder_const_op = uniqued_constants[
            (operation.dialect(), const_value, *operation.result_type_begin())
        ];

        // If there is an existing constant, replace `operation`.
        if folder_const_op {
            operation.replace_all_uses_with(folder_const_op);
            operation.erase();
            return false;
        }

        /*
        Otherwise, we insert `operation`. If `operation` is in the insertion block and is either already at the front of the block, or the previous operation is already a constant we unique'd (i.e. one we inserted), then we don't need to do anything. Otherwise, we move the constant to the insertion block.
        */
        let insert_block = &insert_region.front();
        if  
            block != insert_block
            || (
                &insert_block.front() != operation
                && !self.is_folder_owned_constant(operation.prev_node())
            )
        {
            operation.move_before(&insert_block.front());
        }

        folder_const_op = operation;
        self.referenced_dialects[operation].push(operation.dialect());
        return true;
    }
    
    /**
    Notifies that the given constant `operation` should be remove from this OperationFolder's internal bookkeeping.

    NOTE: this method must be called if a constant operation is to be deleted externally to this OperationFolder. `operation` must be a constant operation.
    */
    pub fn notify_removal(&self, operation: *mut Operation) {
        // Check to see if this operation is uniqued within the folder.
        let dialects = match self.referenced_dialects.find(operation) {
            None => return,
            Some(dialects) => dialects
        };

        /*
        Get the constant value for this operation, this is the value that was used to unique the operation internally.
        */
        let const_value = Attribute::new();
        match_pattern(operation, m_constant(&const_value));
        assert!(const_value);

        // Get the constant map that this operation was uniqued in.
        let uniqued_constants =
            self.fold_scopes[insertion_region(&self.interfaces, &operation.block())];

        // Erase all of the references to this operation.
        let r#type = operation.output(0).r#type();
        for dialect in dialects {
            uniqued_constants.erase((dialect, const_value, r#type));
        }
        self.referenced_dialects.erase(dialects);
    }

    /// Clear out any constants cached inside of the folder.
    pub fn clear(&self) {
        self.fold_scopes.clear();
        self.referenced_dialects.clear();
    }

    /**
    Get or create a constant using the given builder. On success this returns the constant operation, nullptr otherwise.
    */
    pub fn get_or_create_constant(
        &self,
        builder: &Builder,
        dialect: *mut Dialect,
        value: impl Attribute,
        r#type: impl Type,
        location: Location
    ) -> *mut Value
    {

        let fold_guard = Builder::InsertionGuard::new(builder);

        /*
        Use the builder insertion block to find an insertion point for the constant.
        */
        let insert_region =
            insertion_region(&self.interfaces, builder.insertion_block());
        let entry = insert_region.first();
        builder.set_insertion_point(&entry, entry.begin());
      
        // Get the constant map for the insertion region of this operation.
        let uniqued_constants = self.fold_scopes[insert_region];
        let const_op = self.try_get_or_create_constant(
            uniqued_constants, dialect, builder, value, r#type, location);
        match const_op {
            None => null(),
            Some(operation) => operation.output(0)
        }
    }

    /**
    Returns true if the given operation is an already folded constant that is owned by this folder.
    */
    const fn is_folder_owned_constant(
        &self,
        operation: *mut Operation
    ) -> bool
    {
        self.referenced_dialects.count(operation)
    }

    /**
    Try to process a set of fold results, generating constants as necessary.
    Populates `results` on success, otherwise leaves it unchanged.
    */
    fn process_fold_results(
        &self,
        builder: &Builder,
        operation: *mut Operation,
        results: &SmallVector<[Value]>,
        fold_results: &[FoldResult],
        process_generated_constants: fn(*mut Operation)  // function_ref
    ) -> LogicalResult
    {
        // Check to see if the operation was just updated in place.
        if fold_results.is_empty() {
            return Ok(());
        }
        assert!(fold_results.len() == operation.num_outputs());

        // Create a builder to insert new operations into the entry block of the
        // insertion region.
        let insert_region =
            insertion_region(&self.interfaces, builder.insertion_block());
        let entry = insert_region.front();
        let fold_guard = Builder::InsertionGuard::new(builder);
        builder.set_insertion_point(&entry, entry.begin());

        // Get the constant map for the insertion region of this operation.
        let uniqued_constants = self.fold_scopes[insert_region];

        // Create the result constants and replace the results.
        let dialect = operation.dialect();
        for i in 0..operation.num_outputs() {
            assert!(!fold_results[i].is_null(), "Expected valid FoldResult");

            // Check if the result was an SSA value.
            let repl = fold_results[i].dyn_cast::<Value>();
            if repl {
                if repl.r#type() != operation.output(i).r#type() {
                    results.clear();
                    return Err(());
                }
                results.emplace_back(repl);
                continue;
            }

            // Check to see if there is a canonicalised version of this constant.
            let res = operation.output(i);
            let attr_repl = fold_results[i].get::<Attribute>();
            let const_op =
                self.try_get_or_create_constant(
                    uniqued_constants,
                    dialect,
                    builder,
                    attr_repl,
                    res.r#type(),
                    operation.location());
            if const_op {
                /*
                Ensure that this constant dominates the operation we are replacing it with. This may not automatically happen if the operation being folded was inserted before the constant within the insertion block.
                */
                let op_block = operation.block();
                if 
                    op_block == const_op.block()
                    && &op_block.first() != const_op
                {
                    const_op.move_before(&op_block.first());
                }

                results.push(const_op.output(0));
                continue;
            }
            // If materialization fails, cleanup any operations generated for the
            // previous results and return failure.
            for operation in llvm::make_early_inc_range(
                    llvm::make_range(entry.begin(), builder.insertion_point()))
            {
                self.notify_removal(&operation);
                operation.erase();
            }
            results.clear();

            Err(())
        }

        // Process any newly generated operations.
        if process_generated_constants {
            for i in entry.begin()..builder.insertion_point() {
                process_generated_constants(&*i);
            }
        }

        Ok(())
    }

    /**
    Try to get or create a new constant entry. On success this returns the constant operation, nullptr otherwise.
    */
    fn  try_get_or_create_constant(
        &self,
        uniqued_constants: &ConstantMap,
        dialect: *mut Dialect,
        builder: &Builder,
        value: impl Attribute,
        r#type: impl Type,
        location: Location
    ) -> *mut Operation
    {
        // Check if an existing mapping already exists.
        let const_key = (dialect, value, r#type);
        let const_op = uniqued_constants[const_key];
        if const_op {
            return const_op;
        }

        // If one doesn't exist, try to materialize one.
        if !(
            const_op = materialise_constant(dialect, builder, value, r#type, location)
        ) {
            return None;
        }

        // Check to see if the generated constant is in the expected dialect.
        let new_dialect = const_op.dialect();
        if new_dialect == dialect {
            self.referenced_dialects[const_op].push(dialect);
            return const_op;
        }

        /*
        If it isn't, then we also need to make sure that the mapping for the new dialect is valid.
        */
        let new_key = (new_dialect, value, r#type);

        /*
        If an existing operation in the new dialect already exists, delete the materialised operation in favour of the existing one.
        */
        let existing_op = uniqued_constants.lookup(new_key);
        if existing_op {
            const_op.erase();
            self.referenced_dialects[existing_op].push(dialect);
            return const_op = existing_op;
        }

        // Otherwise, update the new dialect to the materialized operation.
        self.referenced_dialects[const_op].assign(dialect, new_dialect);
        let new_it = uniqued_constants.insert(new_key, const_op);
        return new_it.first.second;
    }
}

/**
This map keeps track of uniqued constants by dialect, attribute, and type.
A constant operation materializes an attribute with a type. Dialects may generate different constants with the same input attribute and type, so we also need to track per-dialect.
*/
type ConstantMap =
    DenseMap<(*mut Dialect, dyn Attribute, dyn Type), *mut Operation>;

/**
Given an operation, find the parent region that folded constants should be inserted into.
*/
// static
fn insertion_region(
    interfaces: &DialectInterfaceCollection<dyn DialectFoldInterface>,
    mut insertion_block: *mut Block
) -> Region
{
    while let Some(region) = insertion_block.parent() {
        /*
        Insert in this region for any of the following scenarios:
         
        - The parent is unregistered, or is known to be isolated from above.
        - The parent is a top-level operation.
        */
        if
            let Some(parent_operation) = region.parent_op()
            && (
                parent_operation.might_have_trait::<IsIsolatedFromAbove>()
                || !parent_operation.block()
            )
        {
            return region;
        }

        // Otherwise, check if this region is a desired insertion region.
        if 
            let Some(interface) = interfaces.interface_for(parent_operation)
            && interface.should_materialise_into(region)
        {
            return region;
        }
        

        // Traverse up the parent looking for an insertion region.
        insertion_block = parent_operation.block();
    }
    unreachable!("Expected valid insertion region.");
}

/**
A utility function used to materialize a constant for a given attribute and
type. On success, a valid constant value is returned. Otherwise, null is
returned
*/
// static 
fn materialise_constant(
    dialect: *mut Dialect,
    builder: &Builder,
    value: dyn Attribute,
    r#type: dyn Type,
    location: Location
) -> *mut Operation
{
    let insert_pt = builder.insertion_point();

    // Ask the dialect to materialise a constant operation for this value.
    let const_op
        = dialect.materialise_constant(builder, value, r#type, location);
    if const_op {
        assert!(insert_pt == builder.insertion_point());
        assert!(match_pattern(const_op, m_constant()));
        return const_op;
    }

    None
}
