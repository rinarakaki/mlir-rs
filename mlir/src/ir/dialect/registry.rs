/*!
# Dialect Registration and Extension

This file defines functionality for registring and extending dialects.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/DialectRegistry.h>
*/

use core::{
    any::TypeId,
    ptr::null
};
use alloc::collections::BTreeMap;

use crate::{
    mlir::{
        ir::{
            dialect::Dialect,
            mlir_context::MLIRContext
        },
    },
    llvm::adt::small_vector::SmallVector
};


/**
// ====================
// DialectExtensionBase
// ====================

This class represents an opaque dialect extension. It contains a set of required dialects and an application function. The required dialects control when the extension is applied, i.e. the extension is applied when all required dialects are loaded. The application function can be used to attach additional functionality to attributes, dialects, operations, types, etc., and may also load additional necessary dialects.

// ====================
// DialectExtension
// ====================

This class represents a dialect extension anchored on the given set of dialects. When all of the specified dialects have been loaded, the application function of this extension will be executed.
*/
pub trait DialectExtension<const N: usize>: Clone {
    // /// The names of the dialects affected by this extension.
    // const DIALECT_NAMES: SmallVector<[&'static str; N]>;

    /**
    Return the dialects that our required by this extension to be loaded before applying.
    */
    fn required_dialects(&self) -> &[&'static str; N];

    /// Apply this extension to the given context and the required dialects.
    fn apply(
        &self,
        context: *mut MLIRContext,
        dialects: &mut [impl Dialect]
    );

    /// Override the base apply method to allow providing the exact dialect types.
    fn _apply<const N: usize>(
        &self,
        context: *mut MLIRContext,
        dialects: &mut [impl Dialect; N]
    ) {
        dialects.iter().map(self.apply);
    }
}

type DialectAllocator = dyn Fn(*mut MLIRContext) -> Option<dyn Dialect>;


/**
The DialectRegistry maps a dialect namespace to a constructor for the matching dialect. This allows for decoupling the list of dialects 'available' from the dialects loaded in the Context. The parser in particular will lazily load dialects in the Context as operations are encountered.
*/
pub struct DialectRegistry {
    registry: BTreeMap<String, (TypeId, DialectAllocator)>,
    extensions: Vec<Box<dyn DialectExtension>>
}

impl DialectRegistry {
    /**
    Add a new dynamic dialect constructor in the registry. The constructor provides as argument the created dynamic dialect, and is expected to register the dialect types, attributes, and ops, using the methods defined in ExtensibleDialect such as registerDynamicOperation.
    */
    pub fn insert_dynamic(
        &mut self,
        name: &str,
        ctor: &DynamicDialectPopulationFunction
    ) {
        /*
        This TypeID marks dynamic dialects. We cannot give a TypeID for the
        dialect yet, since the TypeID of a dynamic dialect is defined at its
        construction.
        */
        let type_id = TypeId::of::<void>();

        // Create the dialect, and then call ctor, which allocates its components.
        let constructor = |context: *mut MLIRContext| {
            let dyn_dialect = context.get_or_load_dynamic_dialect(
                name,
                |dialect: *mut DynamicDialect| ctor(context, dialect));
            assert!(
                dyn_dialect,
                "Dynamic dialect creation unexpectedly failed");
            dyn_dialect
        };

        insert(type_id, name, constructor);
    }

    /**
    Return an allocation function for constructing the dialect identified by its namespace, or nullptr if the namespace is not in this registry.
    */
    pub const fn dialect_allocator(&self, name: &str)
    -> fn(*mut MLIRContext) -> *mut Dialect {
        match self.registry.find(name) {
            None => null(),
            Some((_, allocator)) => allocator
        }
    }

    /**
    Register all dialects available in the current registry with the registry
    in the provided context.
    */
    pub const fn append_to(&self, destination: &DialectRegistry) {
        for (name, (type_id, dialect_allocator)) in self.registry {
            destination.insert(
                type_id,
                name,
                dialect_allocator);
        }
        // Merge the extensions.
        for extension in extensions {
            destination.extensions.push(extension.clone());
        }
    }

    /// Return the names of dialects known to this registry.
    pub const fn dialect_names(&self) -> _ {
        self.registry.iter().map(|(name, _): (String, _)| name)
    }

    /**
    Apply any held extensions that require the given dialect. Users are not expected to call this directly.
    */
    pub const fn apply_extensions(&self, dialect: *mut Dialect) {
        let context = dialect.context();
        let dialect_name = dialect.namespace();

        // Functor used to try to apply the given extension.
        let apply_extension = |extension: &DialectExtension| {
            let dialect_names = extension.required_dialects();

            /*
            Handle the simple case of a single dialect name. In this case, the required dialect should be the current dialect.
            */
            if dialect_names.len() == 1 {
                if dialect_names.front() == dialect_name {
                    extension.apply(context, dialect);
                }
                return;
            }

            // Otherwise, check to see if this extension requires this dialect.
            let name = match dialect_names.iter().find(|name| name == dialect_name) {
                None => return,
                Some(name) => name
            };

            /*
            If it does, ensure that all of the other required dialects have been loaded.
            */
            let required_dialects = SmallVector::<[dyn Dialect]>::new();
            required_dialects.reserve(dialect_names.len());
            for name in dialect_names {
                // The current dialect is known to be loaded.
                if name == name_it {
                    required_dialects.push(dialect);
                    continue;
                }
                // Otherwise, check if it is loaded.
                let loaded_dialect = context.loaded_dialect(&name);
                if !loaded_dialect {
                    return;
                }
                required_dialects.push(loaded_dialect);
            }
            extension.apply(context, required_dialects);
        };

        for extension in extensions {
            apply_extension(*extension);
        }
    }

    /**
    Apply any applicable extensions to the given context. Users are not expected to call this directly.
    */
    pub const fn apply_extensions(context: *mut MLIRContext);

    /// Add the given extension to the registry.
    pub fn add_extension(&self, extension: impl DialectExtension) {
        extensions.push(extension);
    }

    // /// Add the given extensions to the registry.
    // template <typename... ExtensionsT>
    // pub fn add_extensions(&mut self) {
    //     (add_extension(std::make_unique<ExtensionsT>()), ...);
    // }

    // /**
    // Add an extension function that requires the given dialects.
    // Note: This bare functor overload is provided in addition to the
    // std::function variant to enable dialect type deduction, e.g.:
    //  registry.add_extension(+[](context: *mut MLIRContext, MyDialect *dialect) { ... })

    // is equivalent to:
    //  registry.add_extension<MyDialect>(
    //     [](context: *mut MLIRContext, MyDialect *dialect){ ... }
    //  )
    // */
    // template <typename... DialectsT>
    // pub fn add_extension(void (*extensionFn)(*mut MLIRContext, DialectsT *...)) {
    //     add_extension<DialectsT...>(
    //         std::function<void(*mut MLIRContext, DialectsT * ...)>(extensionFn));
    // }

    // template <typename... DialectsT>
    // pub fn add_extension(std::function<void(*mut MLIRContext, DialectsT *...)> extensionFn) {
    //     using ExtensionFnT = std::function<void(*mut MLIRContext, DialectsT * ...)>;
    //     add_extension(std::make_unique<Extension>(std::move(extensionFn)));
    // }

    /**
    Returns true if the current registry is a subset of 'rhs', i.e. if 'rhs' contains all of the components of this registry.
    */
    pub const fn is_subset_of(&self, rhs: &DialectRegistry) -> bool;
}

// struct Extension : DialectExtension<DialectsT...> {
//     void apply(*mut MLIRContextcontext, DialectsT *...dialects) const final {
//         extensionFn(context, dialects...);
//     }
//     ExtensionFnT extensionFn;
// }
