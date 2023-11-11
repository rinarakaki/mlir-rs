/*!
This file defines support types for registering dialect extended types.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/TypeSupport.h>
*/

use core::any::TypeId;

use crate::{
    mlir::{
        ir::{
            dialect::Dialect,
            mlir_context::MLIRContext,
            storage_uniquer_support,
            r#type::Type
        },
        support::interface_support::InterfaceMap,
    },
    llvm::adt::{
        function_extras,
        twine
    }
};

/// Base storage class appearing in a Type.
pub struct TypeStorage {
}

/** A utility class to get, or create, unique instances of types within an MLIRContext. This class manages all creation and uniquing of types.
*/
pub struct TypeUniquer;

impl TypeUniquer {
    /// Get an uniqued instance of a type T.
    pub fn get<T: Type, Args>(context: *mut MLIRContext, args: &&Args) -> T {
        Self::with_type_id::<T, Args>(
            context, T::ID,
            std::forward::<Args>(args))
    }

    /**
    Get an uniqued instance of a parametric type T.
    The use of this method is in general discouraged in favour of `get<T, Args>(ctx, args)`.
    */
    // T::ImplType != TypeStorage
    pub fn with_type_id<T: Type, Args>(context: *mut MLIRContext, type_id: TypeId, args: &&Args) {
        context.type_uniquer().get::<T>(
            |storage: *mut TypeStorage| {
                storage.initialise(AbstractType::lookup(type_id, context));
            },
            type_id, std::forward::<Args>(args))
    }

    /**
    Get an uniqued instance of a singleton type T.
    The use of this method is in general discouraged in favor of
    'get<T, Args>(ctx, args)'.
    */
    // T::ImplType == TypeStorage
    pub fn with_type_id<T: Type>(context: *mut MLIRContext, type_id: TypeId) -> T {
        context.type_uniquer().get::<T>(type_id)
    }

    /**
    Change the mutable component of the given type instance in the provided
    context.
    */
    pub fn mutate<T: Type, Args>(
        context: *mut MLIRContext,
        r#type: *mut T,
        args: &&Args
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert!(!r#type.is_null(), "Cannot mutate null type");
        context.type_uniquer().mutate(
            T::ID, r#type, std::forward::<Args>(args))
    }

    /// Register a type instance T with the uniquer.
    pub fn register_type<T: Type>(context: *mut MLIRContext) {
        register_type::<T>(context, T::ID);
    }

    /**
    Register a parametric type instance T with the uniquer.
    The use of this method is in general discouraged in favor of
    'register_type<T>(ctx)'.
    */
    // T::ImplType != TypeStorage
    pub fn register_type(context: *mut MLIRContext, type_id: TypeId) {
        context.type_uniquer().register_parametric_storage_type<T::ImplType>(type_id);
    }
    
    /**
    Register a singleton type instance T with the uniquer.
    The use of this method is in general discouraged in favor of
    'register_type<T>(ctx)'.
    */
    // T::ImplType == TypeStorage
    pub fn register_type(context: *mut MLIRContext, type_id: TypeId) {
        context.type_uniquer().register_singleton_storage_type::<TypeStorage>(
            type_id,
            |storage: *mut TypeStorage| {
                storage.initialise(AbstractType::lookup(type_id, ctx));
            }
        );
    }
}
