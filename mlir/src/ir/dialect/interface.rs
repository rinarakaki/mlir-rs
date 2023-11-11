/*!
# IR Dialect Interfaces

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/DialectInterface.h>
*/

use core::{
    any::{Any, TypeId},
    hash::Hash,
    iter::IntoIterator,
    slice::Iter
};

use crate::{
    mlir::ir::{
        dialect::Dialect,
        mlir_context::MLIRContext
    },
    llvm::adt::{
        dense_map_info::DenseMapInfo,
        dense_set::DenseSet,
        stl_extras
    }
};

pub trait DialectInterface: Any {
    /// The dialect that represents this interface.
    const DIALECT: *mut Dialect;

    /*
    ===================================================================
    DialectInterfaceBase
    ===================================================================

    The base class used for all derived interface types. This class provides utilities necessary for registration.
    */

    // ===================================================================
    // DialectInterface
    // ===================================================================

    /// Return the context that holds the parent dialect of this interface.
    fn context(&self) -> *mut MLIRContext;
}

/**
A collection of dialect interfaces within a context, for a given concrete interface type.
*/
pub trait DialectInterfaceCollection {
    /*
    ===================================================================
    DialectInterfaceCollectionBase
    ===================================================================

    This class is the base class for a collection of instances for a specific interface kind.
    */

    /// A set of registered dialect interface instances.
    const interfaces: DenseSet<*const dyn DialectInterface, dyn InterfaceKeyInfo>;
    /**
    An ordered list of the registered interface instances, necessary for deterministic iteration.
    */
    // NOTE: SetVector does not provide find access, so it can't be used here.
    const ordered_interfaces: Vec<*const dyn DialectInterface>;


    // ===================================================================
    // DialectInterface
    // ===================================================================
}

/// DenseMap info for dialect interfaces that allows lookup by the dialect.
pub trait InterfaceKeyInfo: DenseMapInfo<*const DialectInterface> {
    // using DenseMapInfo<const DialectInterface *>::is_equal;

    // static 
    // fn get_hash_value(key: Option<Dialect>) -> unsigned {
    //     llvm::hash_value(key)
    // }

    // static
    // fn get_hash_value(key: &impl DialectInterface) -> unsigned {
    //     return get_hash_value(key.dialect());
    // }

    // static
    // fn is_equal(lhs: Option<Dialect>, rhs: &impl DialectInterface) -> bool {
    //     if rhs == get_empty_key() || rhs == get_tombstone_key() {
    //         return false;
    //     }
    //     return lhs == rhs.dialect();
    // }
}

// impl<T: DialectInterfaceCollection> Hash for I {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         state.write
//     }
// }

impl<T: DialectInterfaceCollection> IntoIterator for T {
    type Item = *const impl DialectInterface;
    type IntoIter = <T::ordered_interfaces as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self::ordered_interfaces.into_iter()
    }
}
