//! MLIR Location Classes
//!
//! These classes provide the ability to relate MLIR objects back to source location position information.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Location.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/Location.cpp>

use core::ptr::Pointee;

use crate::{
    mlir::ir::{
        attribute::Attribute,
        builtins::{
            dialect,
            location_attributes::FusedLocation
        },
        location,
        sub_element_interfaces,
        visitors::WalkResult
    },
    llvm::{
        adt::{
            set_vector,
            type_switch::TypeSwitch
        },
        support::pointer_like_type_traits
    }
};


/// Location objects represent source locations information in MLIR.
/// LocationAttribute acts as the anchor for all Location based attributes.
pub struct LocationAttribute {

}

impl LocationAttribute {
    /// Walk all of the locations nested under, and including, the current.
    pub fn walk(&self, walk_fn: fn(Location) -> WalkResult) -> WalkResult {
        if walk_fn(&self) == WalkResult::Interrupt {
            WalkResult::Interrupt
        }
    }

    /// Return an instance of the given location type if one is nested under the current location. Returns nullptr if one could not be found.
    pub fn find_instance_of<T>(&self) -> T {
        let mut output = {};
        self.walk(|location: Location| {
            if let typed_loc = llvm::dyn_cast::<T>(location) {
                output = typed_loc;
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        output
    }
}

/// This class defines the main interface for locations in MLIR and acts as a non-nullable wrapper around a LocationAttribute.
#[derive(Clone, Copy)]
pub struct Location;

/* LLVM Utilities */

// impl Pointee for Location {
//     type Metadata = ;
// }
