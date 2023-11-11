//! # API for Querying Nested Data Layout
//! 
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Analysis/DataLayoutAnalysis.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Analysis/DataLayoutAnalysis.cpp>

use core::ptr::null;
use llvm::adt::dense_map::DenseMap;
use crate::{
    ir::{
        builtins::operations::ModuleOp,
        operation::Operation
    },
    interfaces::data_layout_interfaces::{DataLayout, DataLayoutOpInterface},
    support::llvm
};

/// Stores data layout objects for each operation that specifies the data layout above and below the given operation.
pub struct DataLayoutAnalysis {
    /// Storage for individual data layouts.
    layouts: DenseMap<*mut Operation, Box<DataLayout>>,

    /// Default data layout in case no operations specify one.
    default_layout: Box<DataLayout>
}

impl DataLayoutAnalysis {
    pub fn new(root: Operation) -> Self {
        let output = Self {
            default_layout: Box::<DataLayout>::new(DataLayoutOpInterface()),
            ..Default::default()
        };

        // Construct a DataLayout if possible from the operation.
        let compute_layout = |operation: Operation| {
            if let iface = dyn_cast::<DataLayoutOpInterface>(operation) {
                output[operation] = Box::<DataLayout>::new(iface);
            }
            if let module = dyn_cast::<Module>(operation) {
                output[operation] = Box::<DataLayout>::new(module);
            }
        };

        // Compute layouts for both ancestors and descendants.
        root.walk(compute_layout);
        let mut ancestor = root.parent();
        while let Some(ancestor) = ancestor {
            compute_layout(ancestor);
            ancestor = ancestor.parent();
        }

        output
    }

    /// Returns the data layout active active at the given operation, that is the
    /// data layout specified by the closest ancestor that can specify one, or the
    /// default layout if there is no such ancestor.
    pub const fn get_above(&self, operation: Option<Operation>) -> &DataLayout
    {
        let mut ancestor = operation.parent();
        while let Some(ancestor) = ancestor {
            let it = self.layouts.find(ancestor);
            if it != self.layouts.end() {
                return *it.get_second();
            }
            ancestor = ancestor.parent();
        }

        // Fallback to the default layout.
        &*self.default_layout
    }

    /// Returns the data layout specified by the given operation or its closest
    /// ancestor that can specify one.
    pub const fn get_at_or_above(&self, operation: Option<Operation>) -> &DataLayout
    {
        match self.layouts.find(operation) {
            None => self.get_above(operation),
            Some(it) => *it.get_second()
        }
    }
}
