/*!
- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Pass/AnalysisManager.h>
*/

// nclude "mlir/IR/Operation.h"
// #include "mlir/Pass/PassInstrumentation.h"
// #include "mlir/Support/LLVM.h"
// #include "llvm/ADT/DenseMap.h"
// #include "llvm/ADT/MapVector.h"
// #include "llvm/ADT/SmallPtrSet.h"
// #include "llvm/Support/TypeName.h"

/// A utility class to represent the analyses that are known to be preserved.
pub struct PreservedAnalyses {
}

/// The abstract polymorphic base class representing an analysis.
struct AnalysisConcept {
  
}

/// A derived analysis model used to hold a specific analysis object.
struct AnalysisModel<Analysis> {

}

/// This class represents a cache of analyses for a single operation. All
/// computation, caching, and invalidation of analyses takes place here.
pub struct AnalysisMap {

}

/// An analysis map that contains a map for the current operation, and a set of
/// maps for any child operations.
pub struct NestedAnalysisMap {

}

/**
This class represents an analysis manager for a particular operation
instance. It is used to manage and cache analyses on the operation as well
as those for child operations, via nested AnalysisManager instances
accessible via 'slice'. This class is intended to be passed around by value,
and cannot be constructed directly.
*/
pub struct AnalysisManager {

}

/**
An analysis manager class specifically for the top-level operation. This
class contains the memory allocations for all nested analysis managers, and
provides an anchor point. This is necessary because AnalysisManager is
designed to be a thin wrapper around an existing analysis map instance.
*/
pub struct ModuleAnalysisManager {

}
