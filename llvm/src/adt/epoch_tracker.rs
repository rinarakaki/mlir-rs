/*!
This file defines the DebugEpochBase and DebugEpochBase::HandleBase classes.
These can be used to write iterators that are fail-fast when LLVM is built
with asserts enabled.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/EpochTracker.h>
*/

use std::default::Default;

use crate::llvm::config::abi_breaking;

/// A base class for data structure classes wishing to make iterators
/// ("handles") pointing into themselves fail-fast.  When building without
/// asserts, this class is empty and does nothing.
///
/// DebugEpochBase does not by itself track handles pointing into itself.  The
/// expectation is that routines touching the handles will poll on
/// isHandleInSync at appropriate points to assert that the handle they're using
/// is still valid.
#[cfg(LLVM_ENABLE_ABI_BREAKING_CHECKS)]
mod epoch_tracker {
    #[derive(Default)]
    pub struct DebugEpochBase {
        epoch: u64
    }

    impl DebugEpochBase {
    }
}

#[cfg(not(LLVM_ENABLE_ABI_BREAKING_CHECKS))]
mod epoch_tracker {
    pub struct DebugEpochBase;

    impl DebugEpochBase {
        pub fn increment_epoch() -> () {}
    }

    struct HandleBase {
    //   public:
    //     HandleBase() = default;
    //     explicit HandleBase(const DebugEpochBase *) {}
    //     pub const fn isHandleInSync() -> bool { true }
    //     const void *getEpochAddress() const { return nullptr; }
    }
}

pub use epoch_tracker::*;
