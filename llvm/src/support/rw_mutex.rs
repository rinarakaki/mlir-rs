/*!
This file declares the llvm::sys::RWMutex class.

This file implements the llvm::sys::RWMutex class.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/RWMutex.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/RWMutex.cpp>
*/

// #[allow(non_camel_case_types)]
// type unsigned = u16;

use std::sync::{Arc, LockResult, RwLock, RwLockReadGuard};

// use crate::llvm::{
//     config::{
//         config,
//         llvm_config
//     },
//     support::{
//         allocator,
//         threading
//     }
// };

pub type RWMutex<T> = Arc<RwLock<T>>;

// #[cfg(LLVM_USE_RW_MUTEX_IMPL)]
// /// Platform agnostic RWMutex class.
// pub struct RWMutex<T> {
//     #[cfg(all(LLVM_ENABLE_THREADS))]
//     data: Option<T>
// }

// #[cfg(LLVM_USE_RW_MUTEX_IMPL)]
// impl RWMutex {
//     pub fn new() -> Self {
//         Self {
//             #[cfg(all(LLVM_ENABLE_THREADS))]
//             None
//         }
//     }

//     /**
//     Attempts to unconditionally acquire the lock in reader mode. If the
//     lock is held by a writer, this method will wait until it can acquire
//     the lock.
//     @returns false if any kind of error occurs, true otherwise.
//     Unconditionally acquire the lock in reader mode.
//     */
//     pub fn lock_shared(&self) -> bool {
//         true
//     }

//     /**
//     Attempts to release the lock in reader mode.
//     @returns false if any kind of error occurs, true otherwise.
//     Unconditionally release the lock in reader mode.
//     */
//     pub fn unlock_shared(&self) -> bool {
//         true
//     }

//     /**
//     Attempts to unconditionally acquire the lock in reader mode. If the
//     lock is held by any readers, this method will wait until it can
//     acquire the lock.
//     @returns false if any kind of error occurs, true otherwise.
//     Unconditionally acquire the lock in writer mode.
//     */
//     pub fn lock(&self) -> bool {
//         true
//     }

//     /**
//     Attempts to release the lock in writer mode.
//     @returns false if any kind of error occurs, true otherwise.
//     Unconditionally release the lock in write mode.
//     */
//     pub fn unlock(&self) -> bool {
//         true
//     }
// }

// /**
// SmartMutex - An R/W mutex with a compile time constant parameter that
// indicates whether this mutex should become a no-op when we're not
// running in multithreaded mode.
// */
// pub struct SmartRWMutex<const MT_ONLY: bool> {
//     readers: unsigned,  // = 0;
//     writers: unsigned   //= 0;
// }

// #[cfg(LLVM_USE_RW_MUTEX_IMPL)]
// impl<const MT_ONLY: bool> SmartRWMutex<MT_ONLY> {

// }

// /// ScopedWriter - RAII acquisition of a writer lock

// pub type SmartScopedReader<T>
//     // = std::shared_lock<SmartRWMutex<MT_ONLY>>;
//     = LockResult<RwLockReadGuard<'_, T>>;

// #[cfg(LLVM_USE_RW_MUTEX_IMPL)]
// pub struct SmartScopedReader<T, const MT_ONLY: bool> {
//     mutex: &RWMutex<T, MT_ONLY>
// }

// #[cfg(LLVM_USE_RW_MUTEX_IMPL)]
// impl<T, const MT_ONLY: bool> SmartScopedReader<T, MT_ONLY> {
//     pub fn new(mutex: &RWMutex<T>) -> Self {
//         mutex.lock_shared();
//         Self { mutex }
//     }
// }

// /// ScopedWriter - RAII acquisition of a writer lock
// #[cfg(not(LLVM_USE_RW_MUTEX_IMPL))]
// pub type SmartScopedWriter<const MT_ONLY: bool>
//     = std::lock_guard<SmartRWMutex<MT_ONLY>>;

// #[cfg(LLVM_USE_RW_MUTEX_IMPL)]
// pub struct SmartScopedWriter<const MT_ONLY: bool> {
//     mutex: &SmartRWMutex<MT_ONLY>
// }

// #[cfg(LLVM_USE_RW_MUTEX_IMPL)]
// impl<const MT_ONLY: bool> SmartScopedWriter<MT_ONLY> {
//     pub fn new(mutex: &SmartRWMutex<MT_ONLY>) -> Self {
//         mutex.lock();
//         Self { mutex }
//     }
// }
