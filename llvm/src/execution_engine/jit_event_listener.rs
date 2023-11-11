/*!
This file defines the JITEventListener interface, which lets users get callbacks when significant events happen during the JIT compilation process.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ExecutionEngine/JITEventListener.h>
*/

use crate::llvm::object::object_file::ObjectFile;

pub type ObjectKey = u64;

/**
Abstract interface for use by the JIT to notify clients about significant events during compilation. For example, to notify profilers and debuggers that need to know where functions have been emitted.

The default implementation of each method does nothing.
*/
pub trait JITEventListener {
    /**
    Called after an object has had its sections allocated and addresses assigned to all symbols. Note: Section memory will not have been relocated yet. `notify_function_loaded` will not be called for individual functions in the object.

    # ELF-specific information

    The ObjectImage contains the generated object image with section headers updated to reflect the address at which sections were loaded and with relocations performed in-place on debug sections.
    */
    fn notify_object_loaded(
        &self,
        key: ObjectKey,
        obj: &ObjectFile,
        l: &RuntimeDyld::LoadedObjectInfo
    );
}
