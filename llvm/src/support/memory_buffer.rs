/*!
# Memory Buffer Interface

This file defines the MemoryBuffer interface.

This file implements the MemoryBuffer interface.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/MemoryBuffer.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/MemoryBuffer.cpp>
*/

use std::fs::File;

use crate::llvm::{
    adt::twine,
    support::{
        alignment,
        c_binding_wrapping,
        error_or,
        memory_buffer_ref
    }
};

/**
This interface provides simple read-only access to a block of memory, and provides simple methods for reading files and standard input into a memory buffer. In addition to basic access to the characters in the file, this interface guarantees you can read one character past the end of the file, and that this character will read as '\0'.

The '\0' guarantee is needed to support an optimisation -- it's intended to be more efficient for clients which are reading all the data to stop reading when they encounter a '\0' than to continually check the file position to see if it has reached the end of the file.
*/
pub struct MemoryBuffer {
}

impl MemoryBuffer {
    /**
    Open the specified file as a MemoryBuffer, returning a new MemoryBuffer
    if successful, otherwise returning null.

    \param is_text Set to true to indicate that the file should be read in
    text mode.

    \param is_volatile Set to true to indicate that the contents of the file
    can change outside the user's control, e.g. when libclang tries to parse
    while the user is editing/updating the file or if the file is on an NFS.

    \param alignment Set to indicate that the buffer should be aligned to at
    least the specified alignment.
    */
    // static 
    pub fn get_file(
        filename: &str,
        is_text: bool,  // = false,
        requires_null_terminator: bool,  // = true,
        is_volatile: bool,  // = false,
        alignment: Option<Align>,  // = None
    ) -> Result<Self> {

    }

    /**
    Read all of the specified file into a MemoryBuffer as a stream
    (i.e. until EOF reached). This is useful for special files that
    look like a regular file but have 0 size (e.g. /proc/cpuinfo on Linux).
    */
    // static 
    pub fn get_file_as_stream(filename: &str) -> Result<Self> {

    }

    /**
    Given an already-open file descriptor, map some slice of it into a
    MemoryBuffer. The slice is specified by an \p offset and \p map_size.
    Since this is in the middle of a file, the buffer is not null terminated.
    */
    // static 
    pub fn get_open_file_slice(
        fd: File,
        filename: &str,
        map_size: u64,
        offset: i64,
        is_volatile: bool,  // = false,
        alignment: Option<Align>,  // = None
    ) -> Result<Self> {

    }

    /**
    Given an already-open file descriptor, read the file and return a
    MemoryBuffer.

    \param is_volatile Set to true to indicate that the contents of the file
    can change outside the user's control, e.g. when libclang tries to parse
    while the user is editing/updating the file or if the file is on an NFS.

    \param alignment Set to indicate that the buffer should be aligned to at
    least the specified alignment.
    */
    // static
    pub fn get_open_file(
        fd: File,
        filename: &str,
        file_size: u64,
        requires_null_terminator: bool,  // = true,
        is_volatile: bool,  // = false,
        alignment: Option<Align>  // = None
    ) -> Result<Self> {

    }

    /**
    Open the specified memory range as a MemoryBuffer. Note that InputData must be null terminated if requires_null_terminator is true.
    */
    static std::unique_ptr<MemoryBuffer>
    pub fn get_mem_buffer(
        InputData: &str,
        BufferName: &str = "",
        requires_null_terminator: bool  // = true
    );

    static std::unique_ptr<MemoryBuffer>
    pub fn get_mem_buffer(
        Ref: MemoryBufferRef,
        requires_null_terminator: bool  // = true
    );

    /**
    Open the specified memory range as a MemoryBuffer, copying the contents and taking ownership of it. InputData does not have to be null terminated.
    */
    static std::unique_ptr<MemoryBuffer>
    pub fn get_mem_buffer_copy(
        &str InputData,
        const Twine &BufferName = ""
    );

    /// Read all of stdin into a file buffer, and return it.
    // static
    pub fn get_stdin() -> Result<Self> {

    }

    /**
    Open the specified file as a MemoryBuffer, or open stdin if the filename is "-".
    */
    // static
    pub fn get_file_or_stdin(
        filename: &str,
        is_text: bool, // = false,
        requires_null_terminator: bool,  // = true,
        alignment: Option<Align>,  // = None
    ) -> Result<Self>
    {
        SmallString<256> name_buf;
        StringRef name_ref = filename.to_string_ref(name_buf);
      
        if name_ref == "-" {
            return get_stdin();
        }
        get_file(filename, is_text, requires_null_terminator,
                       /*IsVolatile=*/false, alignment)
    }

    /// Map a subrange of the specified file as a MemoryBuffer.
    // static
    pub fn get_file_slice(
        filename: &str,
        u64 MapSize,
        u64 Offset,
        bool IsVolatile = false,
        Option<Align> alignment = None
    ) -> Result<Self> {
        
    }
}

/// The kind of memory backing used to support the MemoryBuffer.
pub enum BufferKind {
    Malloc,
    MMap
}
