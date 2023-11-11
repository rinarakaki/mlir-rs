/*!
This file defines the raw_ostream class.

This implements support for bulk buffered stream output.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/raw_ostream.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/raw_ostream.cpp>
*/

use crate::llvm::{
    adt::{
        small_vector,
        string_extras,
    },
    config::config,
    support::{
        compiler,
        data_types,
        duration,
        error_handling,
        file_system,
        format,
        format_variadic,
        math_extras,
        native_formatting,
        process,
        program,
    }
};

/**
This class implements an extremely fast bulk output stream that can *only*
output to a stream.  It does not support seeking, reopening, rewinding, line
buffered disciplines etc. It is a simple buffer that outputs
a chunk at a time.
*/
pub struct RawOStream {
    kind: OStreamKind,
    /**
    The buffer is handled in such a way that the buffer is
    uninitialised, unbuffered, or out of space when OutBufCur >=
    OutBufEnd. Thus a single comparison suffices to determine if we
    need to take the slow path to write a single character.

    The buffer is in one of three states:
     1. Unbuffered (BufferMode == Unbuffered)
     1. Uninitialized (BufferMode != Unbuffered && OutBufStart == 0).
     2. Buffered (BufferMode != Unbuffered && OutBufStart != 0 &&
                  OutBufEnd - OutBufStart >= 1).

    If buffered, then the raw_ostream owns the buffer if (BufferMode ==
    InternalBuffer); otherwise the buffer has been set via SetBuffer and is
    managed by the subclass.

    If a subclass installs an external buffer using SetBuffer then it can wait
    for a \see write_impl() call to handle the data which has been put into
    this buffer.
    */
    out_buf_start: Option<char>,
    out_buf_end: Option<char>,
    out_buf_cur: Option<char>,
    colour_enabled: bool,  // false

    /**
    Optional stream this stream is tied to. If this stream is written to, the
    tied-to stream will be flushed first.
    */
    tied_stream: Option<Box<Self>>, //= nullptr;

    buffer_mode: BufferKind
}

/// Class kinds to support LLVM-style RTTI.
enum OStreamKind {
    OStream,
    FDStream,
}

enum BufferKind {
    Unbuffered,
    InternalBuffer,
    ExternalBuffer
}

/// Colour order matches ANSI escape sequence, don't change
pub enum Colour {
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    Savedcolor,
    Reset,
}
