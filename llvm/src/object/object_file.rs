/*!
File format independent object file

This file declares a file format independent ObjectFile class.

This file defines a file format independent ObjectFile class.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Object/ObjectFile.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Object/ObjectFile.cpp>
*/

use std::default::Default;

/**
This is a value type class that represents a single relocation in the list of relocations in the object file.
*/
pub struct RelocationRef {

}

/**
This is a value type class that represents a single section in the list of
sections in the object file.
*/
pub struct SectionRef {

}

pub struct SectionedAddress {
    address: u64,
    section_index: u64  // = UndefSection;
}

impl Default for SectionedAddress {
    fn default() -> Self {
        Self { address: 0, section_index: u64::MAX }
    }
}

/**
This is a value type class that represents a single symbol in the list of symbols in the object file.
*/
pub struct SymbolRef {

}

pub enum SymbolRefType {
    Unknown, // Type not specified
    Other,
    Data,
    Debug,
    File,
    Function,
}

/**
This class is the base class for all object file types. Concrete instances of this object are created by createObjectFile, which figures out which type to create.
*/
pub struct ObjectFile {

}
