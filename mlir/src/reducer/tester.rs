/*!
This file defines the Tester class used in the MLIR Reduce tool.

A Tester object is passed as an argument to the reduction passes and it is used to run the interestingness testing script on the different generated reduced variants of the test case.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Reducer/Tester.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Reducer/Tester.cpp>
*/

use crate::{
    mlir::ir::{
        builtins::operations,
        verifier
    },
    llvm::{
        adt::small_string,
        support::{
            error,
            file_system,
            program,
            tool_output_file
        }
    }
};

/**
This class is used to keep track of the testing environment of the tool. It contains a method to run the interestingness testing script on a MLIR test case file.
*/
pub struct Tester {
    test_script: &'static str,
    test_script_args: &'static str
}

pub enum Interestingness {
    True,
    False,
    Untested,
}
