/*!
This file defines an API used to report recoverable errors.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/Error.h>
*/

/**
Lightweight error class with error context and mandatory checking.

Instances of this class wrap a ErrorInfoBase pointer. Failure states are represented by setting the pointer to a ErrorInfoBase subclass instance containing information describing the failure. Success is represented by a null pointer value.

Instances of Error also contains a 'Checked' flag, which must be set before the destructor is called, otherwise the destructor will trigger a runtime error. This enforces at runtime the requirement that all Error instances be checked or returned to the caller.

There are two ways to set the checked flag, depending on what state the Error instance is in. For Error instances indicating success, it is sufficient to invoke the boolean conversion operator. E.g.:

```cpp
  Error foo(<...>);

  if (auto E = foo(<...>))
    return E; // <- Return E if it is in the error state.
  // We have verified that E was in the success state. It can now be safely
  // destroyed.
```

A success value *can not* be dropped. For example, just calling `foo(<...>)` without testing the return value will raise a runtime error, even if foo returns success.

For Error instances representing failure, you must use either the handleErrors or handleAllErrors function with a typed handler. E.g.:

```cpp
  class MyErrorInfo : public ErrorInfo<MyErrorInfo> {
    // Custom error info.
  };

  Error foo(<...>) { return make_error<MyErrorInfo>(...); }

  auto E = foo(<...>); // <- foo returns failure with MyErrorInfo.
  auto NewE =
    handleErrors(E,
      [](const MyErrorInfo &M) {
        // Deal with the error.
      },
      [](std::unique_ptr<OtherError> M) -> Error {
        if (canHandle(*M)) {
          // handle error.
          return Error::success();
        }
        // Couldn't handle this error instance. Pass it up the stack.
        return Error(std::move(M));
      );
  // Note - we must check or return NewE in case any of the handlers
  // returned a new error.
```

The handleAllErrors function is identical to handleErrors, except that it has a void return type, and requires all errors to be handled and no new errors be returned. It prevents errors (assuming they can all be handled) from having to be bubbled all the way to the top-level.

*All* Error instances must be checked before destruction, even if they're moved-assigned or constructed from Success values that have already been checked. This enforces checking through all levels of the call stack.
*/
#[must_use]
pub struct Error {}
