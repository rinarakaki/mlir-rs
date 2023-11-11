/*!
Due to layering constraints (Support depends on llvm-c) this is a thin
wrapper around the implementation that lives in llvm-c, though most clients
can/should think of this as being provided by Support for simplicity (not
many clients are aware of their dependency on llvm-c).

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/DataTypes.h>
*/

// #include "llvm-c/DataTypes.h"
