# llvm

- Homepage https://llvm.org/
- Repository https://github.com/llvm/llvm-project
- License https://llvm.org/LICENSE.txt

# clang

```sh
clang main.c  # a.out
clang main.c -S  # main.s
clang main.c -c -emit-llvm --target=wasm32  # main.bc
clang -Xclang -ast-dump=json -fsyntax-only <file>
```

## Examples

```sh
llc main.bc  # main.s
llvm-dis main.bc  # main.ll

clang test.c -I $(find /usr/lib -name "linux-tools-*")/include
```
