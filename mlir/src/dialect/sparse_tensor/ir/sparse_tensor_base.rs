/*!
Sparse tensor dialect base 

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SparseTensor/IR/SparseTensorBase.td>
*/

use crate::mlir::ir::{
    dialect::Dialect,
    operation::base
};

/**
The `SparseTensor` dialect supports all the attributes, types, operations, and passes that are required to make sparse tensor types first class citizens within the MLIR compiler infrastructure.
The dialect forms a bridge between high-level operations on sparse tensors types and lower-level operations on the actual sparse storage schemes consisting of pointers, indices, and values. Lower-level support may consist of fully generated code or may be provided by means of a small sparse runtime support library.

The concept of **treating sparsity as a property, not a tedious implementation detail**, by letting a **sparse compiler** generate sparse code automatically was pioneered for linear algebra by [Bik96] in MT1 (see <https://www.aartbik.com/sparse.php>) and formalized to tensor algebra by [Kjolstad17,Kjolstad20] in the Sparse Tensor Algebra Compiler (TACO) project (see <http://tensor-compiler.org>).

The MLIR implementation [Biketal22] closely follows the "sparse iteration theory" that forms the foundation of TACO. A rewriting rule is applied to each tensor expression in the Linalg dialect (MLIR's tensor index notation) where the sparsity of tensors is indicated using the per-dimension level types dense/compressed together with a specification of the order on the dimensions (see [Chou18] for an in-depth discussions and possible extensions to these level types). Subsequently, a topologically sorted iteration graph, reflecting the required order on indices with respect to the dimensions of each tensor, is constructed to ensure that all tensors are visited in natural index order. Next, iteration lattices are constructed for the tensor expression for every index in topological order. Each iteration lattice point consists of a conjunction of tensor indices together with a tensor (sub)expression that needs to be evaluated for that conjunction.  Within the lattice, iteration points are ordered according to the way indices are exhausted. As such these iteration lattices drive actual sparse code generation, which consists of a relatively straightforward one-to-one mapping from iteration lattices to combinations of for-loops, while-loops, and if-statements. Sparse tensor outputs that materialize uninitialized are handled with direct insertions if all parallel loops are outermost or insertions that indirectly go through a 1-dimensional access pattern expansion (a.k.a. workspace) where feasible [Gustavson72,Bik96,Kjolstad19].

- [Bik96] Aart J.C. Bik. Compiler Support for Sparse Matrix Computations.
PhD thesis, Leiden University, May 1996.
- [Biketal22] Aart J.C. Bik, Penporn Koanantakool, Tatiana Shpeisman,
Nicolas Vasilache, Bixia Zheng, and Fredrik Kjolstad. Compiler Support
for Sparse Tensor Computations in MLIR. ACM Transactions on Architecture
and Code Optimization, June, 2022. See: <https://dl.acm.org/doi/10.1145/3544559>
- [Chou18] Stephen Chou, Fredrik Berg Kjolstad, and Saman Amarasinghe.
Format Abstraction for Sparse Tensor Algebra Compilers. Proceedings of
the ACM on Programming Languages, October 2018.
- [Chou20] Stephen Chou, Fredrik Berg Kjolstad, and Saman Amarasinghe.
Automatic Generation of Efficient Sparse Tensor Format Conversion Routines.
Proceedings of the 41st ACM SIGPLAN Conference on Programming Language
Design and Implementation, June, 2020.
- [Gustavson72] Fred G. Gustavson. Some basic techniques for solving
sparse systems of linear equations. In Sparse Matrices and Their
Applications, pages 41–52. Plenum Press, New York, 1972.
- [Kjolstad17] Fredrik Berg Kjolstad, Shoaib Ashraf Kamil, Stephen Chou, David
Lugato, and Saman Amarasinghe. The Tensor Algebra Compiler. Proceedings of
the ACM on Programming Languages, October 2017.
- [Kjolstad19] Fredrik Berg Kjolstad, Peter Ahrens, Shoaib Ashraf Kamil,
and Saman Amarasinghe. Tensor Algebra Compilation with Workspaces,
Proceedings of the IEEE/ACM International Symposium on Code Generation
and Optimization, 2019.
- [Kjolstad20] Fredrik Berg Kjolstad. Sparse Tensor Algebra Compilation.
PhD thesis, MIT, February, 2020.
*/
pub trait SparseTensorDialect {
    
}
