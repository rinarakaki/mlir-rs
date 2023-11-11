/*!
This file contains a definition of the TypeId class. This provides a non
RTTI mechanism for producing unique type IDs in LLVM.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Support/TypeID.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Support/TypeID.cpp>
*/

pub use std::any::TypeId;

// use crate::{
//     mlir::support::llvm,
//     llvm::{
//         adt::{
//             dense_map::DenseMap,
//             dense_map_info,
//             hashing,
//             stl_extras,
//             string_ref::StringRef
//         },
//         support::{
//             allocator::SpecificBumpPtrAllocator,
//             pointer_like_type_traits,
//             rw_mutex,
//             type_name::type_name
//         }
//     }
// };

// /**
// This class provides an efficient unique identifier for a specific C++ type.
// This allows for a C++ type to be compared, hashed, and stored in an opaque
// context. This class is similar in some ways to std::type_index, but can be
// used for any type. For example, this class could be used to implement LLVM
// style isa/dyn_cast functionality for a type hierarchy:

// ```cpp
// struct Base {
//   Base(TypeId typeID) : typeID(typeID) {}
//   TypeId typeID;
// };

// struct DerivedA : public Base {
//   DerivedA() : Base(TypeId::get<DerivedA>()) {}

//   static bool classof(const Base *base) {
//     return base->typeID == TypeId::get<DerivedA>();
//   }
// };

// void foo(Base *base) {
//   if (DerivedA *a = llvm::dyn_cast<DerivedA>(base))
//     ...
// }
// ```

// C++ RTTI is a notoriously difficult topic; given the nature of shared
// libraries many different approaches fundamentally break down in either the
// area of support (i.e. only certain types of classes are supported), or in
// terms of performance (e.g. by using string comparison). This class intends
// to strike a balance between performance and the setup required to enable its
// use.

// Assume we are adding support for some class Foo, below are the set of ways
// in which a given C++ type may be supported:

// - Explicitly via `MLIR_DECLARE_EXPLICIT_TYPE_ID` and
//    `MLIR_DEFINE_EXPLICIT_TYPE_ID`

//    - This method explicitly defines the type ID for a given type using the
//      given macros. These should be placed at the top-level of the file (i.e.
//      not within any namespace or class). This is the most effective and
//      efficient method, but requires explicit annotations for each type.

//      Example:

//      ```cpp
//       // Foo.h
//       MLIR_DECLARE_EXPLICIT_TYPE_ID(Foo);

//       // Foo.cpp
//       MLIR_DEFINE_EXPLICIT_TYPE_ID(Foo);
//       ```

// - Explicitly via `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID`
//   - This method explicitly defines the type ID for a given type by
//     annotating the class directly. This has similar effectiveness and
//     efficiency to the above method, but should only be used on internal
//     classes; i.e. those with definitions constrained to a specific library
//     (generally classes in anonymous namespaces).

//     Example:

//     ```cpp
//     namespace {
//     class Foo {
//     public:
//     MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Foo)
//     };
//     } // namespace
//     ```

// - Implicitly via a fallback using the type name
//   - This method implicitly defines a type ID for a given type by using the
//     type name. This method requires nothing explicitly from the user, but
//     pays additional access and initialisation cost. Given that this method
//     uses the name of the type, it may not be used for types defined in
//     anonymous namespaces (which is asserted when it can be detected). String
//     names do not provide any guarantees on uniqueness in these contexts.
// */
// #[derive(Clone, Copy, PartialEq)]
// pub struct TypeId {
//     /// The storage of this type info object.
//     storage: Storage  //  const Storage *
// }

// impl TypeId {
//     /// Construct a type info object for the given type T.
//     pub fn get<T>() -> Self {
//         TypeIdResolver::<T>::resolve_type_id()
//     }
// }

// /**
// This class represents the storage of a type info object.
// Note: We specify an explicit alignment here to allow use with
// PointerIntPair and other utilities/data structures that require a known
// pointer alignment.
// */
// #[repr(align(8))]
// struct Storage {}

// /**
// This class provides a resolver for getting the ID for a given class T. This
// allows for the derived type to specialise its resolution behaviour. The
// default implementation uses the string name of the type to resolve the ID.
// This provides a strong definition, but at the cost of performance (we need
// to do an initial lookup) and is not usable by classes defined in anonymous
// contexts.

// TODO: The use of the type name is only necessary when building in the
// presence of shared libraries. We could add a build flag that guarantees
// "static"-like environments and switch this to a more optimal implementation
// when that is enabled.
// */
// pub struct TypeIdResolver<T, Enable = ()> {

// }

// /**
// This class provides a fallback for resolving `TypeId`s. It uses the string
// name of the type to perform the resolution, and as such does not allow the
// use of classes defined in "anonymous" contexts.
// */
// impl<T, Enable> TypeIdResolver<T, Enable> {
//     /// Register an implicit type ID for the given type name.
//     fn register_implicit_type_id(name: &str) -> TypeId {
//         let registry = ImplicitTypeIdRegistry::new();
//         registry.lookup_or_insert(name)
//     }
// }

// impl<T, Enable> TypeIdResolver<T, Enable> {
//     pub fn resolve_type_id() -> TypeId {
//         // static_assert(is_fully_resolved<T>::value,
//         //               "TypeId::get<> requires the complete definition of `T`");
//         let id = Self::register_implicit_type_id(type_name::<T>());
//         id
//     }
// }

// struct ImplicitTypeIdRegistry {
//     /// A mutex that guards access to the registry.
//     mutex: SmartRWMutex<true>,

//     /// An allocator used for TypeId objects.
//     type_id_allocator: TypeIdAllocator,

//     /// A map type name to TypeId.
//     type_name_to_id: DenseMap<&'static str, TypeId>
// }

// impl ImplicitTypeIdRegistry {
//     /// Lookup or insert a TypeId for the given type name.
//     fn lookup_or_insert(&self, type_name: &str) -> TypeId {
//         {
//             // Try a read-only lookup first.
//             let guard = SmartScopedReader::<true>::new(self.mutex);
//             let it = self.type_name_to_id.find(type_name);
//             if it != self.type_name_to_id.end() {
//                 return it.second;
//             }
//         }
//         let guard = SmartScopedWriter::<true>::new(self.mutex);
//         let it = self.type_name_to_id.try_emplace(type_name, TypeId::new());
//         if it.second {
//             it.first.second = self.type_id_allocator.allocate();
//         }
//         it.first.second
//     }
// }

// /**
// This class provides utilities for resolving the TypeId of a class that provides a `static TypeId resolveTypeId()` method. This allows for simplifying situations when the class can resolve the ID itself. This functionality is separated from the corresponding `TypeIdResolver` specialisation below to enable referencing it more easily in different contexts.
// */
// pub struct InlineTypeIdResolver {

// }

// /**
// This class provides a way to define new `TypeId`s at runtime.
// When the allocator is destructed, all allocated `TypeId`s become invalid and therefore should not be used.
// */
// pub struct TypeIdAllocator {
//     /// The `TypeId`s allocated are the addresses of the different storages.
//     /// Keeping those in memory ensure uniqueness of the `TypeId`s.
//     ids: SpecificBumpPtrAllocator<Storage>
// }

// impl TypeIdAllocator {
//     /// Allocate a new `TypeId`, that is ensured to be unique for the lifetime
//     /// of the `TypeIdAllocator`.
//     pub fn allocate(&self) -> TypeId {
//         TypeId::new(self.ids.allocate())
//     }
// }

// /**
// Defines a TypeId for each instance of this class by using a pointer to the
// instance. Thus, the copy and move constructor are deleted.
// Note: We align by 8 to match the alignment of TypeId::Storage, as we treat
// an instance of this class similarly to TypeId::Storage.
// */
// #[repr(align(8))]
// pub struct SelfOwningTypeId {
// }
