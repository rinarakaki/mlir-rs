/*!
# MLIR Interface Support Classes

This file defines several support classes for defining interfaces.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Support/InterfaceSupport.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Support/InterfaceSupport.cpp>
*/

use core::any::TypeId;

use crate::{
    llvm::{
        adt::{
            dense_map
        },
        support::type_name
    }
};

/**
This class represents an abstract interface. An interface is a simplified mechanism for attaching concept based polymorphism to a class hierarchy. An interface is comprised of two components:

- The derived interface class: This is what users interact with, and invoke
 methods on.
- An interface `Trait` class: This is the class that is attached to the object implementing the interface. It is the mechanism with which models are specialised.

Derived interfaces types must provide the following template types:

- ConcreteType: The CRTP derived type.
- ValueT: The opaque type the derived interface operates on. For example
          `Operation*` for operation interfaces, or `Attribute` for
          attribute interfaces.
- Traits: A class that contains definitions for a 'Concept' and a 'Model'
          class. The 'Concept' class defines an abstract virtual interface,
          where as the 'Model' class implements this interface for a
          specific derived T type. Both of these classes *must* not contain
          non-static data. A simple example is shown below:

```cpp
struct ExampleInterfaceTraits {
    struct Concept {
       virtual unsigned getNumInputs(T t) const = 0;
    };

    template <typename DerivedT> class Model {
       unsigned getNumInputs(T t) const final {
         return cast<DerivedT>(t).getNumInputs();
       }
    };
};
```

- `BaseType`: A desired base type for the interface. This is a class
            that provides specific functionality for the `ValueT`
            value. For instance the specific `Op` that will wrap the
            `Operation*` for an `OpInterface`.
- `BaseTrait`: The base type for the interface trait. This is the base class
             to use for the interface trait that will be attached to each
             instance of `ValueT` that implements this interface.

*/
// <Concrete, Value, Traits, Base, BaseTrait>
pub trait Interface {

}

/// This is a special trait that registers a given interface with an object.
pub trait Trait: BaseTrait<Trait> {
    // type ModelT = Model<ConcreteT>;

    /// Define an accessor for the ID of this interface.
    fn interface_id() -> TypeId {
        TypeId::of::<Self>()
    }
}

/**
This class provides an efficient mapping between a given `Interface` type, and a particular implementation of its concept.
*/
pub struct InterfaceMap {
}

impl InterfaceMap {
    // /// Returns an instance of the concept object for the given interface if it
    // /// was registered to this map, null otherwise.
    // pub const fn lookup<T>(&self) -> *mut T::Concept {
    //     reinterpret_cast::<*mut T::Concept>(self._lookup(T::interface_id()))
    // }

    // /// Returns true if the interface map contains an interface for the given id.
    // pub const fn contains(interface_id: TypeId) -> bool {
    //     lookup(interface_id)
    // }

    /**
    Insert the given models as implementations of the corresponding interfaces for the concrete attribute class.
    */
    template <typename... IfaceModels>
    pub fn insert(&mut self) {
        assert!(all_trivially_destructible<IfaceModels...>::value,
                "interface models must be trivially destructible");
        let elements = [
            (IfaceModels::Interface::interface_id(),
             new (Global.allocate(Layout::<IfaceModels>::new()))) IfaceModels())...
        ];
        insert(elements);
    }
}
