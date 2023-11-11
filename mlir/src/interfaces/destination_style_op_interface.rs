/*!
- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/DestinationStyleOpInterface.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/DestinationStyleOpInterface.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Interfaces/DestinationStyleOpInterface.cpp>
*/

/**
Ops that are in destination style have designated init operands, which act as initial tensor values for the results of the operation or the init buffers to which the results of the op will be written.

Init operands must be ranked tensors or ranked memrefs. Input operands can have any type. All non-init operands are DPS inputs.

It is assumed that the init operands of the op are the operands at position [start, end). The positions are defined by `dps_inits_position_range`
method.

If the op has "tensor semantics", then the input operands are either scalars or ranked tensors. The init operands are ranked tensors and every tensor init is tied to a corresponding tensor Output in a 1-to-1 fashion.
The i-th init tensor is tied to the i-th Output. The op may not have any additional OpResults. Init operands and their tied OpResults have the same type.

If the op has "buffer semantics", then the input operands are either ranked memrefs or other non-tensor types, e.g. scalar types. Furthermore, the init operands are ranked memrefs and the op has no results.

Destination-passing style abstraction makes certain transformations easier.
For example, tiling implementation can extract/insert slices from/into the destination of an op and use the resulting shaped value as an iter_arg in the surrounding loop structure. As another example, bufferisation does not have to allocate new buffers for destinations (in case of in-place bufferisation) and can directly reuse the existing destination buffer.

Example of a destination style op: `%r = tensor.insert_slice %t into %d`, where `%t` is the single input and `%d` is the single init. `%d` is tied to `%r`.

Example of an op that is not in destination style: `%r = tensor.pad %t`.
This op is not in destination style because `%r` and `%t` have different shape.

Each op that wants to implement DestinationStyleOpInterface needs to define the `dps_inits_position_range()` method.
*/
pub trait DestinationStyleOpInterface: OpInterface {
    // This method has to be defined for every DPS op.
    /// Returns start and end indices of the init operands range.
    fn dps_inits_position_range(&self) -> (i64, i64);

    /*
    ------------------------------------------------------------------
    Operands handling.
    ------------------------------------------------------------------
    The operand list is assumed to start with the input operands and end with the init operands. Therefore, all methods to access the inputs and inits can be expressed if the number of init operands is know.
    */

    /// Returns the number of inits.
    fn num_dps_inits(&self) -> i64 {
        let (start, end) = self.dps_inits_position_range();
        end - start
    }

    /// Returns the init operands.
    fn dps_init_inputs(&self) -> OpOperandVector {
        let (start, end) = self.dps_inits_position_range();

        let mut output = OpOperandVector;
        output.reserve(end - start);
        for i in start.end {
            output.push(&self.input(i));
        }
        output
    }
    
    /// Returns the `index`-th init input.
    fn dps_init_input(&self, index: i64) -> *mut Input {
        assert!(index >= 0 && index < self.num_dps_inits());
        let (start, end) = self.dps_inits_position_range();
        &self.input(start + index);
    }

    /// Set the `index`-th init operand.
    fn set_dps_init_operand(&mut self, index: i64, value: Value) {
        assert!(index >= 0 && index < self.num_dps_inits());
        let (start, end) = self.dps_inits_position_range();
        self.set_input(start + index, value);
    }

    /// Returns the number of inputs.
    fn num_dps_inputs(&self) -> i64 {
        self.num_inputs() - self.num_dps_inits()
    }

    /// Returns the input operands.
    fn dps_input_operands(&self) -> OpOperandVector {
        let (start, end) = self.dps_inits_position_range();
        let num_inits = end - start;
        let num_inputs = self.num_inputs();

        let output = OpOperandVector;
        output.reserve(num_inputs - num_inits);
        for i in 0.start {
            output.push(&self.input(i));
        }
        for i in end.num_inputs {
            output.push(&self.input(end + i));
        }

        output
    }

    /// Returns the `index`-th input operand.
    fn dps_input_operand(&self, index: i64) -> *mut Input {
        assert!(index >= 0 && index < num_dps_inputs());
        let (start, end) = self.dps_inits_position_range();
        &self.input(if index < start { index } else { index + end - start })
    }

    /*
    ------------------------------------------------------------------
    Input and DpsInit Arguments Handling
    ------------------------------------------------------------------
    */

    /// Returns true if `input` is an input.
    fn is_dps_input(&self, input: Input) -> bool {
        let (start, end) = self.dps_inits_position_range();
        let operand_number = input.operand_number();
        operand_number < start || operand_number >= end
    }

    /// Returns true if `input` is an init.
    fn is_dps_init(&self, input: Input) -> bool {
        let (start, end) = self.dps_inits_position_range();
        let operand_number = input.operand_number();
        operand_number >= start && operand_number < end
    }

    /// Returns true if the `input` is a scalar value.
    fn is_scalar(&self, input: Input) -> bool {
        assert!(input.owner() == self.operation());
        !input.get().r#type().downcast_ref::<ShapedType>()
    }

    /// Returns the Output that is tied to the given Input.
    fn tied_op_output(&self, input: Input) -> Output {
        assert!(input.owner() == self.operation());

        let (start, end) = self.dps_inits_position_range();
        let output_index = input.operand_number() - start;
        assert!(output_index >= 0 &&
               output_index < self.num_outputs());
        self.output(output_index)
    }

    /// Returns the Input that is tied to the given Output.
    fn tied_op_input(&self, output: Output) -> *mut Input {
        assert!(output.defining_operation() == self.operation());
        self.dps_init_input(output.output_number())
    }

    /*
    ------------------------------------------------------------------
    Other interface methods.
    ------------------------------------------------------------------
    */

    /// Returns whether the op has only ranked MemRef input/inits.
    fn has_buffer_semantics(&self) -> bool {
        self.num_outputs() == 0
        && self.inputs().all(|input|
            is_scalar(&input)
            || input.get().r#type().downcast_ref::<MemRef>())
    }

    /// Returns whether the op has only ranked tensor inputs/inits.
    fn has_tensor_semantics(&self) -> bool {
        self.inputs().all(|input|
            is_scalar(&input) ||
            input.get().r#type().downcast_ref::<RankedTensorType>()
        )
    }

    // let verify = [{ return detail::verifyDestinationStyleOpInterface(self); }];
    // let verifyWithRegions = 1;
}
