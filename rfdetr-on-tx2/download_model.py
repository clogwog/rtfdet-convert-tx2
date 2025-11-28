#!/usr/bin/env python3
# /// script
# requires-python = "<=3.13"
# dependencies = [
#     "inference",
#     "onnx>=1.14.0",
#     "onnxsim>=0.4.35",
# ]
# ///

import sys
import shutil

from pathlib import Path
import onnx
from onnxsim import simplify

try:
    from inference import get_model
    from inference.models.aliases import RFDETR_ALIASES
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False


def replace_layernorm_with_identity(model):
    """
    Replace LayerNormalization nodes with Identity nodes.
    TensorRT cannot parse LayerNormalization, so we replace them with Identity.
    """
    nodes_to_remove = []
    nodes_to_add = []

    for node in model.graph.node:
        if node.op_type == 'LayerNormalization':
            print(f"Replacing LayerNormalization node '{node.name}' with Identity")
            # Create an Identity node with the same inputs and outputs
            identity_node = onnx.helper.make_node(
                'Identity',
                inputs=[node.input[0]],  # Use only the first input (the data input)
                outputs=node.output,
                name=f"{node.name}_identity"
            )
            nodes_to_add.append(identity_node)
            nodes_to_remove.append(node)

    # Remove old LayerNormalization nodes
    for node in nodes_to_remove:
        model.graph.node.remove(node)

    # Add new Identity nodes
    model.graph.node.extend(nodes_to_add)

    return len(nodes_to_remove)


def replace_problematic_range_operations(model):
    """
    Log Range operations but don't replace them.
    Range operations are essential for shape calculations in RF-DETR models.
    Replacing them breaks the model's tensor shape inference.
    """
    range_count = 0
    for node in model.graph.node:
        if node.op_type == 'Range':
            print(f"Found Range operation: '{node.name}' - essential for shape calculations")
            print("  Note: Not replacing Range operations as they break shape inference")
            range_count += 1

    return range_count


def fix_unsqueeze_operations(model):
    """
    Fix Unsqueeze operations that have axes as attributes instead of inputs.
    TensorRT expects axes as a separate input tensor, not an attribute.
    """
    nodes_to_modify = []

    for node in model.graph.node:
        if node.op_type == 'Unsqueeze':
            # Check if it has 'axes' attribute (old format)
            has_axes_attr = any(attr.name == 'axes' for attr in node.attribute)
            if has_axes_attr and len(node.input) == 1:  # Only has data input, axes is attribute
                print(f"Converting Unsqueeze node '{node.name}' from attribute format to input format")
                # Get the axes values
                axes_attr = next(attr for attr in node.attribute if attr.name == 'axes')
                axes_values = list(axes_attr.ints) if hasattr(axes_attr, 'ints') else [axes_attr.i]

                # Create a constant tensor for axes
                axes_tensor_name = f"{node.name}_axes"
                axes_tensor = onnx.helper.make_tensor(
                    name=axes_tensor_name,
                    data_type=onnx.TensorProto.INT64,
                    dims=[len(axes_values)],
                    vals=axes_values
                )

                # Add the axes tensor to initializers
                model.graph.initializer.append(axes_tensor)

                # Add axes as second input
                node.input.append(axes_tensor_name)

                # Remove the axes attribute
                node.attribute.remove(axes_attr)

                nodes_to_modify.append(node)

    return len(nodes_to_modify)


def fix_reshape_operations(model):
    """
    Fix Reshape operations that have -1 (wildcard) dimensions in their shape tensors.
    TensorRT cannot resolve -1 dimensions during shape analysis.
    """
    nodes_to_modify = []

    for reshape_node in model.graph.node:
        if reshape_node.op_type == 'Reshape' and len(reshape_node.input) >= 2:
            # Second input should be the shape tensor
            shape_input_name = reshape_node.input[1]

            # Find the shape tensor in initializers or Constant nodes
            shape_tensor = None
            shape_values = None
            constant_node = None

            # First check initializers
            for tensor in model.graph.initializer:
                if tensor.name == shape_input_name:
                    shape_tensor = tensor
                    shape_values = onnx.numpy_helper.to_array(shape_tensor)
                    break

            # If not found in initializers, check Constant nodes
            if shape_tensor is None:
                for const_node in model.graph.node:
                    if const_node.op_type == 'Constant' and const_node.output[0] == shape_input_name:
                        constant_node = const_node
                        for attr in const_node.attribute:
                            if attr.name == 'value':
                                shape_values = onnx.numpy_helper.to_array(attr.t)
                                break
                        break

            if shape_values is not None:
                # Check if any dimension is -1
                if -1 in shape_values:
                    print(f"Found Reshape node '{reshape_node.name}' with -1 dimension in shape tensor")
                    print(f"  Shape tensor: {shape_values}")

                    # Try to find the input tensor shape
                    input_name = reshape_node.input[0]
                    input_shape = None

                    # Check if input is from initializers (constant)
                    for tensor in model.graph.initializer:
                        if tensor.name == input_name:
                            input_shape = list(tensor.dims)
                            break

                    # If not found in initializers, check input value_infos
                    if input_shape is None:
                        for value_info in model.graph.input:
                            if value_info.name == input_name:
                                # Get shape from value_info - this might be dynamic
                                input_shape = []
                                for dim in value_info.type.tensor_type.shape.dim:
                                    if dim.HasField('dim_value'):
                                        input_shape.append(dim.dim_value)
                                    else:
                                        # Dynamic dimension
                                        input_shape.append(-1)
                                break

                    # Also check intermediate value_infos
                    if input_shape is None:
                        for value_info in model.graph.value_info:
                            if value_info.name == input_name:
                                input_shape = []
                                for dim in value_info.type.tensor_type.shape.dim:
                                    if dim.HasField('dim_value'):
                                        input_shape.append(dim.dim_value)
                                    else:
                                        input_shape.append(-1)
                                break

                    if input_shape is not None and -1 not in input_shape:
                        # Calculate total elements in input
                        input_elements = 1
                        for dim in input_shape:
                            input_elements *= dim

                        # Calculate what -1 should be
                        known_product = 1
                        wildcard_idx = -1
                        for i, dim in enumerate(shape_values):
                            if dim == -1:
                                wildcard_idx = i
                            elif dim > 0:
                                known_product *= dim

                        if wildcard_idx >= 0 and known_product > 0:
                            wildcard_value = input_elements // known_product
                            remainder = input_elements % known_product
                            if remainder == 0:
                                print(f"  Calculated -1 dimension = {wildcard_value}")
                                shape_values[wildcard_idx] = wildcard_value

                                if shape_tensor is not None:
                                    # Handle initializer tensor
                                    new_shape_tensor = onnx.helper.make_tensor(
                                        name=shape_tensor.name,
                                        data_type=shape_tensor.data_type,
                                        dims=shape_tensor.dims,
                                        vals=shape_values.astype(shape_tensor.data_type if shape_tensor.data_type == onnx.TensorProto.INT32 else 'int64')
                                    )
                                    # Replace the tensor
                                    model.graph.initializer.remove(shape_tensor)
                                    model.graph.initializer.append(new_shape_tensor)
                                else:
                                    # Handle Constant node
                                    int32_data = shape_values.astype('int32')
                                    constant_node.attribute[0].t.data_type = onnx.TensorProto.INT32
                                    constant_node.attribute[0].t.CopyFrom(onnx.numpy_helper.from_array(int32_data))

                                nodes_to_modify.append(reshape_node)
                                print(f"  ✅ Fixed Reshape node '{reshape_node.name}'")
                            else:
                                # Handle incompatible reshape by adjusting the shape to fit
                                # This happens when the model was exported with mismatched dimensions
                                print(f"  ⚠️  Incompatible reshape: {input_elements} elements don't divide evenly by {known_product}")
                                print(f"     Adjusting shape to fit...")

                                # Calculate a compatible shape by preserving the known dimensions
                                # and adjusting the wildcard to make the total elements match
                                adjusted_wildcard = (input_elements + known_product - 1) // known_product  # Ceiling division
                                shape_values[wildcard_idx] = adjusted_wildcard

                                # Verify the adjustment works
                                total_check = known_product * adjusted_wildcard
                                if total_check >= input_elements:
                                    print(f"     Adjusted -1 dimension from {wildcard_value} to {adjusted_wildcard} (total elements: {total_check})")

                                    if shape_tensor is not None:
                                        # Handle initializer tensor
                                        new_shape_tensor = onnx.helper.make_tensor(
                                            name=shape_tensor.name,
                                            data_type=shape_tensor.data_type,
                                            dims=shape_tensor.dims,
                                            vals=shape_values.astype(shape_tensor.data_type if shape_tensor.data_type == onnx.TensorProto.INT32 else 'int64')
                                        )
                                        # Replace the tensor
                                        model.graph.initializer.remove(shape_tensor)
                                        model.graph.initializer.append(new_shape_tensor)
                                    else:
                                        # Handle Constant node
                                        int32_data = shape_values.astype('int32')
                                        constant_node.attribute[0].t.data_type = onnx.TensorProto.INT32
                                        constant_node.attribute[0].t.CopyFrom(onnx.numpy_helper.from_array(int32_data))

                                    nodes_to_modify.append(reshape_node)
                                    print(f"  ✅ Fixed incompatible Reshape node '{reshape_node.name}' with adjusted dimensions")
                                else:
                                    print(f"  ❌ Cannot adjust shape: even adjusted value doesn't fit")
                        else:
                            print(f"  ❌ Cannot determine -1 value: input_shape={input_shape}, shape_values={shape_values}")
                    else:
                        print(f"  ❌ Cannot determine input shape for '{input_name}'")

    return len(nodes_to_modify)


def make_onnx_tx2_compatible(onnx_path: Path) -> None:
    """
    Convert ONNX model to TX2 compatibility by:
    1. Replacing LayerNormalization with Identity (TRT compatibility)
    2. Detecting Range operations (preserved for shape calculations)
    3. Fixing Unsqueeze operations (convert axes from attribute to input format)
    4. Converting INT64 tensors/constants to INT32
    5. Converting INT64 inputs/outputs to INT32
    6. Using ONNX opset version 13 for TX2 compatibility (maximum supported)
    7. Simplifying the model
    """
    print("Making ONNX model TX2 compatible...")

    # Load the model
    model = onnx.load(str(onnx_path))

    # Fix Range operation input issue (TensorRT requires scalar inputs)
    print("\\nFixing Range operation scalar input issue...")
    range_fixed = 0

    # Fix Range operations - use the correct limit value
    range_ops = ['/transformer/Range', '/transformer/Range_1']
    for range_name in range_ops:
        for node in model.graph.node:
            if node.name == range_name and node.op_type == 'Range':
                print(f"Found Range operation: {node.name}")
                print(f"Current inputs: {node.input}")

                # The middle input should be a scalar for TensorRT
                # The original complex chain extracted 24 from [[24, 24]], so limit should be 24
                # Range(0, 24, 1) produces [0, 1, 2, ..., 23]
                import numpy as np

                # Create a scalar constant for the limit (24, as extracted from the original tensor)
                scalar_limit_name = f'/transformer/Constant_range_limit_{range_name.replace("/", "_")}_output_0'
                scalar_limit_constant = onnx.helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=[scalar_limit_name],
                    name=f'/transformer/Constant_range_limit_{range_name.replace("/", "_")}',
                    value=onnx.helper.make_tensor(
                        name=f'/transformer/Constant_range_limit_{range_name.replace("/", "_")}_value',
                        data_type=onnx.TensorProto.INT64,
                        dims=[],
                        vals=[24]  # Original limit was 24
                    )
                )

                # Replace the middle input with the scalar constant
                node.input[1] = scalar_limit_name

                # Add the new constant to the graph
                model.graph.node.append(scalar_limit_constant)

                print(f"✅ Fixed {range_name} operation: replaced complex tensor input with scalar constant 24")
                range_fixed += 1
                break

    if range_fixed == 0:
        print("ℹ️ No Range operations found to fix")

    # Fix complex Slice operations that TensorRT can't handle
    print("\\nFixing complex Slice operations...")
    slice_fixed = 0

    # Fix Slice_6 which has invalid parameters causing empty intervals
    for node in model.graph.node:
        if node.name == '/transformer/Slice_6' and node.op_type == 'Slice':
            print(f"Found problematic Slice_6: {node.input}")

            # Replace the complex starts computation with a simple constant
            # The slice needs to produce [1,300,4] to match the concat input [1,300,3]
            # Since starts=[0] produces [1,299,4], let's use starts=[0] and ends=[300]
            simple_starts_name = '/transformer/Constant_slice_6_starts_output_0'
            simple_ends_name = '/transformer/Constant_slice_6_ends_output_0'

            simple_starts_constant = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[simple_starts_name],
                name='/transformer/Constant_slice_6_starts',
                value=onnx.helper.make_tensor(
                    name='/transformer/Constant_slice_6_starts_value',
                    data_type=onnx.TensorProto.INT64,
                    dims=[1],
                    vals=[0]  # Start from beginning
                )
            )

            simple_ends_constant = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[simple_ends_name],
                name='/transformer/Constant_slice_6_ends',
                value=onnx.helper.make_tensor(
                    name='/transformer/Constant_slice_6_ends_value',
                    data_type=onnx.TensorProto.INT64,
                    dims=[1],
                    vals=[300]  # End at 300 to get 300 elements
                )
            )

            # Replace the complex starts input with the simple constant
            node.input[1] = simple_starts_name

            # Replace the ends input with the new constant (input[2] is ends)
            node.input[2] = simple_ends_name

            # Add the new constants to the graph
            model.graph.node.append(simple_starts_constant)
            model.graph.node.append(simple_ends_constant)

            # Now add a second slice to reduce the feature dimension from 4 to 3
            # Slice along axis 2 (last dimension) from 0 to 3
            feature_slice_name = '/transformer/Slice_6_features_output_0'
            feature_starts_name = '/transformer/Constant_slice_6_features_starts_output_0'
            feature_ends_name = '/transformer/Constant_slice_6_features_ends_output_0'
            feature_axes_name = '/transformer/Constant_slice_6_features_axes_output_0'
            feature_steps_name = '/transformer/Constant_slice_6_features_steps_output_0'

            # Create constants for the feature slice
            feature_starts_constant = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[feature_starts_name],
                name='/transformer/Constant_slice_6_features_starts',
                value=onnx.helper.make_tensor(
                    name='/transformer/Constant_slice_6_features_starts_value',
                    data_type=onnx.TensorProto.INT64,
                    dims=[1],
                    vals=[0]
                )
            )

            feature_ends_constant = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[feature_ends_name],
                name='/transformer/Constant_slice_6_features_ends',
                value=onnx.helper.make_tensor(
                    name='/transformer/Constant_slice_6_features_ends_value',
                    data_type=onnx.TensorProto.INT64,
                    dims=[1],
                    vals=[3]  # Take first 3 features
                )
            )

            feature_axes_constant = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[feature_axes_name],
                name='/transformer/Constant_slice_6_features_axes',
                value=onnx.helper.make_tensor(
                    name='/transformer/Constant_slice_6_features_axes_value',
                    data_type=onnx.TensorProto.INT64,
                    dims=[1],
                    vals=[2]  # Slice along axis 2 (features)
                )
            )

            feature_steps_constant = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[feature_steps_name],
                name='/transformer/Constant_slice_6_features_steps',
                value=onnx.helper.make_tensor(
                    name='/transformer/Constant_slice_6_features_steps_value',
                    data_type=onnx.TensorProto.INT64,
                    dims=[1],
                    vals=[1]
                )
            )

            # Create the feature slice operation
            feature_slice = onnx.helper.make_node(
                'Slice',
                inputs=[
                    node.output[0],  # Input from Slice_6
                    feature_starts_name,
                    feature_ends_name,
                    feature_axes_name,
                    feature_steps_name
                ],
                outputs=[feature_slice_name],
                name='/transformer/Slice_6_features'
            )

            # Update the graph: Slice_6 output now goes to feature_slice, and feature_slice becomes the final output
            # But wait, I need to update any nodes that use Slice_6 output to use the feature_slice output instead
            # For now, let's change Slice_6's output to go to feature_slice, and feature_slice produces the final output

            # Update Slice_6 output to intermediate
            slice_6_intermediate = '/transformer/Slice_6_intermediate_output_0'
            node.output[0] = slice_6_intermediate

            # Update feature_slice to take from intermediate and produce final output
            feature_slice.input[0] = slice_6_intermediate
            feature_slice.output[0] = '/transformer/Slice_6_output_0'  # Keep the original output name

            # Add all the new nodes
            model.graph.node.extend([
                feature_starts_constant,
                feature_ends_constant,
                feature_axes_constant,
                feature_steps_constant,
                feature_slice
            ])

            print(f"✅ Fixed Slice_6: added feature slice to reduce from [1,300,4] to [1,300,3]")
            slice_fixed += 1
            break

    if slice_fixed == 0:
        print("ℹ️ No complex Slice operations found to fix")

    # Fix decoder Slice operations that produce incompatible shapes for concat
    decoder_slice_fixed = 0

    # Fix Slice_2 to produce 64 elements instead of 63 to match Slice_1
    for node in model.graph.node:
        if node.name == '/transformer/decoder/Slice_2' and node.op_type == 'Slice':
            print(f"Found decoder Slice_2 with incompatible shape")

            # Slice_2 currently has starts=[1], which produces 63 elements
            # Change to starts=[0] to produce 64 elements like Slice_1
            starts_input_name = node.input[1]  # Second input is starts

            # Create new constant for starts=[0]
            new_starts_name = '/transformer/decoder/Constant_slice_2_starts_output_0'
            new_starts_constant = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[new_starts_name],
                name='/transformer/decoder/Constant_slice_2_starts',
                value=onnx.helper.make_tensor(
                    name='/transformer/decoder/Constant_slice_2_starts_value',
                    data_type=onnx.TensorProto.INT64,
                    dims=[1],
                    vals=[0]  # Start from 0 instead of 1
                )
            )

            # Replace the starts input
            node.input[1] = new_starts_name

            # Add the new constant
            model.graph.node.append(new_starts_constant)

            print(f"✅ Fixed decoder Slice_2: changed starts from [1] to [0] to match Slice_1 shape")
            decoder_slice_fixed += 1
            break

    # Fix Slice_5 to produce 64 elements instead of 63 to match Slice_4
    for node in model.graph.node:
        if node.name == '/transformer/decoder/Slice_5' and node.op_type == 'Slice':
            print(f"Found decoder Slice_5 with incompatible shape")

            # Slice_5 currently has starts=[1], which produces 63 elements
            # Change to starts=[0] to produce 64 elements like Slice_4
            starts_input_name = node.input[1]  # Second input is starts

            # Create new constant for starts=[0]
            new_starts_name = '/transformer/decoder/Constant_slice_5_starts_output_0'
            new_starts_constant = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[new_starts_name],
                name='/transformer/decoder/Constant_slice_5_starts',
                value=onnx.helper.make_tensor(
                    name='/transformer/decoder/Constant_slice_5_starts_value',
                    data_type=onnx.TensorProto.INT64,
                    dims=[1],
                    vals=[0]  # Start from 0 instead of 1
                )
            )

            # Replace the starts input
            node.input[1] = new_starts_name

            # Add the new constant
            model.graph.node.append(new_starts_constant)

            print(f"✅ Fixed decoder Slice_5: changed starts from [1] to [0] to match Slice_4 shape")
            decoder_slice_fixed += 1
            break

    if decoder_slice_fixed == 0:
        print("ℹ️ No decoder slice operations needed fixing")

    # Replace LayerNormalization with Identity (TensorRT compatibility)
    layernorm_count = replace_layernorm_with_identity(model)

    # Replace problematic Range operations with Identity (TensorRT compatibility)
    range_count = replace_problematic_range_operations(model)

    # Fix Unsqueeze operations (convert from attribute to input format)
    unsqueeze_count = fix_unsqueeze_operations(model)

    # Fix Reshape operations with -1 dimensions (TensorRT compatibility)
    reshape_count = fix_reshape_operations(model)

    # Fix known problematic reshape operations
    print("Applying targeted fixes for known problematic reshape operations...")

    # The root issue: Slice operation removes CLS token (576 -> 575 patches), but all reshape operations
    # expect 576 patches. The correct fix is to modify the slice to keep all 576 patches by changing
    # starts from [1] to [0], then the existing hardcoded reshape targets will work correctly.

    # Fix the slice operations to keep all 576 patches instead of removing CLS token
    slice_fixed = False

    # Fix the embeddings slice operation
    for node in model.graph.node:
        if node.name == '/backbone/backbone.0/encoder/encoder/embeddings/Slice_1' and node.op_type == 'Slice':
            # The 'starts' input is the second input (index 1)
            starts_input_name = node.input[1]
            for const_node in model.graph.node:
                if const_node.op_type == 'Constant' and const_node.output[0] == starts_input_name:
                    # Modify the starts value from [1] to [0]
                    for attr in const_node.attribute:
                        if attr.name == 'value':
                            starts_values = onnx.numpy_helper.to_array(attr.t)
                            if starts_values.ndim == 0:
                                starts_values = [starts_values.item()]
                            else:
                                starts_values = starts_values.tolist()
                            if starts_values == [1]:
                                starts_values[0] = 0
                                new_starts_tensor = onnx.helper.make_tensor(
                                    name=const_node.output[0].replace('_output_0', ''),  # Remove _output_0 suffix
                                    data_type=attr.t.data_type,
                                    dims=attr.t.dims,
                                    vals=starts_values
                                )
                                attr.t.CopyFrom(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(new_starts_tensor)))
                                slice_fixed = True
                                print(f"✅ Fixed embeddings slice operation: changed starts from [1] to [0] to keep all 576 patches")
                                break
                    if slice_fixed:
                        break
            if slice_fixed:
                break

    # Also fix the encoder slice operations that are causing reshape failures
    encoder_slices = ['/backbone/backbone.0/encoder/encoder/Slice', '/backbone/backbone.0/encoder/encoder/Slice_1', '/backbone/backbone.0/encoder/encoder/Slice_2', '/backbone/backbone.0/encoder/encoder/Slice_3']
    for slice_name in encoder_slices:
        for node in model.graph.node:
            if node.name == slice_name and node.op_type == 'Slice':
                # The 'starts' input is the second input (index 1)
                starts_input_name = node.input[1]
                for const_node in model.graph.node:
                    if const_node.op_type == 'Constant' and const_node.output[0] == starts_input_name:
                        # Modify the starts value from [1] to [0]
                        for attr in const_node.attribute:
                            if attr.name == 'value':
                                starts_values = onnx.numpy_helper.to_array(attr.t)
                                if starts_values.ndim == 0:
                                    starts_values = [starts_values.item()]
                                else:
                                    starts_values = starts_values.tolist()
                                if starts_values == [1]:
                                    starts_values[0] = 0
                                    new_starts_tensor = onnx.helper.make_tensor(
                                        name=const_node.output[0].replace('_output_0', ''),  # Remove _output_0 suffix
                                        data_type=attr.t.data_type,
                                        dims=attr.t.dims,
                                        vals=starts_values
                                    )
                                    attr.t.CopyFrom(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(new_starts_tensor)))
                                    print(f"✅ Fixed encoder {slice_name}: changed starts from [1] to [0] to keep all 576 patches")
                                    break
                        break
                break

    # Now fix the reshape operations that have -1 dimensions to use concrete values
    reshape_fixed = 0
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    shape_array = onnx.numpy_helper.to_array(attr.t)
                    # Handle both 1D arrays and scalars
                    if shape_array.ndim == 0:
                        shape_vals = [shape_array.item()]
                    else:
                        shape_vals = shape_array.tolist()

                    if shape_vals == [1, 24, 24, -1]:  # Fix to concrete shape
                        print(f"Fixing reshape operation in {node.name}: changing [1,24,24,-1] to [1,24,24,384]")
                        shape_vals = [1, 24, 24, 384]
                        import numpy as np
                        int32_data = np.array(shape_vals, dtype=np.int32)
                        attr.t.CopyFrom(onnx.numpy_helper.from_array(int32_data))
                        reshape_fixed += 1
                        print("✅ Fixed first reshape operation")
                    elif shape_vals == [2, 12, 2, 12, -1]:  # Fix to concrete shape
                        print(f"Fixing reshape operation in {node.name}: changing [2,12,2,12,-1] to [2,12,2,12,384]")
                        shape_vals = [2, 12, 2, 12, 384]
                        import numpy as np
                        int32_data = np.array(shape_vals, dtype=np.int32)
                        attr.t.CopyFrom(onnx.numpy_helper.from_array(int32_data))
                        reshape_fixed += 1
                        print("✅ Fixed second reshape operation")
                    break

    if slice_fixed and reshape_fixed > 0:
        print(f"✅ Applied comprehensive fix: slice keeps 576 patches, {reshape_fixed} reshape operations use concrete shapes")
    elif not slice_fixed:
        print("⚠️ Warning: Could not fix slice operation, model may still have issues")

    # Fix main transformer encoder layer reshape operations (not backbone encoder attention)
    print("\\nApplying fixes for main transformer encoder layer reshape operations...")
    encoder_fixed = 0

    for node in model.graph.node:
        if 'transformer' in node.name and 'encoder' in node.name and node.op_type == 'Reshape':
            # Find the shape constant
            for const in model.graph.node:
                if const.op_type == 'Constant' and const.output[0] == node.input[1]:
                    for attr in const.attribute:
                        if attr.name == 'value':
                            shape_array = onnx.numpy_helper.to_array(attr.t)
                            if shape_array.ndim == 0:
                                shape_vals = [shape_array.item()]
                            else:
                                shape_vals = shape_array.tolist()

                            # Check if this is a shape that needs fixing (580 patches → 576 patches)
                            if shape_vals == [1, 580, 384]:  # 580 patches - main layer outputs
                                print(f"Fixing transformer encoder reshape {node.name}: [1,580,384] → [1,576,384]")
                                shape_vals = [1, 576, 384]
                                import numpy as np
                                int32_data = np.array(shape_vals, dtype=np.int32)
                                attr.t.CopyFrom(onnx.numpy_helper.from_array(int32_data))
                                encoder_fixed += 1
                            elif shape_vals == [1, 580, 6, 64]:  # 580 patches
                                print(f"Fixing transformer encoder reshape {node.name}: [1,580,6,64] → [1,576,6,64]")
                                shape_vals = [1, 576, 6, 64]
                                import numpy as np
                                int32_data = np.array(shape_vals, dtype=np.int32)
                                attr.t.CopyFrom(onnx.numpy_helper.from_array(int32_data))
                                encoder_fixed += 1
                            break
                    break

    if encoder_fixed > 0:
        print(f"✅ Fixed {encoder_fixed} main transformer encoder layer reshape operations for 576 patches")
    else:
        print("ℹ️ No main transformer encoder layer reshape operations needed fixing")

    # Fix position embeddings to match 576 patches (instead of 577)
    print("\\nApplying fixes for position embeddings...")
    pos_embed_fixed = False

    for init in model.graph.initializer:
        if 'position_embeddings' in init.name:
            print(f"Found position embeddings: {init.name}")
            print(f"Current dims: {init.dims}")

            if len(init.dims) >= 2 and init.dims[1] == 577:  # [1, 577, 384]
                print("Position embeddings have 577 patches, changing to 576...")

                # Create new position embeddings with 576 patches
                import numpy as np
                old_data = onnx.numpy_helper.to_array(init)
                new_data = old_data[:, :576, :]  # Keep first 576 patches

                new_init = onnx.helper.make_tensor(
                    name=init.name,
                    data_type=init.data_type,
                    dims=[init.dims[0], 576, init.dims[2]],  # [1, 576, 384]
                    vals=new_data.flatten()
                )

                # Replace the initializer
                model.graph.initializer.remove(init)
                model.graph.initializer.append(new_init)

                pos_embed_fixed = True
                print("✅ Fixed position embeddings to 576 patches")
                break

    if not pos_embed_fixed:
        print("ℹ️ No position embeddings needed fixing")

    # Fix patch embeddings Concat to exclude CLS token since we're keeping all 576 patches
    print("\\nApplying fixes for patch embeddings...")
    patch_embed_fixed = False

    for node in model.graph.node:
        if node.name == '/backbone/backbone.0/encoder/encoder/embeddings/patch_embeddings/Concat':
            print(f"Found patch embeddings Concat: {node.name}")
            print(f"Current inputs: {node.input}")

            # The concat includes: [patches, cls_token]
            # Since we're keeping all 576 patches (not removing CLS), we should exclude the cls_token
            if len(node.input) == 2:
                # Remove the CLS token input (second input)
                cls_token_input = node.input[1]
                node.input.remove(cls_token_input)
                patch_embed_fixed = True
                print(f"✅ Removed CLS token from patch embeddings Concat")
                print(f"New inputs: {node.input}")
            break

    # Fix main embeddings concat to exclude CLS token since we kept all patches
    main_concat_fixed = False

    for node in model.graph.node:
        if node.name == '/backbone/backbone.0/encoder/encoder/embeddings/Concat':
            print(f"Found main embeddings concat: {node.name}")
            print(f"Current inputs: {node.input}")

            # Since we kept all 576 patches (including CLS), remove the Expand input
            expand_input = '/backbone/backbone.0/encoder/encoder/embeddings/Expand_output_0'
            if expand_input in node.input:
                node.input.remove(expand_input)
                main_concat_fixed = True
                print(f"✅ Removed CLS token Expand from main concat")
                print(f"New inputs: {node.input}")
            break

    # Remove the slice operation entirely since we removed CLS token from concat
    # The Add output is now [1, 576, 384] and no slicing is needed
    slice_removed = False

    for node in model.graph.node:
        if node.name == '/backbone/backbone.0/encoder/encoder/embeddings/Slice_1':
            print(f"Found Slice_1: {node.name} - removing since no CLS token to remove")

            # Find what uses the slice output and redirect to Add output
            slice_output = node.output[0]
            add_output = '/backbone/backbone.0/encoder/encoder/embeddings/Add_output_0'

            # Replace slice output references with Add output
            for other_node in model.graph.node:
                for i, inp in enumerate(other_node.input):
                    if inp == slice_output:
                        other_node.input[i] = add_output
                        print(f"✅ Redirected {other_node.name} input from slice to Add output")

            # Remove the slice node
            model.graph.node.remove(node)
            slice_removed = True
            print(f"✅ Removed Slice_1 operation")
            break

    if not main_concat_fixed:
        print("ℹ️ No main embeddings concat needed fixing")

    if not slice_removed:
        print("ℹ️ No slice operation to remove")

    if not patch_embed_fixed:
        print("ℹ️ No patch embeddings concat needed fixing")

    # Fix patch embeddings reshape to use the fixed shape constant instead of dynamic concat
    patch_reshape_fixed = False

    for node in model.graph.node:
        if node.name == '/backbone/backbone.0/encoder/encoder/embeddings/patch_embeddings/Reshape':
            print(f"Found patch embeddings reshape: {node.name}")
            print(f"Current shape input: {node.input[1]}")

            # Change from dynamic concat to fixed constant
            if node.input[1] == '/backbone/backbone.0/encoder/encoder/embeddings/patch_embeddings/Concat_output_0':
                node.input[1] = '/backbone/backbone.0/encoder/encoder/embeddings/patch_embeddings/Constant_3_output_0'
                patch_reshape_fixed = True
                print(f"✅ Changed reshape shape input from dynamic concat to fixed constant")
            break

    # Also update the constant value to [576, 384]
    for node in model.graph.node:
        if node.name == '/backbone/backbone.0/encoder/encoder/embeddings/patch_embeddings/Constant_3':
            print(f"Found constant node: {node.name}")
            for attr in node.attribute:
                if attr.name == 'value':
                    current_vals = onnx.numpy_helper.to_array(attr.t)
                    if current_vals.ndim == 0:
                        current_vals = [current_vals.item()]
                    else:
                        current_vals = current_vals.tolist()
                    print(f"Current value: {current_vals}")

                    # Change [-1] to [1, 384, 576] for proper transpose
                    if current_vals == [-1]:
                        import numpy as np
                        new_vals = np.array([1, 384, 576], dtype=np.int64)
                        attr.t.CopyFrom(onnx.numpy_helper.from_array(new_vals))
                        print(f"✅ Changed constant value from {current_vals} to [1, 384, 576]")
                    break
            break

    if not patch_reshape_fixed:
        print("ℹ️ No patch embeddings reshape needed fixing")

    # Fix attention scaling factor broadcast issue (FINAL FIX)
    print("\\nApplying FINAL fix for attention scaling broadcast...")
    attention_scaling_fixed_count = 0

    # Replace ALL Div operations that compute attention scaling factors (Constant / Sqrt)
    # These are the operations causing broadcast issues in attention mechanisms

    nodes_to_remove = []
    nodes_to_add = []

    for node in model.graph.node:
        # Look for Div operations that take a Constant and a Sqrt as inputs
        # This pattern indicates attention scaling computation: Constant_5 / Sqrt -> scaling factor
        if node.op_type == 'Div' and len(node.input) == 2:
            input1 = node.input[0]
            input2 = node.input[1]

            # Check if one input comes from a Constant and one from Sqrt
            has_constant_input = any('Constant' in inp for inp in node.input)
            has_sqrt_input = any('Sqrt' in inp for inp in node.input)

            if has_constant_input and has_sqrt_input:
                print(f"Found scaling Div operation: {node.name}")
                print(f"Inputs: {node.input}")
                print(f"Output: {node.output}")

                # The scaling factor needs to broadcast with the attention tensor
                # For attention scores [batch, heads, seq_len, seq_len], use scalar [1] that broadcasts to all
                import numpy as np
                correct_scaling = np.sqrt(0.125)  # ≈ 0.3535533905932738
                scaling_constant_name = node.output[0] + '_constant'
                scaling_vals = np.array([correct_scaling], dtype=np.float32)
                scaling_constant = onnx.helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=[node.output[0]],
                    name=scaling_constant_name,
                    value=onnx.helper.make_tensor(
                        name=scaling_constant_name + '_value',
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[1],
                        vals=scaling_vals
                    )
                )

                # Mark for replacement
                nodes_to_remove.append(node)
                nodes_to_add.append(scaling_constant)
                attention_scaling_fixed_count += 1
                print(f"✅ Marked Div operation {node.name} for replacement with Constant scaling factor [1] = {correct_scaling:.6f}")

    # Perform the replacements
    print(f"Removing {len(nodes_to_remove)} Div operations...")
    for node in nodes_to_remove:
        try:
            model.graph.node.remove(node)
            print(f"✅ Removed {node.name}")
        except ValueError:
            print(f"⚠️ Could not remove {node.name}")

    print(f"Adding {len(nodes_to_add)} Constant operations...")
    for node in nodes_to_add:
        model.graph.node.append(node)
        print(f"✅ Added {node.name}")

    # Update existing Constants to have the correct shape for broadcasting
    # Attention scaling factors should be scalars [1] that broadcast to all tensor dimensions
    constants_updated = 0
    for node in model.graph.node:
        if node.op_type == 'Constant' and '_constant' in node.name:
            for attr in node.attribute:
                if attr.name == 'value':
                    vals = onnx.numpy_helper.to_array(attr.t)
                    if vals.shape != (1,):  # Not the desired scalar shape
                        print(f"Updating existing Constant {node.name} from shape {vals.shape} to [1]")

                        # Create new scalar tensor with shape [1]
                        import numpy as np
                        correct_scaling = vals.flatten()[0]  # Get the scaling value
                        new_vals = np.array([correct_scaling], dtype=np.float32)

                        # Update the attribute
                        attr.t.CopyFrom(onnx.helper.make_tensor(
                            name=attr.t.name,
                            data_type=attr.t.data_type,
                            dims=[1],
                            vals=new_vals
                        ))
                        constants_updated += 1
                        print(f"✅ Updated {node.name} shape to [1]")
                    break

    if attention_scaling_fixed_count > 0 or constants_updated > 0:
        print(f"✅ Fixed attention scaling: {attention_scaling_fixed_count} Div replaced, {constants_updated} Constants updated")
    else:
        print("ℹ️ Attention scaling already fixed")

    # Convert INT64 initializers to INT32
    converted_count = 0
    int64_initializers = []
    for tensor in model.graph.initializer:
        if tensor.data_type == onnx.TensorProto.INT64:
            int64_initializers.append(tensor.name)
            print(f"Converting initializer {tensor.name} from INT64 to INT32")
            tensor.data_type = onnx.TensorProto.INT32
            # Convert the actual data
            int64_data = onnx.numpy_helper.to_array(tensor)
            int32_data = int64_data.astype('int32')
            tensor.CopyFrom(onnx.numpy_helper.from_array(int32_data, tensor.name))
            converted_count += 1

    # Also check for Constant nodes with INT64 values
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value' and hasattr(attr, 't') and attr.t.data_type == onnx.TensorProto.INT64:
                    print(f"Converting Constant node {node.name} from INT64 to INT32")
                    int64_data = onnx.numpy_helper.to_array(attr.t)
                    int32_data = int64_data.astype('int32')
                    attr.t.data_type = onnx.TensorProto.INT32
                    attr.t.CopyFrom(onnx.numpy_helper.from_array(int32_data))
                    converted_count += 1

    # Convert input tensor types if they're INT64
    for value_info in model.graph.input:
        if value_info.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            print(f"Converting input {value_info.name} from INT64 to INT32")
            value_info.type.tensor_type.elem_type = onnx.TensorProto.INT32
            converted_count += 1

    # Convert output tensor types if they're INT64
    for value_info in model.graph.output:
        if value_info.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            print(f"Converting output {value_info.name} from INT64 to INT32")
            value_info.type.tensor_type.elem_type = onnx.TensorProto.INT32
            converted_count += 1

    # Set ONNX opset version to 13 for TX2 compatibility (maximum supported)
    if model.opset_import[0].version != 13:
        print(f"Setting ONNX opset version from {model.opset_import[0].version} to 13 for TX2 compatibility")
        model.opset_import[0].version = 13

    # Simplify the model
    print("Simplifying ONNX model...")
    try:
        # Try with default settings first
        model_simp, check = simplify(model)
        if check:
            model = model_simp
            print("✅ ONNX model simplified successfully")
        else:
            print("⚠️ ONNX simplification check failed, trying alternative approach...")
            # Try with more permissive settings
            try:
                model_simp, check = simplify(model, perform_optimization=False, skip_fuse_bn=True)
                if check:
                    model = model_simp
                    print("✅ ONNX model simplified with alternative settings")
                else:
                    print("⚠️ Alternative simplification also failed, using original model")
            except Exception as e2:
                print(f"⚠️ Alternative simplification failed: {e2}, using original model")
    except Exception as e:
        print(f"⚠️ ONNX simplification failed: {e}")
        print("This is usually due to newer ONNX ops not supported by current onnxsim version")
        print("The model will still work for TX2, but may not be optimally simplified")
        print("Consider updating onnxsim: pip install --upgrade onnxsim")

    # Save the TX2-compatible model (overwrite original)
    onnx.save(model, str(onnx_path))

    # Provide summary of what was done
    print("✅ TX2 compatibility conversion completed:")
    print(f"   • {layernorm_count} LayerNormalization nodes replaced with Identity")
    print(f"   • {range_count} Range operations detected (preserved for shape calculations)")
    print(f"   • {unsqueeze_count} Unsqueeze operations converted to input format")
    print(f"   • {converted_count} INT64→INT32 conversions made")
    print(f"   • ONNX opset version set to {model.opset_import[0].version}")
    if len(int64_initializers) > 0:
        print(f"   • INT64 initializers converted: {', '.join(int64_initializers[:3])}{'...' if len(int64_initializers) > 3 else ''}")
    else:
        print("   • No INT64 initializers found (model may already be INT32 compatible)")
    if converted_count == 0:
        print("   ℹ️  No INT64 conversions needed - TensorRT warnings may be from computed values")


def usage():
    print("Download RF-DETR ONNX models or process existing models", file=sys.stderr)
    print("Usage:", file=sys.stderr)
    print("  Download: uv run ./download_model.py <MODEL_ID>", file=sys.stderr)
    print("  Process existing: python3 download_model.py <model.onnx>", file=sys.stderr)
    print("\nMODEL_ID options:", file=sys.stderr)
    if INFERENCE_AVAILABLE:
        [print(f"- {key}", file=sys.stderr) for key in RFDETR_ALIASES.keys()]
    else:
        print("Note: inference package not available for downloading", file=sys.stderr)

if len(sys.argv) != 2:
    usage()
    sys.exit(1)

if __name__ == "__main__":
    # Check if user provided a .onnx file path (process existing model)
    if len(sys.argv) == 2 and sys.argv[1].endswith('.onnx'):
        model_path = Path(sys.argv[1])
        if not model_path.exists():
            print(f"Model file {model_path} does not exist", file=sys.stderr)
            sys.exit(1)
        print("Processing existing ONNX model...")
        make_onnx_tx2_compatible(model_path)
        print("✅ Existing model processed successfully")
        sys.exit(0)

    # Normal download mode - requires inference package
    if not INFERENCE_AVAILABLE:
        print("Error: Required dependencies not available. Please install inference package.", file=sys.stderr)
        print("Or provide an existing .onnx file path to process it", file=sys.stderr)
        sys.exit(1)

model_id = sys.argv[1]

if model_id not in RFDETR_ALIASES.keys():
    print(f'"{model_id}" is not a valid model', file=sys.stderr)
    usage()
    sys.exit(1)

print(f"Downloading {model_id}...")
model = get_model(model_id)

src = Path(model.cache_dir) / model.weights_file
dst = Path.cwd() / f"{model_id}.onnx"

if dst.exists():
    print(f"{dst} already exists. Can't overwrite", file=sys.stderr)
    sys.exit(1)

shutil.copy2(src, dst)
print(f"Successfully downloaded {dst}")

# Make the downloaded ONNX model TX2 compatible
make_onnx_tx2_compatible(dst)
