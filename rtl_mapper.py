#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


# ============================================================================
# Quantization (matching QuantizedMNISTNet.py behavior)
# ============================================================================

def float_to_int16(val: np.ndarray, scale: int = 256) -> np.ndarray:
    """Quantize float array to int16: round(x * scale), clamp to [-32768, 32767]."""
    return np.clip(np.round(val.astype(np.float32) * float(scale)), -32768, 32767).astype(np.int16)


def quantize_tensor(t: torch.Tensor, scale: int = 256) -> torch.Tensor:
    """Project tensor to fixed-point grid: clamp(round(x * scale), -32768, 32767) / scale."""
    return torch.clamp((t * scale).round(), -32768, 32767) / float(scale)


# ============================================================================
# Model Loading
# ============================================================================

def load_model_from_module(module_path: Path, class_name: str) -> nn.Module:
    """Load a Python class (nn.Module) from a filesystem path."""
    module_path = module_path.resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Model module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from: {module_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_path.stem] = mod
    spec.loader.exec_module(mod)

    if not hasattr(mod, class_name):
        raise AttributeError(f"Module '{module_path}' does not define class '{class_name}'.")
    cls = getattr(mod, class_name)
    model = cls()
    if not isinstance(model, nn.Module):
        raise TypeError(f"{class_name} is not a torch.nn.Module (got {type(model)}).")
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    """Load a saved state_dict from checkpoint."""
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    sd = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)


# ============================================================================
# Model Introspection
# ============================================================================

@dataclass
class LayerInfo:
    """Information about a layer in the model."""
    name: str
    layer_type: str  # 'flatten', 'linear', 'relu'
    module_qualname: Optional[str] = None
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    weight: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    in_shape: Optional[Tuple[int, ...]] = None
    out_shape: Optional[Tuple[int, ...]] = None


def extract_layers(model: nn.Module, example_input: torch.Tensor) -> List[LayerInfo]:
    """
    Extract ordered layers (Flatten, Linear, ReLU) from model using torch.fx.
    Returns list of LayerInfo in execution order.
    """
    try:
        gm = torch.fx.symbolic_trace(model)
    except Exception as e:
        raise RuntimeError(f"torch.fx.symbolic_trace failed: {e}") from e

    # Get shapes using ShapeProp
    try:
        from torch.fx.passes.shape_prop import ShapeProp
    except Exception as e:
        raise RuntimeError(f"Failed to import ShapeProp: {e}") from e

    model.eval()
    with torch.no_grad():
        ShapeProp(gm).propagate(example_input)

    layers: List[LayerInfo] = []
    linear_counter = 0
    relu_counter = 0
    flatten_counter = 0

    for node in gm.graph.nodes:
        if node.op == "call_module":
            submod = gm.get_submodule(str(node.target))
            if isinstance(submod, nn.Flatten):
                flatten_counter += 1
                name = f"flatten_{flatten_counter}"
                layer = LayerInfo(
                    name=name,
                    layer_type="flatten",
                    module_qualname=str(node.target)
                )
            elif isinstance(submod, nn.Linear):
                linear_counter += 1
                name = f"fc{linear_counter}"
                layer = LayerInfo(
                    name=name,
                    layer_type="linear",
                    module_qualname=str(node.target),
                    in_features=submod.in_features,
                    out_features=submod.out_features,
                    weight=submod.weight.detach(),
                    bias=submod.bias.detach() if submod.bias is not None else None
                )
            elif isinstance(submod, nn.ReLU):
                relu_counter += 1
                name = f"relu_{relu_counter}"
                layer = LayerInfo(
                    name=name,
                    layer_type="relu",
                    module_qualname=str(node.target)
                )
            else:
                raise NotImplementedError(
                    f"Unsupported module type: {type(submod).__name__} at {node.target}. "
                    "Only Flatten, Linear, and ReLU are supported."
                )
        elif node.op == "call_function":
            tgt = node.target
            tgt_name = getattr(tgt, "__name__", str(tgt))
            if "flatten" in tgt_name.lower():
                flatten_counter += 1
                layer = LayerInfo(
                    name=f"flatten_{flatten_counter}",
                    layer_type="flatten"
                )
            elif "relu" in tgt_name.lower():
                relu_counter += 1
                layer = LayerInfo(
                    name=f"relu_{relu_counter}",
                    layer_type="relu"
                )
            elif "linear" in tgt_name.lower():
                linear_counter += 1
                layer = LayerInfo(
                    name=f"fc{linear_counter}",
                    layer_type="linear"
                )
            else:
                raise NotImplementedError(
                    f"Unsupported function: {tgt_name}. "
                    "Only flatten, relu, and linear are supported."
                )
        elif node.op == "call_method":
            if str(node.target) == "flatten":
                flatten_counter += 1
                layer = LayerInfo(
                    name=f"flatten_{flatten_counter}",
                    layer_type="flatten"
                )
            else:
                raise NotImplementedError(
                    f"Unsupported method: {node.target}. "
                    "Only flatten is supported."
                )
        else:
            continue  # Skip placeholder, output, etc.

        # Extract shapes from node metadata
        tm = node.meta.get("tensor_meta")
        if tm is not None and hasattr(tm, "shape"):
            layer.out_shape = tuple(int(d) for d in tm.shape)

        # Get input shape from first tensor argument
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                tm_in = arg.meta.get("tensor_meta")
                if tm_in is not None and hasattr(tm_in, "shape"):
                    layer.in_shape = tuple(int(d) for d in tm_in.shape)
                    break

        layers.append(layer)

    # Validate we have required info for linear layers
    for layer in layers:
        if layer.layer_type == "linear":
            if layer.weight is None:
                raise RuntimeError(
                    f"Linear layer {layer.name} missing weight. "
                    "Ensure model uses nn.Linear modules, not functional calls."
                )

    return layers


# ============================================================================
# Memory File Generation
# ============================================================================

def write_mem_file(path: Path, arr: np.ndarray) -> None:
    """
    Write int16 array to .mem file as hex (4 hex digits per value).
    Format: int(val) & 0xFFFF, one value per line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = arr.reshape(-1)
    with path.open("w", encoding="utf-8") as f:
        for val in flat:
            f.write(f"{int(val) & 0xFFFF:04x}\n")


def generate_weight_mem(
    weight_matrix: np.ndarray,
    out_path: Path,
    in_features: int,
    out_features: int,
    weight_width: int = 16
) -> None:
    """
    Generate packed weight .mem file.
    
    Layout: For each input feature j (0..in_features-1), pack all neuron weights:
    row_j = { W[out_features-1, j], W[out_features-2, j], ..., W[0, j] }
    
    Each weight is WEIGHT_WIDTH=16 bits.
    File format: Write weights sequentially (one per line) in the order needed for packing.
    The ROM will read them and pack into a wide word.
    """
    # Weight matrix is (out_features, in_features)
    # For each input feature j, we need all neuron weights W[:, j]
    # Write them in reverse order (out_features-1 down to 0) for each j
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for j in range(in_features):
            # Write weights for input feature j in reverse neuron order
            for neuron_idx in range(out_features - 1, -1, -1):
                weight_val = int(weight_matrix[neuron_idx, j]) & 0xFFFF
                f.write(f"{weight_val:04x}\n")


def generate_bias_mem(bias_vector: np.ndarray, out_path: Path) -> None:
    """Generate bias .mem file (one bias per line)."""
    write_mem_file(out_path, bias_vector)


# ============================================================================
# SystemVerilog Generation
# ============================================================================

def generate_sat32_to_16(out_dir: Path) -> Path:
    """Generate sat32_to_16.sv saturation module."""
    content = """`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: sat32_to_16
// Description: Saturates 32-bit signed value to 16-bit signed range
//------------------------------------------------------------------------------
module sat32_to_16 (
    input  logic signed [31:0] val_in,
    output logic signed [15:0] val_out
);

    always_comb begin
        if (val_in > 32767)
            val_out = 16'sd32767;
        else if (val_in < -32768)
            val_out = -16'sd32768;
        else
            val_out = val_in[15:0];
    end

endmodule
"""
    out_path = out_dir / "rtl" / "sat32_to_16.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_fc_in_modified(out_dir: Path, template_path: Path) -> Path:
    """Generate modified fc_in.sv with FRAC_BITS support."""
    template = template_path.read_text(encoding="utf-8")
    
    # Add FRAC_BITS parameter
    template = template.replace(
        "parameter int ACC_WIDTH    = 32    // Bit width of accumulator/output",
        "parameter int ACC_WIDTH    = 32,    // Bit width of accumulator/output\n   parameter int FRAC_BITS    = 8     // Fractional bits for fixed-point scaling"
    )
    
    # Modify bias addition to shift before adding
    old_bias_add = """            else if (out_valid)
               fc_out[i] <= mac_acc[i] + bias[i];    // Add bias to MAC result"""
    
    new_bias_add = """            else if (out_valid)
               fc_out[i] <= (mac_acc[i] >>> FRAC_BITS) + bias[i];    // Shift MAC result, then add bias"""
    
    template = template.replace(old_bias_add, new_bias_add)
    
    out_path = out_dir / "rtl" / "fc_in.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(template, encoding="utf-8")
    return out_path


def generate_weight_rom(
    layer_name: str,
    in_features: int,
    num_neurons: int,
    weight_width: int,
    out_dir: Path
) -> Path:
    """Generate weight_rom_<layer>.sv module."""
    mem_file = f"../SIM/{layer_name}_weights_packed.mem"
    packed_width = num_neurons * weight_width
    # Total memory depth: in_features rows, each with num_neurons weights
    total_depth = in_features * num_neurons
    
    content = f"""`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: weight_rom_{layer_name}
// Description: Weight ROM for {layer_name} layer
//   - Depth: {total_depth} (one weight per line in .mem file)
//   - Packed output: {packed_width} bits (all neurons for one input feature)
//------------------------------------------------------------------------------
module weight_rom_{layer_name} #(
    parameter int ADDR_WIDTH = $clog2({in_features}),
    parameter int DATA_WIDTH = {packed_width},
    parameter int NUM_NEURONS = {num_neurons},
    parameter int WEIGHT_WIDTH = {weight_width}
)(
    input  logic [ADDR_WIDTH-1:0] addr,
    output logic [DATA_WIDTH-1:0] data
);

    // Memory stores individual weights (one per line)
    logic [WEIGHT_WIDTH-1:0] mem [0:{total_depth}-1];
    
    // Pack weights for the addressed input feature
    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i++) begin : PACK_WEIGHTS
            assign data[i*WEIGHT_WIDTH +: WEIGHT_WIDTH] = 
                mem[addr * NUM_NEURONS + (NUM_NEURONS - 1 - i)];
        end
    endgenerate

    initial begin
        $readmemh("{mem_file}", mem);
    end

endmodule
"""
    out_path = out_dir / "rtl" / f"weight_rom_{layer_name}.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_bias_rom(
    layer_name: str,
    num_neurons: int,
    acc_width: int,
    out_dir: Path
) -> Path:
    """Generate bias_rom_<layer>.sv module."""
    mem_file = f"../SIM/{layer_name}_biases.mem"
    packed_width = num_neurons * acc_width
    
    content = f"""`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: bias_rom_{layer_name}
// Description: Bias ROM for {layer_name} layer
//   - Width: {packed_width} bits (packed: {num_neurons} neurons × {acc_width} bits)
//------------------------------------------------------------------------------
module bias_rom_{layer_name} #(
    parameter int DATA_WIDTH = {packed_width}
)(
    output logic [DATA_WIDTH-1:0] data
);

    logic [DATA_WIDTH-1:0] mem [0:0];

    initial begin
        $readmemh("{mem_file}", mem);
    end

    assign data = mem[0];

endmodule
"""
    out_path = out_dir / "rtl" / f"bias_rom_{layer_name}.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_fc_layer_wrapper(
    layer_name: str,
    in_features: int,
    num_neurons: int,
    data_width: int,
    weight_width: int,
    acc_width: int,
    frac_bits: int,
    out_dir: Path
) -> Path:
    """Generate fc_layer_<layer>.sv wrapper module."""
    content = f"""`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: fc_layer_{layer_name}
// Description: Wrapper for {layer_name} FC layer with ROMs
//------------------------------------------------------------------------------
module fc_layer_{layer_name} #(
    parameter int NUM_NEURONS   = {num_neurons},
    parameter int INPUT_SIZE    = {in_features},
    parameter int DATA_WIDTH    = {data_width},
    parameter int WEIGHT_WIDTH  = {weight_width},
    parameter int ACC_WIDTH     = {acc_width},
    parameter int FRAC_BITS     = {frac_bits}
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_in,
    input  logic signed [DATA_WIDTH-1:0] input_data,
    output logic signed [ACC_WIDTH-1:0]   fc_out [NUM_NEURONS],
    output logic                         valid_out
);

    // ROM interfaces
    logic [$clog2(INPUT_SIZE)-1:0] weight_addr;
    logic [NUM_NEURONS*WEIGHT_WIDTH-1:0] weight_row;
    logic [NUM_NEURONS*ACC_WIDTH-1:0] bias_row;

    // Sliced weights and biases
    logic signed [WEIGHT_WIDTH-1:0] weights [NUM_NEURONS];
    logic signed [ACC_WIDTH-1:0]    bias    [NUM_NEURONS];

    genvar i;

    // Weight ROM instantiation
    weight_rom_{layer_name} u_weight_rom (
        .addr(weight_addr),
        .data(weight_row)
    );

    // Bias ROM instantiation
    bias_rom_{layer_name} u_bias_rom (
        .data(bias_row)
    );

    // Slice weight row into individual weights
    generate
        for (i = 0; i < NUM_NEURONS; i++) begin : WEIGHT_SLICE
            assign weights[i] = weight_row[i*WEIGHT_WIDTH +: WEIGHT_WIDTH];
        end
    endgenerate

    // Slice bias row into individual biases
    generate
        for (i = 0; i < NUM_NEURONS; i++) begin : BIAS_SLICE
            assign bias[i] = bias_row[i*ACC_WIDTH +: ACC_WIDTH];
        end
    endgenerate

    // Weight ROM address counter
    always_ff @(posedge clk) begin
        if (!rst_n)
            weight_addr <= '0;
        else if (valid_in) begin
            if (weight_addr == INPUT_SIZE-1)
                weight_addr <= '0;
            else
                weight_addr <= weight_addr + 1'b1;
        end
    end

    // FC compute instantiation
    fc_in #(
        .NUM_NUERONS  (NUM_NEURONS),
        .INPUT_SIZE   (INPUT_SIZE),
        .DATA_WIDTH   (DATA_WIDTH),
        .WEIGHT_WIDTH (WEIGHT_WIDTH),
        .ACC_WIDTH    (ACC_WIDTH),
        .FRAC_BITS    (FRAC_BITS)
    ) u_fc_in (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (valid_in),
        .data_in  (input_data),
        .weights  (weights),
        .bias     (bias),
        .fc_out   (fc_out),
        .out_valid(valid_out)
    );

endmodule
"""
    out_path = out_dir / "rtl" / f"fc_layer_{layer_name}.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_top_module(
    model_name: str,
    layers: List[LayerInfo],
    input_size: int,
    data_width: int,
    weight_width: int,
    acc_width: int,
    frac_bits: int,
    out_dir: Path
) -> Path:
    """Generate top module with streaming control logic."""
    
    # Filter out flatten layers (pass-through in streaming)
    active_layers = [l for l in layers if l.layer_type != "flatten"]
    
    # Build module content
    port_decls = [
        "    // Input interface",
        "    input  logic                         clk,",
        "    input  logic                         rst_n,",
        "    input  logic                         in_valid,",
        f"    input  logic signed [{data_width-1}:0] in_data,"
    ]
    
    signal_decls = []
    instantiations = []
    connections = []
    
    # Track current stream state
    stream_valid = "in_valid"
    stream_data = "in_data"
    fc_layer_idx = 0
    prev_fc_out = None  # Track previous FC output vector name
    prev_fc_valid = None  # Track previous FC valid signal
    
    for i, layer in enumerate(active_layers):
        if layer.layer_type == "linear":
            fc_layer_idx += 1
            fc_name = layer.name
            num_neurons = layer.out_features
            in_features = layer.in_features
            
            # FC layer signals
            fc_valid_in = f"{fc_name}_valid_in"
            fc_data_in = f"{fc_name}_data_in"
            fc_valid_out = f"{fc_name}_valid_out"
            fc_out = f"{fc_name}_out"
            
            signal_decls.extend([
                f"    // {fc_name} layer signals",
                f"    logic                         {fc_valid_in};",
                f"    logic signed [{data_width-1}:0] {fc_data_in};",
                f"    logic                         {fc_valid_out};",
                f"    logic signed [{acc_width-1}:0] {fc_out} [{num_neurons-1}:0];",
                ""
            ])
            
            # Connect input stream to FC
            connections.append(f"    assign {fc_valid_in} = {stream_valid};")
            connections.append(f"    assign {fc_data_in} = {stream_data};")
            connections.append("")
            
            # FC instantiation
            instantiations.extend([
                f"    // {fc_name} layer",
                f"    fc_layer_{fc_name} #(",
                f"        .NUM_NEURONS  ({num_neurons}),",
                f"        .INPUT_SIZE   ({in_features}),",
                f"        .DATA_WIDTH   ({data_width}),",
                f"        .WEIGHT_WIDTH ({weight_width}),",
                f"        .ACC_WIDTH    ({acc_width}),",
                f"        .FRAC_BITS    ({frac_bits})",
                f"    ) u_{fc_name} (",
                f"        .clk       (clk),",
                f"        .rst_n     (rst_n),",
                f"        .valid_in  ({fc_valid_in}),",
                f"        .input_data({fc_data_in}),",
                f"        .fc_out    ({fc_out}),",
                f"        .valid_out ({fc_valid_out})",
                f"    );",
                ""
            ])
            
            # Check if next layer is ReLU
            has_relu = (i + 1 < len(active_layers) and 
                       active_layers[i + 1].layer_type == "relu")
            
            if has_relu:
                # ReLU will be handled next, store FC output for ReLU processing
                prev_fc_out = fc_out
                prev_fc_valid = fc_valid_out
            else:
                # This is the last layer, output directly
                prev_fc_out = fc_out
                prev_fc_valid = fc_valid_out
                break
                
        elif layer.layer_type == "relu":
            # Use the FC output from previous iteration
            if prev_fc_out is None or prev_fc_valid is None:
                raise RuntimeError(f"ReLU {layer.name} has no preceding FC layer output")
            
            # Find preceding FC layer to get dimensions
            prev_fc = None
            for j in range(i-1, -1, -1):
                if active_layers[j].layer_type == "linear":
                    prev_fc = active_layers[j]
                    break
            
            if prev_fc is None:
                raise RuntimeError(f"ReLU {layer.name} has no preceding FC layer")
            
            num_neurons = prev_fc.out_features
            relu_name = layer.name
            
            # ReLU signals - connect to previous FC output
            relu_in_vector = prev_fc_out  # Use the FC output vector directly
            relu_in_valid = prev_fc_valid  # Use the FC valid signal directly
            relu_out_vector = f"{relu_name}_out"
            relu_out_valid = f"{relu_name}_out_valid"
            
            signal_decls.extend([
                f"    // {relu_name} layer signals",
                f"    logic signed [{data_width-1}:0] {relu_out_vector} [{num_neurons-1}:0];",
                f"    logic                         {relu_out_valid};",
                ""
            ])
            
            # ReLU element-wise processing
            instantiations.extend([
                f"    // {relu_name} - element-wise ReLU with saturation",
                f"    genvar relu_i;",
                f"    generate",
                f"        for (relu_i = 0; relu_i < {num_neurons}; relu_i++) begin : {relu_name.upper()}_GEN",
                f"            logic signed [{acc_width-1}:0] relu_in_val;",
                f"            logic signed [{acc_width-1}:0] relu_out_32b;",
                f"            logic signed [{data_width-1}:0] relu_out_16b;",
                f"            logic relu_elem_valid;",
                f"            ",
                f"            assign relu_in_val = {relu_in_vector}[relu_i];",
                f"            ",
                f"            relu_layer #(",
                f"                .DATA_WIDTH({acc_width})",
                f"            ) u_relu_elem (",
                f"                .clk      (clk),",
                f"                .rst_n    (rst_n),",
                f"                .valid_in ({relu_in_valid}),",
                f"                .data_in  (relu_in_val),",
                f"                .data_out (relu_out_32b),",
                f"                .valid_out(relu_elem_valid)",
                f"            );",
                f"            ",
                f"            sat32_to_16 u_sat (",
                f"                .val_in (relu_out_32b),",
                f"                .val_out(relu_out_16b)",
                f"            );",
                f"            ",
                f"            // Register ReLU output when valid",
                f"            always_ff @(posedge clk) begin",
                f"                if (!rst_n) begin",
                f"                    {relu_out_vector}[relu_i] <= '0;",
                f"                end else if (relu_elem_valid) begin",
                f"                    {relu_out_vector}[relu_i] <= relu_out_16b;",
                f"                end",
                f"            end",
                f"        end",
                f"    endgenerate",
                f"    ",
                f"    // Capture valid when all ReLU outputs are ready",
                f"    always_ff @(posedge clk) begin",
                f"        if (!rst_n) begin",
                f"            {relu_out_valid} <= 1'b0;",
                f"        end else begin",
                f"            {relu_out_valid} <= {relu_in_valid};",
                f"        end",
                f"    end",
                ""
            ])
            
            # Serialize ReLU output vector for next FC layer
            serialize_count = f"{relu_name}_serialize_count"
            serialize_valid = f"{relu_name}_serialize_valid"
            serialize_data = f"{relu_name}_serialize_data"
            
            signal_decls.extend([
                f"    // Serialization for {relu_name} output",
                f"    logic [$clog2({num_neurons}+1)-1:0] {serialize_count};",
                f"    logic                         {serialize_valid};",
                f"    logic signed [{data_width-1}:0] {serialize_data};",
                ""
            ])
            
            instantiations.extend([
                f"    // Serialize {relu_name} output vector",
                f"    always_ff @(posedge clk) begin",
                f"        if (!rst_n) begin",
                f"            {serialize_count} <= '0;",
                f"            {serialize_valid} <= 1'b0;",
                f"        end else begin",
                f"            {serialize_valid} <= 1'b0;",
                f"            if ({relu_out_valid}) begin",
                f"                // Start serialization",
                f"                {serialize_count} <= '0;",
                f"                {serialize_valid} <= 1'b1;",
                f"            end else if ({serialize_count} < {num_neurons}-1) begin",
                f"                {serialize_count} <= {serialize_count} + 1'b1;",
                f"                {serialize_valid} <= 1'b1;",
                f"            end",
                f"        end",
                f"    end",
                f"    ",
                f"    assign {serialize_data} = {relu_out_vector}[{serialize_count}];",
                ""
            ])
            
            # Update stream for next layer
            stream_valid = serialize_valid
            stream_data = serialize_data
    
    # Output port
    last_fc = None
    for layer in reversed(active_layers):
        if layer.layer_type == "linear":
            last_fc = layer
            break
    
    if last_fc is None:
        raise RuntimeError("No output FC layer found")
    
    output_size = last_fc.out_features
    port_decls.extend([
        "    // Output interface",
        f"    output logic                         out_valid,",
        f"    output logic signed [{acc_width-1}:0] out_data [{output_size-1}:0]"
    ])
    
    # Connect output
    last_fc_out = f"{last_fc.name}_out"
    last_fc_valid = f"{last_fc.name}_valid_out"
    connections.extend([
        "    // Output assignment",
        f"    assign out_valid = {last_fc_valid};",
        f"    assign out_data = {last_fc_out};"
    ])
    
    # Build complete module
    content = f"""`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: {model_name}_top
// Description: Top-level module for {model_name} neural network
//   Implements streaming pipeline with proper data movement between layers
//------------------------------------------------------------------------------
module {model_name}_top #(
    parameter int DATA_WIDTH   = {data_width},
    parameter int WEIGHT_WIDTH = {weight_width},
    parameter int ACC_WIDTH    = {acc_width},
    parameter int FRAC_BITS    = {frac_bits}
)(
{chr(10).join(port_decls)}
);

{chr(10).join(signal_decls)}

{chr(10).join(connections)}

{chr(10).join(instantiations)}

endmodule
"""
    
    out_path = out_dir / "rtl" / f"{model_name}_top.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_testbench(
    model_name: str,
    input_size: int,
    output_size: int,
    data_width: int,
    acc_width: int,
    out_dir: Path
) -> Path:
    """Generate simple testbench."""
    content = f"""`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Testbench: tb_{model_name}_top
// Description: Simple testbench for {model_name}_top
//------------------------------------------------------------------------------
module tb_{model_name}_top;

    parameter int DATA_WIDTH = {data_width};
    parameter int ACC_WIDTH = {acc_width};
    parameter int INPUT_SIZE = {input_size};
    parameter int OUTPUT_SIZE = {output_size};

    logic clk;
    logic rst_n;
    logic in_valid;
    logic signed [DATA_WIDTH-1:0] in_data;
    logic out_valid;
    logic signed [ACC_WIDTH-1:0] out_data [OUTPUT_SIZE-1:0];

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Reset generation
    initial begin
        rst_n = 0;
        #100;
        rst_n = 1;
    end

    // DUT instantiation
    {model_name}_top u_dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (in_valid),
        .in_data  (in_data),
        .out_valid(out_valid),
        .out_data (out_data)
    );

    // Test stimulus
    initial begin
        in_valid = 0;
        in_data = 0;
        
        wait(rst_n);
        #20;
        
        // Feed dummy input vector
        $display("Starting input stream...");
        for (int i = 0; i < INPUT_SIZE; i++) begin
            @(posedge clk);
            in_valid = 1;
            in_data = $signed(i);  // Simple test pattern
        end
        
        @(posedge clk);
        in_valid = 0;
        
        // Wait for output
        wait(out_valid);
        @(posedge clk);
        
        $display("Output received:");
        for (int i = 0; i < OUTPUT_SIZE; i++) begin
            $display("  out_data[%0d] = %0d", i, out_data[i]);
        end
        
        #100;
        $finish;
    end

    // Monitor
    always @(posedge clk) begin
        if (out_valid) begin
            $display("[%0t] out_valid asserted", $time);
        end
    end

endmodule
"""
    out_path = out_dir / "rtl" / f"tb_{model_name}_top.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


# ============================================================================
# Copy RTL Templates
# ============================================================================

def copy_rtl_templates(template_dir: Path, out_dir: Path) -> List[Path]:
    """Copy RTL template files to output directory."""
    copied = []
    templates = ["mac.sv", "relu_layer.sv"]
    
    rtl_dir = out_dir / "rtl"
    rtl_dir.mkdir(parents=True, exist_ok=True)
    
    for template in templates:
        src = template_dir / template
        if src.exists():
            dst = rtl_dir / template
            dst.write_bytes(src.read_bytes())
            copied.append(dst)
        else:
            LOGGER.warning(f"Template {template} not found at {src}")
    
    return copied


# ============================================================================
# Report Generation
# ============================================================================

def generate_mapping_report(
    out_dir: Path,
    model_name: str,
    layers: List[LayerInfo],
    scale: int,
    data_width: int,
    weight_width: int,
    acc_width: int,
    frac_bits: int
) -> Path:
    """Generate mapping_report.txt."""
    lines = []
    lines.append("RTL Mapping Report")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Model: {model_name}")
    lines.append(f"Scale Factor: {scale}")
    lines.append(f"Data Width: {data_width}")
    lines.append(f"Weight Width: {weight_width}")
    lines.append(f"Accumulator Width: {acc_width}")
    lines.append(f"Fractional Bits: {frac_bits}")
    lines.append("")
    lines.append("Layer Mapping:")
    lines.append("-" * 80)
    
    for i, layer in enumerate(layers):
        line = f"{i+1:2d}. {layer.name:20s} {layer.layer_type:10s}"
        if layer.layer_type == "linear":
            line += f" in_features={layer.in_features:4d} out_features={layer.out_features:4d}"
            lines.append(line)
            lines.append(f"     Weights: SIM/{layer.name}_weights_packed.mem")
            lines.append(f"     Biases:  SIM/{layer.name}_biases.mem")
        elif layer.layer_type == "relu":
            line += " (element-wise)"
            lines.append(line)
        elif layer.layer_type == "flatten":
            line += " (pass-through in streaming)"
            lines.append(line)
        else:
            lines.append(line)
        lines.append("")
    
    lines.append("Generated Files:")
    lines.append("-" * 80)
    lines.append("RTL Modules:")
    lines.append("  - rtl/mac.sv")
    lines.append("  - rtl/fc_in.sv (modified with FRAC_BITS)")
    lines.append("  - rtl/relu_layer.sv")
    lines.append("  - rtl/sat32_to_16.sv")
    
    for layer in layers:
        if layer.layer_type == "linear":
            lines.append(f"  - rtl/fc_layer_{layer.name}.sv")
            lines.append(f"  - rtl/weight_rom_{layer.name}.sv")
            lines.append(f"  - rtl/bias_rom_{layer.name}.sv")
    
    lines.append(f"  - rtl/{model_name}_top.sv")
    lines.append("")
    lines.append("Memory Files:")
    for layer in layers:
        if layer.layer_type == "linear":
            lines.append(f"  - SIM/{layer.name}_weights_packed.mem")
            lines.append(f"  - SIM/{layer.name}_biases.mem")
    
    out_path = out_dir / "mapping_report.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_netlist_json(
    out_dir: Path,
    model_name: str,
    layers: List[LayerInfo]
) -> Path:
    """Generate netlist.json."""
    netlist = {
        "model_name": model_name,
        "layers": []
    }
    
    for layer in layers:
        layer_dict = {
            "name": layer.name,
            "type": layer.layer_type
        }
        if layer.layer_type == "linear":
            layer_dict["in_features"] = layer.in_features
            layer_dict["out_features"] = layer.out_features
            layer_dict["weight_mem"] = f"SIM/{layer.name}_weights_packed.mem"
            layer_dict["bias_mem"] = f"SIM/{layer.name}_biases.mem"
        netlist["layers"].append(layer_dict)
    
    out_path = out_dir / "netlist.json"
    out_path.write_text(json.dumps(netlist, indent=2), encoding="utf-8")
    return out_path


# ============================================================================
# Main CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch MLP to SystemVerilog RTL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--model-module",
        type=str,
        required=True,
        help="Path to Python file containing model class"
    )
    parser.add_argument(
        "--model-class",
        type=str,
        required=True,
        help="Model class name (e.g., SmallMNISTNet)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint .pth file"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for generated RTL and mem files"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=256,
        help="Scale factor for quantization (default: 256)"
    )
    parser.add_argument(
        "--data-width",
        type=int,
        default=16,
        help="Data width in bits (default: 16)"
    )
    parser.add_argument(
        "--weight-width",
        type=int,
        default=16,
        help="Weight width in bits (default: 16)"
    )
    parser.add_argument(
        "--acc-width",
        type=int,
        default=32,
        help="Accumulator width in bits (default: 32)"
    )
    parser.add_argument(
        "--emit-testbench",
        action="store_true",
        help="Generate testbench"
    )
    parser.add_argument(
        "--template-dir",
        type=str,
        default="to_rtl",
        help="Directory containing RTL templates (default: to_rtl)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.scale <= 0:
        raise ValueError("--scale must be positive")
    if args.data_width <= 0 or args.weight_width <= 0 or args.acc_width <= 0:
        raise ValueError("Width parameters must be positive")
    
    frac_bits = 8  # FRAC_BITS = 8 for SCALE_FACTOR = 256
    
    # Setup paths
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = out_dir / "SIM"
    sim_dir.mkdir(parents=True, exist_ok=True)
    rtl_dir = out_dir / "rtl"
    rtl_dir.mkdir(parents=True, exist_ok=True)
    
    template_dir = Path(args.template_dir).resolve()
    
    LOGGER.info("Loading model...")
    model = load_model_from_module(Path(args.model_module), args.model_class)
    load_checkpoint(model, Path(args.checkpoint))
    model.eval()
    
    # Determine input shape from model
    # For MNIST: (1, 1, 28, 28) -> flattened to 784
    example_input = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    
    LOGGER.info("Extracting layers...")
    layers = extract_layers(model, example_input)
    LOGGER.info(f"Found {len(layers)} layers")
    
    # Find input size (from first linear layer or flatten output)
    input_size = None
    for layer in layers:
        if layer.layer_type == "flatten" and layer.out_shape:
            input_size = int(np.prod(layer.out_shape[1:]))  # Skip batch dimension
            break
        elif layer.layer_type == "linear" and layer.in_features:
            input_size = layer.in_features
            break
    
    if input_size is None:
        raise RuntimeError("Could not determine input size")
    
    LOGGER.info(f"Input size: {input_size}")
    
    # Get model name
    model_name = args.model_class
    
    # Process layers: quantize and generate files
    LOGGER.info("Quantizing weights and biases...")
    for layer in layers:
        if layer.layer_type == "linear":
            # Quantize weights and biases
            weight_np = layer.weight.detach().cpu().numpy().astype(np.float32)
            bias_np = layer.bias.detach().cpu().numpy().astype(np.float32) if layer.bias is not None else np.zeros((weight_np.shape[0],), dtype=np.float32)
            
            weight_i16 = float_to_int16(weight_np, args.scale)
            bias_i16 = float_to_int16(bias_np, args.scale)
            
            # Generate .mem files
            weight_mem_path = sim_dir / f"{layer.name}_weights_packed.mem"
            bias_mem_path = sim_dir / f"{layer.name}_biases.mem"
            
            generate_weight_mem(weight_i16, weight_mem_path, layer.in_features, layer.out_features, args.weight_width)
            generate_bias_mem(bias_i16, bias_mem_path)
            
            LOGGER.info(f"  {layer.name}: {layer.in_features} -> {layer.out_features}")
    
    # Copy RTL templates
    LOGGER.info("Copying RTL templates...")
    copy_rtl_templates(template_dir, out_dir)
    
    # Generate modified fc_in.sv
    LOGGER.info("Generating modified fc_in.sv...")
    fc_in_template = template_dir / "fc_in.sv"
    if not fc_in_template.exists():
        raise FileNotFoundError(f"fc_in.sv template not found at {fc_in_template}")
    generate_fc_in_modified(out_dir, fc_in_template)
    
    # Generate saturation module
    LOGGER.info("Generating sat32_to_16.sv...")
    generate_sat32_to_16(out_dir)
    
    # Generate ROM and layer modules for each linear layer
    LOGGER.info("Generating ROM and layer modules...")
    for layer in layers:
        if layer.layer_type == "linear":
            generate_weight_rom(layer.name, layer.in_features, layer.out_features, args.weight_width, out_dir)
            generate_bias_rom(layer.name, layer.out_features, args.acc_width, out_dir)
            generate_fc_layer_wrapper(
                layer.name,
                layer.in_features,
                layer.out_features,
                args.data_width,
                args.weight_width,
                args.acc_width,
                frac_bits,
                out_dir
            )
    
    # Generate top module
    LOGGER.info("Generating top module...")
    generate_top_module(
        model_name,
        layers,
        input_size,
        args.data_width,
        args.weight_width,
        args.acc_width,
        frac_bits,
        out_dir
    )
    
    # Generate testbench if requested
    if args.emit_testbench:
        LOGGER.info("Generating testbench...")
        last_layer = None
        for layer in reversed(layers):
            if layer.layer_type == "linear":
                last_layer = layer
                break
        if last_layer:
            generate_testbench(
                model_name,
                input_size,
                last_layer.out_features,
                args.data_width,
                args.acc_width,
                out_dir
            )
    
    # Generate reports
    LOGGER.info("Generating reports...")
    generate_mapping_report(
        out_dir,
        model_name,
        layers,
        args.scale,
        args.data_width,
        args.weight_width,
        args.acc_width,
        frac_bits
    )
    generate_netlist_json(out_dir, model_name, layers)
    
    LOGGER.info(f"RTL generation complete! Output directory: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
