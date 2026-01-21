// Auto-generated RTL wrapper skeleton by rtl_mapper.py
// NOTE: This is a skeleton. Update ports/handshake to match your RTL blocks.

module rtl_wrapper (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        in_valid,
  input  logic [15:0] in_data,   // TODO: width/packing for your input
  output logic        out_valid,
  output logic [15:0] out_data   // TODO: width/packing for your output
);

  // Internal wires (TODO: define correct widths and buses)
  logic        v0;
  logic [15:0] s0;

  // Example chaining assumes each block consumes/produces streaming samples.
  // Replace with your actual interface (AXI-stream, valid/ready, etc).

  // Input assignment
  assign v0 = in_valid;
  assign s0 = in_data;

  logic        v1;
  logic [15:0] s1;  // TODO: adjust width/bus for RTL_FLATTEN
  RTL_FLATTEN u_fx_flatten (/* TODO ports */);

  logic        v2;
  logic [15:0] s2;  // TODO: adjust width/bus for RTL_FC
  // params: weight=params/fc1_weights.mem bias=params/fc1_biases.mem
  RTL_FC u_fc1 (/* TODO ports */);

  logic        v3;
  logic [15:0] s3;  // TODO: adjust width/bus for RTL_RELU
  RTL_RELU u_fx_relu (/* TODO ports */);

  logic        v4;
  logic [15:0] s4;  // TODO: adjust width/bus for RTL_FC
  // params: weight=params/fc2_weights.mem bias=params/fc2_biases.mem
  RTL_FC u_fc2 (/* TODO ports */);

  logic        v5;
  logic [15:0] s5;  // TODO: adjust width/bus for RTL_RELU
  RTL_RELU u_fx_relu_1 (/* TODO ports */);

  logic        v6;
  logic [15:0] s6;  // TODO: adjust width/bus for RTL_FC
  // params: weight=params/fc3_weights.mem bias=params/fc3_biases.mem
  RTL_FC u_fc3 (/* TODO ports */);

  assign out_valid = v6;
  assign out_data  = s6;
endmodule
