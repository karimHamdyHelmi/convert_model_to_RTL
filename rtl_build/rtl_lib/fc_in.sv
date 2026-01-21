`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: fc_in
// Description:
//   Fully Connected (FC) input layer with multiple output neurons.
//   - Each output neuron is implemented using a dedicated MAC unit
//   - Accumulates INPUT_SIZE valid input samples
//   - Adds a per-neuron bias
//   - Asserts out_valid when the FC output is ready
//
// Functionality:
//   For each neuron i:
//     fc_out[i] = bias[i] + Σ (data_in × weights[i])
//------------------------------------------------------------------------------
module fc_in #(
   parameter int NUM_NUERONS     = 8,    // Number of output neurons (MAC units)
   parameter int INPUT_SIZE   = 16,   // Number of valid inputs per FC operation
   parameter int DATA_WIDTH   = 8,    // Bit width of input data
   parameter int WEIGHT_WIDTH = 8,    // Bit width of weights
   parameter int ACC_WIDTH    = 32    // Bit width of accumulator/output
)(
   input  logic                           clk,                // System clock
   input  logic                           rst_n,              // Active-low synchronous reset

   input  logic                           in_valid,           // Indicates valid input data
   input  logic signed [DATA_WIDTH-1:0]   data_in,            // Input activation

   input  logic signed [WEIGHT_WIDTH-1:0] weights [NUM_NUERONS], // Weight array
   input  logic signed [ACC_WIDTH-1:0]    bias    [NUM_NUERONS], // Bias array

   output logic signed [ACC_WIDTH-1:0]    fc_out  [NUM_NUERONS], // FC output array
   output logic                           out_valid           // Output valid indicator
);

   //--------------------------------------------------------------------------
   // Internal signals
   //--------------------------------------------------------------------------

   // Accumulated outputs from each MAC unit (before bias addition)
   logic signed [ACC_WIDTH-1:0] mac_acc [NUM_NUERONS];

   // Counter to track number of valid inputs processed
   logic [$clog2(INPUT_SIZE+1)-1:0] in_count;

   // MAC control signals
   logic mac_enable;   // Enables MAC accumulation
   logic mac_clear;    // Clears MAC accumulator

   genvar i;

   //--------------------------------------------------------------------------
   // Instantiate one MAC unit per output neuron
   //--------------------------------------------------------------------------
   // Each MAC:
   //   - Multiplies data_in by its corresponding weight
   //   - Accumulates results over multiple valid input cycles
   generate
      for (i = 0; i < NUM_NUERONS; i++) begin : FC_MACS
         mac #(
            .A_WIDTH(DATA_WIDTH),
            .B_WIDTH(WEIGHT_WIDTH),
            .ACC_WIDTH(ACC_WIDTH)
         ) mac_inst (
            .clk    (clk),
            .rst_n  (rst_n),
            .enable (mac_enable),
            .clear  (mac_clear),
            .a      (data_in),
            .b      (weights[i]),
            .acc    (mac_acc[i])
         );
      end
   endgenerate

   //--------------------------------------------------------------------------
   // Input counter and output valid control logic
   //--------------------------------------------------------------------------
   // - Counts the number of valid input samples
   // - When INPUT_SIZE samples are received:
   //     * out_valid is asserted for one clock cycle
   //     * MAC accumulators are cleared for the next operation
   always_ff @(posedge clk) begin
      if (!rst_n) begin
         in_count  <= '0;      // Reset input counter
         out_valid <= 1'b0;    // Deassert output valid
      end else begin
         out_valid <= 1'b0;    // Default deassert

         if (in_valid) begin
            if (in_count == INPUT_SIZE-1) begin
               in_count  <= '0;  // Reset counter after full accumulation
               out_valid <= 1'b1; // Signal output is valid
            end else begin
               in_count  <= in_count + 1'b1; // Increment input counter
            end
         end
      end
   end

   // MAC control signal assignments
   assign mac_enable = in_valid;  // Accumulate only on valid input
   assign mac_clear  = out_valid; // Clear MACs after output is produced

   //--------------------------------------------------------------------------
   // Bias addition stage
   //--------------------------------------------------------------------------
   // Adds bias to each MAC output when out_valid is asserted
   generate
      for (i = 0; i < NUM_NUERONS; i++) begin : FC_BIAS_ADD
         always_ff @(posedge clk) begin
            if (!rst_n)
               fc_out[i] <= '0;                      // Clear output on reset
            else if (out_valid)
               fc_out[i] <= mac_acc[i] + bias[i];    // Add bias to MAC result
         end
      end
   endgenerate

endmodule
