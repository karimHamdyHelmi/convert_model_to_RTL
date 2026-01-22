`timescale 1ns/1ps
//------------------------------------------------------------------------------
// Module: mac
// Description:
//   Pipelined Multiply-Accumulate (MAC) unit.
//   - Stage 1: Multiplies inputs 'a' and 'b'
//   - Stage 2: Accumulates the multiplication result
//------------------------------------------------------------------------------
// Parameters:
//   A_WIDTH   : Bit width of input 'a'
//   B_WIDTH   : Bit width of input 'b'
//   ACC_WIDTH : Bit width of accumulator output
//------------------------------------------------------------------------------
module mac #(
   parameter int A_WIDTH   = 8,
   parameter int B_WIDTH   = 8,
   parameter int ACC_WIDTH = 32
)(
   input  logic                         clk,     // System clock
   input  logic                         rst_n,   // Active-low synchronous reset

   input  logic                         enable,  // Enable MAC operation
   input  logic                         clear,   // Clear accumulator

   input  logic signed [A_WIDTH-1:0]    a,       // Signed multiplicand input
   input  logic signed [B_WIDTH-1:0]    b,       // Signed multiplier input

   output logic signed [ACC_WIDTH-1:0]  acc      // Accumulator output
);

   // Register to hold multiplication result (pipeline stage 1)
   // Width = A_WIDTH + B_WIDTH to prevent overflow
   logic signed [A_WIDTH+B_WIDTH-1:0] mult_reg;

   //--------------------------------------------------------------------------
   // Stage 1: Multiply
   //--------------------------------------------------------------------------
   // Performs signed multiplication of inputs 'a' and 'b'
   // Result is registered to improve timing (pipeline stage)
   always_ff @(posedge clk) begin
      if (!rst_n)
         mult_reg <= '0;               // Clear multiplication register on reset
      else if (enable)
         mult_reg <= a * b;            // Register multiplication result
   end

   //--------------------------------------------------------------------------
   // Stage 2: Accumulate
   //--------------------------------------------------------------------------
   // Adds the multiplication result to the accumulator
   always_ff @(posedge clk) begin
      if (!rst_n)
         acc <= '0;                    // Clear accumulator on reset
      else if (clear)
         acc <= '0;                    // Clear accumulator explicitly
      else if (enable)
         acc <= acc + mult_reg;        // Accumulate multiplication result
   end

endmodule