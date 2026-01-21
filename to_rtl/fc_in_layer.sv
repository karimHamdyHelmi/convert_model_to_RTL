`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: fc_proj_in
// Description:
//   FC projection input block
//   - Reads weights from ROM per input sample
//   - Reads bias from ROM once
//   - Feeds data into fc_in
//------------------------------------------------------------------------------
module fc_in_layer #(
   parameter int NUM_NUERONS     = 8,
   parameter int INPUT_SIZE   = 16,
   parameter int DATA_WIDTH   = 8,
   parameter int WEIGHT_WIDTH = 8,
   parameter int ACC_WIDTH    = 32
)(
   input  logic                         clk,
   input  logic                         rst_n,

   input  logic                         valid_in,
   input  logic signed [DATA_WIDTH-1:0] input_data,

   output logic signed [ACC_WIDTH-1:0]  fc_out   [NUM_NUERONS],
   output logic                         valid_out
);

   //--------------------------------------------------------------------------
   // ROM interfaces
   //--------------------------------------------------------------------------

   // Weight ROM
   logic [$clog2(INPUT_SIZE)-1:0] weight_addr;
   logic [NUM_NUERONS*WEIGHT_WIDTH-1:0] weight_row;

   // Bias ROM
   logic [NUM_NUERONS*ACC_WIDTH-1:0] bias_row;

   //--------------------------------------------------------------------------
   // weights and biases
   //--------------------------------------------------------------------------

   logic signed [WEIGHT_WIDTH-1:0] weights [NUM_NUERONS];
   logic signed [ACC_WIDTH-1:0]    bias    [NUM_NUERONS];

   genvar i;

   //--------------------------------------------------------------------------
   // ROM instantiation based on INPUT_SIZE
   //--------------------------------------------------------------------------
   generate

      if (INPUT_SIZE == 720) begin : ROM_720
         weight_rom_720 u_weight_rom (
            .addr (weight_addr),
            .data (weight_row)
         );

         bias_rom_720 u_bias_rom (
            .data (bias_row)
         );
      end
      else if (INPUT_SIZE == 45) begin : ROM_45
         weight_rom_45 u_weight_rom (
            .addr (weight_addr),
            .data (weight_row)
         );

         bias_rom_45 u_bias_rom (
            .data (bias_row)
         );
      end
      else if (INPUT_SIZE == 5) begin : ROM_5
         weight_rom_5 u_weight_rom (
            .addr (weight_addr),
            .data (weight_row)
         );

         bias_rom_5 u_bias_rom (
            .data (bias_row)
         );
      end
      else begin : ROM_UNSUPPORTED
         // Catch unsupported INPUT_SIZE values at compile time
         initial begin
            $error("Unsupported INPUT_SIZE = %0d", INPUT_SIZE);
         end
      end
  endgenerate


   //--------------------------------------------------------------------------
   // Slice weight ROM row into individual weights
   //--------------------------------------------------------------------------
   generate
      for (i = 0; i < NUM_NUERONS; i++) begin : WEIGHT_SLICE
         assign weights[i] =
            weight_row[i*WEIGHT_WIDTH +: WEIGHT_WIDTH];
      end
   endgenerate

   //--------------------------------------------------------------------------
   // Slice bias ROM row into individual bias values
   //--------------------------------------------------------------------------
   generate
      for (i = 0; i < NUM_NUERONS; i++) begin : BIAS_SLICE
         assign bias[i] =
            bias_row[i*ACC_WIDTH +: ACC_WIDTH];
      end
   endgenerate

   //--------------------------------------------------------------------------
   // Input counter for weight ROM addressing
   //--------------------------------------------------------------------------
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

   //--------------------------------------------------------------------------
   // FC input layer instantiation
   //--------------------------------------------------------------------------
   fc_in #(
      .NUM_NUERONS  (NUM_NUERONS),
      .INPUT_SIZE   (INPUT_SIZE),
      .DATA_WIDTH   (DATA_WIDTH),
      .WEIGHT_WIDTH (WEIGHT_WIDTH),
      .ACC_WIDTH    (ACC_WIDTH)
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
