`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: fc_layer_fc2
// Description: Wrapper for fc2 FC layer with ROMs
//------------------------------------------------------------------------------
module fc_layer_fc2 #(
    parameter int NUM_NEURONS   = 16,
    parameter int INPUT_SIZE    = 16,
    parameter int DATA_WIDTH    = 16,
    parameter int WEIGHT_WIDTH  = 8,
    parameter int ACC_WIDTH     = 32,
    parameter int FRAC_BITS     = 8
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
    weight_rom_fc2 u_weight_rom (
        .addr(weight_addr),
        .data(weight_row)
    );

    // Bias ROM instantiation
    bias_rom_fc2 u_bias_rom (
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
