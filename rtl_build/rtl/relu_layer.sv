`timescale 1ns / 1ps

//------------------------------------------------------------------------------
// Module: relu_layer
// Description: Applies ReLU activation function element-wise to input array
//------------------------------------------------------------------------------
module relu_layer #(
    parameter int DATA_WIDTH   = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic signed [DATA_WIDTH-1:0] data_in ,

    output logic signed [DATA_WIDTH-1:0] data_out,
    output logic                        valid_out
);

    // Pipeline valid signal
    logic valid_pipeline;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_pipeline <= 1'b0;
        else
            valid_pipeline <= valid_in;
    end

    assign valid_out = valid_pipeline;

    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
           data_out <= '0;
           else if (valid_in)
           data_out <= (data_in < 0) ? 0 : data_in;
            end
       

endmodule
