`timescale 1ns/1ps

module binaryclass_NN #(
    parameter int DATA_WIDTH       = 8,
    parameter int FC1_NEURONS      = 30,
    parameter int FC2_NEURONS      = 5,
    parameter int FC3_NEURONS      = 1,
    parameter int FC1_INPUT_SIZE   = 720,
    parameter int FC2_INPUT_SIZE   = 45,
    parameter int FC3_INPUT_SIZE   = 5,
    parameter int WEIGHT_WIDTH     = 8,
    parameter int BIAS_WIDTH       = 32,
    parameter int ACC_WIDTH        = 32,
    parameter int OUT_WIDTH        = 32,
    parameter int FC1_ROM_DEPTH    = 45,
    parameter int FC2_ROM_DEPTH    = 5,
    parameter int FC3_ROM_DEPTH    = 1
)(
    input  logic                      clk,
    input  logic                      rst_n,

    // AXI4-Stream input
    input  logic [DATA_WIDTH-1:0]     axi_in_tdata,
    input  logic                      axi_in_tvalid,
    input  logic                      axi_in_tlast,

    // AXI4-Stream output
    output logic [OUT_WIDTH-1:0]      predictions,
    output logic                      predictions_valid,
    output logic                      predictions_tlast
);

    // ------------------------------------------------------------
    // Layer 1 signals
    // ------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] fc1_out  [FC1_NEURONS];
    logic                  fc1_valid_in, fc1_valid_out;

    logic signed [OUT_WIDTH-1:0] fc1_relu_in;
	logic signed [OUT_WIDTH-1:0] fc1_relu_out;
    logic                  fc1_relu_valid_in, fc1_relu_valid_out;

    // ------------------------------------------------------------
    // Layer 2 signals
    // ------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] fc2_out  [FC2_NEURONS];
    logic                  fc2_valid_in, fc2_valid_out;

    logic signed [OUT_WIDTH-1:0] fc2_relu_in;
	logic signed [OUT_WIDTH-1:0]   fc2_relu_out;
    logic                  fc2_relu_valid_in, fc2_relu_valid_out;

    // ------------------------------------------------------------
    // Layer 3 signals
    // ------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] fc3_out  [FC3_NEURONS];
    logic                  fc3_valid_in, fc3_valid_out;

    // ------------------------------------------------------------
    // tlast propagation
    // ------------------------------------------------------------
    logic fc1_tlast, fc2_tlast, fc3_tlast;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fc1_tlast <= 1'b0;
            fc2_tlast <= 1'b0;
            fc3_tlast <= 1'b0;
        end else begin
            if (axi_in_tvalid) fc1_tlast <= axi_in_tlast;
            else if (fc1_valid_out & fc1_tlast) fc1_tlast <= 1'b0;

            if (fc1_relu_valid_out) fc2_tlast <= fc1_tlast;
            else if (fc2_valid_out & fc2_tlast) fc2_tlast <= 1'b0;

            if (fc2_relu_valid_out) fc3_tlast <= fc2_tlast;
            else if (fc3_valid_out & fc3_tlast) fc3_tlast <= 1'b0;
        end
    end

    // ------------------------------------------------------------
    // Layer 1: fc_proj_in -> fc_proj_out -> relu
    // ------------------------------------------------------------
    fc_in_layer #(
        .NUM_NEURONS(FC1_NEURONS),
        .INPUT_SIZE(FC1_INPUT_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) fc_proj_in_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(axi_in_tvalid),
        .input_data(axi_in_tdata),
        .fc_out(fc1_out),
        .valid_out(fc1_valid_out)
    );

    fc_out_layer #(
        .NUM_INPUTS(FC1_NEURONS),
        .DATA_WIDTH(ACC_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .BIAS_WIDTH(BIAS_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .ROM_DEPTH(FC1_ROM_DEPTH)
    ) fc_proj_out_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fc1_valid_out),
        .fc_in(fc1_out),
        .fc_out(fc1_relu_in),
        .valid_out(fc1_relu_valid_in)
    );

    relu_layer #(
        .DATA_WIDTH(OUT_WIDTH)
    ) relu1_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fc1_relu_valid_in),
        .data_in(fc1_relu_in),
        .data_out(fc1_relu_out),
        .valid_out(fc1_relu_valid_out)
    );

    // ------------------------------------------------------------
    // Layer 2: fc_2_proj_in -> fc_2_proj_out -> relu
    // ------------------------------------------------------------
    fc_in_layer #(
        .NUM_NEURONS(FC2_NEURONS),
        .INPUT_SIZE(FC2_INPUT_SIZE),
        .DATA_WIDTH(OUT_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) fc_2_proj_in_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fc1_relu_valid_out),
        .input_data(fc1_relu_out),
        .fc_out(fc2_out),
        .valid_out(fc2_valid_out)
    );

    fc_out_layer #(
        .NUM_INPUTS(FC2_NEURONS),
        .DATA_WIDTH(ACC_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .BIAS_WIDTH(BIAS_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .ROM_DEPTH(FC2_ROM_DEPTH)
    ) fc_2_proj_out_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fc2_valid_out),
        .fc_in(fc2_out),
        .fc_out(fc2_relu_in),
        .valid_out(fc2_relu_valid_in)
    );

    relu_layer #(
        .DATA_WIDTH(OUT_WIDTH)
    ) relu2_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fc2_relu_valid_in),
        .data_in(fc2_relu_in),
        .data_out(fc2_relu_out),
        .valid_out(fc2_relu_valid_out)
    );

    // ------------------------------------------------------------
    // Layer 3: fc_3_proj_in -> fc_3_proj_out
    // ------------------------------------------------------------
    fc_in_layer #(
        .NUM_NEURONS(FC3_NEURONS),
        .INPUT_SIZE(FC3_INPUT_SIZE),
        .DATA_WIDTH(OUT_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) fc_3_proj_in_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fc2_relu_valid_out),
        .input_data(fc2_relu_out),
        .fc_out(fc3_out),
        .valid_out(fc3_valid_out)
    );

    fc_out_layer #(
        .NUM_INPUTS(FC3_NEURONS),
        .DATA_WIDTH(ACC_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .BIAS_WIDTH(BIAS_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .ROM_DEPTH(FC3_ROM_DEPTH)
    ) fc_3_proj_out_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fc3_valid_out),
        .fc_in(fc3_out),
        .fc_out(predictions),
        .valid_out(predictions_valid)
    );

    assign predictions_tlast = fc3_tlast;

endmodule
