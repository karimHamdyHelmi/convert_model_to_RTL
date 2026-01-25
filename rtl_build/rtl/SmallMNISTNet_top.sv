`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: SmallMNISTNet_top
// Description: Top-level module for SmallMNISTNet neural network
//   Implements streaming pipeline with proper data movement between layers
//------------------------------------------------------------------------------
module SmallMNISTNet_top #(
    parameter int DATA_WIDTH   = 16,
    parameter int WEIGHT_WIDTH = 8,
    parameter int ACC_WIDTH    = 32,
    parameter int FRAC_BITS    = 8
)(
    // Input interface
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         in_valid,
    input  logic signed [15:0] in_data,
    // Output interface
    output logic                         out_valid,
    output logic signed [31:0] out_data [9:0]
);

    // fc1 layer signals
    logic                         fc1_valid_in;
    logic signed [15:0] fc1_data_in;
    logic                         fc1_valid_out;
    logic signed [31:0] fc1_out [15:0];

    // relu_1 layer signals
    logic signed [15:0] relu_1_out [15:0];
    logic                         relu_1_out_valid;

    // Serialization for relu_1 output
    logic [$clog2(16+1)-1:0] relu_1_serialize_count;
    logic                         relu_1_serialize_valid;
    logic signed [15:0] relu_1_serialize_data;

    // fc2 layer signals
    logic                         fc2_valid_in;
    logic signed [15:0] fc2_data_in;
    logic                         fc2_valid_out;
    logic signed [31:0] fc2_out [15:0];

    // relu_2 layer signals
    logic signed [15:0] relu_2_out [15:0];
    logic                         relu_2_out_valid;

    // Serialization for relu_2 output
    logic [$clog2(16+1)-1:0] relu_2_serialize_count;
    logic                         relu_2_serialize_valid;
    logic signed [15:0] relu_2_serialize_data;

    // fc3 layer signals
    logic                         fc3_valid_in;
    logic signed [15:0] fc3_data_in;
    logic                         fc3_valid_out;
    logic signed [31:0] fc3_out [9:0];


    assign fc1_valid_in = in_valid;
    assign fc1_data_in = in_data;

    assign fc2_valid_in = relu_1_serialize_valid;
    assign fc2_data_in = relu_1_serialize_data;

    assign fc3_valid_in = relu_2_serialize_valid;
    assign fc3_data_in = relu_2_serialize_data;

    // Output assignment
    assign out_valid = fc3_valid_out;
    assign out_data = fc3_out;

    // fc1 layer
    fc_layer_fc1 #(
        .NUM_NEURONS  (16),
        .INPUT_SIZE   (784),
        .DATA_WIDTH   (16),
        .WEIGHT_WIDTH (8),
        .ACC_WIDTH    (32),
        .FRAC_BITS    (8)
    ) u_fc1 (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (fc1_valid_in),
        .input_data(fc1_data_in),
        .fc_out    (fc1_out),
        .valid_out (fc1_valid_out)
    );

    // relu_1 - element-wise ReLU with saturation
    genvar relu_i;
    generate
        for (relu_i = 0; relu_i < 16; relu_i++) begin : RELU_1_GEN
            logic signed [31:0] relu_in_val;
            logic signed [31:0] relu_out_32b;
            logic signed [15:0] relu_out_16b;
            logic relu_elem_valid;
            
            assign relu_in_val = fc1_out[relu_i];
            
            relu_layer #(
                .DATA_WIDTH(32)
            ) u_relu_elem (
                .clk      (clk),
                .rst_n    (rst_n),
                .valid_in (fc1_valid_out),
                .data_in  (relu_in_val),
                .data_out (relu_out_32b),
                .valid_out(relu_elem_valid)
            );
            
            sat32_to_16 u_sat (
                .val_in (relu_out_32b),
                .val_out(relu_out_16b)
            );
            
            // Register ReLU output when valid
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    relu_1_out[relu_i] <= '0;
                end else if (relu_elem_valid) begin
                    relu_1_out[relu_i] <= relu_out_16b;
                end
            end
        end
    endgenerate
    
    // Capture valid when all ReLU outputs are ready
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            relu_1_out_valid <= 1'b0;
        end else begin
            relu_1_out_valid <= fc1_valid_out;
        end
    end

    // Serialize relu_1 output vector
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            relu_1_serialize_count <= '0;
            relu_1_serialize_valid <= 1'b0;
        end else begin
            relu_1_serialize_valid <= 1'b0;
            if (relu_1_out_valid) begin
                // Start serialization
                relu_1_serialize_count <= '0;
                relu_1_serialize_valid <= 1'b1;
            end else if (relu_1_serialize_count < 16-1) begin
                relu_1_serialize_count <= relu_1_serialize_count + 1'b1;
                relu_1_serialize_valid <= 1'b1;
            end
        end
    end
    
    assign relu_1_serialize_data = relu_1_out[relu_1_serialize_count];

    // fc2 layer
    fc_layer_fc2 #(
        .NUM_NEURONS  (16),
        .INPUT_SIZE   (16),
        .DATA_WIDTH   (16),
        .WEIGHT_WIDTH (8),
        .ACC_WIDTH    (32),
        .FRAC_BITS    (8)
    ) u_fc2 (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (fc2_valid_in),
        .input_data(fc2_data_in),
        .fc_out    (fc2_out),
        .valid_out (fc2_valid_out)
    );

    // relu_2 - element-wise ReLU with saturation
    genvar relu_i;
    generate
        for (relu_i = 0; relu_i < 16; relu_i++) begin : RELU_2_GEN
            logic signed [31:0] relu_in_val;
            logic signed [31:0] relu_out_32b;
            logic signed [15:0] relu_out_16b;
            logic relu_elem_valid;
            
            assign relu_in_val = fc2_out[relu_i];
            
            relu_layer #(
                .DATA_WIDTH(32)
            ) u_relu_elem (
                .clk      (clk),
                .rst_n    (rst_n),
                .valid_in (fc2_valid_out),
                .data_in  (relu_in_val),
                .data_out (relu_out_32b),
                .valid_out(relu_elem_valid)
            );
            
            sat32_to_16 u_sat (
                .val_in (relu_out_32b),
                .val_out(relu_out_16b)
            );
            
            // Register ReLU output when valid
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    relu_2_out[relu_i] <= '0;
                end else if (relu_elem_valid) begin
                    relu_2_out[relu_i] <= relu_out_16b;
                end
            end
        end
    endgenerate
    
    // Capture valid when all ReLU outputs are ready
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            relu_2_out_valid <= 1'b0;
        end else begin
            relu_2_out_valid <= fc2_valid_out;
        end
    end

    // Serialize relu_2 output vector
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            relu_2_serialize_count <= '0;
            relu_2_serialize_valid <= 1'b0;
        end else begin
            relu_2_serialize_valid <= 1'b0;
            if (relu_2_out_valid) begin
                // Start serialization
                relu_2_serialize_count <= '0;
                relu_2_serialize_valid <= 1'b1;
            end else if (relu_2_serialize_count < 16-1) begin
                relu_2_serialize_count <= relu_2_serialize_count + 1'b1;
                relu_2_serialize_valid <= 1'b1;
            end
        end
    end
    
    assign relu_2_serialize_data = relu_2_out[relu_2_serialize_count];

    // fc3 layer
    fc_layer_fc3 #(
        .NUM_NEURONS  (10),
        .INPUT_SIZE   (16),
        .DATA_WIDTH   (16),
        .WEIGHT_WIDTH (8),
        .ACC_WIDTH    (32),
        .FRAC_BITS    (8)
    ) u_fc3 (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (fc3_valid_in),
        .input_data(fc3_data_in),
        .fc_out    (fc3_out),
        .valid_out (fc3_valid_out)
    );


endmodule
