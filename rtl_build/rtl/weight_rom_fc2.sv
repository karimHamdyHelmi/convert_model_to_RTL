`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: weight_rom_fc2
// Description: Weight ROM for fc2 layer
//   - Depth: 256 (one weight per line in .mem file)
//   - Packed output: 128 bits (all neurons for one input feature)
//------------------------------------------------------------------------------
module weight_rom_fc2 #(
    parameter int ADDR_WIDTH = $clog2(16),
    parameter int DATA_WIDTH = 128,
    parameter int NUM_NEURONS = 16,
    parameter int WEIGHT_WIDTH = 8
)(
    input  logic [ADDR_WIDTH-1:0] addr,
    output logic [DATA_WIDTH-1:0] data
);

    // Memory stores individual weights (one per line)
    logic [WEIGHT_WIDTH-1:0] mem [0:256-1];
    
    // Pack weights for the addressed input feature
    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i++) begin : PACK_WEIGHTS
            assign data[i*WEIGHT_WIDTH +: WEIGHT_WIDTH] = 
                mem[addr * NUM_NEURONS + (NUM_NEURONS - 1 - i)];
        end
    endgenerate

    initial begin
        $readmemh("../SIM/fc2_weights_packed.mem", mem);
    end

endmodule
