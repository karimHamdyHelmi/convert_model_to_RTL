`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: weight_rom_fc3
// Description: Weight ROM for fc3 layer
//   - Depth: 160 (one weight per line in .mem file)
//   - Packed output: 80 bits (all neurons for one input feature)
//------------------------------------------------------------------------------
module weight_rom_fc3 #(
    parameter int ADDR_WIDTH = $clog2(16),
    parameter int DATA_WIDTH = 80,
    parameter int NUM_NEURONS = 10,
    parameter int WEIGHT_WIDTH = 8
)(
    input  logic [ADDR_WIDTH-1:0] addr,
    output logic [DATA_WIDTH-1:0] data
);

    // Memory stores individual weights (one per line)
    logic [WEIGHT_WIDTH-1:0] mem [0:160-1];
    
    // Pack weights for the addressed input feature
    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i++) begin : PACK_WEIGHTS
            assign data[i*WEIGHT_WIDTH +: WEIGHT_WIDTH] = 
                mem[addr * NUM_NEURONS + (NUM_NEURONS - 1 - i)];
        end
    endgenerate

    initial begin
        $readmemh("../SIM/fc3_weights_packed.mem", mem);
    end

endmodule
