`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: bias_rom_fc1
// Description: Bias ROM for fc1 layer
//   - Width: 512 bits (packed: 16 neurons × 32 bits)
//------------------------------------------------------------------------------
module bias_rom_fc1 #(
    parameter int DATA_WIDTH = 512
)(
    output logic [DATA_WIDTH-1:0] data
);

    logic [DATA_WIDTH-1:0] mem [0:0];

    initial begin
        $readmemh("../SIM/fc1_biases.mem", mem);
    end

    assign data = mem[0];

endmodule
