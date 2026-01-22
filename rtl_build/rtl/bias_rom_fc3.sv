`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: bias_rom_fc3
// Description: Bias ROM for fc3 layer
//   - Width: 320 bits (packed: 10 neurons × 32 bits)
//------------------------------------------------------------------------------
module bias_rom_fc3 #(
    parameter int DATA_WIDTH = 320
)(
    output logic [DATA_WIDTH-1:0] data
);

    logic [DATA_WIDTH-1:0] mem [0:0];

    initial begin
        $readmemh("../SIM/fc3_biases.mem", mem);
    end

    assign data = mem[0];

endmodule
