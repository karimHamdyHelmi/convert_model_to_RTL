`timescale 1ns/1ps

//------------------------------------------------------------------------------
// Module: sat32_to_16
// Description: Saturates 32-bit signed value to 16-bit signed range
//------------------------------------------------------------------------------
module sat32_to_16 (
    input  logic signed [31:0] val_in,
    output logic signed [15:0] val_out
);

    always_comb begin
        if (val_in > 32767)
            val_out = 16'sd32767;
        else if (val_in < -32768)
            val_out = -16'sd32768;
        else
            val_out = val_in[15:0];
    end

endmodule
