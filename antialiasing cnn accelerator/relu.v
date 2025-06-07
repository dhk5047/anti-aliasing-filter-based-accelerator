`timescale 1ns / 1ps

// AA-ReLU activation function with LUT and linear interpolation (Q25.7 fixed-point)
module relu #(
    parameter N = 32,                         // Bit width for Q25.7
    parameter Q = 7,                          // Fractional bits
    parameter ALPHA = 32'sd16640,               // 6.0 ¡¿ 128 in Q25.7
    parameter BETA  = 32'sd80046,              // 28.868 ¡¿ 128 in Q25.7
    parameter LUT_SIZE = 256,                 // LUT size
    parameter STEP = 249                      // Approx. (BETA - ALPHA) / (LUT_SIZE - 1)
)(
    input  wire signed [N-1:0] din_relu,       // Input value (Q25.7)
    output reg  signed [N-1:0] dout_relu       // Output after AA-ReLU (Q25.7)
);

    // LUT holds Q25.7 encoded function values
    reg signed [N-1:0] lut [0:LUT_SIZE-1];
    initial $readmemh("aa_relu_lut_130.mem", lut);

    // Intermediate wires/regs
    wire signed [N-1:0] x = din_relu;
    reg signed [N-1:0] x_shifted;
    reg [$clog2(LUT_SIZE)-1:0] index;
    reg [Q-1:0] frac;
    reg signed [N-1:0] y0, y1;
    reg signed [N+Q-1:0] interp;

    always @(*) begin
        if (x < 0) begin
            dout_relu = 0;
        end else if (x < ALPHA) begin
            dout_relu = x;
        end else if (x >= BETA) begin
            dout_relu = ALPHA <<< 1; // 2 ¡¿ ALPHA
        end else begin
            x_shifted = x - ALPHA;
            index = x_shifted / STEP;
            frac = x_shifted % STEP;

            if (index >= LUT_SIZE - 1) begin
                y0 = lut[LUT_SIZE - 1];
                y1 = lut[LUT_SIZE - 1];
            end else begin
                y0 = lut[index];
                y1 = lut[index + 1];
            end

            // Linear interpolation: y = y0 + (y1 - y0) * frac / STEP
            interp = y0 + (((y1 - y0) * frac) >>> Q); // >>> Q == divide by 2^Q
            dout_relu = interp[N-1:0];
        end
    end

endmodule
