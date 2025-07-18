module bram_w3(clk, rst, dout, addr, en, din, wen);
  
  input clk;
  input rst;
  output reg [31:0] dout;
  input  [3:0]  wen;  
  input  [31:0] addr;
  input  en;
  input  [31:0] din;
  wire [31:0] addrW;
  reg [31:0] mem [0:15000];
  
  initial begin
     $readmemh("out_conv3_32.hex", mem);
  end
  
  always @(posedge clk) begin
    if(wen == 4'b1111 && en) mem[addrW] <= din;
  end

  always @(posedge clk) begin
    if(en) dout <= mem[addrW];
  end

  assign addrW = addr >> 2;



    
endmodule