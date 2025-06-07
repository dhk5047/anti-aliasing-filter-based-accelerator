`timescale 1ns/10ps

`define CYCLE 15.4

// ────────────────────────────────────────────────────────────────────────────
// cnn_top.v: 최상위 래퍼 (FPGA) 
//  - include 지시문을 제거하고, Vivado에 소스로 파일을 추가한다.
//  - Vivado 프로젝트에 cnn.v, bram.v, PE.v, cnn_top.v 를 모두 "Add Sources" 해야 함.
// ────────────────────────────────────────────────────────────────────────────

module cnn_top (
    input  wire        clk_100m,    // FPGA 보드 100MHz 클럭 (핀 R4)
    input  wire        rst_btn,     // 리셋 버튼    (핀 U7)
    input  wire        start_btn,   // 시작 버튼    (핀 M21)
    output wire [7:0]  led_result  // 결과 LED[7:0] (핀 Y18, AA18, AB18, W19, Y19, AA19, W20, AA20)
);
    // BRAM 인터페이스: cnn.v 모듈과 그대로 매핑
    wire [31:0] BRAM_IF1_ADDR;
    wire [ 3:0] BRAM_IF1_WE;
    wire        BRAM_IF1_EN;
    wire [31:0] BRAM_IF1_DOUT;
    wire [31:0] BRAM_IF1_DIN;

    // IF2 BRAM 인터페이스
    wire [31:0] BRAM_IF2_ADDR;
    wire [ 3:0] BRAM_IF2_WE;
    wire        BRAM_IF2_EN;
    wire [31:0] BRAM_IF2_DOUT;
    wire [31:0] BRAM_IF2_DIN;

    // W1 BRAM 인터페이스
    wire [31:0] BRAM_W1_ADDR;
    wire [ 3:0] BRAM_W1_WE;
    wire        BRAM_W1_EN;
    wire [31:0] BRAM_W1_DOUT;
    wire [31:0] BRAM_W1_DIN;

    // W2 BRAM 인터페이스
    wire [31:0] BRAM_W2_ADDR;
    wire [ 3:0] BRAM_W2_WE;
    wire        BRAM_W2_EN;
    wire [31:0] BRAM_W2_DOUT;
    wire [31:0] BRAM_W2_DIN;

    // W3 BRAM 인터페이스
    wire [31:0] BRAM_W3_ADDR;
    wire [ 3:0] BRAM_W3_WE;
    wire        BRAM_W3_EN;
    wire [31:0] BRAM_W3_DOUT;
    wire [31:0] BRAM_W3_DIN;

    // W4 BRAM 인터페이스
    wire [31:0] BRAM_W4_ADDR;
    wire [ 3:0] BRAM_W4_WE;
    wire        BRAM_W4_EN;
    wire [31:0] BRAM_W4_DOUT;
    wire [31:0] BRAM_W4_DIN;

    // W5 BRAM 인터페이스
    wire [31:0] BRAM_W5_ADDR;
    wire [ 3:0] BRAM_W5_WE;
    wire        BRAM_W5_EN;
    wire [31:0] BRAM_W5_DOUT;
    wire [31:0] BRAM_W5_DIN;
  //─────────────────────────────────────────────────────────────────────────────
  // 1) 내부 신호
  //─────────────────────────────────────────────────────────────────────────────
  wire        cnn_done;
  wire [7:0]  cnn_result;

  // 리셋 동기화
  reg rst_meta, rst_sync;
  always @(posedge clk_100m) begin
    rst_meta <= rst_btn;
    rst_sync <= rst_meta;
  end

  // start 버튼에 단일 펄스 생성
  reg start_meta, start_prev;
  wire start_pulse;
  always @(posedge clk_100m) begin
    start_meta <= start_btn;
    start_prev <= start_meta;
  end
  assign start_pulse = (start_meta && !start_prev) ? 1'b1 : 1'b0;

  //─────────────────────────────────────────────────────────────────────────────
  // 2) Inferable BRAM 인스턴스
  //    - Vivado 합성 시, $readmemh로 초기화된 메모리가 Block RAM으로 인식됨
  //─────────────────────────────────────────────────────────────────────────────

  // Conv1 가중치 (out_conv1_32.hex)
  bram_w1 bram_w1_inst (
    .clk   (clk_100m),
    .rst   (rst_sync),
    .wen   (BRAM_W1_WE),
    .addr  (BRAM_W1_ADDR),
    .en    (BRAM_W1_EN),
    .dout  (BRAM_W1_DOUT),
    .din   (BRAM_W1_DIN)
  );

  // Conv2 가중치 (out_conv2_32.hex)
  bram_w2 bram_w2_inst (
    .clk   (clk_100m),
    .rst   (rst_sync),
    .wen   (BRAM_W2_WE),
    .addr  (BRAM_W2_ADDR),
    .en    (BRAM_W2_EN),
    .dout  (BRAM_W2_DOUT),
    .din   (BRAM_W2_DIN)
  );

  // Conv3 가중치 (out_conv3_32.hex)
  bram_w3 bram_w3_inst (
    .clk   (clk_100m),
    .rst   (rst_sync),
    .wen   (BRAM_W3_WE),
    .addr  (BRAM_W3_ADDR),
    .en    (BRAM_W3_EN),
    .dout  (BRAM_W3_DOUT),
    .din   (BRAM_W3_DIN)
  );

  // FC1 가중치 (out_fc1_32.hex)
  bram_w4 bram_w4_inst (
    .clk   (clk_100m),
    .rst   (rst_sync),
    .wen   (BRAM_W4_WE),
    .addr  (BRAM_W4_ADDR),
    .en    (BRAM_W4_EN),
    .dout  (BRAM_W4_DOUT),
    .din   (BRAM_W4_DIN)
  );

  // FC2 가중치 (out_fc2_32.hex)
  bram_w5 bram_w5_inst (
    .clk   (clk_100m),
    .rst   (rst_sync),
    .wen   (BRAM_W5_WE),
    .addr  (BRAM_W5_ADDR),
    .en    (BRAM_W5_EN),
    .dout  (BRAM_W5_DOUT),
    .din   (BRAM_W5_DIN)
  );

  // IF1 BRAM: 입력 이미지  (in_32.hex)
  bram_if1 bram_if1_inst (
    .clk   (clk_100m),
    .rst   (rst_sync),
    .wen   (BRAM_IF1_WE),
    .addr  (BRAM_IF1_ADDR),
    .en    (BRAM_IF1_EN),
    .dout  (BRAM_IF1_DOUT),
    .din   (BRAM_IF1_DIN)
  );

  // IF2 BRAM: 중간 Feature Map 저장
  bram_if2 bram_if2_inst (
    .clk   (clk_100m),
    .rst   (rst_sync),
    .wen   (BRAM_IF2_WE),
    .addr  (BRAM_IF2_ADDR),
    .en    (BRAM_IF2_EN),
    .dout  (BRAM_IF2_DOUT),
    .din   (BRAM_IF2_DIN)
  );

  //─────────────────────────────────────────────────────────────────────────────
  // 3) cnn.v 모듈 인스턴스화
  //─────────────────────────────────────────────────────────────────────────────
  cnn cnn_inst (
    .clk          (clk_100m),
    .rst          (rst_sync),
    .start        (start_pulse),
    .ready        (1'b1),
    .done         (cnn_done),
    .result       (cnn_result),

    .BRAM_IF1_ADDR(BRAM_IF1_ADDR),
    .BRAM_IF2_ADDR(BRAM_IF2_ADDR),

    .BRAM_W1_ADDR (BRAM_W1_ADDR),
    .BRAM_W2_ADDR (BRAM_W2_ADDR),
    .BRAM_W3_ADDR (BRAM_W3_ADDR),
    .BRAM_W4_ADDR (BRAM_W4_ADDR),
    .BRAM_W5_ADDR (BRAM_W5_ADDR),

    .BRAM_IF1_WE  (BRAM_IF1_WE),
    .BRAM_IF2_WE  (BRAM_IF2_WE),
    .BRAM_W1_WE   (BRAM_W1_WE),
    .BRAM_W2_WE   (BRAM_W2_WE),
    .BRAM_W3_WE   (BRAM_W3_WE),
    .BRAM_W4_WE   (BRAM_W4_WE),
    .BRAM_W5_WE   (BRAM_W5_WE),

    .BRAM_IF1_EN  (BRAM_IF1_EN),
    .BRAM_IF2_EN  (BRAM_IF2_EN),
    .BRAM_W1_EN   (BRAM_W1_EN),
    .BRAM_W2_EN   (BRAM_W2_EN),
    .BRAM_W3_EN   (BRAM_W3_EN),
    .BRAM_W4_EN   (BRAM_W4_EN),
    .BRAM_W5_EN   (BRAM_W5_EN),

    .BRAM_IF1_DOUT(BRAM_IF1_DOUT),
    .BRAM_IF2_DOUT(BRAM_IF2_DOUT),

    .BRAM_W1_DOUT (BRAM_W1_DOUT),
    .BRAM_W2_DOUT (BRAM_W2_DOUT),
    .BRAM_W3_DOUT (BRAM_W3_DOUT),
    .BRAM_W4_DOUT (BRAM_W4_DOUT),
    .BRAM_W5_DOUT (BRAM_W5_DOUT),

    .BRAM_IF1_DIN (BRAM_IF1_DIN),
    .BRAM_IF2_DIN (BRAM_IF2_DIN),
    .BRAM_W1_DIN  (BRAM_W1_DIN),
    .BRAM_W2_DIN  (BRAM_W2_DIN),
    .BRAM_W3_DIN  (BRAM_W3_DIN),
    .BRAM_W4_DIN  (BRAM_W4_DIN),
    .BRAM_W5_DIN  (BRAM_W5_DIN)
  );


  assign led_result = cnn_result;



endmodule
