`timescale 1ns/10ps
`define CYCLE 15.4

// ------------------------------
// cnn_top_tb.v
//  - 박사님이 보내주신 cnn_top.v (내부에 bram_w1~w5, bram_if1~if2, cnn 모듈 포함)를
//    그대로 시뮬레이션하기 위한 테스트벤치
//  - BRAM 초기화용 Hex 파일들 (out_conv1_32.hex, out_conv2_32.hex, …, in_32.hex 등)이
//    시뮬레이션 실행 디렉토리에 있어야 합니다.
//
//  - 시뮬레이션 순서:
//      1) clk 생성
//      2) rst를 잠시 걸어두었다가 해제
//      3) start 한 사이클 펄스
//      4) cnn_done (led_result) 신호가 올라올 때까지 대기
//      5) 최종 led_result 출력
//      6) $finish
// ------------------------------

module cnn_top_tb;

  // 1) 클럭, 리셋, 스타트 신호
  reg         clk_100m;
  reg         rst_btn;
  reg         start_btn;

  // 2) DUT(cnn_top)의 출력(LED 결과)
  wire [7:0]  led_result;
  wire        cnn_done;      // 내부 cnn에서 done 신호 (cnn_top 내부에서 묵시적 생성)
  
  // -------------------------------------------------------------
  // 3) cnn_top 인스턴스
  //    - 내부에서 bram_w1~w5, bram_if1~if2, cnn 모듈을 모두 인스턴스화함
  //    - 따라서 TB에서는 clk_100m, rst_btn, start_btn, led_result만 연결해주면 된다.
  // -------------------------------------------------------------
  cnn_top UUT (
    .clk_100m   (clk_100m),
    .rst_btn    (rst_btn),
    .start_btn  (start_btn),
    .led_result (led_result)
    // cnn_top 내부에서 bram_* 과 cnn 인스턴스가 모두 선언되어 있음
  );

  // -------------------------------------------------------------
  // 4) 시뮬레이션 절차
  // -------------------------------------------------------------
  initial begin
    // 초기화
    clk_100m   = 0;
    rst_btn    = 1;
    start_btn  = 0;

    // 20ns 후 리셋 해제
    #20 rst_btn = 0;

    // 잠깐 기다렸다가 start 펄스
    #50 start_btn = 1;
    #(`CYCLE) start_btn = 0;

    // cnn_done (led_result 유효) 대기
    //   → cnn_top 내부에서 done 신호를 만들긴 하지만, 최종 결과는 led_result에 표시됨.
    //   → 대체로 "끝나는 시점"은 내부 FSM이 DONE 상태에 진입한 시점. 
    //   → cnn_top 코드에서는 "cnn_done" 신호를 내부에서 생성한 뒤 "led_result"에 최종 값을 담아줌.
    //   → 여기서는 단순히 led_result가 바뀌는 시점을 잡지 않고, 충분히 긴 시간(예: 200us) 기다리거나, 
    //     혹은 내부 cnn_done 플래그를 환기시켜도 무방. 예제에서는 간단히 #1_000_000으로 충분히 늘려두고 
    //     모니터링용 $display를 출력한다.
    //
    // ★ 원하는 경우 UUT.cnn_inst.done 신호를 직접 참조해도 되지만, 
    //   cnn_top 내부 구조가 바뀌면 하부 inst 이름이 달라질 수 있으므로 편의상 시간을 넉넉히 주고 멈춤.
    #200_000;  // 약 200us 정도 시뮬레이션 진행
  
    $display("\n================ Simulation End ================");
    $display("  led_result = %d", led_result);
    $finish;
  end

  // -------------------------------------------------------------
  // 5) 100MHz 클럭 생성 (주기 = `CYCLE 로 정의됨)
  // -------------------------------------------------------------
  always #(`CYCLE/2) clk_100m = ~clk_100m;

  // -------------------------------------------------------------
  // 6) 가급적이면 waveform(dump) 설정
  // -------------------------------------------------------------
  initial begin
    $dumpfile("cnn_top_tb.vcd");
    $dumpvars(0, cnn_top_tb);
  end

endmodule
