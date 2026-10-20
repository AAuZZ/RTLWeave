module kgveri_iverilog_driver;
  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic sample_pre_i = 1'b0;
  logic [31:0] cycle_i = 32'b0;
  logic done_o;
  logic pass_o;
  integer cycle;

  kgveri_minitb_top top (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .sample_pre_i(sample_pre_i),
    .cycle_i(cycle_i),
    .done_o(done_o),
    .pass_o(pass_o)
  );

  task automatic stop_if_done;
    begin
      if (done_o) begin
        if (pass_o) $finish;
        else $fatal(1, "MiniTB target check failed");
      end
    end
  endtask

  initial begin
    // Drive an explicit active reset edge.  Unlike Verilator's C++ driver,
    // Icarus does not necessarily schedule an edge for a declaration-time
    // X-to-0 initializer at time zero.
    #1 rst_ni = 1'b1;
    #1 rst_ni = 1'b0;
    repeat ({{RESET_CYCLES}}) begin
      #1 clk_i = 1'b1;
      #1 clk_i = 1'b0;
    end
    // Keep the last reset falling edge distinct from functional sampling.
    #1 rst_ni = 1'b1;

    for (cycle = 0; cycle < {{TIMEOUT_CYCLES}}; cycle = cycle + 1) begin
      cycle_i = cycle;
      #1;
      sample_pre_i = 1'b1;
      #1;
      sample_pre_i = 1'b0;
      #1;
      stop_if_done();
      clk_i = 1'b1;
      #1;
      clk_i = 1'b0;
      #1;
      stop_if_done();
    end

    $display("KGVERI_MINITB_TIMEOUT cycle=%0d", {{TIMEOUT_CYCLES}});
    $display("KGVERI_MINITB_END status=TIMEOUT");
    $fatal(2, "MiniTB timed out");
  end
endmodule
