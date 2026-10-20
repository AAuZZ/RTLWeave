module kgveri_minitb_top (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic sample_pre_i,
  input  logic [31:0] cycle_i,
  output logic done_o,
  output logic pass_o
);
{{DECLARATIONS}}

  {{TARGET_MODULE}} {{DUT_PARAMETERS}}dut (
    {{PORT_CONNECTIONS}}
  );

  always_comb begin
{{INITIAL_ASSIGNMENTS}}
  end

  always @(negedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      done_o = 1'b0;
      pass_o = 1'b1;
      kgveri_started = 1'b0;
      kgveri_checked = '0;
    end else if (!done_o && cycle_i >= 32'd4) begin
      kgveri_started = 1'b1;
      $display("KGVERI_MINITB_START instance={{INSTANCE_ID}} template={{TEMPLATE_ID}}");
      done_o = 1'b1;
      pass_o = 1'b1;
      $display("KGVERI_MINITB_END status=MODULE_SMOKE_ONLY");
    end
  end
endmodule
