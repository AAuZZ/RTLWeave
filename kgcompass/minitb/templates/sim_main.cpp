#include "Vkgveri_minitb_top.h"
#include "verilated.h"

// Older Verilator releases require this symbol when repository RTL uses
// $time, even when the MiniTB itself does not model wall-clock time.
double sc_time_stamp() { return 0.0; }

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Vkgveri_minitb_top top;
  top.clk_i = 0;
  top.rst_ni = 0;
  top.sample_pre_i = 0;
  top.cycle_i = 0;
  top.eval();

  constexpr int kResetCycles = {{RESET_CYCLES}};
  constexpr int kTimeoutCycles = {{TIMEOUT_CYCLES}};

  for (int reset_cycle = 0; reset_cycle < kResetCycles; ++reset_cycle) {
    top.clk_i = 1;
    top.eval();
    top.clk_i = 0;
    top.eval();
  }
  top.rst_ni = 1;

  for (int cycle = 0; cycle < kTimeoutCycles && !Verilated::gotFinish(); ++cycle) {
    top.cycle_i = cycle;
    // Settle cycle-dependent inputs and optionally sample combinational
    // protocol responses before the DUT's active clock edge.
    top.eval();
    top.sample_pre_i = 1;
    top.eval();
    top.sample_pre_i = 0;
    top.eval();
    if (top.done_o) {
      const int status = top.pass_o ? 0 : 1;
      top.final();
      return status;
    }
    top.clk_i = 1;
    top.eval();
    top.clk_i = 0;
    top.eval();
    if (top.done_o) {
      const int status = top.pass_o ? 0 : 1;
      top.final();
      return status;
    }
  }
  VL_PRINTF("KGVERI_MINITB_TIMEOUT cycle=%d\n", kTimeoutCycles);
  VL_PRINTF("KGVERI_MINITB_END status=TIMEOUT\n");
  top.final();
  return 2;
}
