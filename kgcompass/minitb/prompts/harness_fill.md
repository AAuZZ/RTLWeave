Domain: deterministic hardware RTL simulation assembly only.
Return exactly one JSON object with these string fields and no prose:

declarations, dut_parameters, port_connections, initial_assignments,
pre_stimulus_body, stimulus_body, check_body

The required fragments were deterministically derived from a validated
BehaviorSpec and target interface. Preserve their semantics exactly and add no
new behavior, I/O primitives, loops, expected values, hierarchical references,
or source entries. This is a bounded template-fill step, not a redesign of the
test.

Input:

{{INPUT_JSON}}
