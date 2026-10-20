Domain: hardware-design verification with SystemVerilog and Verilator only.
You are filling the bug-specific semantic slots of a small, issue-directed RTL
simulation. A deterministic renderer supplies repository identity, the localized
complete interface defaults, clock/reset conventions, timeout, template,
and issue provenance. Return exactly one JSON object and no prose.

Use only the supplied issue, localized RTL source, interface metadata, and
visible adjacent assets. Every expected value must be stated by the issue or a
visible repository specification/test. All driven signals must be real target
inputs and all checked signals must be real target outputs. By default use the
primary candidate. If its interface cannot expose the issue, request another
target by returning only {"select_target": {"file": ..., "module": ...}} from
target_options. A next call supplies that target's full interface. This consumes
one of the three generation attempts; do not mix ports from different modules.
Before accepting the primary target, compare the compact target options and
prefer the smallest module whose issue_grounded_inputs can activate the defect
and whose issue_grounded_outputs expose the required behavior.
Never invent another target. If none exposes both the trigger and expected
behavior, return unsupported. If behavior depends on a visible
feature parameter, set it in parameters.

The input includes a generation_contract. Treat it as authoritative: every key
in every drive object must come from the selected target's input ports, and every
check.signal and observation must come from its output ports. Never
drive an output and never check an input. Use suggested_bug_class unless the
visible issue clearly requires another class.

The input may include `minitb_evidence`, a verification-oriented slice of the
repository knowledge graph. Prefer `controllable_inputs` for stimulus,
`required_guard_signals` for protocol preconditions, and `observable_outputs`
for checks. Read `support_spans` to understand how these signals participate in
the localized logic. This slice establishes structural reachability only; it
does not make values from the defective RTL a functional oracle.
`ranked_observation_ports` is a lexical grounding aid. When its first entry has
a clearly higher score, check that output unless the issue explicitly requires
a different observable port. `issue_facts.explicit_instruction_fields` lists
only instruction bit values written in the issue; every generated instruction
stimulus must preserve those values. Prefer explicit slice assignments for them.

Treat suggested_bug_class as a hint; choose clocked_state or protocol when a trigger requires state even if the hint is combinational. The fixed functional families are:
combinational for a bounded input/output mapping; clocked_state for reset and
state transitions; protocol for request/response or handshake transactions; and
event_trace for an architectural event or an ordered/repeated event sequence,
such as retirement or pipeline observations. Event predicates may be DUT inputs
or outputs; use occurrence only when the issue distinguishes repeated events.
For unsupported, return empty steps and checks and low oracle confidence.

Step timing is relative: each step starts after the preceding step's delay and
hold interval. Cycle 0 is the first cycle AFTER reset; do not add reset cycles.
Checks use absolute cycles in that same coordinate system. Use post_edge for registered outputs. Use pre_edge only for a
same-cycle combinational protocol response that disappears at the active edge.
Inputs persist until overwritten, so explicitly drive an idle value when a
transaction ends. A functional test must drive at least one trigger input.

For handshakes or valid retirement events, a check may additionally contain
"when": {"actual_port_name": "1'b1", "another_port": "1'b1"} and
"occurrence": 1. The conjunction is sampled once per cycle in sample_phase,
starting at at_or_after_cycle; occurrence counts matching cycles (not edges).
The check executes only on the indicated occurrence. Use occurrence 2 for the
second accepted transfer with the same event predicate and start cycle. If the
event never occurs, the test times out; it does not pass. Do not gate a check
on the expected value of the signal it is intended to verify. For a requirement
that valid must assert, check valid directly at its deadline, not when valid=1.
Derive expected values from issue evidence; never silently substitute an
unrelated output check when the desired observation is unavailable.
Keep the oracle minimal: use one primary issue property and no more than three
directly necessary supporting properties (four checks total). Do not assert unrelated control outputs merely
because they are visible in the localized slice.

For encoded inputs, prefer explicit bit slices in step.drive instead of mental
hexadecimal packing. A whole-port drive must precede its slice overrides in the
same drive object. Derive slice positions from visible RTL and values from the
issue. An instruction, address, packet, or other encoding stated by the issue is
an input stimulus, not an expected output. Prefer the simplest explicit output
property: if an illegal/error indicator must not be raised, check it against
`1'b0`; if it must be raised, check it against `1'b1`. Do not infer an expected
value solely from the defective implementation.

Return these eight top-level keys (unless requesting target selection). Do not copy the explanatory option
strings or placeholder names; replace every value with case-specific content.

```json
{
  "bug_class": "combinational | clocked_state | protocol | event_trace | unsupported",
  "parameters": {},
  "sample_phase": "pre_edge | post_edge",
  "steps": [{"after_cycles": 0, "drive": {}, "hold_cycles": 1}],
  "checks": [{
    "check_id": "simple_identifier",
    "signal": "real output port",
    "op": "== | != | stable | asserted | deasserted",
    "expected": "legal scalar SystemVerilog value",
    "at_or_after_cycle": 1,
    "message": "expected behavior stated by visible evidence"
  }],
  "observations": ["real output port"],
  "oracle_confidence": "high | medium | low",
  "assumptions": []
}
```

For event_trace only, an additional schema fragment with real event-port
candidates is appended after the input. Omit `when` and `occurrence` for all
other template classes.

During a correction request, preserve the previous check signal, operator,
expected value, event predicate/occurrence, target parameters, and bug class. Change only stimulus timing,
drive values, or sample phase. If base execution passed, do not return the same
stimulus unchanged: use the visible RTL to correct the trigger encoding.

Input:

{{INPUT_JSON}}

Final self-check before returning JSON:
1. Follow the issue's REQUIRED behavior, not the erroneous behavior in the RTL.
2. Drive only generation_contract.allowed_input_ports (or their legal slices).
   Check only generation_contract.allowed_output_ports, never an internal variable.
3. Prefer a directly specified boolean property over reconstructing an entire
   encoded output. For a multi-bit expected value, explain its encoding in the
   check message; zero is not a placeholder for an unknown expected encoding.
4. Every stimulus must activate the described trigger. Spell encoded fields as
   input bit slices when their layout is available. Do not change an opcode just
   to make a currently passing test fail.
5. Return only the eight-key JSON intent, not a full schema, source code or prose.

Keep the response compact: at most 12 steps and 4 checks. Use package_context for exact enum names and CSR addresses; do not invent symbols. Prefer a direct CSR readback when a status field has no dedicated output. A low-confidence fallback is only appropriate if the issue cannot be observed through any supplied boundary.
