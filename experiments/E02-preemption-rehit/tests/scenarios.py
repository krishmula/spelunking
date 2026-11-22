"""Literal arrival scripts for the real scheduler. Each entry: step -> [(request id, prompt ids, max tokens)].

SELF: pool of 24 blocks. Request d needs a 6th block at raw progress 80 and is last, so it
preempts itself with position 79 still in flight.
TWO_VICTIMS: q2's prefill chunk needs 2 blocks with none free; each small request frees 1.
REPEATED: q4 is preempted mid-prefill and again during decode; it shares a 32-token prefix.
EQUIVALENT: q1 resumes onto a prefix block recomputed elsewhere after its own copy was reused.
REUSED_ID: q1's hit names its original block ID 8, but that ID was reallocated and recomputed.
ZERO_C: q3 is preempted with its whole first prefill chunk in flight, so C is 0.
BEFORE_REGAIN: q1 resumes onto part of its prefix and is preempted again mid re-prefill, below its
earlier greatest completed position.
"""

from engine_fixture import prompt

SELF = dict(num_blocks=24, max_num_batched_tokens=64, arrivals={
    0: [("a", prompt(1, 40), 60), ("b", prompt(2, 40), 60)],
    1: [("c", prompt(3, 40), 60), ("d", prompt(4, 40), 60)],
})

TWO_VICTIMS = dict(num_blocks=12, max_num_batched_tokens=96, long_prefill_token_threshold=32, arrivals={
    0: [("q1", prompt(101, 8), 10)],
    1: [("q2", prompt(102, 160), 10)],
    3: [("q0", prompt(100, 12), 10), ("q3", prompt(103, 8), 20)],
})

REPEATED = dict(num_blocks=10, max_num_batched_tokens=64, arrivals={
    1: [("q2", prompt(102, 16) + prompt(202, 8), 40), ("q3", prompt(2, 32) + prompt(203, 8), 20)],
    5: [("q0", prompt(100, 16) + prompt(200, 16), 40), ("q4", prompt(2, 32) + prompt(204, 24), 20)],
    6: [("q1", prompt(2, 32) + prompt(201, 8), 20)],
})

EQUIVALENT = dict(num_blocks=10, max_num_batched_tokens=64, arrivals={
    4: [("q3", prompt(103, 16) + prompt(203, 8), 30)],
    5: [("q0", prompt(100, 16) + prompt(200, 8), 40), ("q2", prompt(1, 32) + prompt(202, 16), 20)],
    6: [("q1", prompt(1, 32) + prompt(201, 24), 20)],
})

REUSED_ID = dict(num_blocks=12, max_num_batched_tokens=64, arrivals={
    1: [("q3", prompt(103, 16) + prompt(203, 16), 40), ("q4", prompt(104, 16) + prompt(204, 16), 40)],
    4: [("q0", prompt(4, 32) + prompt(200, 24), 30), ("q1", prompt(4, 32) + prompt(201, 8), 20)],
    5: [("q2", prompt(102, 16) + prompt(202, 16), 30)],
})

ZERO_C = dict(num_blocks=12, max_num_batched_tokens=64, arrivals={
    1: [("q0", prompt(100, 16) + prompt(200, 16), 20), ("q2", prompt(102, 16) + prompt(202, 8), 40)],
    4: [("q1", prompt(2, 32) + prompt(201, 16), 30)],
    6: [("q3", prompt(103, 16) + prompt(203, 16), 20)],
})

BEFORE_REGAIN = dict(num_blocks=12, max_num_batched_tokens=32, arrivals={
    0: [("q0", prompt(100, 48), 60)],
    1: [("q3", prompt(103, 8), 60)],
    3: [("q2", prompt(102, 16), 20)],
    5: [("q1", prompt(101, 48), 60)],
})

ALL = {"self": SELF, "two_victims": TWO_VICTIMS, "repeated": REPEATED, "equivalent": EQUIVALENT,
       "reused_id": REUSED_ID, "zero_c": ZERO_C, "before_regain": BEFORE_REGAIN}


def engine_kwargs(script: dict) -> dict:
    return {k: v for k, v in script.items() if k != "arrivals"}
