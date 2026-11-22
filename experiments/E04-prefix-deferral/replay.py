"""Deterministic physical block accounting for one compression event."""
import hashlib
import json
import platform
from datetime import datetime
from itertools import product
from pathlib import Path


def replay(free, shared, finish, bound):
    target = set(range(8))
    sibling = set(range(shared)) | {8}
    independent = set(range(9, 13))

    def live():
        return len(target | sibling | independent)

    peak = live()
    capacity = peak + free
    events = []
    completed = []
    compressed = False
    retained = (0, 2, 4, 6)
    for tick in range(7):
        if tick == finish:
            sibling.clear()
            completed.append("sibling")
        if not compressed:
            common = target & sibling
            new = min(len(common), len(retained))
            if new > capacity - live():
                if tick < bound:
                    continue
                events.append((tick, "independent"))
                independent.clear()
            destinations = set(range(13, 13 + new))
            reused = sorted(target - common)[:len(retained) - new]
            before = target.copy()
            target.update(destinations)
            peak = max(peak, live())
            assert live() <= capacity
            # Each selected token is copied into its assigned physical block.
            mapping = dict(zip(retained, reused + sorted(destinations)))
            target = set(mapping.values())
            assert len(target) == len(retained)
            freed = len(before - target - sibling)
            compressed = True
            compression_tick = tick
        if tick == 5:
            target.clear()
            completed.append("target")
            independent.update(range(9, 13))
            assert live() <= capacity
        if tick == 6:
            independent.clear()
            completed.append("independent")
    assert live() == 0 and completed == ["sibling", "target", "independent"]
    return dict(free=free, shared=shared, sibling_finish=finish, bound=bound,
                compressed_at=compression_tick, destination_blocks_new=new,
                original_blocks_freed=freed, peak_live=peak, preemptions=events,
                completed=completed, token_to_destination=mapping)


if __name__ == "__main__":
    rows = []
    wins = 0
    for free, shared, finish in product(range(6), range(9), range(1, 5)):
        early, late = [replay(free, shared, finish, bound) for bound in (0, 2)]
        rows.extend((early, late))
        saved = len(early["preemptions"]) - len(late["preemptions"])
        assert saved == int(finish <= 2 and free < min(shared, 4))
        wins += saved
    run = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-grid")
    run.mkdir(parents=True)
    (run / "cases.json").write_text(json.dumps(rows, indent=2))
    env = dict(python=platform.python_version(), platform=platform.platform(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               timing="untimed integer counts")
    (run / "env.json").write_text(json.dumps(env, indent=2))
    print(run.name, wins, len(rows) // 2)
