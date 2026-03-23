#!/usr/bin/env python3
"""Generate Sudoku training data for TWM.

State representation: 27 triples (9 rows + 9 cols + 9 boxes), each with a
sorted string of filled digits as the value token.

  ["row1", "digits", "13458"]  →  row 1 has digits 1,3,4,5,8 filled

Transition: state_t → state_t+1 by filling the easiest naked single
(cell where only one digit is possible given row/col/box constraints).

From each completed grid, we generate a solution sequence by:
1. Remove cells to create a puzzle
2. Repeatedly find naked singles and fill them
3. Each fill is one training example
"""

import json
import random
from pathlib import Path

random.seed(42)

ALL_DIGITS = frozenset(range(1, 10))


# --- Sudoku grid generation ---

def _fill_grid(grid: list[list[int]], pos: int = 0) -> bool:
    """Backtracking solver to generate a random complete grid."""
    if pos == 81:
        return True
    r, c = divmod(pos, 9)
    if grid[r][c] != 0:
        return _fill_grid(grid, pos + 1)

    digits = list(range(1, 10))
    random.shuffle(digits)
    for d in digits:
        if _is_valid(grid, r, c, d):
            grid[r][c] = d
            if _fill_grid(grid, pos + 1):
                return True
            grid[r][c] = 0
    return False


def _is_valid(grid, r, c, d):
    for i in range(9):
        if grid[r][i] == d or grid[i][c] == d:
            return False
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if grid[i][j] == d:
                return False
    return True


def generate_complete_grid() -> list[list[int]]:
    grid = [[0] * 9 for _ in range(9)]
    _fill_grid(grid)
    return grid


# --- Constraint helpers ---

def row_digits(grid, r) -> frozenset:
    return frozenset(d for d in grid[r] if d != 0)


def col_digits(grid, c) -> frozenset:
    return frozenset(grid[r][c] for r in range(9) if grid[r][c] != 0)


def box_digits(grid, r, c) -> frozenset:
    br, bc = 3 * (r // 3), 3 * (c // 3)
    return frozenset(grid[i][j] for i in range(br, br + 3)
                     for j in range(bc, bc + 3) if grid[i][j] != 0)


def box_index(r, c) -> int:
    return 3 * (r // 3) + (c // 3)


def candidates(grid, r, c) -> frozenset:
    if grid[r][c] != 0:
        return frozenset()
    return ALL_DIGITS - row_digits(grid, r) - col_digits(grid, c) - box_digits(grid, r, c)


def find_naked_singles(grid) -> list[tuple[int, int, int]]:
    """Find all cells with exactly one candidate. Returns [(r, c, digit), ...]."""
    singles = []
    for r in range(9):
        for c in range(9):
            cands = candidates(grid, r, c)
            if len(cands) == 1:
                singles.append((r, c, next(iter(cands))))
    return singles


# --- Bitmask triple encoding ---

def digits_token(digits: frozenset) -> str:
    """Encode a set of digits as a sorted string token. Empty set = '0'."""
    if not digits:
        return "0"
    return "".join(str(d) for d in sorted(digits))


def grid_to_triples(grid) -> list[list[str]]:
    """Encode grid state as 27 triples (9 rows + 9 cols + 9 boxes)."""
    triples = []
    for r in range(9):
        triples.append(["row" + str(r), "digits", digits_token(row_digits(grid, r))])
    for c in range(9):
        triples.append(["col" + str(c), "digits", digits_token(col_digits(grid, c))])
    for b in range(9):
        br, bc = 3 * (b // 3), 3 * (b % 3)
        triples.append(["box" + str(b), "digits", digits_token(box_digits(grid, br, bc))])
    return triples


# --- Training data generation ---

def create_puzzle(grid, n_remove) -> list[list[int]]:
    """Remove n cells from a complete grid."""
    puzzle = [row[:] for row in grid]
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)
    for r, c in cells[:n_remove]:
        puzzle[r][c] = 0
    return puzzle


def generate_solution_sequence(puzzle) -> list[dict]:
    """Generate training examples by repeatedly filling naked singles."""
    grid = [row[:] for row in puzzle]
    examples = []

    while True:
        singles = find_naked_singles(grid)
        if not singles:
            break

        # Pick the one to fill (random among available naked singles)
        r, c, digit = random.choice(singles)

        # Record state before
        state_t = [["#mode", "type", "advance"]] + grid_to_triples(grid)

        # Fill the cell
        grid[r][c] = digit

        # Record state after
        state_t1 = grid_to_triples(grid)

        examples.append({"state_t": state_t, "state_t+1": state_t1})

    return examples


def write_jsonl(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  {path.name}: {len(examples)} examples")


# --- Main ---

def main():
    out_dir = Path("data/sudoku")
    n_grids = 500
    removals = [40, 45, 50, 55]  # varying difficulty

    print(f"Generating Sudoku training data")
    print(f"  Grids: {n_grids}")
    print(f"  Removals per grid: {removals}")

    all_examples = []
    grids_with_no_singles = 0

    for i in range(n_grids):
        grid = generate_complete_grid()
        n_remove = random.choice(removals)
        puzzle = create_puzzle(grid, n_remove)
        examples = generate_solution_sequence(puzzle)

        if not examples:
            grids_with_no_singles += 1
        all_examples.extend(examples)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_grids} grids, {len(all_examples)} examples so far")

    random.shuffle(all_examples)
    print()
    write_jsonl(out_dir / "train.jsonl", all_examples)

    # Stats
    max_in = max_out = 0
    tokens = set()
    for line in open(out_dir / "train.jsonl"):
        ex = json.loads(line)
        max_in = max(max_in, len(ex["state_t"]))
        max_out = max(max_out, len(ex["state_t+1"]))
        for t in ex["state_t"] + ex["state_t+1"]:
            tokens.update(t)

    print(f"\n  Train: {len(all_examples)}")
    print(f"  Grids with no naked singles: {grids_with_no_singles}")
    print(f"  Avg examples per grid: {len(all_examples) / n_grids:.1f}")
    print(f"\n  Max input triples: {max_in}")
    print(f"  Max output triples: {max_out}")
    print(f"  Unique tokens: {len(tokens)}")
    print(f"  Tokens: {sorted(tokens)}")


if __name__ == "__main__":
    main()
