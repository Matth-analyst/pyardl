# The native backend — measured first, built second

`pyardl.backend` · `rust/`

pyardl is pure Python. `pip install pyardl` needs no compiler, and the
NumPy path is not a fallback — it is **the reference implementation**,
the one every test runs against and the one the optional native kernel
is checked *by*, never the reverse.

## Where the time actually goes

The obvious thing to accelerate in a bootstrap is "the bootstrap". That
turns out to be wrong here, and the way to find out was to measure
before writing any Rust. Profiling `bootstrap_bounds_test` at
`T = 1000`, `B = 9999`, `k = 3` — 26.8 s under `cProfile`:

| where | time | share |
|---|---|---|
| `numpy.linalg.qr` | 9.5 s | 36% |
| **`simulate_paths`** (the null recursion) | 7.3 s | 27% |
| `numpy.stack` (building the designs) | 2.9 s | 11% |
| ~42 000 small allocations inside the loop | ~2 s | 7% |

The largest item is **already LAPACK**. Rewriting a QR in Rust would
call the same LAPACK and gain nothing; the way to make that faster is
algorithmic — not computing the full `Q` when only `Q'y` and `R` are
needed — and it belongs in the NumPy path.

The one place where Python genuinely costs something is the recursion
that regenerates the null paths. It is **sequential in `t`**: period `t`
needs period `t-1`, so NumPy cannot vectorise over time and the loop
pays 1050 interpreter turns plus an allocation per period on a `(B, k)`
array.

So exactly one function was ported. Not "the bootstrap".

## What the kernel does differently

It inverts the loops. NumPy advances all `B` replications through one
period at a time, because that is the only axis it can vectorise. The
kernel walks one replication through all periods, then moves to the
next — so a replication's whole working set (33 KB at these dimensions)
stays in L2 cache, and the replications, being independent, run in
parallel across cores with no synchronisation at all.

Measured on the recursion alone:

| T | B | k | NumPy | native | speed-up |
|---|---|---|---|---|---|
| 100 | 999 | 1 | 0.023 s | 0.005 s | 4.6× |
| 300 | 2999 | 1 | 0.233 s | 0.057 s | 4.1× |
| 1000 | 2999 | 3 | 1.856 s | 0.368 s | 5.0× |
| 1000 | 9999 | 3 | 6.167 s | 1.322 s | 4.7× |

And on the whole test, which is the number that matters:

| T | B | k | NumPy | native | speed-up |
|---|---|---|---|---|---|
| 100 | 2999 | 1 | 0.428 s | 0.295 s | **1.45×** |
| 300 | 2999 | 1 | 0.900 s | 0.591 s | **1.52×** |
| 1000 | 2999 | 3 | 6.210 s | 4.435 s | **1.40×** |
| 1000 | 9999 | 3 | 20.341 s | 14.795 s | **1.37×** |

**About 1.4×, and that is Amdahl's law doing exactly what it says.**
Speeding up 27% of a run by 5× cannot give more. Reporting the 5×
alone would be true and misleading; the honest headline is the smaller
number, and the useful conclusion is that the next optimisation is the
QR, not the language.

### The benchmark had to be fixed before these numbers meant anything

The first version of `validation/backend_benchmark.py` timed all of
NumPy, then all of Rust, three repetitions each. It reported **0.83×**
at `T = 1000, B = 2999, k = 3` — the native kernel *slower* than NumPy —
while the kernel-only measurement on the same configuration said 4.2×.
Both could not be true.

The fault was the protocol, not the code. The script ran straight after
a 25-minute test suite, the machine was still shedding heat, and
whichever backend was measured first absorbed the drift. Re-measured
with seven repetitions **alternating** the two: 6.73 s against 4.93 s,
1.37×.

The script now alternates, discards a warm-up run of each, and takes the
minimum rather than the mean. A benchmark that does not alternate
measures thermal drift as much as it measures code — and it does so in a
form that looks exactly like a result.

## Equivalence is exact, not distributional

The architecture called for a distributional check between backends — a
Kolmogorov-Smirnov test with `p > 0.99`. The design here allows
something much stronger, so the KS is not the lock.

**The kernel draws nothing.** Innovations are resampled on the Python
side by a seeded `numpy.random.Generator` and passed in. Both backends
therefore see the identical numbers and must return identical
trajectories. The test suite pins the gap at **1e-12**; measured, it is
about 4e-14 across the five deterministic cases, ragged lag orders and
several `k` — pure summation-order rounding, since NumPy sums pairwise
and the kernel sequentially.

That matters because a KS test on 2000 draws cannot separate two
distributions that differ by 1e-9. It would have passed a sign error on
a coefficient that is rarely active. The KS is still run — on the
end-to-end bootstrap distributions, where it answers its own question:
that substituting the kernel does not move the law of the decisions.

The end-to-end check is stronger still: with the same seed, the two
backends return the same critical values to 1e-15 and the same p-values
bit for bit.

## Using it

```python
from pyardl.bootstrap import bootstrap_bounds_test

res = bootstrap_bounds_test(y, x, case=3, backend="auto")
```

- `"numpy"` (**default**) — the reference. Unchanged behaviour, no
  compiler needed.
- `"rust"` — the kernel, and an `ImportError` telling you how to build
  it if it is missing. It does **not** fall back silently: someone
  timing a speed-up has to know which implementation they just timed.
- `"auto"` — the kernel when present, NumPy otherwise.

```pycon
>>> from pyardl import backend
>>> backend.resolve("numpy")
'numpy'
>>> backend.resolve("auto") in ("numpy", "rust")
True

```

`backend.why_unavailable()` returns the import error rather than a bare
`False` — a library built for another Python version fails very
differently from a missing file, and the distinction saves an hour.

## Building it

```bash
python rust/build.py          # cargo build --release, then install
python rust/build.py --check  # report the current state
```

The crate is **not** built by `pip install`. maturin wants to be the
package's build system, and it cannot be one here: pyardl must stay a
pure-Python, universal wheel installable without a Rust toolchain. So
`rust/build.py` does the only necessary thing — `cargo build --release`,
then copy the shared library to the name Python expects.

The compiled artefact is gitignored and excluded from the wheel. It is
specific to one platform *and* one Python version; shipping the one
built on a release machine would produce a package that loads nowhere
else.

## What is deliberately not in the kernel

**Random number generation.** Keeping it in Python keeps reproducibility
under one generator, and is what makes the equivalence exact rather than
statistical.

**The NARDL decomposition.** It passes a Python callback (`expand`)
invoked once per period. Crossing the boundary a thousand times would
cost more than the loop saves, so that case stays on NumPy. The fallback
is silent, because it is a performance decision and not a
methodological caveat — and a test asserts that it produces exactly the
NumPy result.

## References

- Amdahl, G. M. (1967). Validity of the single processor approach to
  achieving large scale computing capabilities. *AFIPS Conference
  Proceedings*, 30, 483-485.
