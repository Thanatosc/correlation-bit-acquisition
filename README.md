# Correlation bit acquisition

Version 1.0.0. Siyu Cai, School of Computing and Artificial Intelligence,
Southwest Jiaotong University. ORCID: https://orcid.org/0009-0003-3716-0008.
Contact: zero@my.swjtu.edu.cn.

## Installation and replay

Use Python 3.13. The verified environment uses Python 3.13.7, NumPy 2.2.6,
and SymPy 1.14.0. No network access is needed after installing dependencies.

The software and dataset ZIPs both contain the directory
`correlation-bit-acquisition/`. Extract both into the same parent directory,
then run these commands inside that directory. A Git checkout already
contains both parts.

```text
python -m pip install -r requirements.txt
python verify_manifest.py
python run_checks.py --output-dir results/replay --compare data/expected
```

Choose a new output directory for each run. The runner copies code and
inputs into a temporary sibling directory, executes all seven suites, and
compares the 37 generated JSON and bit-plane files byte for byte with the
frozen results. It exits unsuccessfully on any assertion or mismatch.
Runtime logs and temporary test files are not part of the dataset.

## Files and operations

| Location | Operation |
|---|---|
| `pacd/skeleton.py`, `pacd/intervals.py` | Exact order-one graph and sound integer scatter intervals |
| `pacd/semantic_moments.py`, `pacd/semantic_acquisition.py` | Joint moment-state stopping and column acquisition |
| `pacd/positive_correlation_corner.py` | Sufficient coordinate-derivative corner certificate |
| `pacd/codec.py` | Encoder and prefix-only reader for bit-plane files |
| `pacd/whole_column_stopping_hardness.py` | Three-column integer construction with one unknown low bit per row |
| `scripts/check_extension.py` | Determinant/Schur comparisons, small prefix boxes, and higher-order interval acquisition |
| `scripts/check_lower.py`, `scripts/check_growth.py` | Fixed-order integer constructions, label switches, and interval certificates |
| `check_strengthening.py` | Symbolic identities, explicit integer construction, and corner certificates |
| `scripts/check_semantic_moment_oracle.py` | Exact moment states, resource-limit behavior, and codec traces |
| `scripts/check_semantic_margin.py` | Rational intervals, margin fixtures, and exhaustive small-input optima |
| `scripts/check_whole_column_stopping_hardness.py` | Scalar checks of the three-column construction and aligned prefixes |
| `data/` | Construction coefficients, frozen numerical results, input files, and format description |

## Algorithm conditions

Each whole-column query reveals the next most significant plane and costs
one unit. Marginal neighborhoods are frozen once; conditioning eligibility
uses their union. Threshold equality deletes and singular tests retain.
`skeleton.py` implements order one. The determinant routines in
`check_extension.py` also evaluate larger fixed conditioning orders.

Moment states contain jointly attainable first and second moments. An
unlimited completed computation is exact. With resource limits, the
returned status distinguishes a certificate, ambiguity, and an incomplete
check. A run that reads further after an incomplete check does not acquire
the unrestricted semantic competitive guarantee. The prototype uses sparse
Python dictionaries for reachable states; the dense direct-address table
is an alternative implementation for the stated arithmetic upper bound.

Interval and derivative routines return sufficient decisions and may remain
undecided on a semantically sufficient prefix. General moment-state running
time is not polynomial in binary bit width. Polynomial interval guarantees
require a positive margin on a set of decisions that fixes the graph by
three-valued evaluation; unused equalities are allowed.

The file codec supports 1–32 bits and blocks of at most 65,535 rows.
The supplied whole-column traces use a single block per column. Higher bit
widths use mathematical prefix simulation, as identified in each record.
Byte and request counts represent logical returned data, not disk traffic
or elapsed-time benchmarks.

The hardness constructor selects its scale by interval-box containment.
The associated explicit rational scale conditions are another sufficient
rule and may select a different scale. Integer vectors, endpoint alignment,
and graph decisions follow the same construction. Its checks enumerate all
item assignments and selected auxiliary changes, not all column completions.

Finite checks verify the listed instances. They do not certify universal
quantifiers, optimal constants, asymptotic claims, or causal recovery accuracy.

## Licensing

Software and operating documentation: MIT (`LICENSE`). Generated inputs,
certificates, and results: CC BY 4.0 (`data/LICENSE.txt`). NumPy and SymPy
are installed separately under their own licenses; their source is not
bundled. This directory contains no manuscript or third-party research data.
