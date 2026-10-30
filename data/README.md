# Constructed inputs and frozen results

Version 1.0.0. Creator: Siyu Cai, ORCID 0009-0003-3716-0008.
License: CC BY 4.0; see `LICENSE.txt`.

Extract the matching software archive into the same parent directory.
Run `python verify_manifest.py` and then the replay command in the root
README. The software supplies all deterministic fixture generators.

`inputs/higher_order_parameters.json` contains 20 rational Gram coefficient
records. Fields `tau` and `k` are the correlation threshold and conditioning
order; `L` is a denominator-clearing scale; `common` and `private` are integer
coefficient arrays. The software checks their exact Gram identities before
constructing Walsh rows. Other fixture weights, targets, seeds and scale
choices are explicit in their checkers and resulting JSON records.

`expected/` contains 37 files: 22 JSON reports/traces and 15 binary inputs.
JSON integers are exact; rational numbers use `numerator/denominator`
strings. Row and column indices and graph edge endpoints are zero-based.
`status`, comparison counts and Boolean checks describe arithmetic validation.
`sources` records SHA-256 hashes of the code used for that report.

| Subdirectory | Stored coverage |
|---|---|
| `extension` | 7,378 determinant/Schur comparisons; 704 small-box completions; 700 interval atoms; 30 coarse constructions; 9 acquisition traces |
| `lower` | 54 threshold/order pairs; 1,685 formula checks; 20 integer cases; 205 labelings; 4,440 comparisons; 40 complete-prefix interval certificates and 10,100 atoms |
| `growth` | 6 parameter cases and 12 matrices, with larger blocks and 32 additional bit planes |
| `strengthening` | 5 symbolic identities, 6 interval margins, 48 integer cases, and 8 separation cases |
| `moment` | 10 unit checks, 6 derivative polynomials, moment-state compression controls, and 5 encoded acquisition traces |
| `margin` | 103 rational interval comparisons; 6 encoded and 4 simulated large-bit traces; 32 certificate comparisons in a universe of 262,144 matrices |
| `hardness` | 7 thresholds, 34 instances, 6,748 scalar completion checks, and 4 encoded prefix checks |

Counts describe different suites and should not be added as independent
samples. An interval certificate covers all completions of its stated box;
selected completion checks alone do not establish constancy.

## Binary input format

Files ending in `.bp` are decoded by `pacd.codec.PrefixReader`. The header
uses little-endian struct `<8sIHHH`: magic `PACDBP01`, row count, column
count, bit width, and block-row count. It is followed in column/block/plane
order by records with struct `<IH` (block index and plane index), then the
plane bits packed most significant bit first. Planes run from the most
significant to the least significant. Unused bits at a final byte are zero.

The paired trace records include column depths, actions, decided edges,
logical returned-byte counts, and incomplete moment-check counts where
applicable. Larger-bit simulation traces have no `.bp` counterpart.
