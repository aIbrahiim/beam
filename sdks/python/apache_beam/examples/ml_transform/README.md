# MLTransform Examples

This directory contains Apache Beam examples for MLTransform pipelines.

## MLTransform - Generate Vocab (Batch only)

`mltransform_generate_vocab.py` builds a deterministic vocabulary artifact from
batch input rows.

### What it does

1. Reads input rows from JSONL (`--input_file`) or BigQuery (`--input_table`).
2. Extracts specified columns (`--columns`).
3. Normalizes text (`trim`, optional lowercasing).
4. Tokenizes text (`whitespace` or `regex` tokenizer).
5. Counts global token frequencies.
6. Applies `--min_frequency`, ranks deterministically, and keeps top
   `--vocab_size`.
7. Ensures `--oov_token` is included first.
8. Writes the vocabulary as one token per line.

### Required arguments

- `--output_vocab`
- `--columns`
- and one of:
  - `--input_file`
  - `--input_table`

### Optional arguments

- `--vocab_size` (default: `50000`)
- `--min_frequency` (default: `1`)
- `--lowercase` (default: `true`)
- `--tokenizer` (`whitespace` or `regex`, default: `whitespace`)
- `--oov_token` (default: `<UNK>`)
- `--input_expand_factor` (default: `1`, useful for perf/load testing)

### Local batch example

```sh
python -m apache_beam.examples.ml_transform.mltransform_generate_vocab \
  --input_file=/tmp/input.jsonl \
  --output_vocab=/tmp/vocab.txt \
  --columns=text,category \
  --vocab_size=5 \
  --min_frequency=1 \
  --lowercase=true \
  --tokenizer=whitespace \
  --oov_token=<UNK> \
  --input_expand_factor=1 \
  --runner=DirectRunner
```

### Input format

JSONL input with object rows, for example:

```json
{"id":"1","text":"Beam beam ML pipeline"}
{"id":"2","text":"Beam pipeline dataflow"}
{"id":"3","text":"ML transform beam"}
{"id":"4","text":"vocab vocab vocab test"}
{"id":"5","text":"rare_token_once"}
{"id":"6","text":""}
{"id":"7","text":null}
```

The same sample is available at:
`apache_beam/examples/ml_transform/testdata/vocab_test_input.jsonl`

### Output format

One token per line, deterministic order:

1. `oov_token` first
2. remaining tokens sorted by:
   - frequency descending
   - token ascending (for ties)

Example output:

```txt
<UNK>
beam
ml
```

For this sample and config:

```sh
--columns=text --min_frequency=2 --vocab_size=3
```

the expected output is:

```txt
<UNK>
beam
vocab
ml
```

### Empty vocabulary behavior

If all tokens are filtered out by `--min_frequency`, the pipeline writes only
the reserved `--oov_token` and logs a warning.

### Additional test datasets

- Happy path: `testdata/vocab_test_input.jsonl`
- Tie-break verification: `testdata/vocab_tie_break_input.jsonl`
- Null/empty/missing column: `testdata/vocab_edge_nulls_input.jsonl`

### Performance testing pattern

- Small local files: functional correctness and deterministic-order tests.
- Large GCS files (or moderate file + `--input_expand_factor`): throughput/cost
  benchmarking on Dataflow.

