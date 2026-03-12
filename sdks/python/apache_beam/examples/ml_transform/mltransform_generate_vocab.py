#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Batch-only deterministic vocabulary generation pipeline.

This pipeline creates a deterministic vocabulary artifact from one or more
input columns.

Key properties:
- Batch only (no streaming path).
- Deterministic ordering:
  1) token frequency descending
  2) token text ascending for tie-breaks
- Reserved OOV token is always written first.
- Output format: one token per line.
"""

import argparse
import json
import logging
import re
from collections.abc import Iterable
from typing import Any
from typing import Optional

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

SUPPORTED_TOKENIZERS = ('whitespace', 'regex')
DEFAULT_REGEX_PATTERN = r"[A-Za-z0-9_]+"


def parse_bool_flag(value: str) -> bool:
  value_lc = value.strip().lower()
  if value_lc in ('1', 'true', 't', 'yes', 'y'):
    return True
  if value_lc in ('0', 'false', 'f', 'no', 'n'):
    return False
  raise ValueError(
      f'Invalid boolean value {value!r}. Expected true/false style value.')


def normalize_text(value: Any, lowercase: bool = True) -> str:
  if value is None:
    return ''
  text = str(value).strip()
  if lowercase:
    text = text.lower()
  return text


def tokenize_text(
    text: str,
    tokenizer: str = 'whitespace',
    regex_pattern: str = DEFAULT_REGEX_PATTERN) -> list[str]:
  if not text:
    return []
  if tokenizer == 'whitespace':
    return [token for token in text.split() if token]
  if tokenizer == 'regex':
    return re.findall(regex_pattern, text)
  raise ValueError(
      f'Unsupported tokenizer {tokenizer!r}. '
      f'Supported tokenizers: {", ".join(SUPPORTED_TOKENIZERS)}')


def rank_and_select_tokens(
    token_counts: list[tuple[str, int]],
    vocab_size: int,
    min_frequency: int,
) -> list[str]:
  filtered = [(token, count) for token, count in token_counts
              if count >= min_frequency]
  ranked = sorted(filtered, key=lambda item: (-item[1], item[0]))
  return [token for token, _ in ranked[:vocab_size]]


def _parse_json_line(line: str) -> dict[str, Any]:
  try:
    parsed = json.loads(line)
  except json.JSONDecodeError:
    # Treat plain-text rows as values for the default "text" column.
    return {'text': line}
  if not isinstance(parsed, dict):
    raise ValueError(f'Input JSON line must decode to an object, got: {parsed!r}')
  return parsed


def _extract_column_values(row: dict[str, Any], columns: list[str]) -> list[Any]:
  values = []
  for col in columns:
    if col not in row:
      continue
    val = row[col]
    if val is None:
      continue
    if isinstance(val, list):
      values.extend(val)
    else:
      values.append(val)
  return values


def _tokenize_row_values(
    values: list[Any],
    lowercase: bool,
    tokenizer: str,
) -> list[str]:
  tokens: list[str] = []
  for value in values:
    normalized = normalize_text(value, lowercase=lowercase)
    tokens.extend(tokenize_text(normalized, tokenizer=tokenizer))
  return tokens


def parse_known_args(argv):
  parser = argparse.ArgumentParser(
      description='Generate deterministic vocabulary from batch input.')
  parser.add_argument('--input_file', help='Input JSONL file path.')
  parser.add_argument(
      '--input_table',
      help='Input BigQuery table path in PROJECT:DATASET.TABLE format.')
  parser.add_argument('--output_vocab', help='Output vocab file prefix/path.')
  parser.add_argument(
      '--columns',
      help='Comma-separated source columns to include in vocabulary.')
  parser.add_argument(
      '--vocab_size',
      type=int,
      default=50000,
      help='Maximum vocabulary size (top-K by frequency).')
  parser.add_argument(
      '--min_frequency',
      type=int,
      default=1,
      help='Minimum token frequency required to keep token.')
  parser.add_argument(
      '--lowercase',
      default='true',
      help='Whether to lowercase text before tokenization (default: true).')
  parser.add_argument(
      '--tokenizer',
      default='whitespace',
      help='Tokenizer strategy: whitespace or regex.')
  parser.add_argument(
      '--oov_token',
      default='<UNK>',
      help='Reserved out-of-vocabulary token to write first.')
  parser.add_argument(
      '--input_expand_factor',
      type=int,
      default=1,
      help=(
          'Batch-only: repeat each input line this many times to scale volume '
          'for load/perf testing.'))
  return parser.parse_known_args(argv)


def validate_args(args) -> list[str]:
  has_input_file = bool(args.input_file)
  has_input_table = bool(args.input_table)
  if not has_input_file and not has_input_table:
    raise ValueError('One of --input_file or --input_table is required.')
  if has_input_file and has_input_table:
    raise ValueError('Use exactly one of --input_file or --input_table.')
  if not args.output_vocab:
    raise ValueError('--output_vocab is required.')
  if not args.columns:
    raise ValueError('--columns is required.')
  if args.vocab_size is None or args.vocab_size <= 0:
    raise ValueError('--vocab_size must be > 0.')
  if args.min_frequency is None or args.min_frequency < 1:
    raise ValueError('--min_frequency must be >= 1.')
  if args.tokenizer not in SUPPORTED_TOKENIZERS:
    raise ValueError(
        f'Unsupported tokenizer {args.tokenizer!r}. '
        f'Supported tokenizers: {", ".join(SUPPORTED_TOKENIZERS)}')
  if not args.oov_token:
    raise ValueError('--oov_token must be non-empty.')
  if args.input_expand_factor is None or args.input_expand_factor < 1:
    raise ValueError('--input_expand_factor must be >= 1.')
  return [col.strip() for col in args.columns.split(',') if col.strip()]


def run(argv=None, test_pipeline=None):
  known_args, pipeline_args = parse_known_args(argv)
  columns = validate_args(known_args)
  lowercase = parse_bool_flag(known_args.lowercase)

  options = PipelineOptions(pipeline_args)
  pipeline = test_pipeline or beam.Pipeline(options=options)

  if known_args.input_file:
    lines = (
        pipeline
        | 'ReadInputFile' >> beam.io.ReadFromText(known_args.input_file))
    if known_args.input_expand_factor > 1:
      lines = (
          lines
          | 'ExpandInputForPerf' >> beam.FlatMap(
              lambda line, n: [line] * n, known_args.input_expand_factor))
    rows = lines | 'ParseJSON' >> beam.Map(_parse_json_line)
  else:
    rows = pipeline | 'ReadInputTable' >> beam.io.ReadFromBigQuery(
        table=known_args.input_table)

  tokens = (
      rows
      | 'ExtractColumnValues' >> beam.Map(
          lambda row: _extract_column_values(row, columns))
      | 'TokenizeRowValues' >> beam.Map(
          lambda values: _tokenize_row_values(
              values,
              lowercase=lowercase,
              tokenizer=known_args.tokenizer))
      | 'FlattenTokens' >> beam.FlatMap(lambda token_list: token_list)
      | 'DropEmptyTokens' >> beam.Filter(bool))

  ranked_tokens = (
      tokens
      | 'CountTokensGlobally' >> beam.combiners.Count.PerElement()
      | 'CollectTokenCounts' >> beam.combiners.ToList()
      | 'RankAndSelectTokens' >> beam.Map(
          lambda counts: rank_and_select_tokens(
              counts,
              vocab_size=known_args.vocab_size,
              min_frequency=known_args.min_frequency)))

  def build_output_vocab(tokens_list: list[str], oov_token: str) -> list[str]:
    # If everything is filtered out, still write the reserved token.
    if not tokens_list:
      logging.warning(
          'No tokens remained after filtering; writing only reserved token %r.',
          oov_token)
      return [oov_token]
    output = [oov_token]
    output.extend(token for token in tokens_list if token != oov_token)
    return output

  output_lines = (
      ranked_tokens
      | 'BuildOutputVocabulary' >> beam.Map(
          lambda token_list: build_output_vocab(token_list, known_args.oov_token))
      | 'FlattenOutputLines' >> beam.FlatMap(lambda token_list: token_list))

  _ = output_lines | 'WriteVocab' >> beam.io.WriteToText(
      known_args.output_vocab, shard_name_template='')

  result = pipeline.run()
  if not test_pipeline:
    result.wait_until_finish()
  return result


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()

