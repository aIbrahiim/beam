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

import json
import os
import tempfile
import unittest

from apache_beam.examples.ml_transform import mltransform_generate_vocab


class MLTransformGenerateVocabUnitTest(unittest.TestCase):
  def test_normalize_and_tokenize_whitespace(self):
    text = mltransform_generate_vocab.normalize_text('  Hello Beam  ', True)
    self.assertEqual(text, 'hello beam')
    tokens = mltransform_generate_vocab.tokenize_text(text, 'whitespace')
    self.assertEqual(tokens, ['hello', 'beam'])

  def test_tokenize_regex(self):
    tokens = mltransform_generate_vocab.tokenize_text(
        'beam,beam! 123', tokenizer='regex')
    self.assertEqual(tokens, ['beam', 'beam', '123'])

  def test_rank_select_and_tie_break_order(self):
    # Tie at frequency=3 should be alpha sorted between apple and apricot.
    counts = [('banana', 2), ('apricot', 3), ('apple', 3), ('zebra', 1)]
    ranked = mltransform_generate_vocab.rank_and_select_tokens(
        counts, vocab_size=3, min_frequency=1)
    self.assertEqual(ranked, ['apple', 'apricot', 'banana'])

  def test_min_frequency_and_top_k(self):
    counts = [('a', 10), ('b', 5), ('c', 4), ('d', 2)]
    ranked = mltransform_generate_vocab.rank_and_select_tokens(
        counts, vocab_size=2, min_frequency=4)
    self.assertEqual(ranked, ['a', 'b'])

  def test_null_and_empty_handling_helpers(self):
    normalized_none = mltransform_generate_vocab.normalize_text(None, True)
    self.assertEqual(normalized_none, '')
    self.assertEqual(
        mltransform_generate_vocab._tokenize_row_values(
            [None, '', '   ', 'Beam'], lowercase=True, tokenizer='whitespace'),
        ['beam'])


class MLTransformGenerateVocabCliValidationTest(unittest.TestCase):
  def test_missing_required_args(self):
    args, _ = mltransform_generate_vocab.parse_known_args([])
    with self.assertRaisesRegex(ValueError, 'input_file or --input_table'):
      mltransform_generate_vocab.validate_args(args)

  def test_invalid_numeric_values(self):
    args, _ = mltransform_generate_vocab.parse_known_args([
        '--input_file=a.jsonl',
        '--output_vocab=/tmp/vocab',
        '--columns=text',
        '--vocab_size=0',
        '--min_frequency=0',
    ])
    with self.assertRaisesRegex(ValueError, 'vocab_size'):
      mltransform_generate_vocab.validate_args(args)

  def test_invalid_tokenizer(self):
    args, _ = mltransform_generate_vocab.parse_known_args([
        '--input_file=a.jsonl',
        '--output_vocab=/tmp/vocab',
        '--columns=text',
        '--tokenizer=custom',
    ])
    with self.assertRaisesRegex(ValueError, 'Unsupported tokenizer'):
      mltransform_generate_vocab.validate_args(args)

  def test_invalid_input_expand_factor(self):
    args, _ = mltransform_generate_vocab.parse_known_args([
        '--input_file=a.jsonl',
        '--output_vocab=/tmp/vocab',
        '--columns=text',
        '--input_expand_factor=0',
    ])
    with self.assertRaisesRegex(ValueError, 'input_expand_factor'):
      mltransform_generate_vocab.validate_args(args)


class MLTransformGenerateVocabIntegrationTest(unittest.TestCase):
  def test_batch_pipeline_exact_output_order(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      input_path = os.path.join(tmpdir, 'input.jsonl')
      output_prefix = os.path.join(tmpdir, 'vocab.txt')

      rows = [
          {'id': '1', 'text': 'Beam beam ML pipeline'},
          {'id': '2', 'text': 'Beam pipeline dataflow'},
          {'id': '3', 'text': 'ML transform beam'},
          {'id': '4', 'text': 'vocab vocab vocab test'},
          {'id': '5', 'text': 'rare_token_once'},
          {'id': '6', 'text': ''},
          {'id': '7', 'text': None},
      ]
      with open(input_path, 'w', encoding='utf-8') as f:
        for row in rows:
          f.write(json.dumps(row) + '\n')

      mltransform_generate_vocab.run([
          f'--input_file={input_path}',
          f'--output_vocab={output_prefix}',
          '--columns=text',
          '--vocab_size=3',
          '--min_frequency=2',
          '--lowercase=true',
          '--tokenizer=whitespace',
          '--oov_token=<UNK>',
          '--runner=DirectRunner',
      ])

      output_path = output_prefix + '-00000-of-00001'
      with open(output_path, 'r', encoding='utf-8') as f:
        output_tokens = [line.rstrip('\n') for line in f]

      # Counts:
      # beam=4, vocab=3, ml=2, pipeline=2, others=1
      # After min_frequency=2 + top_k=3 + tie-break alphabetical:
      # beam, vocab, ml
      self.assertEqual(output_tokens, ['<UNK>', 'beam', 'vocab', 'ml'])

  def test_tie_break_ordering_alphabetical_for_equal_frequency(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      input_path = os.path.join(tmpdir, 'input.jsonl')
      output_prefix = os.path.join(tmpdir, 'vocab.txt')
      rows = [
          {'text': 'apple banana'},
          {'text': 'banana apple'},
          {'text': 'cat dog'},
          {'text': 'dog cat'},
      ]
      with open(input_path, 'w', encoding='utf-8') as f:
        for row in rows:
          f.write(json.dumps(row) + '\n')

      mltransform_generate_vocab.run([
          f'--input_file={input_path}',
          f'--output_vocab={output_prefix}',
          '--columns=text',
          '--vocab_size=4',
          '--min_frequency=2',
          '--lowercase=true',
          '--tokenizer=whitespace',
          '--oov_token=<UNK>',
          '--runner=DirectRunner',
      ])

      output_path = output_prefix + '-00000-of-00001'
      with open(output_path, 'r', encoding='utf-8') as f:
        output_tokens = [line.rstrip('\n') for line in f]
      self.assertEqual(output_tokens, ['<UNK>', 'apple', 'banana', 'cat', 'dog'])

  def test_empty_filtered_result_writes_only_reserved_token(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      input_path = os.path.join(tmpdir, 'input.jsonl')
      output_prefix = os.path.join(tmpdir, 'vocab.txt')
      with open(input_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'text': 'beam'}) + '\n')

      mltransform_generate_vocab.run([
          f'--input_file={input_path}',
          f'--output_vocab={output_prefix}',
          '--columns=text',
          '--vocab_size=10',
          '--min_frequency=2',
          '--oov_token=<UNK>',
          '--runner=DirectRunner',
      ])

      output_path = output_prefix + '-00000-of-00001'
      with open(output_path, 'r', encoding='utf-8') as f:
        output_tokens = [line.rstrip('\n') for line in f]
      self.assertEqual(output_tokens, ['<UNK>'])


if __name__ == '__main__':
  unittest.main()

