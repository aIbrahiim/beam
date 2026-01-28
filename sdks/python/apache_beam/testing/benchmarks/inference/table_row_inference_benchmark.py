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

"""Benchmark test for table row inference pipeline.

This benchmark measures the performance of RunInference with continuous
table row inputs, including throughput, latency, and cost metrics.
"""

import logging

from apache_beam.examples.inference import table_row_inference
from apache_beam.testing.load_tests.dataflow_cost_benchmark import DataflowCostBenchmark


class TableRowInferenceBenchmarkTest(DataflowCostBenchmark):
  """Benchmark for continuous table row inference with RunInference.

  This benchmark measures:
  - Mean Inference Batch Size: Average batch size for inference
  - Mean Inference Batch Latency: Average time per batch inference
  - Mean Load Model Latency: Time to load the model
  - Throughput: Elements processed per second
  - Cost: Estimated cost on Dataflow
  """
  def __init__(self):
    self.metrics_namespace = 'BeamML_TableInference'
    super().__init__(
        metrics_namespace=self.metrics_namespace,
        pcollection='FormatOutput.out0')

  def test(self):
    """Execute the table row inference pipeline for benchmarking."""
    # The load test framework passes arguments through the pipeline options.
    # We extract them here to pass to the actual pipeline implementation.
    
    # Use getattr to safely get options that might not be registered in the parser
    options = self.pipeline.get_pipeline_options().get_all_options()
    
    mode = options.get('mode') or 'batch'
    extra_opts = {'mode': mode}

    if mode == 'streaming':
      extra_opts['input_subscription'] = options.get('input_subscription')
      extra_opts['window_size_sec'] = int(options.get('window_size_sec') or 60)
      extra_opts['trigger_interval_sec'] = int(options.get('trigger_interval_sec') or 30)
    else:
      extra_opts['input_file'] = options.get('input_file')

    for opt in ['output_table', 'model_path', 'feature_columns']:
      val = options.get(opt)
      if val:
        extra_opts[opt] = val

    # Convert the dictionary of options back into a list of strings for the run() function
    argv = []
    for k, v in extra_opts.items():
      if v is not None:
        argv.extend([f'--{k}', str(v)])

    self.result = table_row_inference.run(
        argv,
        test_pipeline=self.pipeline)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  TableRowInferenceBenchmarkTest().run()
