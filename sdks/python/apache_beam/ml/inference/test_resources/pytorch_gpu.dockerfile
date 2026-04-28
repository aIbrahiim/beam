# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dataflow custom SDK container for PyTorch GPU benchmarks.
# Do not install NVIDIA kernel drivers in the image; Dataflow mounts them.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      tzdata \
      python3.10-full \
      python3.10-distutils \
      python3.10-dev \
      build-essential && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    python3 -m pip install --upgrade pip setuptools wheel

# Beam worker harness files.
COPY --from=gcr.io/apache-beam-testing/beam-sdk/beam_python3.10_sdk:latest \
     /opt/apache/beam /opt/apache/beam

# Install Beam SDK (built in workflow) and benchmark dependencies.
COPY ./sdks/python/build/apache-beam.tar.gz /tmp/beam.tar.gz
COPY ./sdks/python/apache_beam/ml/inference/torch_tests_requirements.txt /tmp/torch_tests_requirements.txt
RUN python3 -m pip install --no-cache-dir "/tmp/beam.tar.gz[gcp]" && \
    python3 -m pip install --no-cache-dir -r /tmp/torch_tests_requirements.txt && \
    python3 -m pip check

# Ensure mounted NVIDIA libraries are discoverable by PyTorch.
ENV PYTHONPATH=/opt/apache/beam:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENTRYPOINT ["/opt/apache/beam/boot"]
