# artifact-secretary

![https://github.com/converged-computing/artifact-secretary/blob/main/img/artifact-secretary-small.png](https://github.com/converged-computing/artifact-secretary/blob/main/img/artifact-secretary-small.png)

[![PyPI - Version](https://img.shields.io/pypi/v/artifact-secretary)](https://badge.fury.io/py/artifact-secretary)


This library serves as a discovery tool to look into opaque objects (like containers) to figure out what is needed to run, build, etc. Our use case is for scheduling. We want to find the best spot for said opaque container.


## Details

### Container Discovery

For each image in a catalog:

1. Start the container running.
2. Copy our inspection code (Python) into it.
3. An agent drives the inspection (find binary, read ELF and linkage, how compiled).
4. Writes results into an artifact.


## Install

Use the devcontainer (`.devcontainer/`).

```bash
python3 -m pip install -e .
python3 -m pip install -e .[aws]

# Claude
export ANTHROPIC_API_KEY=sk-ant-...

# AWS
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
```

Test connection:

```bash
aws sts get-caller-identity
aws bedrock list-inference-profiles --region "$AWS_DEFAULT_REGION" \
  --query 'inferenceProfileSummaries[].inferenceProfileId' --output text
```
```bash
python3 -c "import boto3; print(boto3.client('sts').get_caller_identity()['Arn'])"
python3 -c "import boto3,os; c=boto3.client('bedrock', region_name=os.environ.get('AWS_DEFAULT_REGION','us-east-1')); print([p['inferenceProfileId'] for p in c.list_inference_profiles()['inferenceProfileSummaries']])"
```

And test:

```bash
python3 - <<'PY'
from strands import Agent
from strands.models import BedrockModel
m = BedrockModel(model_id="us.anthropic.claude-sonnet-5",  # paste a real one from step 3
                 region_name="us-east-1")
print(Agent(model=m)("say ok"))
PY
```

## Usage

Here is a one-off example:

```bash
artifact-secretary profile --backend aws \
  --model us.anthropic.claude-sonnet-5 \
  --catalog ghcr.io/converged-computing/metric-lammps-cpu:zen4-reax \
  --out lookup.json --keep-images
```

By default it deletes any image it had to pull so your disk doesn't fill up.
Images you already had are left alone. Use `--keep-images` to keep everything.

The inspection library works on its own if you just want the facts and don't
want an agent or a key:

```python
from secretary import Target, Inspector, derive_capability
insp = Inspector(Target("/opt/lammps"))
elf = insp.inspect_elf("/build/lmp")
cap = derive_capability(elf["needed"], elf["rpath"])
```

Here is how to use the artifact secretary to list packages and tags from GitHub packages:

```bash
export GITHUB_TOKEN=...   # needs read:packages
artifact-secretary list --org converged-computing --repo performance-study
```


## License

HPCIC DevTools is distributed under the terms of the MIT license.
All new contributions must be made under this license.

See [LICENSE](https://github.com/converged-computing/cloud-select/blob/main/LICENSE),
[COPYRIGHT](https://github.com/converged-computing/cloud-select/blob/main/COPYRIGHT), and
[NOTICE](https://github.com/converged-computing/cloud-select/blob/main/NOTICE) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE- 842614
