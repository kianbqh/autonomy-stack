"""CI package identity only; never constructs a simulator or runs a model."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('M3 CPU CI requires its isolated virtual environment')
    expected = {'numpy': '2.2.6', 'gymnasium': '1.2.0', 'gymnasium_robotics': '1.4.1',
                'mujoco': '3.3.7', 'torch': '2.15.0+cpu', 'matplotlib': '3.10.3',
                'yaml': '6.0.2', 'pytest': '8.4.1', 'packaging': '25.0'}
    modules = []
    prefix = Path(sys.prefix).resolve()
    for name, version in expected.items():
        package = importlib.import_module(name)
        actual = str(package.__version__)
        source = Path(package.__file__).resolve()
        if actual != version or not source.is_relative_to(prefix):
            raise RuntimeError(f'{name}: unexpected package identity {actual} at {source}')
        modules.append({'module': name, 'version': actual, 'source': str(source)})
    torch = importlib.import_module('torch')
    available = torch.cuda.is_available()
    built = torch.backends.cuda.is_built()
    if torch.version.cuda is not None or available or built:
        raise RuntimeError('M3 CPU CI must use the official CPU-only Torch build')
    distributions = []
    for dist in sorted(metadata.distributions(), key=lambda item: item.metadata['Name'].lower()):
        name = re.sub(r'[-_.]+', '-', dist.metadata['Name']).lower()
        location = Path(dist._path).resolve()
        if name.startswith(('nvidia', 'cuda', 'triton')) or not location.is_relative_to(prefix):
            raise RuntimeError(f'Unexpected GPU or external distribution: {name} at {location}')
        distributions.append({'name': dist.metadata['Name'], 'version': dist.version, 'location': str(location)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'scope': 'CI package identity; CPU unit fixtures are separate from formal experiments',
              'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
              'github_sha': os.environ.get('GITHUB_SHA'), 'github_run_id': os.environ.get('GITHUB_RUN_ID'),
              'modules': modules, 'distributions': distributions, 'torch_version_cuda': torch.version.cuda,
              'torch_cuda_available': available, 'torch_cuda_is_built': built,
              'simulator_created': False, 'tensor_or_model_executed_by_identity_check': False}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
