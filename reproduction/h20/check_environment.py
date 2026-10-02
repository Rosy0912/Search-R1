"""Validate imports and permit exactly the tested Torch/cuBLAS dependency exception.

No model loading, GPU allocation, NCCL operation, or package installation.
"""
import importlib
import importlib.metadata as metadata
import json
import platform
import sys
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


EXPECTED = {
    "torch": "2.4.0", "vllm": "0.6.3", "transformers": "4.46.3",
    "datasets": "3.2.0", "ray": "2.40.0", "click": "8.2.1", "tensordict": "0.5.0",
    "numpy": "1.26.4", "faiss-cpu": "1.8.0.post1", "setuptools": "70.3.0",
    "flash-attn": "2.7.4.post1", "nvidia-cublas-cu12": "12.4.5.8",
    "xformers": "0.0.27.post2", "torchdata": "0.8.0",
}


def inspect_dependencies():
    environment = default_environment()
    environment["extra"] = ""
    exceptions, unexpected = [], []
    for distribution in metadata.distributions():
        name = canonicalize_name(distribution.metadata["Name"])
        for text in distribution.requires or []:
            requirement = Requirement(text)
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            try:
                installed = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                installed = None
            if installed is not None and Version(installed) in requirement.specifier:
                continue
            finding = {"package": name, "requirement": str(requirement), "installed": installed}
            if (name == "torch" and canonicalize_name(requirement.name) == "nvidia-cublas-cu12"
                    and str(requirement.specifier) == "==12.1.3.1" and installed == "12.4.5.8"):
                exceptions.append(finding)
            else:
                unexpected.append(finding)
    return exceptions, unexpected


def main():
    assert platform.system() == "Linux" and platform.machine() == "x86_64"
    assert sys.version_info[:2] == (3, 10)
    actual = {name: metadata.version(name) for name in EXPECTED}
    assert all(Version(actual[n]).public == v for n, v in EXPECTED.items()), actual
    for module in ["torch", "vllm", "flash_attn", "ray", "ray.scripts.scripts", "transformers", "datasets", "faiss", "verl"]:
        importlib.import_module(module)
    exceptions, unexpected = inspect_dependencies()
    result = {"versions": actual, "permitted_dependency_exceptions": exceptions,
              "unexpected_dependency_failures": unexpected, "gpu_integration_tested": False}
    print(json.dumps(result, indent=2))
    assert len(exceptions) == 1 and not unexpected, "Unexpected dependency conflicts; see JSON"


if __name__ == "__main__":
    main()
