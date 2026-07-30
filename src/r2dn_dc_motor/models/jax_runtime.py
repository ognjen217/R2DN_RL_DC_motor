"""JAX accelerator discovery and fail-fast CUDA checks for Phase 6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any


@dataclass(frozen=True)
class JAXRuntime:
    """Serializable evidence for the accelerator used by JAX."""

    jax_version: str
    jaxlib_version: str
    backend: str
    device_count: int
    device_platforms: tuple[str, ...]
    device_kinds: tuple[str, ...]
    cuda12_plugin_version: str | None
    cuda12_pjrt_version: str | None
    cuda_active: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        devices = ", ".join(
            f"{platform}:{kind}"
            for platform, kind in zip(
                self.device_platforms,
                self.device_kinds,
                strict=True,
            )
        )
        return "\n".join(
            (
                "PHASE 6 JAX RUNTIME",
                f"jax/jaxlib: {self.jax_version}/{self.jaxlib_version}",
                f"backend: {self.backend}",
                f"devices: {devices or 'none'}",
                f"jax-cuda12-plugin: {self.cuda12_plugin_version or 'not installed'}",
                f"CUDA active: {'yes' if self.cuda_active else 'no'}",
            )
        )

    def require_cuda(self) -> None:
        """Reject CPU fallback or a non-CUDA JAX GPU backend."""

        if self.cuda_active:
            return
        raise RuntimeError(
            "CUDA was required for Phase-6 training, but JAX is not using the "
            f"CUDA backend (backend={self.backend!r}, "
            f"jax-cuda12-plugin={self.cuda12_plugin_version or 'not installed'}). "
            'Install the locked CUDA dependencies with: '
            'python -m pip install -e ".[dev,phase6,phase6-cuda12]"'
        )


def inspect_jax_runtime(*, require_cuda: bool = False) -> JAXRuntime:
    """Initialize JAX, execute a probe, and report the actual runtime backend."""

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise ImportError(
            'Phase-6 runtime inspection requires: python -m pip install -e ".[phase6]"'
        ) from error

    # A synchronized, differentiated matrix operation proves that the selected
    # backend can compile and execute the same core workload used by the R2DN MLP.
    def probe_loss(matrix: Any) -> Any:
        return jnp.sum(jnp.tanh(matrix @ matrix))

    matrix = jnp.ones((32, 32), dtype=jnp.float32)
    probe_value, probe_gradient = jax.jit(jax.value_and_grad(probe_loss))(matrix)
    probe_value.block_until_ready()
    probe_gradient.block_until_ready()

    devices = tuple(jax.devices())
    cuda_plugin_version = _optional_package_version("jax-cuda12-plugin")
    cuda_pjrt_version = _optional_package_version("jax-cuda12-pjrt")
    runtime = JAXRuntime(
        jax_version=str(jax.__version__),
        jaxlib_version=_optional_package_version("jaxlib") or "unknown",
        backend=str(jax.default_backend()),
        device_count=len(devices),
        device_platforms=tuple(str(device.platform) for device in devices),
        device_kinds=tuple(str(device.device_kind) for device in devices),
        cuda12_plugin_version=cuda_plugin_version,
        cuda12_pjrt_version=cuda_pjrt_version,
        cuda_active=(
            jax.default_backend() == "gpu"
            and bool(devices)
            and all(device.platform == "gpu" for device in devices)
            and cuda_plugin_version is not None
        ),
    )
    if require_cuda:
        runtime.require_cuda()
    return runtime


def _optional_package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None
