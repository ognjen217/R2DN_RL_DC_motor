import pytest

pytest.importorskip("jax")

from r2dn_dc_motor.models.jax_runtime import JAXRuntime, inspect_jax_runtime


def _runtime(*, backend: str, plugin: str | None) -> JAXRuntime:
    platform = "gpu" if backend == "gpu" else "cpu"
    kind = "NVIDIA test GPU" if backend == "gpu" else "test CPU"
    return JAXRuntime(
        jax_version="0.5.3",
        jaxlib_version="0.5.3",
        backend=backend,
        device_count=1,
        device_platforms=(platform,),
        device_kinds=(kind,),
        cuda12_plugin_version=plugin,
        cuda12_pjrt_version=plugin,
        cuda_active=backend == "gpu" and plugin is not None,
    )


def test_runtime_inspection_executes_on_available_backend():
    runtime = inspect_jax_runtime()

    assert runtime.device_count >= 1
    assert runtime.backend in {"cpu", "gpu", "tpu"}
    assert runtime.jax_version == "0.5.3"
    assert runtime.backend in runtime.device_platforms


def test_required_cuda_rejects_cpu_fallback():
    runtime = _runtime(backend="cpu", plugin=None)

    with pytest.raises(RuntimeError, match="CUDA was required"):
        runtime.require_cuda()


def test_required_cuda_accepts_active_cuda_plugin():
    runtime = _runtime(backend="gpu", plugin="0.5.3")

    runtime.require_cuda()
