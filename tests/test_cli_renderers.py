import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from pixelshift.cli import app

runner = CliRunner()


@pytest.fixture
def images(tmp_path):
    rng = np.random.default_rng(0)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)).save(a)
    Image.fromarray(rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)).save(b)
    return a, b


def run(args):
    return runner.invoke(app, ["run", *args])


SMALL = ["--size", "8", "--scale", "2", "--duration", "0.4", "--fps", "10"]


def test_cpu_renderer_can_be_forced(images, tmp_path):
    a, b = images
    out = tmp_path / "cpu.gif"
    r = run(["-s", str(a), "-t", str(b), "-o", str(out), *SMALL, "--renderer", "cpu"])
    assert r.exit_code == 0, r.output
    assert "Renderer: CPU" in r.output and out.exists()


def test_gpu_renderer_works_or_fails_cleanly(images, tmp_path):
    a, b = images
    out = tmp_path / "gpu.mp4"
    r = run(["-s", str(a), "-t", str(b), "-o", str(out), *SMALL, "--renderer", "gpu"])
    if r.exit_code == 0:
        assert "Renderer: GPU" in r.output and out.exists()
    else:
        assert r.exit_code == 2 and "error:" in r.output


def test_auto_renderer_always_produces_a_file(images, tmp_path):
    a, b = images
    out = tmp_path / "auto.gif"
    r = run(["-s", str(a), "-t", str(b), "-o", str(out), *SMALL])
    assert r.exit_code == 0, r.output
    assert "Renderer:" in r.output and out.exists()


def test_gpu_renderer_rejects_pixel_style_and_bad_renderer(images, tmp_path):
    a, b = images
    base = ["-s", str(a), "-t", str(b), "-o", str(tmp_path / "o.gif"), "--size", "8"]
    r = run([*base, "--style", "pixel", "--renderer", "gpu"])
    assert r.exit_code == 2 and "only supports" in r.output
    r = run([*base, "--renderer", "quantum"])
    assert r.exit_code == 2 and "renderer" in r.output


@pytest.fixture
def without_gpu_packages(monkeypatch):
    """Make `import moderngl` and `import glfw` fail, as on a base install."""
    import sys

    monkeypatch.setitem(sys.modules, "moderngl", None)
    monkeypatch.setitem(sys.modules, "glfw", None)


def test_auto_renderer_falls_back_to_cpu_without_the_gpu_extra(
    images, tmp_path, without_gpu_packages
):
    a, b = images
    out = tmp_path / "fallback.gif"
    r = run(["-s", str(a), "-t", str(b), "-o", str(out), *SMALL])
    assert r.exit_code == 0, r.output
    assert "Renderer: CPU" in r.output and out.exists()


def test_forcing_gpu_without_the_extra_fails_with_install_hint(
    images, tmp_path, without_gpu_packages
):
    a, b = images
    r = run(
        ["-s", str(a), "-t", str(b), "-o", str(tmp_path / "o.gif"), *SMALL, "--renderer", "gpu"]
    )
    assert r.exit_code == 2 and "pip install" in r.output


def test_live_without_the_extra_fails_with_install_hint(images, without_gpu_packages):
    a, b = images
    r = runner.invoke(app, ["live", "-s", str(a), "-t", str(b), "--size", "8"])
    assert r.exit_code == 2 and "pip install" in r.output


def test_live_command_is_documented_and_validates_mode(images):
    a, b = images
    r = runner.invoke(app, ["live", "--help"])
    assert r.exit_code == 0 and "SPACE pause" in r.output
    r = runner.invoke(app, ["live", "-s", str(a), "-t", str(b), "--mode", "bounce"])
    assert r.exit_code == 2 and "mode" in r.output
