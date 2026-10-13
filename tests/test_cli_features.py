import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from pixelshift.animate import TRANSITIONS
from pixelshift.cli import app

runner = CliRunner()
SMALL = ["--size", "8", "--scale", "2", "--duration", "0.4", "--fps", "10", "--renderer", "cpu"]


@pytest.fixture
def pictures(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for name in "abc":
        p = tmp_path / f"{name}.png"
        Image.fromarray(rng.integers(0, 256, (24, 30, 3), dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


def frames_of(path):
    with Image.open(path) as gif:
        out = []
        for i in range(gif.n_frames):
            gif.seek(i)
            out.append(np.asarray(gif.convert("RGB")).copy())
    return out


@pytest.mark.parametrize("name", sorted(TRANSITIONS))
def test_every_transition_renders(pictures, tmp_path, name):
    a, b, _ = pictures
    out = tmp_path / f"{name}.gif"
    r = runner.invoke(
        app, ["run", "-s", str(a), "-t", str(b), "-o", str(out), *SMALL, "--transition", name]
    )
    assert r.exit_code == 0, r.output
    assert out.exists()


def test_explicit_options_override_the_preset(pictures, tmp_path):
    a, b, _ = pictures
    out = tmp_path / "o.gif"
    r = runner.invoke(
        app,
        ["run", "-s", str(a), "-t", str(b), "-o", str(out), *SMALL, "--transition", "iris",
         "--swirl", "2.0", "--burst", "0.5", "--wave-shape", "vertical", "--delay", "0.3"],
    )  # fmt: skip
    assert r.exit_code == 0, r.output


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--transition", "teleport"], "unknown transition"),
        (["--wave-shape", "zigzag"], "unknown wave_shape"),
    ],
)
def test_bad_transition_options_fail_cleanly(pictures, tmp_path, extra, message):
    a, b, _ = pictures
    r = runner.invoke(
        app, ["run", "-s", str(a), "-t", str(b), "-o", str(tmp_path / "o.gif"), *SMALL, *extra]
    )
    assert r.exit_code == 2 and message in r.output


def test_loop_ends_exactly_where_it_began(pictures, tmp_path):
    a, b, _ = pictures
    out = tmp_path / "loop.gif"
    r = runner.invoke(app, ["run", "-s", str(a), "-t", str(b), "-o", str(out), *SMALL, "--loop"])
    assert r.exit_code == 0 and "seamless loop" in r.output, r.output
    frames = frames_of(out)
    assert np.array_equal(frames[0], frames[-1])
    assert not np.array_equal(frames[0], frames[len(frames) // 2])  # it really goes somewhere


def test_chain_writes_one_video_through_all_pictures(pictures, tmp_path):
    out = tmp_path / "chain.gif"
    r = runner.invoke(
        app,
        ["chain", *map(str, pictures), "-o", str(out), *SMALL, "--hold", "0.2", "--settle", "0.2"],
    )
    assert r.exit_code == 0, r.output
    assert "morph 1 of 2" in r.output and "morph 2 of 2" in r.output
    frames = frames_of(out)
    from pixelshift.io import load_square

    first = np.kron(load_square(pictures[0], 8), np.ones((2, 2, 1), dtype=np.uint8))
    last = np.kron(load_square(pictures[-1], 8), np.ones((2, 2, 1), dtype=np.uint8))
    assert len(frames) > 10 and frames[0].shape == (16, 16, 3)
    assert np.abs(frames[0].astype(int) - first).mean() < 4  # GIF palette rounding only
    assert np.abs(frames[-1].astype(int) - last).mean() < 4


def test_chain_loop_adds_a_morph_back_to_the_start(pictures, tmp_path):
    out = tmp_path / "ring.gif"
    r = runner.invoke(
        app, ["chain", *map(str, pictures), "-o", str(out), *SMALL, "--loop", "--settle", "0"]
    )
    assert r.exit_code == 0, r.output
    assert "morph 3 of 3" in r.output


def test_chain_validates_its_input(pictures, tmp_path):
    out = str(tmp_path / "c.gif")
    r = runner.invoke(app, ["chain", str(pictures[0]), "-o", out, *SMALL])
    assert r.exit_code == 2 and "at least two" in r.output
    r = runner.invoke(app, ["chain", str(pictures[0]), str(pictures[1]), "-o", "x.txt", *SMALL])
    assert r.exit_code == 2 and "unsupported output" in r.output
    r = runner.invoke(app, ["chain", str(pictures[0]), str(tmp_path / "nope.png"), "-o", out])
    assert r.exit_code == 2 and "not found" in r.output
    r = runner.invoke(
        app, ["chain", str(pictures[0]), str(pictures[1]), "-o", out, "--transition", "x"]
    )
    assert r.exit_code == 2 and "unknown transition" in r.output


def test_live_rejects_an_unknown_transition(pictures):
    a, b, _ = pictures
    r = runner.invoke(app, ["live", "-s", str(a), "-t", str(b), "--transition", "teleport"])
    assert r.exit_code == 2 and "unknown transition" in r.output
