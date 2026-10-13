import imageio.v2 as imageio
import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from pixelshift.cli import app
from pixelshift.export import save_mp4

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


def test_help_lists_commands():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0 and "run" in r.stdout


def test_version():
    assert runner.invoke(app, ["version"]).stdout.strip() != ""


def test_png_output_is_final_image_upscaled(images, tmp_path):
    a, b = images
    out = tmp_path / "final.png"
    r = run(["-s", str(a), "-t", str(b), "-o", str(out), "--size", "8", "--scale", "3"])
    assert r.exit_code == 0, r.output
    assert np.asarray(Image.open(out)).shape == (24, 24, 3)


def test_mp4_output_is_a_playable_video(images, tmp_path):
    a, b = images
    out = tmp_path / "x.mp4"
    r = run(
        [
            "-s",
            str(a),
            "-t",
            str(b),
            "-o",
            str(out),
            "--size",
            "8",
            "--scale",
            "3",
            "--duration",
            "0.5",
            "--fps",
            "10",
            "--png",
            str(tmp_path / "end.png"),
        ]
    )
    assert r.exit_code == 0, r.output
    reader = imageio.get_reader(str(out))
    frames = [f for f in reader]
    assert len(frames) >= 6 and frames[0].shape[:2] == (24, 24)
    assert (tmp_path / "end.png").exists()


def test_gif_output(images, tmp_path):
    a, b = images
    out = tmp_path / "x.gif"
    r = run(
        [
            "-s",
            str(a),
            "-t",
            str(b),
            "-o",
            str(out),
            "--size",
            "8",
            "--duration",
            "0.5",
            "--fps",
            "10",
        ]
    )
    assert r.exit_code == 0, r.output
    with Image.open(out) as gif:
        assert gif.n_frames >= 2


def test_same_seed_gives_identical_output(images, tmp_path):
    a, b = images
    outs = []
    for name in ("one.png", "two.png"):
        out = tmp_path / name
        assert (
            run(
                ["-s", str(a), "-t", str(b), "-o", str(out), "--size", "10", "--seed", "4"]
            ).exit_code
            == 0
        )
        outs.append(np.asarray(Image.open(out)))
    assert np.array_equal(*outs)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["-o", "x.txt"], "unsupported output"),
        (["--solver", "nope"], "unknown solver"),
        (["--easing", "bounce"], "unknown easing"),
        (["--color-space", "hsv"], "color-space"),
        (["--solver", "exact", "--size", "70"], "limited to"),
    ],
)
def test_bad_options_fail_cleanly(images, tmp_path, extra, message):
    a, b = images
    base = ["-s", str(a), "-t", str(b), "-o", str(tmp_path / "o.png")]
    r = run([*base, *extra] if "-o" not in extra else ["-s", str(a), "-t", str(b), *extra])
    assert r.exit_code == 2
    assert message in r.output


def test_missing_input_file(tmp_path, images):
    _, b = images
    r = run(["-s", str(tmp_path / "missing.png"), "-t", str(b), "-o", str(tmp_path / "o.png")])
    assert r.exit_code == 2 and "not found" in r.output


def test_save_mp4_pads_odd_sizes_and_rejects_empty(tmp_path):
    frames = [np.zeros((5, 7, 3), dtype=np.uint8)] * 3
    out = tmp_path / "odd.mp4"
    assert save_mp4(out, frames, fps=5, hold_start=0, hold_end=0) == 3
    assert next(iter(imageio.get_reader(str(out)))).shape[:2] == (6, 8)
    with pytest.raises(ValueError):
        save_mp4(tmp_path / "e.mp4", [])
