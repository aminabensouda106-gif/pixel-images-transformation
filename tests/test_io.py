import numpy as np
import pytest
from PIL import Image

from pixelshift.io import load_square, save_image


def test_load_square_crops_and_resizes(tmp_path):
    arr = np.zeros((40, 100, 3), dtype=np.uint8)
    arr[:, :30] = (255, 0, 0)
    arr[:, 70:] = (0, 0, 255)
    p = tmp_path / "wide.png"
    Image.fromarray(arr).save(p)
    out = load_square(p, 16)
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8
    # center crop of a 100x40 image keeps columns 30..70, which is all black
    assert out.max() == 0


def test_load_handles_gray(tmp_path):
    p = tmp_path / "gray.png"
    Image.new("L", (20, 20), 128).save(p)
    assert load_square(p, 8).shape == (8, 8, 3)


def test_save_round_trip(tmp_path):
    arr = np.random.default_rng(1).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    p = tmp_path / "x.png"
    save_image(p, arr)
    assert np.array_equal(load_square(p, 8), arr)


def test_invalid_size(tmp_path):
    p = tmp_path / "x.png"
    Image.new("RGB", (4, 4)).save(p)
    with pytest.raises(ValueError):
        load_square(p, 0)
