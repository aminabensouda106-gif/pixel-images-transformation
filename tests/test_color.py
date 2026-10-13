import numpy as np
import pytest

from pixelshift import color


@pytest.fixture
def all_colors():
    g = np.linspace(0, 255, 17).astype(np.uint8)
    r, gr, b = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([r, gr, b], axis=-1).reshape(-1, 3)


@pytest.mark.parametrize(
    ("to_lab", "from_lab"),
    [(color.rgb_to_oklab, color.oklab_to_rgb), (color.rgb_to_lab, color.lab_to_rgb)],
)
def test_round_trip_is_lossless_in_uint8(all_colors, to_lab, from_lab):
    back = color.float_to_rgb_u8(from_lab(to_lab(color.rgb_u8_to_float(all_colors))))
    assert np.array_equal(back, all_colors)


def test_oklab_reference_values():
    white = color.rgb_to_oklab(np.array([1.0, 1.0, 1.0]))
    black = color.rgb_to_oklab(np.array([0.0, 0.0, 0.0]))
    assert white == pytest.approx([1.0, 0.0, 0.0], abs=1e-4)
    assert black == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    red = color.rgb_to_oklab(np.array([1.0, 0.0, 0.0]))
    assert red == pytest.approx([0.6280, 0.2249, 0.1258], abs=1e-3)


def test_cielab_reference_values():
    white = color.rgb_to_lab(np.array([1.0, 1.0, 1.0]))
    assert white == pytest.approx([100.0, 0.0, 0.0], abs=1e-2)
    red = color.rgb_to_lab(np.array([1.0, 0.0, 0.0]))
    assert red == pytest.approx([53.24, 80.09, 67.20], abs=0.05)


def test_shapes_preserved():
    img = np.random.default_rng(0).random((4, 5, 3))
    assert color.rgb_to_oklab(img).shape == (4, 5, 3)
    assert color.rgb_to_lab(img).shape == (4, 5, 3)
