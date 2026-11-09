import cv2
import numpy as np
import pytest

from src.preprocessing import enhance_image


def _legacy_enhance_image(img):
    """The pre-fix implementation copied in every training script (kept to pin the regression)."""
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y_channel = img_yuv[:, :, 0]
    if y_channel.dtype != np.uint8:
        y_channel = (y_channel * 255).astype(np.uint8)
    img_yuv[:, :, 0] = cv2.equalizeHist(y_channel)
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)


@pytest.fixture
def fundus_like():
    """Low-contrast reddish image in [0, 1], like a fundus photo after rescale=1/255."""
    rng = np.random.default_rng(0)
    img = np.zeros((64, 64, 3), np.float32)
    img[..., 0] = 0.45 + 0.1 * rng.random((64, 64))
    img[..., 1] = 0.20 + 0.05 * rng.random((64, 64))
    img[..., 2] = 0.10 + 0.05 * rng.random((64, 64))
    return img


def test_legacy_version_broke_the_scale(fundus_like):
    # ML-1: documents the bug the fix addresses.
    assert _legacy_enhance_image(fundus_like.copy()).max() > 1.5


def test_float_input_stays_in_unit_range(fundus_like):
    out = enhance_image(fundus_like)
    assert out.dtype == np.float32
    assert out.shape == fundus_like.shape
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_contrast_increases(fundus_like):
    out = enhance_image(fundus_like)
    assert out.mean(axis=2).std() > fundus_like.mean(axis=2).std() * 2


def test_colour_is_preserved(fundus_like):
    out = enhance_image(fundus_like)
    means = out.reshape(-1, 3).mean(axis=0)
    assert means[0] > means[1] > means[2]  # still red-dominant, not collapsed to grey


def test_uint8_roundtrip_matches_float_path(fundus_like):
    u8 = np.rint(fundus_like * 255).astype(np.uint8)
    out_u8 = enhance_image(u8)
    assert out_u8.dtype == np.uint8
    np.testing.assert_allclose(out_u8 / 255.0, enhance_image(fundus_like), atol=1 / 255 + 1e-6)


def test_does_not_mutate_input(fundus_like):
    before = fundus_like.copy()
    enhance_image(fundus_like)
    np.testing.assert_array_equal(before, fundus_like)


@pytest.mark.parametrize("bad", [np.full((8, 8, 3), 200.0, np.float32), np.zeros((8, 8), np.uint8)])
def test_rejects_unscaled_float_or_non_rgb(bad):
    with pytest.raises(ValueError):
        enhance_image(bad)
