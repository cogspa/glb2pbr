"""
Image math shared by the extractor.

All arrays are float32 in [0, 1], shape (H, W) for grayscale or (H, W, C).
"""
from __future__ import annotations

import numpy as np
from PIL import Image


# --------------------------------------------------------------- conversion
def to_float(img: Image.Image, mode: str = "RGBA") -> np.ndarray:
    """PIL image -> float32 array in [0,1]. Always returns 3 channels for RGB, 4 for RGBA, 1 for L."""
    if mode == "L":
        return np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    arr = np.asarray(img.convert(mode), dtype=np.float32) / 255.0
    return arr


def to_image(arr: np.ndarray, bits: int = 8) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2:
        if bits == 16:
            return Image.fromarray((arr * 65535.0 + 0.5).astype(np.uint16))
        return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="L")
    c = arr.shape[2]
    arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
    if c == 3:
        return Image.fromarray(arr8, mode="RGB")
    if c == 4:
        return Image.fromarray(arr8, mode="RGBA")
    raise ValueError(f"unsupported channel count {c}")


def resize(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize a float array to size x size with high-quality resampling."""
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr
    if arr.ndim == 2:
        im = Image.fromarray(arr.astype(np.float32)).resize((size, size), Image.LANCZOS)
        return np.asarray(im, dtype=np.float32)
    chans = [resize(arr[..., i], size) for i in range(arr.shape[2])]
    return np.stack(chans, axis=-1)


def constant(size: int, value, channels: int) -> np.ndarray:
    if channels == 1:
        return np.full((size, size), float(value), dtype=np.float32)
    v = np.asarray(value, dtype=np.float32).reshape(-1)
    if v.size == 1:
        v = np.repeat(v, channels)
    return np.broadcast_to(v[:channels], (size, size, channels)).astype(np.float32).copy()


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


# ------------------------------------------------------------ normal maps
def normal_gl_to_dx(n: np.ndarray) -> np.ndarray:
    """Flip the green channel: OpenGL (+Y up, glTF/Blender) <-> DirectX (-Y, Substance default)."""
    out = n.copy()
    out[..., 1] = 1.0 - out[..., 1]
    return out


def apply_normal_scale(n: np.ndarray, scale: float) -> np.ndarray:
    """glTF normalTexture.scale: scales the XY components, then renormalises."""
    if abs(scale - 1.0) < 1e-6:
        return n
    v = n[..., :3] * 2.0 - 1.0
    v[..., 0] *= scale
    v[..., 1] *= scale
    v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)
    out = n.copy()
    out[..., :3] = v * 0.5 + 0.5
    return out


def height_from_normal(normal_rgb: np.ndarray, mask: np.ndarray | None = None,
                       highpass: float = 1.0 / 24.0) -> np.ndarray:
    """
    Frankot-Chellappa integration of a tangent-space (OpenGL, +Y up) normal map
    into a height field. Global, FFT-based.

    mask     : optional (H,W) array in [0,1]; gradients outside the UV islands are
               zeroed so texture padding / dilation doesn't streak across the map.
    highpass : sigma as a fraction of the map size used to remove the low-frequency
               drift that global integration accumulates across island borders.
               0 disables it.
    Result normalised to [0,1] with 0.5 as the mean plane.
    """
    v = normal_rgb[..., :3].astype(np.float64) * 2.0 - 1.0
    nz = np.clip(v[..., 2], 0.05, 1.0)
    # surface gradients: dz/dx = -nx/nz, dz/dy = -ny/nz  (y up)
    p = -v[..., 0] / nz
    q = -v[..., 1] / nz
    q = -q  # image rows go down, so flip the y-gradient into image space
    if mask is not None:
        m = mask.astype(np.float64)
        if m.shape != p.shape:
            m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(p.shape[::-1], Image.BILINEAR)) / 255.0
        p *= m
        q *= m
    h, w = p.shape
    wx = np.fft.fftfreq(w) * 2.0 * np.pi
    wy = np.fft.fftfreq(h) * 2.0 * np.pi
    WX, WY = np.meshgrid(wx, wy)
    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)
    denom = WX ** 2 + WY ** 2
    denom[0, 0] = 1.0
    Z = (-1j * WX * P - 1j * WY * Q) / denom
    Z[0, 0] = 0.0
    z = np.real(np.fft.ifft2(Z))
    if highpass and highpass > 0:
        import cv2
        z = z - cv2.GaussianBlur(z, (0, 0), max(1.0, highpass * max(h, w)))
    z -= z.mean()
    amp = np.percentile(np.abs(z), 99.5)
    if amp < 1e-9:
        return np.full(z.shape, 0.5, dtype=np.float32)
    z = np.clip(z / (2.0 * amp) + 0.5, 0.0, 1.0)
    return z.astype(np.float32)


def normal_from_height(height: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Inverse of the above (used for the self-test)."""
    h = height.astype(np.float64)
    dzdx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5 * strength
    dzdy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5 * strength
    nx = -dzdx
    ny = dzdy  # +Y up in OpenGL convention with image rows going down
    nz = np.ones_like(h)
    n = np.stack([nx, ny, nz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


def curvature_from_normal(normal_rgb: np.ndarray) -> np.ndarray:
    """Cheap curvature (divergence of the normal XY) — useful as an SD mask input."""
    v = normal_rgb[..., :3].astype(np.float32) * 2.0 - 1.0
    dx = np.roll(v[..., 0], -1, axis=1) - np.roll(v[..., 0], 1, axis=1)
    dy = np.roll(v[..., 1], 1, axis=0) - np.roll(v[..., 1], -1, axis=0)
    c = (dx + dy) * 0.25
    s = np.percentile(np.abs(c), 99.0) + 1e-6
    return np.clip(c / (2 * s) + 0.5, 0, 1).astype(np.float32)


# ------------------------------------------------------------- subsurface
def subsurface_weight(metallic: np.ndarray, transmission: np.ndarray | None, gain: float = 1.0) -> np.ndarray:
    """
    glTF has no subsurface channel. Heuristic: dielectric, non-transmissive
    surfaces scatter; metals don't. Scale with --sss-gain.
    """
    w = 1.0 - np.clip(metallic, 0, 1)
    if transmission is not None:
        w *= 1.0 - np.clip(transmission, 0, 1)
    return np.clip(w * gain, 0, 1).astype(np.float32)


def subsurface_radius(basecolor_rgb: np.ndarray, tint: float = 0.75) -> np.ndarray:
    """
    Principled BSDF 'Subsurface Radius' colour. Blender's default is (1, 0.2, 0.1)
    (red travels furthest). Mix that default toward the base colour so pink/skin
    materials keep their hue in the scatter and dark/blue materials don't glow red.
    """
    default = np.array([1.0, 0.2, 0.1], dtype=np.float32)
    bc = basecolor_rgb[..., :3]
    return np.clip(default * (1.0 - tint) + bc * tint, 0, 1).astype(np.float32)
