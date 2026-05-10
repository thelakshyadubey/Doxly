"""
app/services/esp32_image_service.py
=====================================
Converts raw ESP32 camera frames to JPEG bytes.

Supports RGB565 (2 bytes/pixel) and Grayscale/L (1 byte/pixel) formats.
Pure in-memory conversion — no I/O, no side effects.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


class ESP32ImageService:
    """Converts ESP32 raw binary frames to JPEG bytes."""

    def to_jpeg_bytes(
        self,
        raw: bytes,
        width: int,
        height: int,
        fmt: str = "RGB565",
        quality: int = 85,
    ) -> tuple[bytes, int, int]:
        """
        Convert a raw ESP32 camera frame to JPEG bytes.

        Automatically corrects height if the frame was truncated (fewer bytes
        than width × height × bytes_per_pixel — common with real hardware).

        Args:
            raw:     Raw binary pixel data from the ESP32 camera.
            width:   Frame width declared by the device.
            height:  Frame height declared by the device.
            fmt:     Pixel format — ``"RGB565"`` (2 bytes/px) or ``"L"`` (1 byte/px).
            quality: JPEG compression quality (0–95).

        Returns:
            Tuple of ``(jpeg_bytes, actual_width, actual_height)``.
            actual_height may differ from the declared height if the frame
            was truncated.

        Raises:
            ValueError: If the frame is empty after height correction.
        """
        bytes_per_pixel = 2 if fmt == "RGB565" else 1
        actual_height = len(raw) // (width * bytes_per_pixel)

        if actual_height == 0:
            raise ValueError(
                f"Frame is empty after height correction "
                f"(raw={len(raw)} bytes, width={width}, bpp={bytes_per_pixel})."
            )

        if fmt == "RGB565":
            img = self._rgb565_to_image(raw, width, actual_height)
        else:
            img = self._grayscale_to_image(raw, width, actual_height)

        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue(), width, actual_height

    @staticmethod
    def _rgb565_to_image(raw: bytes, width: int, height: int) -> Image.Image:
        pixels = np.frombuffer(raw, dtype=np.dtype(">u2")).reshape((height, width))
        r = ((pixels >> 11) & 0x1F).astype(np.uint8) << 3
        g = ((pixels >> 5)  & 0x3F).astype(np.uint8) << 2
        b = ( pixels        & 0x1F).astype(np.uint8) << 3
        return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")

    @staticmethod
    def _grayscale_to_image(raw: bytes, width: int, height: int) -> Image.Image:
        return Image.frombytes("L", (width, height), raw)
