import logging
import uvicorn
import cloudinary
import cloudinary.uploader
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from PIL import Image
import numpy as np
import io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ===== CONFIGURE THESE =====
cloudinary.config(
    cloud_name = "dvuznvhpo",
    api_key     = "422642487119727",
    api_secret  = "JadFS5_oDmWUF5fy7RoVP0CuYVU",
    secure     = True
)
# ============================

app = FastAPI(title="ESP32 Raw Image Receiver")


def rgb565_to_image(raw: bytes, width: int, height: int) -> Image.Image:
    pixels = np.frombuffer(raw, dtype=np.dtype('>u2')).reshape((height, width))
    r = ((pixels >> 11) & 0x1F).astype(np.uint8) << 3
    g = ((pixels >> 5)  & 0x3F).astype(np.uint8) << 2
    b = ( pixels        & 0x1F).astype(np.uint8) << 3
    return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")


def grayscale_to_image(raw: bytes, width: int, height: int) -> Image.Image:
    return Image.frombytes("L", (width, height), raw)


@app.post("/capture")
async def receive_raw(
    image:  UploadFile = File(...),
    mac:    str        = Form(...),
    width:  int        = Form(...),
    height: int        = Form(...),
    format: str        = Form(default="RGB565"),
):
    raw = await image.read()
    ts  = datetime.now(timezone.utc).isoformat()
    bytes_per_pixel = 2 if format == "RGB565" else 1
    safe_mac  = mac.replace(":", "")
    file_ts   = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')

    log.info("=" * 50)
    log.info("  Timestamp  : %s", ts)
    log.info("  Device MAC : %s", mac)
    log.info("  Format     : %s (%d bytes/px)", format, bytes_per_pixel)
    log.info("  Dimensions : %dx%d", width, height)
    log.info("  Raw size   : %d bytes (expected %d)", len(raw), width * height * bytes_per_pixel)

    # ── Upload raw binary first ───────────────────────────────────────────────
    raw_public_id = f"esp32/raw/{safe_mac}_{file_ts}"
    raw_result = cloudinary.uploader.upload(
        io.BytesIO(raw),
        public_id     = raw_public_id,
        resource_type = "raw",        # non-image binary
        overwrite     = True,
        tags          = [safe_mac, "esp32", "raw", format.lower()],
    )
    raw_url = raw_result.get("secure_url", "")
    log.info("  Raw URL    : %s", raw_url)

    # ── Correct height if frame was truncated ─────────────────────────────────
    actual_height = len(raw) // (width * bytes_per_pixel)
    if actual_height != height:
        log.warning("  Height corrected: %d -> %d", height, actual_height)
        height = actual_height

    if height == 0:
        log.error("  Height is 0 after correction — bad frame")
        return JSONResponse({"status": "error", "detail": "empty frame"}, status_code=400)

    # ── Convert raw -> PIL Image ──────────────────────────────────────────────
    if format == "RGB565":
        img = rgb565_to_image(raw, width, height)
    else:
        img = grayscale_to_image(raw, width, height)

    img = img.convert("RGB")

    # ── Encode as JPEG ────────────────────────────────────────────────────────
    jpeg_buf = io.BytesIO()
    img.save(jpeg_buf, format="JPEG", quality=85)
    jpeg_buf.seek(0)
    jpeg_size = jpeg_buf.getbuffer().nbytes
    log.info("  JPEG size  : %d bytes (%.1f KB)", jpeg_size, jpeg_size / 1024)

    # ── Upload JPEG to Cloudinary ─────────────────────────────────────────────
    public_id = f"esp32/jpeg/{safe_mac}_{file_ts}"
    log.info("  Uploading JPEG: %s", public_id)
    result = cloudinary.uploader.upload(
        jpeg_buf,
        public_id     = public_id,
        resource_type = "image",
        overwrite     = True,
        tags          = [safe_mac, "esp32", "jpeg", format.lower()],
    )
    secure_url = result.get("secure_url", "")
    log.info("  JPEG URL   : %s", secure_url)
    log.info("=" * 50)

    return JSONResponse({
        "status":         "ok",
        "mac":            mac,
        "format":         format,
        "dimensions":     f"{width}x{height}",
        "jpeg_bytes":     jpeg_size,
        "cloudinary_url": secure_url,
        "raw_url":        raw_url,
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)