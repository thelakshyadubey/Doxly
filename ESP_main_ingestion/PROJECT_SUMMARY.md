# ESP32 Image Upload Server - Project Summary

## 🎯 Project Overview

A **FastAPI-based microservice** that receives raw image data from ESP32 microcontroller devices, converts them to JPEG, and uploads both raw and processed versions to Cloudinary cloud storage. Acts as an intermediary between IoT devices and cloud storage.

---

## 📋 Core Functionality

### **Main Purpose**
- Receive raw binary image data from ESP32 devices
- Automatically convert image formats (RGB565 or Grayscale → RGB → JPEG)
- Upload to Cloudinary for persistent cloud storage
- Return URLs for downstream integration

### **Key Features**
- ✅ Handles two image formats: RGB565 (2 bytes/pixel) and Grayscale (1 byte/pixel)
- ✅ Automatic frame truncation detection and height correction
- ✅ JPEG compression (85% quality) for bandwidth efficiency
- ✅ Dual file storage: raw binary + JPEG image
- ✅ Comprehensive logging with timestamps
- ✅ Cloud storage organization by device MAC address
- ✅ Health check endpoint for monitoring

---

## 🔌 API Endpoints

### **1. POST `/capture` - Main Endpoint**
**Purpose:** Receive and process image data

**Input (Form-Data):**
```
{
  "image": <binary_file>,           // Raw image file (required)
  "mac": "AA:BB:CC:DD:EE:FF",      // Device MAC (required)
  "width": 240,                     // Image width in pixels (required)
  "height": 320,                    // Image height in pixels (required)
  "format": "RGB565"                // "RGB565" or "L" (optional, default: RGB565)
}
```

**Output (JSON):**
```json
{
  "status": "ok",
  "mac": "AA:BB:CC:DD:EE:FF",
  "format": "RGB565",
  "dimensions": "240x320",
  "jpeg_bytes": 6684,
  "cloudinary_url": "https://res.cloudinary.com/.../esp32/jpeg/AABBCCDDEEFF_20260510_111514_480685.jpg",
  "raw_url": "https://res.cloudinary.com/.../esp32/raw/AABBCCDDEEFF_20260510_111514_480685"
}
```

### **2. GET `/health` - Health Check**
**Purpose:** Monitor server status

**Output:**
```json
{"status": "ok"}
```

---

## 📥 Input Specifications

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `image` | File | Yes | Raw binary image data | Binary file (RGB565: 153.6KB for 240x320) |
| `mac` | String | Yes | Device MAC address | "AA:BB:CC:DD:EE:FF" |
| `width` | Integer | Yes | Image width in pixels | 240 |
| `height` | Integer | Yes | Image height in pixels | 320 |
| `format` | String | No | Image format (RGB565 or L) | "RGB565" (default) or "L" |

### **Image Format Details:**
- **RGB565**: Each pixel = 2 bytes. Format: RRRRRGGG GGGBBBBB
  - Red: 5 bits (0-31)
  - Green: 6 bits (0-63)
  - Blue: 5 bits (0-31)
  - Expected file size: `width × height × 2` bytes

- **Grayscale (L)**: Each pixel = 1 byte (0-255 intensity)
  - Expected file size: `width × height` bytes

---

## 📤 Output Specifications

### **Success Response (200 OK):**
```json
{
  "status": "ok",
  "mac": "AA:BB:CC:DD:EE:FF",
  "format": "RGB565",
  "dimensions": "240x320",
  "jpeg_bytes": 6684,
  "cloudinary_url": "https://res.cloudinary.com/dvuznvhpo/image/upload/v1778411715/esp32/jpeg/AABBCCDDEEFF_20260510_111514_480685.jpg",
  "raw_url": "https://res.cloudinary.com/dvuznvhpo/raw/upload/v1778411714/esp32/raw/AABBCCDDEEFF_20260510_111514_480685"
}
```

### **Error Response (400 Bad Request):**
```json
{
  "status": "error",
  "detail": "empty frame"
}
```

### **Key Output Fields:**
- `cloudinary_url`: **Main output** - Direct link to JPEG image (use this for your frontend/other software)
- `raw_url`: Link to raw binary file
- `jpeg_bytes`: Compressed file size
- `dimensions`: Final image dimensions (may be corrected if truncated)

---

## 🔧 Technology Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| FastAPI | 0.115.0 | Web framework |
| Uvicorn | 0.30.6 | ASGI server |
| Pillow | 10.4.0 | Image processing |
| NumPy | 1.26.4 | Binary data conversion |
| Cloudinary | 1.40.0 | Cloud storage |
| Python | 3.x | Runtime |

---

## ☁️ Cloudinary Integration

### **Configuration:**
```python
cloudinary.config(
    cloud_name = "dvuznvhpo",
    api_key     = "422642487119727",
    api_secret  = "JadFS5_oDmWUF5fy7RoVP0CuYVU",
    secure     = True
)
```

### **Storage Structure:**
```
Cloudinary Account
├── esp32/jpeg/
│   └── {MAC_ADDRESS}_{TIMESTAMP}.jpg    (JPEG image)
└── esp32/raw/
    └── {MAC_ADDRESS}_{TIMESTAMP}.bin    (Raw binary)
```

### **Metadata Tags:**
Each file is tagged with: `[mac_address, "esp32", "jpeg"/"raw", format]`

---

## 🚀 How to Run

### **Local Development:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

Server starts on: `http://localhost:8000`

### **Production Deployment:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 📡 Integration with Your Frontend/Other Software

### **Step 1: Send Image Data**
```bash
curl -X POST http://localhost:8000/capture \
  -F "image=@image.bin" \
  -F "mac=AA:BB:CC:DD:EE:FF" \
  -F "width=240" \
  -F "height=320" \
  -F "format=RGB565"
```

### **Step 2: Receive Response**
You'll get back the `cloudinary_url` (JPEG image link):
```
https://res.cloudinary.com/dvuznvhpo/image/upload/v.../esp32/jpeg/AABBCCDDEEFF_20260510_111514_480685.jpg
```

### **Step 3: Use in Your Software**
- **Display in UI:** Use the `cloudinary_url` to show the image
- **Store reference:** Save the URL in your database
- **Process further:** Download and process the image as needed
- **Share:** Share the JPEG URL with users/systems

---

## 🔄 Processing Pipeline

```
ESP32 Device
     ↓
  [POST /capture with raw binary image]
     ↓
  [App receives & validates]
     ↓
  [Detect & correct frame truncation]
     ↓
  [Convert RGB565 or Grayscale → RGB]
     ↓
  [Encode as JPEG (85% quality)]
     ↓
  [Upload raw binary to Cloudinary]
     ↓
  [Upload JPEG to Cloudinary]
     ↓
  [Return both URLs to caller]
     ↓
Your Software/Frontend
```

---

## 📊 Performance Metrics

### **Tested Configuration:**
- Image size: 240×320 pixels
- Format: RGB565
- Raw file: 153.6 KB
- JPEG output: 6.68 KB
- Compression ratio: **95.6%**
- Processing time: < 500ms

---

## ✅ Data Flow Example

1. **ESP32 captures image** → 153.6 KB RGB565 raw data
2. **Sends to `/capture`** with MAC, width, height, format
3. **Server processes:**
   - Converts RGB565 → RGB (8-bit)
   - Encodes to JPEG
   - Uploads both to Cloudinary
4. **Returns JSON with URLs**
5. **Your software gets JPEG URL:**
   ```
   https://res.cloudinary.com/dvuznvhpo/image/upload/v1778411715/esp32/jpeg/AABBCCDDEEFF_20260510_111514_480685.jpg
   ```
6. **Display or process the JPEG image**

---

## 🔐 Security Notes

⚠️ **Current Status:** 
- Cloudinary credentials are **hardcoded** in `app.py`
- No authentication on the `/capture` endpoint
- Production: Move credentials to environment variables

**For integration:**
```python
# Better approach (use environment variables)
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key     = os.getenv("CLOUDINARY_API_KEY"),
    api_secret  = os.getenv("CLOUDINARY_API_SECRET"),
    secure     = True
)
```

---

## 📦 Dependencies

```
fastapi==0.115.0
uvicorn==0.30.6
pillow==10.4.0
python-multipart==0.0.9
aiofiles==23.2.1
cloudinary==1.40.0
numpy==1.26.4
```

---

## 🎯 Integration Checklist for Your Project

- [ ] Copy `app.py` to your project
- [ ] Copy dependencies from `requirements.txt`
- [ ] Configure Cloudinary credentials (or use env variables)
- [ ] Run the server
- [ ] Send POST requests to `/capture` endpoint
- [ ] Receive and use the `cloudinary_url` in your software
- [ ] Implement URL storage/display in your frontend

---

## 📝 Example Integration Code

```python
# Your software calling this service
import requests

response = requests.post(
    "http://localhost:8000/capture",
    files={"image": open("raw_image.bin", "rb")},
    data={
        "mac": "AA:BB:CC:DD:EE:FF",
        "width": 240,
        "height": 320,
        "format": "RGB565"
    }
)

result = response.json()
if result["status"] == "ok":
    jpeg_url = result["cloudinary_url"]  # ← USE THIS IN YOUR FRONTEND
    print(f"Image uploaded to: {jpeg_url}")
    # Store jpeg_url in database or pass to frontend
else:
    print(f"Error: {result['detail']}")
```

---

## 🤝 Ready for Merge

This microservice is **independently deployable** and can:
- Run on the same server as your project
- Run on a separate Docker container
- Be called via HTTP from your frontend/backend
- Return direct image URLs for immediate use

**Output for your frontend:** `cloudinary_url` (JPEG image link)

