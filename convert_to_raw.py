from PIL import Image
import numpy as np

img = Image.open(r"C:\Users\Lakshya\Desktop\YASH\OCR\TEST ocr\test ocr.jpeg").convert("RGB").resize((240, 320))
arr = np.array(img)

r = (arr[:,:,0] >> 3).astype(np.uint16)
g = (arr[:,:,1] >> 2).astype(np.uint16)
b = (arr[:,:,2] >> 3).astype(np.uint16)
rgb565 = ((r << 11) | (g << 5) | b).astype(np.dtype(">u2"))

output_path = r"C:\Users\Lakshya\Desktop\YASH\OCR\test_frame.bin"
rgb565.tofile(output_path)
print(f"Written {rgb565.nbytes} bytes to {output_path}")
