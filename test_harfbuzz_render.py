import freetype
import uharfbuzz as hb
import numpy as np
from PIL import Image, ImageDraw

font_path = "static/fonts/NotoSansDevanagari.ttf"
font_size = 40

# Load freetype face
ft_face = freetype.Face(font_path)
ft_face.set_char_size(font_size * 64)

# Load HarfBuzz font
with open(font_path, "rb") as f:
    font_bytes = f.read()

hb_blob = hb.Blob(font_bytes)
hb_face = hb.Face(hb_blob)
hb_font = hb.Font(hb_face)
upem = hb_face.upem or 1000
hb_font.scale = (upem, upem)

# Shape Devanagari text
text = "क्या कोई गिरफ्तार हुआ?"
hb_buf = hb.Buffer()
hb_buf.add_str(text)
hb_buf.guess_segment_properties()
hb.shape(hb_font, hb_buf)

infos = hb_buf.glyph_infos
positions = hb_buf.glyph_positions

scale = font_size / upem
total_width = int(sum(pos.x_advance for pos in positions) * scale) + 60
height = 160
img = Image.new("RGBA", (total_width + 40, height), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Background pill
draw.rounded_rectangle([10, 20, total_width + 30, 120], radius=12, fill=(20, 20, 24, 220))

baseline_y = 80
curr_x = 30.0

for info, pos in zip(infos, positions):
    gid = info.codepoint
    ft_face.load_glyph(gid, freetype.FT_LOAD_RENDER)
    slot = ft_face.glyph
    bm = slot.bitmap
    
    gx = int(curr_x + (pos.x_offset * scale) + slot.bitmap_left)
    gy = int(baseline_y - (pos.y_offset * scale) - slot.bitmap_top)
    
    if bm.width > 0 and bm.rows > 0:
        glyph_arr = np.array(bm.buffer, dtype=np.uint8).reshape((bm.rows, bm.width))
        glyph_img = Image.fromarray(glyph_arr, mode="L")
        # Colorize white
        solid_white = Image.new("RGBA", (bm.width, bm.rows), (255, 255, 255, 255))
        img.paste(solid_white, (gx, gy), glyph_img)
        
    curr_x += pos.x_advance * scale

img.save("static/hindi_harfbuzz_test.png")
print("Saved shaped text to static/hindi_harfbuzz_test.png, width:", total_width)
