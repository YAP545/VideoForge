"""Caption generator and video compositor.
Features:
- Professional HarfBuzz + FreeType text shaping using Noto Sans Devanagari.
  (All matras and conjunct ligatures like 'क्य' and 'फ्त' join with 100% typographical precision).
- Transparent PNG subtitle overlays per scene.
- Seamless crossfade video concatenation via imageio-ffmpeg.
- Background music mixing (mp3/wav) with gentle fade in and out.
- Dual export: final video with music and clean video without music.
- SRT subtitle generation.
"""
import unicode_compat
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import imageio_ffmpeg
from PIL import Image, ImageDraw

import freetype
import uharfbuzz as hb

BASE = Path(__file__).parent
FONTS_DIR = BASE / "static" / "fonts"
NOTO_FONT_PATH = FONTS_DIR / "NotoSansDevanagari.ttf"


def get_font_path() -> Path:
    if NOTO_FONT_PATH.exists():
        return NOTO_FONT_PATH
    return Path("C:/Windows/Fonts/Nirmala.ttc")


def shape_and_render_caption(
    text: str,
    font_path: Path,
    font_size: int = 34,
    width: int = 1280,
    height: int = 720,
    text_color: tuple = (255, 255, 255, 255),
    bg_box_color: tuple = (12, 14, 20, 160),
) -> Image.Image:
    """Render shaped Devanagari or Latin text onto a transparent image with rounded background box."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if not text.strip():
        return img

    ft_face = freetype.Face(str(font_path))
    ft_face.set_char_size(font_size * 64)

    with open(font_path, "rb") as f:
        font_bytes = f.read()

    hb_blob = hb.Blob(font_bytes)
    hb_face = hb.Face(hb_blob)
    hb_font = hb.Font(hb_face)
    upem = hb_face.upem or 1000
    hb_font.scale = (upem, upem)

    hb_buf = hb.Buffer()
    hb_buf.add_str(text)
    hb_buf.guess_segment_properties()
    hb.shape(hb_font, hb_buf)

    infos = hb_buf.glyph_infos
    positions = hb_buf.glyph_positions

    scale = font_size / upem
    text_width = sum(pos.x_advance for pos in positions) * scale
    pad_x = 24
    pad_y = 12
    box_w = int(text_width + pad_x * 2)
    box_h = int(font_size * 1.5 + pad_y * 2)
    box_x = max(20, (width - box_w) // 2)
    box_y = max(10, height - box_h - int(height * (0.22 if height > width else 0.12)))

    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([box_x, box_y, box_x + box_w, box_y + box_h], radius=12, fill=bg_box_color)

    baseline_y = box_y + pad_y + int(font_size * 1.05)
    curr_x = float(box_x + pad_x)

    for info, pos in zip(infos, positions):
        gid = info.codepoint
        ft_face.load_glyph(gid, freetype.FT_LOAD_RENDER)
        slot = ft_face.glyph
        bm = slot.bitmap

        gx = int(curr_x + (pos.x_offset * scale) + slot.bitmap_left)
        gy = int(baseline_y - (pos.y_offset * scale) - slot.bitmap_top)

        if bm.width > 0 and bm.rows > 0:
            glyph_arr = np.array(bm.buffer, dtype=np.uint8).reshape((bm.rows, bm.width))
            glyph_mask = Image.fromarray(glyph_arr, mode="L")
            solid = Image.new("RGBA", (bm.width, bm.rows), text_color)
            img.paste(solid, (gx, gy), glyph_mask)

        curr_x += pos.x_advance * scale

    return img


def shape_and_render_multiline_prompt(
    text: str,
    font_path: Path,
    font_size: int = 34,
    width: int = 1280,
    height: int = 720,
    margin_pct: float = 0.06,
    text_color: tuple = (255, 255, 255, 255),
    bg_box_color: tuple = (12, 14, 20, 160),
    border_color: tuple = (255, 107, 53, 110),
) -> Image.Image:
    """Render shaped multiline prompt text with safe margins and HarfBuzz Devanagari shaping."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if not text.strip():
        return img

    ft_face = freetype.Face(str(font_path))
    ft_face.set_char_size(font_size * 64)

    with open(font_path, "rb") as f:
        font_bytes = f.read()

    hb_blob = hb.Blob(font_bytes)
    hb_face = hb.Face(hb_blob)
    hb_font = hb.Font(hb_face)
    upem = hb_face.upem or 1000
    hb_font.scale = (upem, upem)
    scale = font_size / upem

    def measure_line(s: str) -> float:
        if not s:
            return 0.0
        buf = hb.Buffer()
        buf.add_str(s)
        buf.guess_segment_properties()
        hb.shape(hb_font, buf)
        return sum(pos.x_advance for pos in buf.glyph_positions) * scale

    # Safe margin: at least 6% on left and right
    margin_x = int(width * max(0.06, margin_pct))
    max_text_w = width - (margin_x * 2) - 48

    # Wrap words into lines
    words = text.strip().split()
    lines = []
    curr_line = []
    for word in words:
        test = " ".join(curr_line + [word])
        w = measure_line(test)
        if w <= max_text_w or not curr_line:
            curr_line.append(word)
        else:
            lines.append(" ".join(curr_line))
            curr_line = [word]
    if curr_line:
        lines.append(" ".join(curr_line))

    line_widths = [measure_line(l) for l in lines]
    max_measured_w = max(line_widths) if line_widths else 0

    pad_x = 24
    pad_y = 12
    line_height = int(font_size * 1.30)

    box_w = min(width - (margin_x * 2), int(max_measured_w + pad_x * 2))
    box_h = int(len(lines) * line_height + pad_y * 2)

    box_x = (width - box_w) // 2
    # Safe area placement: 22% from bottom for 9:16 vertical videos, 12% for landscape
    box_y = max(10, height - box_h - int(height * (0.22 if height > width else 0.12)))

    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_h],
        radius=12,
        fill=bg_box_color,
        outline=border_color,
        width=2,
    )

    for i, line_str in enumerate(lines):
        line_w = line_widths[i]
        line_x = box_x + (box_w - line_w) // 2
        baseline_y = box_y + pad_y + (i * line_height) + int(font_size * 0.95)

        buf = hb.Buffer()
        buf.add_str(line_str)
        buf.guess_segment_properties()
        hb.shape(hb_font, buf)

        infos = buf.glyph_infos
        positions = buf.glyph_positions

        curr_x = float(line_x)
        for info, pos in zip(infos, positions):
            gid = info.codepoint
            ft_face.load_glyph(gid, freetype.FT_LOAD_RENDER)
            slot = ft_face.glyph
            bm = slot.bitmap

            gx = int(curr_x + (pos.x_offset * scale) + slot.bitmap_left)
            gy = int(baseline_y - (pos.y_offset * scale) - slot.bitmap_top)

            if bm.width > 0 and bm.rows > 0:
                glyph_arr = np.array(bm.buffer, dtype=np.uint8).reshape((bm.rows, bm.width))
                glyph_mask = Image.fromarray(glyph_arr, mode="L")
                solid = Image.new("RGBA", (bm.width, bm.rows), text_color)
                img.paste(solid, (gx, gy), glyph_mask)

            curr_x += pos.x_advance * scale

    return img


def export_srt(scenes: List[Dict[str, str]], clip_duration: float, srt_path: Path):
    """Write subtitle file in standard .srt format."""
    def fmt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    curr = 0.0
    idx = 1
    for s in scenes:
        caption = s.get("caption", "").strip()
        end = curr + clip_duration
        if caption:
            lines.append(str(idx))
            lines.append(f"{fmt_time(curr)} --> {fmt_time(end)}")
            lines.append(caption)
            lines.append("")
            idx += 1
        curr = end

    srt_path.write_text("\n".join(lines), encoding="utf-8")


def stitch_clips(
    clip_paths: List[Path],
    scene_captions: List[str],
    output_path: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    clip_duration: float = 5.0,
    crossfade_sec: float = 0.3,
    music_path: Optional[Path] = None,
    no_music_output_path: Optional[Path] = None,
) -> None:
    """Burn captions into clips, concatenate with crossfades, and optionally mix music."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    font_path = get_font_path()
    temp_dir = output_path.parent / f"tmp_{output_path.stem}"
    temp_dir.mkdir(exist_ok=True)

    try:
        processed_clips = []
        for i, clip in enumerate(clip_paths):
            caption = scene_captions[i] if i < len(scene_captions) else ""
            overlay_clip = temp_dir / f"clip_proc_{i}.mp4"

            if caption.strip():
                png_path = temp_dir / f"sub_{i}.png"
                cap_img = shape_and_render_caption(caption, font_path, font_size=36, width=width, height=height)
                cap_img.save(str(png_path), "PNG")

                cmd = [
                    ffmpeg_exe, "-y",
                    "-i", str(clip),
                    "-i", str(png_path),
                    "-filter_complex",
                    f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}[base];"
                    f"[base][1:v]overlay=0:0[v]",
                    "-map", "[v]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(overlay_clip),
                ]
            else:
                cmd = [
                    ffmpeg_exe, "-y",
                    "-i", str(clip),
                    "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(overlay_clip),
                ]

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processed_clips.append(overlay_clip)

        n = len(processed_clips)
        target_no_music = no_music_output_path or (output_path if not music_path else temp_dir / "clean_nomusic.mp4")

        if n == 1:
            shutil.copy2(processed_clips[0], target_no_music)
        else:
            inputs = []
            for c in processed_clips:
                inputs.extend(["-i", str(c)])

            filter_parts = []
            last_out = "0:v"
            offset = clip_duration - crossfade_sec

            for i in range(1, n):
                next_in = f"{i}:v"
                curr_out = f"v{i}"
                filter_parts.append(
                    f"[{last_out}][{next_in}]xfade=transition=fade:duration={crossfade_sec}:offset={offset:.2f}[{curr_out}]"
                )
                last_out = curr_out
                offset += clip_duration - crossfade_sec

            filter_str = ";".join(filter_parts)
            cmd_stitch = [
                ffmpeg_exe, "-y",
                *inputs,
                "-filter_complex", filter_str,
                "-map", f"[{last_out}]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(target_no_music),
            ]
            subprocess.run(cmd_stitch, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if music_path and music_path.exists():
            # Mix music with low volume and in/out fades
            cmd_music = [
                ffmpeg_exe, "-y",
                "-i", str(target_no_music),
                "-i", str(music_path),
                "-filter_complex",
                "[1:a]volume=0.22,afade=t=in:ss=0:d=1,afade=t=out:st=16:d=2[music];"
                "[0:v]copy[v]",
                "-map", "[v]", "-map", "[music]",
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                str(output_path),
            ]
            subprocess.run(cmd_music, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif target_no_music != output_path:
            shutil.copy2(target_no_music, output_path)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
