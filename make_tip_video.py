"""Render one 'AI tip' Short (1080x1920 MP4) from a tip JSON file.

Usage: python make_tip_video.py tip.json out_dir     -> out_dir/tip.mp4
Voice: Kokoro (free, local). Visuals: Pillow. Encoding: ffmpeg.
"""
import sys, os, json, math, subprocess, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import make_video as mv

W, H, FPS, TSR, SR = mv.W, mv.H, mv.FPS, mv.TSR, 44100
SANS, font, ease = mv.SANS, mv.font, mv.ease
WHITE = (255, 255, 255); ACC = (255, 196, 61); RED = (255, 107, 107); MUTE = (170, 178, 210)

def narrate(tip):
    from kokoro_onnx import Kokoro
    k = Kokoro(*mv.ensure_models())
    chunks, ev, t = [], {}, [0.0]
    def sil(d): chunks.append(np.zeros(int(d * TSR), dtype=np.float32)); t[0] += d
    sil(0.3)
    for key, text in tip["voice"]:
        s, sr = k.create(text, voice=mv.VOICE, speed=1.0, lang="en-us"); s = s.astype(np.float32); assert sr == TSR
        d = len(s) / TSR; ev[key] = (round(t[0], 3), round(t[0] + d, 3)); chunks.append(s); t[0] += d
        sil(0.45 if key not in ("hook",) else 0.7)
    return np.concatenate(chunks), ev, round(t[0] + 0.3, 3)

def make_bg():
    t = np.linspace(0, 1, H)[:, None, None]
    c1, c2 = np.array((12, 18, 44)), np.array((46, 26, 104))
    arr = (c1 * (1 - t) + c2 * t) * np.ones((1, W, 1))
    im = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
    gd.ellipse((W - 520, 260, W + 280, 1060), fill=(120, 80, 255, 70)); gd.ellipse((-300, 1200, 520, 2000), fill=(40, 160, 255, 55))
    im = Image.alpha_composite(im, glow.filter(ImageFilter.GaussianBlur(120)))
    d = ImageDraw.Draw(im)
    for x in range(40, W, 60):
        for y in range(40, H, 60): d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 255, 255, 22))
    return im.convert("RGB")

class Tip:
    def __init__(self, tip, ev):
        self.tip, self.ev = tip, ev; self.bg = make_bg()
        self.pstart = ev["intro"][0]; self.type_end = ev["intro"][1] + 3.2
        self.text = "\n".join(tip["prompt"]); self.n = len(self.text)

    def layer(self):  # RGBA overlay helper
        return Image.new("RGBA", (W, H), (0, 0, 0, 0))

    def rich(self, d, words, cx, y, size, alpha=255, maxw=900):
        f = font(SANS, size, 800); lines, cur = [], []
        for w, col in words:
            test = " ".join(x for x, _ in cur + [(w, col)])
            if f.getlength(test) <= maxw: cur.append((w, col))
            else: lines.append(cur); cur = [(w, col)]
        lines.append(cur)
        for li, ln in enumerate(lines):
            tot = f.getlength(" ".join(x for x, _ in ln)); x = cx - tot / 2
            for w, col in ln:
                d.text((x, y + li * int(size * 1.18)), w, font=f, fill=tuple(col) + (alpha,)); x += f.getlength(w + " ")
        return len(lines)

    def frame(self, t):
        im = self.bg.convert("RGBA"); ov = self.layer(); d = ImageDraw.Draw(ov)
        ev = self.ev
        # brand pill
        d.rounded_rectangle((W // 2 - 190, 90, W // 2 + 190, 170), radius=40, fill=ACC + (255,))
        d.text((W // 2, 130), "AI TIP OF THE DAY", font=font(SANS, 34, 800), fill=(30, 24, 70, 255), anchor="mm")
        # hook
        h0, h1 = ev["hook"]; fade_out = 1 - ease((t - (ev["intro"][0] - 0.1)) / 0.4)
        if fade_out > 0:
            a = int(255 * ease(t / 0.4) * fade_out); yoff = (1 - fade_out) * -80
            self.rich(d, self.tip["hook_words"], W // 2, 640 + yoff, 118, a, 880)
        # prompt card
        if t >= self.pstart:
            u = ease((t - self.pstart) / 0.5)
            d.rounded_rectangle((50, 300, W - 50, 1110), radius=44, fill=(255, 255, 255, int(28 * u)), outline=ACC + (int(220 * u),), width=4)
            d.text((100, 352), "COPY THIS PROMPT", font=font(SANS, 38, 800), fill=ACC + (int(255 * u),))
            d.text((W - 100, 352), "tap to save", font=font(SANS, 30, 600), fill=MUTE + (int(255 * u),), anchor="ra")
            shown = int(self.n * min(1, max(0, (t - self.pstart - 0.3) / (self.type_end - self.pstart - 0.3))))
            txt = self.text[:shown]
            if t % 0.8 < 0.45 and shown < self.n: txt += "|"
            f = font(SANS, 50, 600); y = 430
            for line in txt.split("\n"):
                col = ACC if line[:2] in ("1.", "2.", "3.") else WHITE
                d.text((100, y), line, font=f, fill=col + (255,)); y += 78
        # benefits
        for i, (key, label) in enumerate(self.tip["benefits"]):
            s, e = ev[key]
            if t < s: continue
            u = ease((t - s) / 0.4); y = 1190 + i * 150 + (1 - u) * 50; active = s <= t < e
            d.rounded_rectangle((60, y, W - 60, y + 124), radius=36, fill=(255, 255, 255, int((60 if active else 30) * u)),
                                outline=(ACC if active else MUTE) + (int(255 * u),), width=4 if active else 2)
            d.ellipse((92, y + 22, 172, y + 102), fill=ACC + (int(255 * u),))
            d.text((132, y + 60), str(i + 1), font=font(SANS, 48, 800), fill=(30, 24, 70, int(255 * u)), anchor="mm")
            d.text((210, y + 60), label, font=font(SANS, 46, 700), fill=WHITE + (int(255 * u),), anchor="lm")
        # trick line
        if "trick" in ev and t >= ev["trick"][0]:
            u = ease((t - ev["trick"][0]) / 0.4)
            self.rich(d, self.tip["trick_words"], W // 2, 1680 + (1 - u) * 40, 54, int(255 * u), 940)
        im = Image.alpha_composite(im, ov)
        # call to action
        c0 = ev["cta"][0]
        if t >= c0 - 0.2:
            u = ease((t - (c0 - 0.2)) / 0.5); cv = self.layer(); cd = ImageDraw.Draw(cv)
            cd.rectangle((0, 0, W, H), fill=(10, 12, 34, int(235 * u)))
            bounce = 1 + 0.06 * math.sin((t - c0) * 6) * u
            bx, by = W // 2, 640
            s = 1.0 * bounce
            pts = [(bx - 90 * s, by - 130 * s), (bx + 90 * s, by - 130 * s), (bx + 90 * s, by + 130 * s), (bx, by + 60 * s), (bx - 90 * s, by + 130 * s)]
            cd.polygon(pts, fill=ACC + (int(255 * u),))
            cd.text((W // 2, 930), "SAVE THIS", font=font(SANS, 150, 800), fill=WHITE + (int(255 * u),), anchor="mm")
            cd.text((W // 2, 1090), "FOLLOW FOR A NEW", font=font(SANS, 64, 700), fill=MUTE + (int(255 * u),), anchor="mm")
            cd.text((W // 2, 1175), "AI TRICK EVERY DAY", font=font(SANS, 80, 800), fill=ACC + (int(255 * u),), anchor="mm")
            im = Image.alpha_composite(im, cv)
        return im.convert("RGB")

def mix_audio(nar24, tip_obj, total, path):
    n = int(total * SR); ev = tip_obj.ev
    nar = np.interp(np.arange(int(len(nar24) * SR / TSR)) * TSR / SR, np.arange(len(nar24)), nar24)
    rng = np.random.default_rng(5)
    def add(buf, t0, sig):
        i = int(t0 * SR)
        if 0 <= i < len(buf): buf[i:i + len(sig)] += sig[:len(buf) - i]
    def tone(buf, t0, f, dur, decay, vol):
        tt = np.arange(int(dur * SR)) / SR; add(buf, t0, np.sin(2 * np.pi * f * tt) * np.exp(-decay * tt) * vol)
    def swoosh(buf, t0):
        m = int(0.6 * SR); x = np.convolve(rng.standard_normal(m), np.ones(120) / 120, "same"); u = np.arange(m) / m
        x = x * np.sin(np.pi * u) ** 2; add(buf, t0, x / np.abs(x).max() * 0.2)
    music = np.zeros(n + SR); beat = 60 / 96; bass = [98, 98, 130.8, 116.5]; pad = [392, 466.2, 523.3, 466.2]; t = 0.0; k = 0
    while t < total:
        if k % 2 == 0: tone(music, t, bass[(k // 2) % 4], beat * 0.95, 4, 0.55)
        tone(music, t, pad[k % 4], 0.4, 7, 0.28); t += beat / 2; k += 1
    sfx = np.zeros(n + SR)
    swoosh(sfx, ev["intro"][0] - 0.2)
    ts, te = tip_obj.pstart + 0.3, tip_obj.type_end
    for j in range(int((te - ts) * 14)): tone(sfx, ts + j / 14, 2200 + (j % 3) * 150, 0.02, 120, 0.35)
    for key, _ in tip_obj.tip["benefits"]: tone(sfx, ev[key][0], 880, 0.5, 6, 0.35); tone(sfx, ev[key][0], 1320, 0.5, 6, 0.18)
    swoosh(sfx, ev["cta"][0] - 0.3)
    for f, v in ((784, 0.4), (1175, 0.25), (1568, 0.15)): tone(sfx, ev["cta"][0], f, 0.9, 4, v)
    voice = np.zeros(n + SR); voice[:len(nar)] = nar[:n + SR]
    nz = lambda a: a / max(1e-9, np.abs(a).max())
    mix = nz(voice) * 0.95 + nz(sfx) * 0.3 + nz(music) * 0.07
    mix = mix[:n]; mix = mix / np.abs(mix).max() * 0.9
    w = wave.open(path, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()

def main(tip_path, out_dir):
    tip = json.load(open(tip_path)); os.makedirs(out_dir, exist_ok=True)
    nar, ev, total = narrate(tip); tp = Tip(tip, ev)
    wav = os.path.join(out_dir, "tip.wav"); mix_audio(nar, tp, total, wav)
    silent = os.path.join(out_dir, "tip.silent.mp4"); final = os.path.join(out_dir, "tip.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-crf", "19", "-preset", "veryfast", "-pix_fmt", "yuv420p", silent], stdin=subprocess.PIPE)
    for i in range(int(total * FPS)): p.stdin.write(tp.frame(i / FPS).tobytes())
    p.stdin.close(); assert p.wait() == 0, "ffmpeg video encode failed"
    tmp = os.path.join(out_dir, "tip.tmp.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-shortest", tmp], check=True)
    os.replace(tmp, final); os.remove(silent); print("wrote", final, "duration", total)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
