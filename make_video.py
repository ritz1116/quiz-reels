"""Render one quiz Reel (1080x1920 MP4) from a quiz JSON file.

Usage: python make_video.py quiz.json out_dir
Writes out_dir/quiz.mp4 and out_dir/timings.json.
Voice: Kokoro (free, local). Visuals: Pillow. Encoding: ffmpeg.
"""
import sys, os, json, math, subprocess, wave, urllib.request
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SANS = os.path.join(HERE, "assets", "fonts", "DMSans[opsz,wght].ttf")
SERIF = os.path.join(HERE, "assets", "fonts", "Gelasio[wght].ttf")
MODEL_DIR = os.environ.get("KOKORO_DIR", os.path.join(HERE, ".kokoro"))
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
VOICE = os.environ.get("QUIZ_VOICE", "am_michael")
W, H, FPS, TSR = 1080, 1920, 30, 24000
PAPER = (252, 251, 247); INK = (24, 28, 40); GREY = (110, 114, 124); RED = (214, 48, 48); GREEN = (24, 150, 84)

def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for f in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        p = os.path.join(MODEL_DIR, f)
        if not os.path.exists(p):
            print("downloading", f, flush=True)
            urllib.request.urlretrieve(MODEL_URL + f, p + ".part"); os.replace(p + ".part", p)
    return os.path.join(MODEL_DIR, "kokoro-v1.0.onnx"), os.path.join(MODEL_DIR, "voices-v1.0.bin")

# ---------------- narration + timings ----------------
def narrate(quiz):
    from kokoro_onnx import Kokoro
    k = Kokoro(*ensure_models())
    def say(t):
        s, sr = k.create(t, voice=VOICE, speed=1.0, lang="en-us"); assert sr == TSR; return s.astype(np.float32)
    chunks, ev, t = [], [], [0.0]
    def sil(d): chunks.append(np.zeros(int(d * TSR), dtype=np.float32)); t[0] += d
    def add(kind, text, **kw):
        s = say(text); d = len(s) / TSR
        ev.append(dict(kind=kind, start=round(t[0], 3), end=round(t[0] + d, 3), **kw)); chunks.append(s); t[0] += d
    sil(0.4)
    for i, q in enumerate(quiz["questions"]):
        add("question", q["text"], q=i); sil(0.3)
        ev.append(dict(kind="think", start=round(t[0], 3), end=round(t[0] + 2.0, 3), q=i)); sil(2.0)
        add("answer", f"The answer is {q['options'][q['answer']]}!", q=i); sil(1.1)
    add("outro", "How many did you get? Comment your score!"); sil(0.7)
    return np.concatenate(chunks), ev, round(t[0], 3)

# ---------------- worksheet visuals ----------------
_fc = {}
def font(path, size, w):
    key = (path, size, w)
    if key not in _fc:
        f = ImageFont.truetype(path, size)
        try: f.set_variation_by_axes([14, w] if path == SANS else [w])
        except Exception: pass
        _fc[key] = f
    return _fc[key]
QF = lambda: font(SANS, 60, 700)
OF = lambda: font(SANS, 56, 600)

def wrap(text, f, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if f.getlength(t) <= maxw: cur = t
        else: lines.append(cur); cur = w
    lines.append(cur); return lines

def ease(u): u = max(0, min(1, u)); return u * u * (3 - 2 * u)

# (background top, background bottom, accent)  one palette per subject, so each day looks different
PALETTES = [((38, 30, 110), (120, 52, 170), (255, 170, 40)), ((10, 70, 120), (20, 150, 160), (255, 205, 60)),
            ((120, 25, 70), (220, 80, 80), (255, 215, 90)), ((20, 90, 70), (60, 170, 110), (255, 140, 60)),
            ((30, 50, 130), (70, 120, 220), (255, 120, 140)), ((90, 40, 130), (200, 70, 150), (90, 220, 200))]
BADGE = [(255, 99, 99), (60, 140, 255), (255, 176, 32)]
CX0, CX1 = 44, W - 44          # card edges
TX = 92                         # question text x
RH, RS = 102, 118              # option row height / step

def palette_for(cat): return PALETTES[sum(map(ord, cat)) % len(PALETTES)]

class Worksheet:
    def __init__(self, quiz, events):
        self.quiz, self.ev, self.qs = quiz, events, quiz["questions"]
        self.pal = palette_for(quiz["category"])
        y, self.blk = 330, []
        for i, q in enumerate(self.qs):
            b = dict(top=y, qlines=[], opts=[]); y += 122
            for l in wrap(q["text"], QF(), 890):
                b["qlines"].append((l, y)); y += 86
            y += 22
            for o in q["options"]:
                b["opts"].append((o, y)); y += RS
            b["bottom"] = y - (RS - RH) + 34; y = b["bottom"] + 52; self.blk.append(b)
        ph = max(H, y + 40); self.ph = ph
        c1, c2, acc = self.pal
        t = np.linspace(0, 1, ph)[:, None, None]
        arr = (np.array(c1)[None, None, :] * (1 - t) + np.array(c2)[None, None, :] * t) * np.ones((1, W, 1))
        im = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
        sh = Image.new("RGBA", (W, ph), (0, 0, 0, 0)); sd = ImageDraw.Draw(sh)
        for b in self.blk: sd.rounded_rectangle((CX0 + 4, b["top"] + 12, CX1 + 4, b["bottom"] + 12), radius=44, fill=(0, 0, 0, 70))
        im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(14))); d = ImageDraw.Draw(im)
        # header banner
        hv = Image.new("RGBA", (W, ph), (0, 0, 0, 0)); ImageDraw.Draw(hv).rounded_rectangle((CX0, 44, CX1, 262), radius=44, fill=(0, 0, 0, 70), outline=(255, 255, 255, 160), width=3)
        im = Image.alpha_composite(im, hv); d = ImageDraw.Draw(im)
        cat = quiz["category"].upper(); fs = 96
        while font(SANS, fs, 800).getlength(cat) > 900: fs -= 4
        d.text((W // 2, 128), cat, font=font(SANS, fs, 800), fill=(255, 255, 255), anchor="mm")
        d.text((W // 2, 215), "5 QUESTIONS  \u2022  CAN YOU SCORE 5/5?", font=font(SANS, 38, 700), fill=acc, anchor="mm")
        for i, b in enumerate(self.blk):
            d.rounded_rectangle((CX0, b["top"], CX1, b["bottom"]), radius=44, fill=(255, 255, 255))
            d.rounded_rectangle((CX0 + 36, b["top"] + 30, CX0 + 36 + 300, b["top"] + 92), radius=31, fill=c1)
            d.text((CX0 + 36 + 150, b["top"] + 61), f"QUESTION {i+1}/5", font=font(SANS, 34, 800), fill=(255, 255, 255), anchor="mm")
            for l, ly in b["qlines"]: d.text((TX, ly), l, font=QF(), fill=INK)
            for k, (o, oy) in enumerate(b["opts"]):
                d.rounded_rectangle((CX0 + 32, oy, CX1 - 32, oy + RH), radius=30, fill=(243, 245, 252), outline=(214, 219, 235), width=3)
                bc = BADGE[k]; cx, cy = CX0 + 32 + 58, oy + RH // 2
                d.ellipse((cx - 36, cy - 36, cx + 36, cy + 36), fill=bc)
                d.text((cx, cy - 2), "ABC"[k], font=font(SANS, 44, 800), fill=(255, 255, 255), anchor="mm")
                d.text((CX0 + 32 + 124, cy - 2), o, font=OF(), fill=INK, anchor="lm")
        self.page = im.convert("RGB"); maxs = ph - H
        self.keys = [(0.0, 0.0)]
        for e in events:
            if e["kind"] == "question":
                self.keys.append((max(0, e["start"] - 0.25), max(0, min(maxs, self.blk[e["q"]]["top"] - 330))))

    def cam(self, t):
        pos = self.keys[0][1]
        for i in range(1, len(self.keys)):
            ts, tg = self.keys[i]; prev = self.keys[i - 1][1]
            if t >= ts: pos = prev + (tg - prev) * ease((t - ts) / 0.8)
            else: break
        return pos

    @staticmethod
    def _poly(d, pts, p, color, width):
        if p <= 0: return
        p = min(1, p); segs = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        tg = p * sum(segs); out = [pts[0]]; acc = 0
        for i, s in enumerate(segs):
            if acc + s <= tg: out.append(pts[i + 1]); acc += s
            else:
                r = (tg - acc) / s if s else 0
                out.append((pts[i][0] + (pts[i + 1][0] - pts[i][0]) * r, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * r)); break
        if len(out) > 1: d.line(out, fill=color, width=width, joint="curve")
        for q in (out[0], out[-1]): d.ellipse((q[0] - width / 2, q[1] - width / 2, q[0] + width / 2, q[1] + width / 2), fill=color)

    def frame(self, t):
        c = int(round(self.cam(t))); im = self.page.crop((0, c, W, c + H)).convert("RGBA")
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); od = ImageDraw.Draw(ov); solid = []
        for e in self.ev:
            if e["kind"] not in ("question", "answer") or t < e["start"]: continue
            b = self.blk[e["q"]]
            if e["kind"] == "answer":
                o, oy = b["opts"][self.qs[e["q"]]["answer"]]; y0 = oy - c
                a = ease((t - e["start"]) / 0.3)
                od.rounded_rectangle((CX0 + 32, y0, CX1 - 32, y0 + RH), radius=30, fill=(46, 204, 113, int(235 * a)))
                solid.append(("tick", e["start"], y0, o))
            else: solid.append(("ul", e, b, c))
        im = Image.alpha_composite(im, ov).convert("RGB"); d = ImageDraw.Draw(im)
        for s in solid:
            if s[0] == "ul":
                _, e, b, cc = s; tot = sum(len(l) for l, _ in b["qlines"]); cum = 0
                p_all = (t - e["start"]) / (e["end"] - e["start"] - 0.05)
                for li, (l, ly) in enumerate(b["qlines"]):
                    share = len(l) / tot; p = (p_all - cum) / share; cum += share
                    x1 = TX + QF().getlength(l); y = ly - cc + 70
                    pts = [(TX + (x1 - TX) * k / 40, y + 3.5 * math.sin(k * 0.7 + li * 2.0)) for k in range(41)]
                    self._poly(d, pts, p, RED, 9)
            else:
                _, st, y0, otxt = s; cy = y0 + RH // 2
                d.ellipse((CX0 + 32 + 58 - 36, cy - 36, CX0 + 32 + 58 + 36, cy + 36), fill=(255, 255, 255))
                d.text((CX0 + 32 + 58, cy - 2), "ABC"[self.qs[[x["q"] for x in self.ev if x["kind"] == "answer" and x["start"] == st][0]]["answer"]], font=font(SANS, 44, 800), fill=(24, 150, 84), anchor="mm")
                d.text((CX0 + 32 + 124, cy - 2), otxt, font=OF(), fill=(255, 255, 255), anchor="lm")
                cx = W - 130
                self._poly(d, [(cx - 30, cy + 2), (cx - 8, cy + 26), (cx + 34, cy - 30)], (t - st) / 0.4, (255, 255, 255), 16)
        for e in self.ev:
            if e["kind"] == "think" and e["start"] <= t < e["end"]:
                mid = self.blk[e["q"]]["top"] - c + 61; cx, r = 880, 46; el = t - e["start"]
                d.ellipse((cx - r, mid - r, cx + r, mid + r), fill=(255, 255, 255), outline=(225, 228, 238), width=8)
                d.arc((cx - r, mid - r, cx + r, mid + r), -90, -90 + 360 * (1 - el / 2.0), fill=RED, width=10)
                d.text((cx, mid - 2), "2" if el < 1 else "1", font=font(SANS, 58, 800), fill=INK, anchor="mm")
        return im

# ---------------- audio mix (voice + light music + sfx) ----------------
SR = 44100
def mix_audio(nar24, ev, total, path):
    n = int(total * SR)
    nar = np.interp(np.arange(int(len(nar24) * SR / TSR)) * TSR / SR, np.arange(len(nar24)), nar24)
    rng = np.random.default_rng(7)
    def add(buf, t0, sig):
        i = int(t0 * SR)
        if 0 <= i < len(buf): buf[i:i + len(sig)] += sig[:len(buf) - i]
    def tone(buf, t0, f, dur, decay, vol):
        tt = np.arange(int(dur * SR)) / SR; add(buf, t0, np.sin(2 * np.pi * f * tt) * np.exp(-decay * tt) * vol)
    def scribble(buf, t0, dur, vol):
        m = int(dur * SR); x = rng.standard_normal(m); x = x - np.convolve(x, np.ones(40) / 40, "same")
        tt = np.arange(m) / SR
        env = (0.55 + 0.45 * np.sin(2 * np.pi * 11 * tt)) * np.minimum(1, np.minimum(tt / 0.03, (dur - tt) / 0.03)); add(buf, t0, x * env * vol)
    def swoosh(buf, t0):
        m = int(0.7 * SR); x = np.convolve(rng.standard_normal(m), np.ones(120) / 120, "same"); u = np.arange(m) / m
        x = x * np.sin(np.pi * u) ** 2; add(buf, t0, x / np.abs(x).max() * 0.12)
    music = np.zeros(n + SR); beat = 60 / 108; bass = [110, 110, 130.8, 164.8]; arp = [440, 523.3, 659.3, 523.3]; t = 0.0; k = 0
    while t < total:
        if k % 2 == 0: tone(music, t, bass[(k // 2) % 4], beat * 0.9, 5, 0.55)
        tone(music, t, arp[k % 4] * (1 if (k // 8) % 2 == 0 else 0.75), 0.35, 9, 0.30)
        t += beat / 2; k += 1
    sfx = np.zeros(n + SR)
    for e in ev:
        if e["kind"] == "think":
            for j in range(4): tone(sfx, e["start"] + j * 0.5, 1500 if j % 2 == 0 else 1000, 0.05, 70, 0.5)
        if e["kind"] == "answer":
            for f, v in ((880, 0.5), (1760, 0.25), (2640, 0.12)): tone(sfx, e["start"] - 0.05, f, 0.8, 5, v)
            scribble(sfx, e["start"], 0.4, 0.18)
        if e["kind"] == "question":
            scribble(sfx, e["start"], e["end"] - e["start"], 0.07)
            if e["q"] > 0: swoosh(sfx, e["start"] - 0.3)
    voice = np.zeros(n + SR); voice[:len(nar)] = nar[:n + SR]
    nz = lambda a: a / max(1e-9, np.abs(a).max())
    mix = nz(voice) * 0.95 + nz(sfx) * 0.35 + nz(music) * 0.07
    mix = mix[:n]; mix = mix / np.abs(mix).max() * 0.9
    w = wave.open(path, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()

def main(quiz_path, out_dir):
    quiz = json.load(open(quiz_path)); os.makedirs(out_dir, exist_ok=True)
    assert len(quiz["questions"]) == 5 and all(len(q["options"]) == 3 for q in quiz["questions"]), "need 5 questions x 3 options"
    nar, ev, total = narrate(quiz)
    json.dump(dict(events=ev, total=total), open(os.path.join(out_dir, "timings.json"), "w"), indent=1)
    ws = Worksheet(quiz, ev)
    wav = os.path.join(out_dir, "mix.wav"); mix_audio(nar, ev, total, wav)
    silent = os.path.join(out_dir, "video.tmp.mp4"); final = os.path.join(out_dir, "quiz.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-crf", "19", "-preset", "veryfast", "-pix_fmt", "yuv420p", silent], stdin=subprocess.PIPE)
    for i in range(int(total * FPS)): p.stdin.write(ws.frame(i / FPS).tobytes())
    p.stdin.close(); assert p.wait() == 0, "ffmpeg video encode failed"
    tmp = os.path.join(out_dir, "quiz.tmp.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-shortest", tmp], check=True)
    os.replace(tmp, final); os.remove(silent); print("wrote", final, "duration", total)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
