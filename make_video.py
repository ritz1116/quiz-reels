"""Render one quiz Reel (1080x1920 MP4) from a quiz JSON file.

Usage: python make_video.py quiz.json out_dir
Writes out_dir/quiz.mp4 and out_dir/timings.json.
Voice: Kokoro (free, local). Visuals: Pillow. Encoding: ffmpeg.
"""
import sys, os, json, math, subprocess, wave, urllib.request
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SANS = os.path.join(HERE, "assets", "fonts", "DMSans[opsz,wght].ttf")
SERIF = os.path.join(HERE, "assets", "fonts", "Gelasio[wght].ttf")
MODEL_DIR = os.environ.get("KOKORO_DIR", os.path.join(HERE, ".kokoro"))
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
VOICE = os.environ.get("QUIZ_VOICE", "am_michael")
W, H, FPS, TSR = 1080, 1920, 30, 24000
PAPER = (232, 242, 252); INK = (24, 28, 40); GREY = (110, 114, 124); RED = (214, 48, 48); GREEN = (24, 150, 84)

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
QF = lambda: font(SANS, 58, 700)
OF = lambda: font(SANS, 54, 500)

def wrap(text, f, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if f.getlength(t) <= maxw: cur = t
        else: lines.append(cur); cur = w
    lines.append(cur); return lines

def ease(u): u = max(0, min(1, u)); return u * u * (3 - 2 * u)

class Worksheet:
    def __init__(self, quiz, events):
        self.quiz, self.ev, self.qs = quiz, events, quiz["questions"]
        y, self.blk = 300, []
        for i, q in enumerate(self.qs):
            b = dict(top=y, qlines=[], opts=[])
            for l in wrap(f"{i+1} ) {q['text']}", QF(), 930):
                b["qlines"].append((l, y)); y += 86
            y += 18
            for o in q["options"]:
                b["opts"].append((o, y)); y += 88
            y += 64; self.blk.append(b)
        ph = max(H, y + 40); self.ph = ph
        rng = np.random.default_rng(3)
        arr = np.full((ph, W, 3), PAPER, dtype=np.int16) + rng.integers(-3, 4, (ph, W, 1)).astype(np.int16)
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
        cat = quiz["category"].upper(); fs = 88
        while font(SERIF, fs, 700).getlength(cat) > 940: fs -= 4
        d.text((W // 2, 95), cat, font=font(SERIF, fs, 700), fill=INK, anchor="mm")
        d.text((W // 2, 185), quiz.get("title", "Can you score 5/5?"), font=font(SANS, 46, 500), fill=GREY, anchor="mm")
        d.line([(70, 240), (W - 70, 240)], fill=(200, 200, 205), width=3)
        for b in self.blk:
            for l, ly in b["qlines"]: d.text((70, ly), l, font=QF(), fill=INK)
            for o, oy in b["opts"]:
                d.ellipse((74, oy + 6, 118, oy + 50), outline=INK, width=4)
                d.text((150, oy), o, font=OF(), fill=INK)
        self.page = im; maxs = ph - H
        self.keys = [(0.0, 0.0)]
        for e in events:
            if e["kind"] == "question":
                self.keys.append((max(0, e["start"] - 0.25), max(0, min(maxs, self.blk[e["q"]]["top"] - 300))))

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
                od.rounded_rectangle((70, y0 - 4, W - 70, y0 + 74), radius=14, fill=(60, 200, 120, int(95 * a)))
                solid.append(("tick", e["start"], y0))
            else: solid.append(("ul", e, b, c))
        im = Image.alpha_composite(im, ov).convert("RGB"); d = ImageDraw.Draw(im)
        for s in solid:
            if s[0] == "ul":
                _, e, b, cc = s; tot = sum(len(l) for l, _ in b["qlines"]); cum = 0
                p_all = (t - e["start"]) / (e["end"] - e["start"] - 0.05)
                for li, (l, ly) in enumerate(b["qlines"]):
                    share = len(l) / tot; p = (p_all - cum) / share; cum += share
                    x1 = 70 + QF().getlength(l); y = ly - cc + 68
                    pts = [(70 + (x1 - 70) * k / 40, y + 3.5 * math.sin(k * 0.7 + li * 2.0)) for k in range(41)]
                    self._poly(d, pts, p, RED, 9)
            else:
                _, st, y0 = s; cx, cy = 96, y0 + 28
                self._poly(d, [(cx - 26, cy + 2), (cx - 6, cy + 28), (cx + 36, cy - 34)], (t - st) / 0.4, GREEN, 14)
        for e in self.ev:
            if e["kind"] == "think" and e["start"] <= t < e["end"]:
                mid = self.blk[e["q"]]["opts"][1][1] - c + 36; cx, r = 900, 78; el = t - e["start"]
                d.ellipse((cx - r, mid - r, cx + r, mid + r), fill=(255, 255, 255), outline=(210, 212, 218), width=10)
                d.arc((cx - r, mid - r, cx + r, mid + r), -90, -90 + 360 * (1 - el / 2.0), fill=RED, width=12)
                d.text((cx, mid - 4), "2" if el < 1 else "1", font=font(SANS, 92, 800), fill=INK, anchor="mm")
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
