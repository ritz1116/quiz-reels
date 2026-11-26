"""Generate and verify one 5-question quiz with the Claude API.
# (re-run trigger: credits added)

Usage: python generate_quiz.py --out out_dir [--topic "Science & Space"]
Needs env ANTHROPIC_API_KEY. Writes out_dir/quiz.json (renderer input) and out_dir/verification.json.
Exit code 1 if five verified questions could not be produced (no video should be made).
"""
import os, sys, json, re, random, argparse, datetime
from concurrent.futures import ThreadPoolExecutor
import anthropic

MODEL = os.environ.get("QUIZ_MODEL", "claude-sonnet-5-5")
HERE = os.path.dirname(os.path.abspath(__file__))
TOPICS = [
    ("General Knowledge", "simple, well-known facts: geography, history, everyday world knowledge, famous firsts.", True),
    ("Science & Space", "simple science and astronomy facts taught in school; planets, human body, animals, chemistry basics.", True),
    ("Psychology & Human Behaviour", "ONLY well-established, non-controversial psychology findings and terms (e.g. named biases, memory, sleep). Avoid pop-psychology myths and anything disputed or replication-crisis related.", True),
    ("Technology & AI", "simple technology, internet, computing and AI facts that will not change soon; avoid anything time-sensitive.", True),
    ("Brain Teasers & Logic (Cognitive)", "simple riddles, number sequences, odd-one-out, easy logic. Each must have exactly ONE unambiguous answer that can be solved by reasoning alone.", False),
    ("English (Grammar, Spelling, Vocabulary)", "unambiguous English: correct spelling, correct word choice (their/there), word meanings, common mistakes. Use American English and avoid anything where British/American usage differs.", False),
]
usage = dict(input=0, output=0, searches=0)
client = None

def get_client():
    global client
    if client is None: client = anthropic.Anthropic()
    return client

def call(system, user, tools=None, max_tokens=2500):
    msgs = [{"role": "user", "content": user}]
    n_search = 0; all_urls = []
    for _ in range(6):
        kw = dict(model=MODEL, max_tokens=max_tokens, system=system, messages=msgs)
        if tools: kw["tools"] = tools
        r = get_client().messages.create(**kw)
        usage["input"] += r.usage.input_tokens; usage["output"] += r.usage.output_tokens
        st = getattr(getattr(r.usage, "server_tool_use", None), "web_search_requests", 0) or 0
        usage["searches"] += st; n_search += st
        for b in r.content:
            if getattr(b, "type", "") == "web_search_tool_result" and isinstance(b.content, list):
                for it in b.content:
                    u = getattr(it, "url", None)
                    if u and u not in all_urls: all_urls.append(u)
        if r.stop_reason == "pause_turn":
            msgs.append({"role": "assistant", "content": r.content}); continue
        return r, n_search, all_urls
    raise RuntimeError("too many pause_turn continuations")

def text_of(r): return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
def urls_of(r):
    out = []
    for b in r.content:
        for c in (getattr(b, "citations", None) or []):
            u = getattr(c, "url", None)
            if u and u not in out: out.append(u)
    return out
def extract_json(t):
    m = re.search(r"\{.*\}", t, re.S)
    if not m: raise ValueError("no JSON in reply")
    return json.loads(m.group(0))

def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

def validate(q):
    errs = []; t = (q.get("text") or "").strip(); o = q.get("options"); a = q.get("answer")
    if not t or len(t) > 75: errs.append("question missing or longer than 75 characters")
    if not isinstance(o, list) or len(o) != 3 or any((not isinstance(x, str)) or not x.strip() or len(x) > 22 for x in o):
        errs.append("need exactly 3 non-empty options of 22 characters or fewer")
    elif len({norm(x) for x in o}) != 3: errs.append("duplicate options")
    elif any(re.search(r"all of the above|none of the above|\bboth\b", x, re.I) for x in o): errs.append("banned option wording")
    if a not in (0, 1, 2): errs.append("answer must be 0, 1 or 2")
    if re.search(r"\b(current|currently|latest|newest|most recent|today|this year|right now)\b", t, re.I): errs.append("time-sensitive wording")
    return errs

GEN_SYS = "You write simple, accurate multiple-choice quiz questions for short vertical videos. Reply with JSON only."
def generate(topic, guidance, n, avoid, failures, want_caption):
    p = f"""Write {n} multiple-choice questions for the category: {topic}.
Style: {guidance}
Rules:
- Each question has exactly 3 options and exactly one clearly correct answer. "answer" is the 0-based index of the correct option.
- Question text at most 70 characters. Each option at most 18 characters.
- Easy to medium difficulty, ordered easy first. Questions must be about stable facts (no 'current', 'latest', 'today').
- No 'all of the above', no 'both'. Do not repeat or closely paraphrase any of these existing questions:
{json.dumps(avoid[-80:], ensure_ascii=False)}
"""
    if failures:
        p += "\nEarlier attempts failed checks; avoid these problems and these topics:\n" + json.dumps(failures[-10:], ensure_ascii=False) + "\n"
    shape = '{"questions":[{"text":"...","options":["...","...","..."],"answer":0}]'
    if want_caption: shape += ',"caption":"one engaging line, max 120 chars, ends with a question to comment on","hashtags":["#quiz","..."]'
    p += f"\nReturn JSON exactly shaped like: {shape}}}"
    return extract_json(text_of(call(GEN_SYS, p)[0]))

VER_SYS = ("You are a strict fact-checker. You will get a multiple-choice question WITHOUT being told the intended answer. "
           "Decide the correct option yourself. For factual questions you MUST run at least one web search and base your answer on the results, "
           "even if you already know the answer. If the question is ambiguous, two options are defensible, or the premise is disputed, say so. "
           'Reply with JSON only: {"answer":"A"|"B"|"C"|null,"confidence":"high"|"medium"|"low","source_url":"url of a page you found","notes":"short"}')
def verify_one(q, must_search):
    L = "ABC"; opts = "\n".join(f"{L[i]}) {o}" for i, o in enumerate(q["options"]))
    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
    extra = ""
    for attempt in range(2):
        r, n, urls = call(VER_SYS, f"Question: {q['text']}\n{opts}{extra}", tools=tools, max_tokens=1500)
        if n > 0 or not must_search: break
        extra = "\n\nYou did not run a web search. Run a web search now, then answer."
    j = extract_json(text_of(r)); a = j.get("answer")
    found = L.index(a) if a in ("A", "B", "C") else None
    retrieved = list(dict.fromkeys(urls_of(r) + urls))
    src = j.get("source_url") or ""
    if src not in retrieved: src = retrieved[0] if retrieved else ""   # only accept sources that were really retrieved
    return dict(found=found, confidence=j.get("confidence", "low"), source=src, searches=n, notes=j.get("notes", ""), all_sources=retrieved[:3])

def passes(q, v, needs_source):
    if v["found"] != q["answer"]: return False, f"verifier chose {v['found']} but intended {q['answer']}: {v['notes']}"
    if v["confidence"] == "low": return False, "verifier confidence low"
    if needs_source and (v.get("searches", 0) < 1 or not v["source"]): return False, "no real web search/source for factual claim"
    return True, ""

def finalize_order(questions, seed):
    rng = random.Random(seed)
    for _ in range(50):
        qs = []
        for q in questions:
            opts = list(q["options"]); correct = opts[q["answer"]]
            if all(re.fullmatch(r"[\d,.\s]+", o) for o in opts):
                order = sorted(opts, key=lambda s: float(re.sub(r"[^\d.]", "", s) or 0))
            else:
                order = opts[:]; rng.shuffle(order)
            qs.append(dict(text=q["text"], options=order, answer=order.index(correct)))
        counts = [sum(1 for q in qs if q["answer"] == i) for i in range(3)]
        if max(counts) <= 3: return qs
    return qs

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--topic")
    ap.add_argument("--history", default=os.path.join(HERE, "data", "history.json")); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    today = datetime.date.today()
    topic, guidance, needs_source = (next(t for t in TOPICS if t[0] == a.topic) if a.topic else TOPICS[today.toordinal() % len(TOPICS)])
    history = json.load(open(a.history)) if os.path.exists(a.history) else []
    seen = {norm(h) for h in history}
    good, failures, log, caption, tags = [], [], [], None, None
    for rnd in range(1, 5):
        need = 5 - len(good)
        if need == 0: break
        g = generate(topic, guidance, need + 1, history + [x["text"] for x in good], failures, caption is None)
        if caption is None: caption, tags = g.get("caption"), g.get("hashtags")
        cands = []
        for q in g.get("questions", []):
            errs = validate(q)
            if norm(q.get("text", "")) in seen or any(norm(q.get("text", "")) == norm(x["text"]) for x in good): errs.append("repeat")
            if errs: failures.append(f"{q.get('text')}: {'; '.join(errs)}"); log.append(dict(round=rnd, text=q.get("text"), ok=False, reason="; ".join(errs))); continue
            cands.append(q)
        with ThreadPoolExecutor(4) as ex: vs = list(ex.map(lambda q: verify_one(q, needs_source), cands))
        for q, v in zip(cands, vs):
            ok, why = passes(q, v, needs_source)
            log.append(dict(round=rnd, text=q["text"], intended=q["answer"], ok=ok, reason=why, **v))
            if ok and len(good) < 5: good.append(dict(q, verification=v))
            elif not ok: failures.append(f"{q['text']}: {why}")
    status = dict(topic=topic, date=str(today), verified=len(good), model=MODEL, usage=usage, log=log)
    json.dump(status, open(os.path.join(a.out, "verification.json"), "w"), indent=1, ensure_ascii=False)
    if len(good) < 5:
        print(f"FAILED: only {len(good)} verified questions", file=sys.stderr); sys.exit(1)
    final = finalize_order(good, f"{today}-{topic}")
    quiz = dict(category=topic, title="Can you score 5/5?", questions=final, caption=caption or f"{topic} quiz - how many did you get?",
                hashtags=tags or ["#quiz", "#trivia"], meta=dict(date=str(today), model=MODEL))
    json.dump(quiz, open(os.path.join(a.out, "quiz.json"), "w"), indent=1, ensure_ascii=False)
    if os.environ.get("QUIZ_RECORD") == "1":
        json.dump((history + [q["text"] for q in final])[-400:], open(a.history, "w"), indent=1, ensure_ascii=False)
    print("OK", topic, "usage", usage)

if __name__ == "__main__":
    main()
