# Fletch
Clipforge app
git remote add origin <url> && git push -u origin main.
"""ClipForge: long video in, captioned vertical short clips out.
Run: ANTHROPIC_API_KEY=... uvicorn app:app --port 8000
Optional env: SECRET_KEY, SIGNUP_CODE (invite code), MAX_UPLOAD_MB, DAILY_JOBS,
JOB_TTL_HOURS, COOKIE_SECURE=1 (set when serving over https)
Needs ffmpeg installed on the machine.
"""
import hashlib, hmac, json, os, re, secrets, shutil, sqlite3, subprocess, threading, time, uuid
from pathlib import Path

import anthropic
import cv2
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel

DATA = Path("jobs")
DATA.mkdir(exist_ok=True)
LLM = os.getenv("CLIP_MODEL", "claude-sonnet-5-5")
MAX_CLIPS = 5
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
TTL_HOURS = int(os.getenv("JOB_TTL_HOURS", "24"))
DAILY_LIMIT = int(os.getenv("DAILY_JOBS", "5"))
SIGNUP_CODE = os.getenv("SIGNUP_CODE", "")
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "0") == "1"
SESSION_DAYS = 7
SECRET = os.getenv("SECRET_KEY") or secrets.token_hex(32)
if not os.getenv("SECRET_KEY"):
    print("Warning: SECRET_KEY is not set, so logins reset whenever the server restarts.")

app = FastAPI()
jobs: dict = {}
_whisper = None


def whisper():
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel("small", compute_type="int8")
    return _whisper


def transcribe(video: Path):
    segments, _ = whisper().transcribe(str(video), word_timestamps=True)
    return [
        {"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)}
        for seg in segments
        for w in seg.words
    ]


def pick_clips(words):
    # Group words into timestamped lines so the model can cite exact times.
    lines = []
    for i in range(0, len(words), 14):
        chunk = words[i : i + 14]
        lines.append(f"[{chunk[0]['s']}] " + " ".join(w["w"] for w in chunk))
    prompt = (
        f"Transcript with start times in seconds:\n" + "\n".join(lines) +
        f"\n\nPick up to {MAX_CLIPS} self-contained moments of 20-60 seconds that "
        "would work as short-form videos (strong hook in the first sentence, a "
        "complete thought, no need for outside context). Return ONLY a JSON list "
        'of objects: {"start": number, "end": number, "title": string}.'
    )
    msg = anthropic.Anthropic().messages.create(
        model=LLM, max_tokens=1500, messages=[{"role": "user", "content": prompt}]
    )
    text = msg.content[0].text
    picks = json.loads(text[text.find("[") : text.rfind("]") + 1])
    out = []
    for p in picks[:MAX_CLIPS]:
        s, e = float(p["start"]), float(p["end"])
        if 10 <= e - s <= 90:
            out.append({"start": s, "end": e, "title": str(p.get("title", "Clip"))})
    return out


def ass_time(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def make_ass(words, start, end, path: Path):
    head = (
        "[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,OutlineColour,Bold,BorderStyle,"
        "Outline,Shadow,Alignment,MarginV\n"
        "Style: Cap,Arial,84,&H00FFFFFF,&H00000000,1,1,6,0,2,520\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Text\n"
    )
    inside = [w for w in words if w["s"] >= start and w["e"] <= end]
    rows = []
    for i in range(0, len(inside), 3):
        g = inside[i : i + 3]
        rows.append(
            f"Dialogue: 0,{ass_time(g[0]['s'] - start)},{ass_time(g[-1]['e'] - start)},"
            f"Cap,{' '.join(w['w'] for w in g).upper()}"
        )
    path.write_text(head + "\n".join(rows), encoding="utf-8")


def face_path(video: Path, start, end):
    """Sample the clip once a second, follow the largest face, smooth the motion.
    Returns (crop_width, [(seconds_into_clip, crop_x), ...]) or None if no tracking is needed."""
    cap = cv2.VideoCapture(str(video))
    iw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ih = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cw = int(ih * 9 / 16) // 2 * 2
    if iw <= cw + 2:  # already vertical
        cap.release()
        return None
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    scale, last, ema, pts, t = 480 / iw, iw / 2, None, [], start
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok:
            small = cv2.cvtColor(cv2.resize(frame, (480, int(ih * scale))), cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(small, 1.1, 5, minSize=(30, 30))
            if len(faces):
                x, _, w, _ = max(faces, key=lambda f: f[2] * f[3])
                last = (x + w / 2) / scale
        ema = last if ema is None else 0.5 * ema + 0.5 * last
        pts.append((t - start, min(max(ema - cw / 2, 0), iw - cw)))
        t += 1.0
    cap.release()
    if len(pts) < 2:
        pts.append((pts[0][0] + 1, pts[0][1]))
    return cw, pts


def x_expr(pts):
    """Piecewise-linear crop position over time, as an ffmpeg expression."""
    expr = f"{pts[-1][1]:.1f}"
    for (t0, x0), (t1, x1) in reversed(list(zip(pts, pts[1:]))):
        expr = f"if(lt(t,{t1:.2f}),{x0:.1f}+({x1 - x0:.1f})*(t-{t0:.2f})/{t1 - t0:.2f},{expr})"
    return expr


def cut(video: Path, words, clip, out: Path):
    ass = out.with_suffix(".ass")
    make_ass(words, clip["start"], clip["end"], ass)
    track = face_path(video, clip["start"], clip["end"])
    if track:
        cw, pts = track
        crop = f"crop=w={cw}:h=ih:x='{x_expr(pts)}':y=0"
    else:
        crop = "crop=w=min(iw\\,ih*9/16):h=min(ih\\,iw*16/9)"
    script = out.with_suffix(".vf")  # filter script avoids shell and escaping trouble
    script.write_text(f"{crop},scale=1080:1920,ass={ass}", encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(clip["start"]), "-t", str(clip["end"] - clip["start"]),
         "-i", str(video), "-filter_script:v", str(script), "-c:v", "libx264",
         "-preset", "veryfast", "-c:a", "aac", str(out)],
        check=True, capture_output=True,
    )


def process(job_id: str, video: Path):
    job = jobs[job_id]
    try:
        job["status"] = "Transcribing"
        words = transcribe(video)
        job["status"] = "Finding the best moments"
        picks = pick_clips(words)
        for i, clip in enumerate(picks):
            job["status"] = f"Rendering clip {i + 1} of {len(picks)}"
            cut(video, words, clip, video.parent / f"clip{i}.mp4")
            job["clips"].append(clip["title"])
        job["status"] = "done"
    except Exception as exc:  # surface the reason in the UI
        job["status"] = "error"
        job["error"] = str(exc)[:300]


# ---------- accounts ----------
db = sqlite3.connect("users.db", check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS users (name TEXT PRIMARY KEY, pw TEXT)")
db_lock = threading.Lock()
fails: dict = {}


def hash_pw(pw: str, salt: str = ""):
    salt = salt or secrets.token_hex(16)
    h = hashlib.scrypt(pw.encode(), salt=salt.encode(), n=2**14, r=8, p=1).hex()
    return f"{salt}${h}"


def check_pw(pw: str, stored: str):
    return hmac.compare_digest(hash_pw(pw, stored.split("$")[0]), stored)


def sign(msg: str):
    return hmac.new(SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()


def current_user(request: Request) -> str:
    try:
        user, exp, sig = request.cookies.get("session", "").rsplit(".", 2)
        if hmac.compare_digest(sig, sign(f"{user}.{exp}")) and int(exp) > time.time():
            return user
    except ValueError:
        pass
    raise HTTPException(401, "Please log in")


class Creds(BaseModel):
    username: str
    password: str
    code: str = ""


def session_response(user: str):
    exp = int(time.time()) + SESSION_DAYS * 86400
    r = JSONResponse({"user": user})
    r.set_cookie("session", f"{user}.{exp}.{sign(f'{user}.{exp}')}", httponly=True,
                 samesite="lax", secure=COOKIE_SECURE, max_age=SESSION_DAYS * 86400)
    return r


@app.post("/api/signup")
def signup(c: Creds):
    name = c.username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{3,30}", name):
        raise HTTPException(400, "Username must be 3 to 30 letters, numbers or underscores")
    if len(c.password) < 10:
        raise HTTPException(400, "Password must be at least 10 characters")
    if SIGNUP_CODE and not hmac.compare_digest(c.code, SIGNUP_CODE):
        raise HTTPException(403, "That invite code is not valid")
    with db_lock:
        try:
            db.execute("INSERT INTO users VALUES (?, ?)", (name, hash_pw(c.password)))
            db.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(409, "That username is taken")
    return session_response(name)


@app.post("/api/login")
def login(c: Creds):
    name, now = c.username.strip().lower(), time.time()
    recent = [t for t in fails.get(name, []) if now - t < 900]
    if len(recent) >= 8:
        raise HTTPException(429, "Too many attempts. Try again in 15 minutes.")
    with db_lock:
        row = db.execute("SELECT pw FROM users WHERE name = ?", (name,)).fetchone()
    if not row or not check_pw(c.password, row[0]):
        fails[name] = recent + [now]
        raise HTTPException(401, "Wrong username or password")
    fails.pop(name, None)
    return session_response(name)


@app.post("/api/logout")
def logout():
    r = JSONResponse({"ok": True})
    r.delete_cookie("session")
    return r


def used_today(user: str):
    return sum(1 for j in jobs.values() if j["owner"] == user and time.time() - j["created"] < 86400)


@app.get("/api/me")
def me(user: str = Depends(current_user)):
    return {"user": user, "used": used_today(user), "limit": DAILY_LIMIT}


# ---------- upload cap ----------
@app.middleware("http")
async def guard_uploads(request: Request, call_next):
    """Reject oversized or anonymous uploads before the body is read."""
    if request.method == "POST" and request.url.path == "/api/jobs":
        try:
            current_user(request)
        except HTTPException:
            return JSONResponse({"detail": "Please log in"}, status_code=401)
        length = request.headers.get("content-length", "")
        if not length.isdigit() or int(length) > MAX_MB * 1024 * 1024 + 65536:
            return JSONResponse({"detail": f"File too large. The limit is {MAX_MB} MB."}, status_code=413)
    return await call_next(request)


# ---------- jobs ----------
def owned(job_id: str, user: str):
    job = jobs.get(job_id)
    if not job or job["owner"] != user:
        raise HTTPException(404)
    return job


@app.post("/api/jobs")
async def create(file: UploadFile, tasks: BackgroundTasks, user: str = Depends(current_user)):
    if used_today(user) >= DAILY_LIMIT:
        raise HTTPException(429, f"You have used your {DAILY_LIMIT} videos for today. Try again tomorrow.")
    if not (file.filename or "").lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".m4v")):
        raise HTTPException(400, "Upload an mp4, mov, mkv, m4v or webm video")
    job_id = uuid.uuid4().hex[:12]
    folder = DATA / job_id
    folder.mkdir()
    video, size = folder / "source.mp4", 0
    try:
        with video.open("wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_MB * 1024 * 1024:
                    raise HTTPException(413, f"File too large. The limit is {MAX_MB} MB.")
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    jobs[job_id] = {"owner": user, "created": time.time(), "status": "Queued", "clips": []}
    tasks.add_task(process, job_id, video)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def status(job_id: str, user: str = Depends(current_user)):
    j = owned(job_id, user)
    return {k: j.get(k) for k in ("status", "clips", "error") if k in j}


@app.get("/api/jobs/{job_id}/clips/{n}")
def clip(job_id: str, n: int, user: str = Depends(current_user)):
    owned(job_id, user)
    path = DATA / job_id / f"clip{n}.mp4"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4", filename=f"clip{n + 1}.mp4")


# ---------- cleanup ----------
def sweep():
    """Delete job folders older than the retention window, every 10 minutes."""
    while True:
        cutoff = time.time() - TTL_HOURS * 3600
        for folder in DATA.iterdir():
            try:
                if folder.stat().st_mtime < cutoff:
                    shutil.rmtree(folder, ignore_errors=True)
                    if folder.name in jobs:
                        jobs[folder.name].update(status="expired", clips=[])
            except OSError:
                pass
        time.sleep(600)


@app.on_event("startup")
def start_sweeper():
    threading.Thread(target=sweep, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index():
    return (PAGE.replace("__MB__", str(MAX_MB)).replace("__H__", str(TTL_HOURS))
            .replace("__CODE__", "" if SIGNUP_CODE else "hidden"))


PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ClipForge</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;max-width:720px;margin:0 auto;padding:24px;background:#111418;color:#eef2f4}
h1{margin:0 0 4px}p.s{color:#9aa7ad;margin:0 0 20px}
[hidden]{display:none!important}
input{display:block;width:100%;max-width:340px;margin:0 0 10px;padding:10px 12px;border-radius:10px;border:1px solid #3a4650;background:#1a2026;color:inherit;font:inherit}
button{padding:10px 16px;border-radius:10px;border:0;background:#5ad1b5;color:#08201a;font:inherit;font-weight:600;cursor:pointer;margin-right:8px}
button.alt{background:transparent;color:#5ad1b5;border:1px solid #5ad1b5}
.drop{display:block;border:2px dashed #3a4650;border-radius:16px;padding:36px;text-align:center;cursor:pointer}
.drop:hover,.drop:focus-visible{border-color:#5ad1b5;outline:none}
#st{margin:18px 0;color:#5ad1b5}.err{color:#ff8a7a}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px}
video{width:100%;border-radius:12px;background:#000;aspect-ratio:9/16}
a{color:#5ad1b5}
</style></head><body>
<h1>ClipForge</h1>
<section id="auth" hidden>
  <p class="s">Log in or create an account to make clips.</p>
  <form id="af">
    <input id="u" placeholder="Username" autocomplete="username" required>
    <input id="p" type="password" placeholder="Password (10+ characters)" autocomplete="current-password" required>
    <input id="c" placeholder="Invite code" __CODE__>
    <button type="submit">Log in</button><button type="button" class="alt" id="su">Create account</button>
  </form>
  <p class="err" id="ae"></p>
</section>
<section id="app" hidden>
  <p class="s"><span id="who"></span>: <span id="q"></span>. <a href="#" id="lo">Log out</a></p>
  <label class="drop" tabindex="0"><input id="f" type="file" accept="video/*" hidden>Choose a video (up to __MB__ MB)</label>
  <p class="s" style="margin-top:8px">Clips are deleted after __H__ hours, so download the ones you want.</p>
  <div id="st"></div><div class="grid" id="g"></div>
</section>
<script>
const $=id=>document.getElementById(id),st=$('st'),g=$('g');
async function api(p,o){const r=await fetch(p,o);let d={};try{d=await r.json()}catch{}
  if(!r.ok)throw Object.assign(new Error(typeof d.detail==='string'?d.detail:'Check the details you entered'),{status:r.status});return d}
function show(u){$('auth').hidden=!!u;$('app').hidden=!u;if(u){$('who').textContent=u.user;$('q').textContent=u.used+' of '+u.limit+' videos used today'}}
async function boot(){try{show(await api('/api/me'))}catch{show(null)}}
async function sign(kind){$('ae').textContent='';
  try{await api('/api/'+kind,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:$('u').value,password:$('p').value,code:$('c').value})});$('p').value='';boot()}
  catch(e){$('ae').textContent=e.message}}
$('af').onsubmit=e=>{e.preventDefault();sign('login')};
$('su').onclick=()=>sign('signup');
$('lo').onclick=async e=>{e.preventDefault();await api('/api/logout',{method:'POST'});g.innerHTML='';st.textContent='';show(null)};
function fail(e){st.className='err';st.textContent=e.message;if(e.status===401)show(null)}
function poll(id){const t=setInterval(async()=>{
  try{const j=await api('/api/jobs/'+id);
    if(j.status==='error'){clearInterval(t);return fail(new Error('Failed: '+j.error))}
    if(j.status==='expired'){clearInterval(t);return fail(new Error('These clips have expired.'))}
    st.textContent=j.status==='done'?'Done':j.status+'...';
    g.innerHTML=j.clips.map((c,i)=>`<div><video controls preload="metadata" src="/api/jobs/${id}/clips/${i}"></video><div>${c.replace(/</g,'&lt;')}</div><a href="/api/jobs/${id}/clips/${i}" download>Download</a></div>`).join('');
    if(j.status==='done')clearInterval(t)}
  catch(e){clearInterval(t);fail(e)}},2500)}
$('f').onchange=async e=>{const file=e.target.files[0];e.target.value='';if(!file)return;
  g.innerHTML='';st.className='';st.textContent='Uploading...';
  try{const fd=new FormData();fd.append('file',file);const {id}=await api('/api/jobs',{method:'POST',body:fd});boot();poll(id)}catch(err){fail(err)}};
boot();
</script></body></html>"""
