#!/usr/bin/env python3
"""
cc_history.py — Export Claude Code prompts + final responses to a polished Markdown doc.

Modes:
  Regenerate (default): scan ALL of a project's session transcripts, merge in
    chronological order, rewrite the whole doc.
        python3 cc_history.py --project "/Users/you/code/myapp"
  Append (--append): read ONLY the current session's transcript, append new
    exchanges, dedupe via a state file. Cheap; run it from a Stop hook.
        python3 cc_history.py --append --out "$CLAUDE_PROJECT_DIR/claude-history.md"

Stdlib only. Hooks run this as a shell command (zero model tokens).
"""

import argparse
import contextlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECTS_DIR = Path.home() / ".claude" / "projects"
TITLE_MAX = 72

NOISE_PREFIXES = (
    "<command-name>", "<command-message>", "<command-args>",
    "<local-command-stdout>", "<local-command-stderr>",
    "<bash-input>", "<bash-stdout>", "<bash-stderr>",
    "[Request interrupted by user", "Caveat:",
    "This session is being continued from a previous conversation",  # auto-compaction preamble
    "<system-reminder>",
)


def die(m):
    print(m, file=sys.stderr)
    sys.exit(1)


# ---- locating transcripts ---------------------------------------------------

def encode_project_path(p: str) -> str:
    s = str(Path(p).expanduser().resolve())
    for ch in ("/", "\\", ":"):
        s = s.replace(ch, "-")
    return s


def friendly(name: str) -> str:
    name = name.rstrip("-")
    seg = name.split("-")[-1] if "-" in name else name
    return seg or name


def projects_available():
    if not PROJECTS_DIR.is_dir():
        return []
    return [d for d in PROJECTS_DIR.iterdir() if d.is_dir()]


def find_project_dir(args) -> Path:
    if args.project_dir:
        cand = Path(args.project_dir)
        if cand.is_dir():
            return cand
        cand = PROJECTS_DIR / args.project_dir
        if cand.is_dir():
            return cand
        die(f"--project-dir not found: {args.project_dir}")
    raw = args.project or os.getcwd()
    cand = PROJECTS_DIR / encode_project_path(raw)
    if cand.is_dir():
        return cand
    msg = [f"No transcript folder for: {raw}", f"(looked in {cand})", "",
           "Available projects:"]
    msg += ["  " + d.name for d in sorted(projects_available())] or ["  (none)"]
    msg += ["", "Re-run with:  --project-dir=<name-from-above>"]
    die("\n".join(msg))


def session_files(proj_dir: Path):
    return sorted(p for p in proj_dir.glob("*.jsonl") if not p.name.startswith("agent-"))


# ---- content helpers --------------------------------------------------------

def as_blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return []


def text_of(content) -> str:
    parts = [b.get("text", "") for b in as_blocks(content)
             if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(p for p in parts if p)


def has_tool_result(content) -> bool:
    return any(isinstance(b, dict) and b.get("type") == "tool_result"
              for b in as_blocks(content))


def is_real_prompt(e) -> bool:
    if e.get("type") != "user" or e.get("isMeta") or e.get("isSidechain"):
        return False
    content = (e.get("message") or {}).get("content")
    if has_tool_result(content):
        return False
    txt = text_of(content).strip()
    return bool(txt) and not txt.startswith(NOISE_PREFIXES)


def is_assistant(e) -> bool:
    return e.get("type") == "assistant" and not e.get("isSidechain")


def parse_file(path: Path):
    out = []
    try:
        fh = path.open(encoding="utf-8")
    except OSError:
        return out
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---- core extraction --------------------------------------------------------

def _clean_title(s, limit=60):
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit].rstrip() + "…"


def _bad_summary(s):
    return s.startswith("API Error") or '"type":"error"' in s or "invalid_request_error" in s


def session_title(entries, fallback=""):
    """Mirror Claude Code's resume-list label: customTitle > generated summary
    (the one whose leafUuid is this conversation's leaf) > first prompt > id."""
    uuid_pos, custom, summaries, first_prompt = {}, None, [], None
    for idx, e in enumerate(entries):
        u = e.get("uuid")
        if u and u not in uuid_pos:
            uuid_pos[u] = idx
        if e.get("customTitle"):
            custom = e.get("customTitle")
        if e.get("type") == "summary" and e.get("summary"):
            summaries.append((e.get("leafUuid"), e["summary"]))
        if first_prompt is None and is_real_prompt(e):
            first_prompt = text_of((e.get("message") or {}).get("content")).strip()
    if custom:
        return _clean_title(custom)
    best, best_pos = None, -1
    for leaf, s in summaries:                 # pick the summary tied to the latest in-file leaf
        if _bad_summary(s):
            continue
        pos = uuid_pos.get(leaf, -1)
        if pos > best_pos:
            best, best_pos = s, pos
    if best is not None and best_pos >= 0:
        return _clean_title(best)
    if first_prompt:
        return _clean_title(first_prompt, limit=40)
    return (fallback or "session")[:8]


def extract_exchanges(files, full, seen_prompts):
    exchanges = []
    for f in files:
        entries = parse_file(f)
        title = session_title(entries, fallback=f.stem)
        cur = None
        for e in entries:
            if is_real_prompt(e):
                if cur:
                    exchanges.append(cur)
                    cur = None
                uid = e.get("uuid")
                if uid in seen_prompts:
                    continue
                if uid:
                    seen_prompts.add(uid)
                msg = e.get("message") or {}
                cur = {"prompt": text_of(msg.get("content")).strip(),
                       "ts": e.get("timestamp", ""),
                       "session": title,
                       "steps": [], "final": ""}
            elif is_assistant(e) and cur is not None:
                t = text_of((e.get("message") or {}).get("content")).strip()
                if t:
                    cur["steps"].append(t)
                    cur["final"] = t
        if cur:
            exchanges.append(cur)
    exchanges.sort(key=lambda x: x["ts"])
    for x in exchanges:
        x["response"] = "\n\n".join(x["steps"]) if full else x["final"]
    return exchanges


# ---- rendering --------------------------------------------------------------

def fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts or "?"


def make_title(prompt: str):
    s = prompt.strip()
    lines = s.splitlines()
    first = " ".join(lines[0].split()) if lines else "(empty prompt)"
    truncated = len(first) > TITLE_MAX
    title = (first[:TITLE_MAX].rstrip() + "…") if truncated else (first or "(empty prompt)")
    show_full = truncated or len([ln for ln in lines if ln.strip()]) > 1
    return title, show_full


def block_body(i, x) -> str:
    title, show_full = make_title(x["prompt"])
    out = [f"## {i}. [{x['session']}] {title}\n\n`{fmt_ts(x['ts'])}`\n"]
    if show_full:
        quoted = "\n".join(("> " + ln) if ln.strip() else ">"
                           for ln in x["prompt"].strip().splitlines())
        out.append(f"\n> **Prompt**\n>\n{quoted}\n")
    out.append("\n" + (x["response"] or "_(no text response captured)_") + "\n")
    return "".join(out)


SEP = "\n---\n\n"


def render_doc(exchanges, title, full) -> str:
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    top = (f"# Claude Code Log · {title}\n\n"
           f"`{len(exchanges)} exchanges`  ·  `updated {now}`\n\n"
           f"> Prompts and final responses across Claude Code sessions, in chronological order."
           f"{' Full responses.' if full else ''}\n")
    return top + "".join(SEP + block_body(i, x) for i, x in enumerate(exchanges, 1)) + "\n"


def doc_header(title, full) -> str:
    return (f"# Claude Code Log · {title}\n\n"
            f"> Prompts and final responses, appended live across Claude Code sessions."
            f"{' Full responses.' if full else ''}\n")


# ---- state (append) ---------------------------------------------------------

def load_state(p: Path):
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return set(d.get("seen", [])), int(d.get("count", 0))
    except Exception:
        return set(), 0


def save_state(p: Path, seen, count):
    p.write_text(json.dumps({"seen": sorted(seen), "count": count}), encoding="utf-8")


# ---- cross-platform advisory lock (serializes concurrent writers) -----------

@contextlib.contextmanager
def file_lock(target: Path, timeout=10.0):
    """Serialize processes writing to the same doc (e.g. parallel Stop hooks).
    Keyed on a sibling .lock file. Best-effort on exotic filesystems."""
    lock_path = Path(str(target) + ".lock")
    f = open(lock_path, "a+")
    deadline = time.time() + timeout
    try:
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    if time.time() > deadline:
                        break  # give up rather than hang the turn
                    time.sleep(0.05)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        f.close()


# ---- modes ------------------------------------------------------------------

def run_regenerate(args):
    proj = find_project_dir(args)
    title = args.title or friendly(proj.name)
    seen = set()
    ex = extract_exchanges(session_files(proj), args.full_response, seen)
    out = Path(args.out) if args.out else Path.cwd() / "claude-history.md"
    state_path = out.with_suffix(out.suffix + ".state")
    with file_lock(out):
        out.write_text(render_doc(ex, title, args.full_response), encoding="utf-8")
        save_state(state_path, seen, len(ex))   # so the append hook continues from here
    print(f"Wrote {len(ex)} exchanges -> {out}")


def run_all(args):
    dirs = sorted(projects_available())
    if not dirs:
        die("No projects found under ~/.claude/projects")
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    # pretty names, but fall back to the unique encoded name on collisions
    counts = {}
    for d in dirs:
        nm = friendly(d.name)
        counts[nm] = counts.get(nm, 0) + 1

    total = 0
    for proj in dirs:
        ex = extract_exchanges(session_files(proj), args.full_response, set())
        if not ex:
            continue  # skip projects with no captured prompts
        nm = friendly(proj.name)
        fname = nm if counts[nm] == 1 else proj.name
        out = out_dir / f"{fname}.md"
        with file_lock(out):
            out.write_text(render_doc(ex, nm, args.full_response), encoding="utf-8")
        total += 1
        print(f"{len(ex):5d} exchanges -> {out}")
    print(f"Done. Wrote {total} project log(s) to {out_dir}")


def resolve_transcript(args) -> Path:
    if args.transcript:
        return Path(args.transcript)
    if not sys.stdin.isatty():
        try:
            data = json.load(sys.stdin)
            tp = data.get("transcript_path")
            if tp:
                return Path(tp).expanduser()
        except Exception:
            pass
    die("append mode needs a transcript: pass --transcript PATH, or run from a Stop "
        "hook (which provides transcript_path on stdin).")


def rotate_if_needed(out: Path, max_mb):
    """If the live doc has reached max_mb, archive it with a timestamp so a fresh
    file starts. Called under the lock. State is untouched, so numbering/dedupe
    carry across the rotation."""
    if not max_mb or max_mb <= 0:
        return
    try:
        size = out.stat().st_size
    except OSError:
        return  # nothing to rotate yet
    if size < max_mb * 1024 * 1024:
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = out.with_name(f"{out.stem}-{stamp}{out.suffix}")
    i = 1
    while archive.exists():
        archive = out.with_name(f"{out.stem}-{stamp}-{i}{out.suffix}")
        i += 1
    out.rename(archive)
    print(f"Rotated {out.name} -> {archive.name} ({size/1048576:.1f} MB)")


def run_append(args):
    transcript = resolve_transcript(args)
    out = Path(args.out) if args.out else Path.cwd() / "claude-history.md"
    state_path = Path(args.state) if args.state else out.with_suffix(out.suffix + ".state")
    title = args.title or friendly(transcript.parent.name)

    with file_lock(out):
        rotate_if_needed(out, args.max_mb)
        seen, count = load_state(state_path)
        new = extract_exchanges([transcript], args.full_response, seen)
        if not new:
            return
        if not out.exists():
            out.write_text(doc_header(title, args.full_response), encoding="utf-8")
        with out.open("a", encoding="utf-8") as fh:
            for x in new:
                count += 1
                fh.write(SEP + block_body(count, x))
        save_state(state_path, seen, count)
    print(f"Appended {len(new)} exchange(s) -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Export Claude Code prompts + final responses to Markdown.")
    ap.add_argument("--append", action="store_true", help="append new exchanges from the current session (Stop hook)")
    ap.add_argument("--transcript", help="[append] path to the session .jsonl (else read from hook stdin)")
    ap.add_argument("--state", help="[append] dedupe state file (default: <out>.state)")
    ap.add_argument("--max-mb", type=float, default=0,
                    help="[append] rotate the doc to a timestamped archive once it reaches this many MB (0 = never)")
    ap.add_argument("--all", action="store_true", help="generate one doc per project for every project in ~/.claude/projects")
    ap.add_argument("--out-dir", help="[--all] directory to write the per-project docs into (default: current dir)")
    ap.add_argument("--project", help="[regenerate] absolute path of the project (default: current dir)")
    ap.add_argument("--project-dir", help="[regenerate] folder under ~/.claude/projects, or a path to it")
    ap.add_argument("--out", help="output Markdown file (default: ./claude-history.md)")
    ap.add_argument("--title", help="display name in the doc header (default: derived from project)")
    ap.add_argument("--full-response", action="store_true", help="include all assistant text, not just the final answer")
    ap.add_argument("--list", action="store_true", help="list available project folders and exit")
    args = ap.parse_args()

    if args.list:
        for d in sorted(projects_available()):
            print(d.name)
        return
    if args.all:
        run_all(args)
    elif args.append:
        run_append(args)
    else:
        run_regenerate(args)


if __name__ == "__main__":
    main()
