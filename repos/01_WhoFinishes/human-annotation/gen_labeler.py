from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CODE = HERE.parent
DEFAULT_CASES = CODE / "data" / "judge" / "rq1_judge_cases_population_12k.json"
LABELS = ["Direct fix", "Related touch", "Unrelated/mislabeled", "Indeterminate"]


def anchor(url: str, filename: str) -> str:
    """GitHub Files-changed deep-link, anchored to a file (sha256 of path)."""
    h = hashlib.sha256(filename.encode()).hexdigest()
    return f"{url}/files#diff-{h}"


def _embed(obj) -> str:
    """JSON safe to inline in a <script>: escape `<`/`>`/`&` (a patch containing
    `</script>` would otherwise break out) and the JS line separators."""
    return (json.dumps(obj)
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def build_judge_records(key_path: Path, cases_path: Path) -> list[dict]:
    """One record per judged pair: the case's diffs + the judge's (hidden) call."""
    import pandas as pd

    from prfix.rq1_fix_rates.judge.attribution import pair_provenance

    key = pd.read_csv(key_path)
    judged = {int(r.pair_id): r for r in key.itertuples()}
    cases = {int(c["pair_id"]): c for c in json.loads(cases_path.read_text(encoding="utf-8"))}

    out = []
    for pid, k in judged.items():
        c = cases.get(pid)
        if c is None:
            continue
        shared = []
        for f in c.get("shared", []):
            shared.append({
                "name": f["name"],
                "boilerplate": bool(f.get("boilerplate")),
                "q_patch": (f.get("q_patch") or "")[:6000],
                "p_patch": (f.get("p_patch") or "")[:6000],
                "q_link": anchor(c["q_url"], f["name"]),
                "p_link": anchor(c["p_url"], f["name"]),
            })
        # "Substantive" = non-boilerplate AND carries a diff; an empty-patch
        # non-boilerplate file is demoted so it doesn't inflate the evidence count.
        def _has_patch(f):
            return bool((f["q_patch"] or "").strip() or (f["p_patch"] or "").strip())
        n_sub = sum(1 for f in shared if not f["boilerplate"] and _has_patch(f))
        out.append({
            "pair_id": pid, "agent": c.get("agent", ""), "day_gap": c.get("day_gap"),
            "q_type": c.get("q_type", ""), "q_references_p": bool(c.get("q_references_p")),
            "p_url": c["p_url"], "p_title": c.get("p_title", ""),
            "q_url": c["q_url"], "q_title": c.get("q_title", ""),
            "n_shared_commitunion": c.get("n_shared"),
            "fetch_ok": True,
            "shared": shared,
            "n_net_shared": len(shared), "n_net_substantive": n_sub,
            "context_complete": bool(c.get("context_complete")),
            "prov_warn": pair_provenance(c.get("shared", [])).get("warning", ""),
            "judge": {
                "label": k.label, "confidence": float(k.confidence),
                "rationale": k.rationale,
                "evidence": [e for e in str(k.evidence_files or "").split(";") if e],
            },
        })
    out.sort(key=lambda r: r["pair_id"])
    return out


# One template for all modes; __JUDGE__ (true/false) gates the judge-only parts
# (reveal panel, header agreement counter, extra export columns).
TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--bd:#d8dce2;--ink:#1c2530;--mut:#6b7480;--acc:#2f6f4f;--add:#e6ffec;--del:#ffebe9}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--bd);padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:5}
header b{font-size:15px}.grow{flex:1}
input[type=text]{padding:6px 8px;border:1px solid var(--bd);border-radius:6px;font:inherit}
button{padding:7px 12px;border:1px solid var(--acc);background:var(--acc);color:#fff;border-radius:6px;cursor:pointer;font:inherit}
button.sec{background:#fff;color:var(--acc)}
button:disabled{opacity:.45;cursor:not-allowed}
.bar{height:6px;background:#e6e9ee;border-radius:4px;width:160px;overflow:hidden}.bar>div{height:100%;background:var(--acc);width:0}
.legend{font-size:12px;color:var(--mut)}
main{max-width:920px;margin:16px auto;padding:0 16px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px 16px;margin:0 0 14px}
.card.done{border-color:var(--acc)}
.row{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.pid{font-weight:700}.tag{font-size:11px;background:#eef1f4;color:var(--mut);border-radius:4px;padding:1px 6px}
.tag.ref{background:#fff4d6;color:#7a5b00}.tag.warn{background:#ffe0e0;color:#9b1c1c}
.pr{margin:8px 0;padding:8px 10px;background:#fafbfc;border:1px solid #eef1f4;border-radius:8px}
.pr a{color:#1a5fb4;text-decoration:none;font-weight:600}.pr a:hover{text-decoration:underline}
.pr .ttl{color:var(--ink);font-weight:400}
.files{margin-top:8px}
.fhdr{font-size:12px;font-weight:700;color:var(--mut);margin:8px 0 4px;text-transform:uppercase;letter-spacing:.03em}
.fname{font-family:ui-monospace,Menlo,monospace;font-size:12px;overflow-wrap:anywhere;word-break:break-word}
.flink{font-size:11px;margin-left:6px}.flink a{color:#1a5fb4;text-decoration:none}.flink a:hover{text-decoration:underline}
details.diff{margin:4px 0 8px;border:1px solid #eef1f4;border-radius:6px;background:#fbfcfd}
details.diff>summary{cursor:pointer;padding:5px 8px;font-family:ui-monospace,Menlo,monospace;font-size:12px}
.pwrap{padding:0 8px 8px}.plab{font-size:11px;font-weight:700;color:var(--mut);margin:6px 0 2px;text-transform:uppercase}
pre.patch{margin:0;padding:8px;overflow-x:auto;font:11px/1.45 ui-monospace,Menlo,monospace;border:1px solid #eef1f4;border-radius:4px;background:#fff}
pre.patch .a{background:var(--add)}pre.patch .d{background:var(--del)}pre.patch .h{color:#8250df}
.boil{font-size:12px;color:var(--mut)}
.note{font-size:12px;padding:6px 9px;border-radius:6px;margin-top:6px}
.note.warn{background:var(--del);color:#7d1a13}.note.muted{background:#f1f3f6;color:var(--mut)}
.opts{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}
.opts label{border:1px solid var(--bd);border-radius:20px;padding:5px 12px;cursor:pointer;user-select:none}
.opts label:hover{background:#eef1f4}.opts input{display:none}
.opts label:has(input:checked){background:var(--acc);color:#fff;border-color:var(--acc)}
.reason{margin-top:8px;width:100%}
.jbtn{margin-top:10px}
.judge{margin-top:10px;padding:10px 12px;border:1px dashed #b9c2cc;border-radius:8px;background:#f7f9fb}
.judge.agree{border-color:#2f6f4f;background:#eef7f1}.judge.disagree{border-color:#c0392b;background:#fdeeec}
.jverdict{font-weight:700}.jr{font-size:13px;margin-top:4px;white-space:pre-wrap}
.badge{font-size:11px;font-weight:700;border-radius:4px;padding:1px 7px;margin-left:6px}
.badge.ok{background:#2f6f4f;color:#fff}.badge.no{background:#c0392b;color:#fff}
</style></head><body>
<header>
  <b>__HDR__</b>
  <span class="legend">__LEGEND__</span>
  <span class="grow"></span>
  Annotator: <input id="who" type="text" placeholder="your name" style="width:120px">
  <div class="bar"><div id="prog"></div></div><span id="cnt" class="legend">0/0</span>
  <span id="agr" class="legend"></span>
  <button class="sec" id="save">Save</button>
  <button id="exp">Export CSV</button>
</header>
<main id="app"></main>
<script>
const PAIRS=__DATA__, LABELS=__LABELS__, JUDGE=__JUDGE__;
const who=document.getElementById('who'), app=document.getElementById('app');
function key(){return '__LSKEY__'+(who.value.trim()||'anon')}
let state={};
function load(){try{state=JSON.parse(localStorage.getItem(key()))||{}}catch(e){state={}}}
function save(){localStorage.setItem(key(),JSON.stringify(state));prog()}
function prog(){const n=PAIRS.length,d=PAIRS.filter(p=>state[p.pair_id]&&state[p.pair_id].label).length;
  document.getElementById('cnt').textContent=d+'/'+n;document.getElementById('prog').style.width=(100*d/n)+'%';
  let rev=0,ok=0;
  PAIRS.forEach(p=>{const s=state[p.pair_id]||{};
    if(JUDGE&&s.label&&s.revealed){rev++;if(s.label===p.judge.label)ok++;}
    const c=document.getElementById('c'+p.pair_id);if(c)c.classList.toggle('done',!!s.label)});
  document.getElementById('agr').textContent=rev?('agree '+ok+'/'+rev+' ('+Math.round(100*ok/rev)+'%)'):'';}
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;')}
function patch(txt){return esc(txt).split('\n').map(l=>{
  const c=l[0]; const cls=c==='+'?'a':c==='-'?'d':(c==='@'?'h':'');
  return cls?`<span class="${cls}">${l}</span>`:l;}).join('\n');}
function patchBlock(label,txt){return txt&&txt.trim().length
  ?`<div class="plab">${label}</div><pre class="patch">${patch(txt)}</pre>`:''}
function fileBlock(f){
  const links=`<span class="flink">[<a href="${esc(f.q_link)}" target="_blank" rel="noopener">Q diff</a>] `
            +`[<a href="${esc(f.p_link)}" target="_blank" rel="noopener">P diff</a>]</span>`;
  const body=patchBlock('Q patch (candidate fix)',f.q_patch)+patchBlock('P patch (original)',f.p_patch);
  if(body){
    return `<details class="diff"><summary><span class="fname">${esc(f.name)}</span>${links}</summary>`
         +`<div class="pwrap">${body}</div></details>`;}
  return `<div style="margin:3px 0"><span class="fname">${esc(f.name)}</span>${links} `
       +`<span class="legend">(no patch — file removed/binary/too large)</span></div>`;}
function judgeHtml(p,st){
  if(!st.label) return `<div class="legend jbtn">Pick a label to unlock the judge's call.</div>`;
  if(!st.revealed) return `<button class="sec jbtn" data-reveal="${p.pair_id}">Reveal judge ▸</button>`;
  const agree=st.label===p.judge.label;
  const badge=`<span class="badge ${agree?'ok':'no'}">${agree?'AGREE':'DISAGREE'}</span>`;
  return `<div class="judge ${agree?'agree':'disagree'}">
    <div class="jverdict">Judge: ${esc(p.judge.label)} <span class="legend">(conf ${esc(p.judge.confidence)})</span>${badge}</div>
    <div class="jr">${esc(p.judge.rationale)}</div>
    ${p.judge.evidence.length?`<div class="legend" style="margin-top:4px">evidence: ${p.judge.evidence.map(esc).join(' · ')}</div>`:''}
  </div>`;}
function render(){app.innerHTML='';PAIRS.forEach(p=>{
  const st=state[p.pair_id]||{};
  const opts=LABELS.map(l=>`<label><input type="radio" name="l${p.pair_id}" value="${esc(l)}" ${st.label===l?'checked':''}><span>${esc(l)}</span></label>`).join('');
  const has=f=>!!(((f.q_patch||'').trim())||((f.p_patch||'').trim()));
  const sub=(p.shared||[]).filter(f=>!f.boilerplate&&has(f));
  const nopatch=(p.shared||[]).filter(f=>!f.boilerplate&&!has(f));
  const boil=(p.shared||[]).filter(f=>f.boilerplate);
  let filesHtml='';
  if(p.fetch_ok===false){
    filesHtml=`<div class="note muted">Diff unavailable (PR private/deleted). `
      +`Per-commit shared count was ${esc(p.n_shared_commitunion)}; open both PRs to judge manually.</div>`;
  }else if((p.n_net_shared||0)===0){
    filesHtml=`<div class="note warn">⚠ P and Q share <b>no net-diff file</b> — they were linked only via `
      +`per-commit / merge-commit history${p.n_shared_commitunion!=null?` (${esc(p.n_shared_commitunion)} per-commit overlap)`:''}. `
      +`Likely not a genuine fix; verify via the PRs above.</div>`;
  }else{
    if(sub.length){filesHtml+=`<div class="fhdr">Substantive shared files (${sub.length})</div>`+sub.map(fileBlock).join('');}
    else{filesHtml+=`<div class="note warn">No substantive shared file with a captured diff — verify via the PR links above.</div>`;}
    if(nopatch.length){filesHtml+=`<div class="fhdr">Shared files with no captured diff (${nopatch.length})</div>`
      +`<div class="note muted">Counted as shared but no patch was stored (file dropped/oversized/binary) — open the diff links to check on GitHub.</div>`
      +nopatch.map(fileBlock).join('');}
    if(boil.length){filesHtml+=`<div class="fhdr">Boilerplate shared files (${boil.length})</div>`
      +`<div class="boil">${boil.map(f=>esc(f.name)).join(' · ')}</div>`;}
  }
  const refTag=p.q_references_p?`<span class="tag ref">★ Q references P</span>`:'';
  const netTag=p.fetch_ok===false?`<span class="tag warn">diff n/a</span>`
    :`<span class="tag">${sub.length} subst${nopatch.length?` · ${nopatch.length} no-diff`:''} · ${boil.length} boiler (net)</span>`;
  const ctxTag=p.context_complete===false?`<span class="tag warn">ctx truncated</span>`:'';
  const div=document.createElement('div');div.className='card';div.id='c'+p.pair_id;
  div.innerHTML=`<div class="row"><span class="pid">#${p.pair_id}</span>
    <span class="tag">${esc(p.agent)}</span><span class="tag">gap ${esc(p.day_gap)}d</span>
    ${p.q_type?`<span class="tag">Q: ${esc(p.q_type)}</span>`:''}${netTag}${refTag}${ctxTag}</div>
    <div class="pr"><b>P (original):</b> <a href="${esc(p.p_url)}" target="_blank" rel="noopener">${esc(p.p_url)}</a>
      <span class="flink">[<a href="${esc(p.p_url)}/files" target="_blank" rel="noopener">Files ▸</a>]</span>
      <br><span class="ttl">${esc(p.p_title)}</span></div>
    <div class="pr"><b>Q (candidate fix):</b> <a href="${esc(p.q_url)}" target="_blank" rel="noopener">${esc(p.q_url)}</a>
      <span class="flink">[<a href="${esc(p.q_url)}/files" target="_blank" rel="noopener">Files ▸</a>]</span>
      <br><span class="ttl">${esc(p.q_title)}</span>
      <div class="files">${filesHtml}</div></div>
    ${p.prov_warn?`<div class="note warn">⚠ ${esc(p.prov_warn)}</div>`:''}
    <div class="opts">${opts}</div>
    <input class="reason" type="text" placeholder="reason (≤15 words)" value="${esc(st.reason||'')}" data-r="${p.pair_id}">
    ${JUDGE?`<div class="jpanel" id="j${p.pair_id}">${judgeHtml(p,st)}</div>`:''}`;
  app.appendChild(div);
});
const byId={};PAIRS.forEach(p=>byId[p.pair_id]=p);
function refreshJudge(id){const p=byId[id];document.getElementById('j'+id).innerHTML=judgeHtml(p,state[id]||{})}
app.querySelectorAll('input[type=radio]').forEach(r=>r.addEventListener('change',e=>{
  const id=e.target.name.slice(1);state[id]=state[id]||{};state[id].label=e.target.value;save();if(JUDGE)refreshJudge(id)}));
app.querySelectorAll('input[data-r]').forEach(t=>t.addEventListener('input',e=>{
  const id=e.target.dataset.r;state[id]=state[id]||{};state[id].reason=e.target.value;save()}));
app.addEventListener('click',e=>{const b=e.target.closest('[data-reveal]');if(!b)return;
  const id=b.dataset.reveal;state[id]=state[id]||{};state[id].revealed=true;save();refreshJudge(id)});
prog()}
who.addEventListener('change',()=>{load();render()});
document.getElementById('save').onclick=save;
document.getElementById('exp').onclick=()=>{
  const head=['pair_id','agent','p_url','q_url','label','reason','annotator'];
  if(JUDGE)head.push('judge_label','agree');
  const rows=[head];
  PAIRS.forEach(p=>{const s=state[p.pair_id]||{};
    const r=[p.pair_id,p.agent,p.p_url,p.q_url,s.label||'',s.reason||'',who.value.trim()];
    if(JUDGE){const agree=s.label?(s.label===p.judge.label?1:0):'';r.push(p.judge.label,agree);}
    rows.push(r)});
  const csv=rows.map(r=>r.map(x=>'"'+String(x==null?'':x).replace(/"/g,'""')+'"').join(',')).join('\n');
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  a.download='__EXPORT_PREFIX__'+(who.value.trim()||'anon')+'.csv';a.click()};
load();render();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the offline HTML labeler for a validation set")
    ap.add_argument("--mode", choices=["net-diff", "judge"], default="net-diff",
                    help="which validation set to render (default: net-diff, the paper's)")
    ap.add_argument("--n", type=int, default=50,
                    help="[judge] sample size (matches sample_pairs.py judge)")
    ap.add_argument("--tag", default="",
                    help="[judge] stem suffix used at sampling time (e.g. humanP)")
    ap.add_argument("--sample-mode", default="random",
                    help="[judge] sampling mode used: random | stratified | enriched")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES,
                    help="[judge] population cases JSON with the net-diff patches the judge saw")
    args = ap.parse_args()

    if args.mode == "judge":
        tag = f"_{args.tag}" if getattr(args, "tag", None) else ""
        stem = f"rq1_judge_sample{args.n}_{args.sample_mode}{tag}"
        key_path = HERE / "labels" / "judge" / f"{stem}_model_key.csv"
        if not key_path.exists():
            raise SystemExit(f"model key not found: {key_path}\n"
                             f"  run: uv run python human-annotation/sample_pairs.py judge "
                             f"--mode {args.sample_mode} --n {args.n}")
        pairs = build_judge_records(key_path, args.cases)
        out_path = DATA / "judge" / f"{stem}_labeler.html"
        fills = {
            "__JUDGE__": "true",
            "__TITLE__": (f"RQ1 judge sample ({len(pairs)}, {args.sample_mode}) "
                          "— human-vs-judge labeler"),
            "__HDR__": f"RQ1 judge sample — {len(pairs)} pairs ({args.sample_mode})",
            "__LEGEND__": ('Label BLIND first; then "Reveal judge" to compare '
                           "with the LLM&#39;s call."),
            "__LSKEY__": f"{stem}_",
            "__EXPORT_PREFIX__": f"{stem}_annotator_",
        }
    else:
        mode_dir = DATA / args.mode.replace("-", "_")
        enriched = mode_dir / "pairs_enriched.json"
        if not enriched.exists():
            raise SystemExit(f"enriched pairs not found: {enriched}\n"
                             f"  run: uv run python human-annotation/sample_pairs.py "
                             f"gold --linker {args.mode}")
        pairs = json.loads(enriched.read_text(encoding="utf-8"))
        out_path = mode_dir / "labeler.html"
        fills = {
            "__JUDGE__": "false",
            "__TITLE__": f"RQ1 follow-up validation labeler ({args.mode})",
            "__HDR__": "RQ1 follow-up validation",
            "__LEGEND__": ("Does Q genuinely fix the originating PR P? "
                           "Read Q&#39;s diff of the <em>shared files</em> below."),
            "__LSKEY__": f"rq1_followup_validation_{args.mode}_",
            "__EXPORT_PREFIX__": "rq1_followup_validation_annotator_",
        }

    html = TEMPLATE.replace("__DATA__", _embed(pairs)).replace("__LABELS__", _embed(LABELS))
    for k, v in fills.items():
        html = html.replace(k, v)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    n_sub = sum(1 for p in pairs if p.get("n_net_substantive"))
    print(f"wrote {out_path.relative_to(HERE)}  "
          f"({len(pairs)} pairs, {n_sub} with a substantive shared file)")


if __name__ == "__main__":
    main()
