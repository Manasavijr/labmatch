'use client';
import { useEffect, useRef, useState } from 'react';
import Chat from './Chat';

const EXAMPLES = [
  'Graph neural networks for protein structure and drug discovery',
  'Reinforcement learning for robot manipulation',
  'LLMs for clinical notes and medical question answering',
  'Fairness and privacy in machine learning',
];
const PLACEHOLDERS = [
  'Paste your research interests…',
  'or the abstract of a paper you loved…',
  'or a sentence about what you want to build.',
];

const fmtMoney = (n) => (n >= 1e6 ? `$${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M` : n >= 1e3 ? `$${Math.round(n / 1e3)}K` : `$${Math.round(n)}`);
const fmtDate = (s) => (s ? new Date(s + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', year: 'numeric' }) : '');
const monthsLeft = (s) => (s ? (new Date(s) - new Date()) / (1000 * 60 * 60 * 24 * 30.4) : 0);

function Topo() {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current, ctx = c.getContext('2d');
    let raf, t = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const size = () => { c.width = innerWidth * dpr; c.height = innerHeight * dpr; };
    size(); addEventListener('resize', size);
    const col = () => getComputedStyle(document.documentElement).getPropertyValue('--topo').trim();
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const draw = () => {
      const w = c.width, h = c.height;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = col(); ctx.lineWidth = 1.1 * dpr;
      for (let k = 0; k < 15; k++) {
        ctx.beginPath();
        for (let x = 0; x <= w; x += 10 * dpr) {
          const u = x / w;
          const y = h * (0.06 + k * 0.058)
            + Math.sin(u * 5.2 + k * 0.55 + t) * 22 * dpr
            + Math.sin(u * 11 - k * 0.3 - t * 1.4) * 9 * dpr
            + Math.cos(u * 2.1 + k + t * 0.6) * 30 * dpr;
          x ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.stroke();
      }
      t += 0.0028;
      if (!reduce) raf = requestAnimationFrame(draw);
    };
    draw();
    return () => { cancelAnimationFrame(raf); removeEventListener('resize', size); };
  }, []);
  return <canvas id="topo" ref={ref} aria-hidden="true" />;
}

function useCount(target, ms = 1400) {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (!target) return;
    let raf; const t0 = performance.now();
    const step = (now) => { const p = Math.min(1, (now - t0) / ms); setV(Math.round(target * (1 - Math.pow(1 - p, 4)))); if (p < 1) raf = requestAnimationFrame(step); };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

function Dial({ value, delay }) {
  const r = 32, C = 2 * Math.PI * r;
  const [on, setOn] = useState(false);
  useEffect(() => { const id = setTimeout(() => setOn(true), 80 + delay); return () => clearTimeout(id); }, [delay]);
  return (
    <div className="dial" title="Match score: research fit, active funding and recent output">
      <svg width="76" height="76" viewBox="0 0 76 76" aria-hidden="true">
        <circle className="trk" cx="38" cy="38" r={r} fill="none" strokeWidth="6" />
        <circle className="val" cx="38" cy="38" r={r} fill="none" strokeWidth="6" strokeDasharray={C} strokeDashoffset={on ? C * (1 - value) : C} />
      </svg>
      <b><span>{Math.round(value * 100)}<small>match</small></span></b>
    </div>
  );
}

function Card({ r, i, open, onToggle }) {
  const d = i * 90;
  const soon = r.funded_until && monthsLeft(r.funded_until) < 12;
  const agencies = (r.agencies || '').split('/').filter(Boolean);
  return (
    <li className={`card${open ? ' open' : ''}`} style={{ animationDelay: `${d}ms` }} onClick={onToggle}
        tabIndex={0} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), onToggle())} aria-expanded={open}>
      <div className="rank">{r.rank}</div>
      <div className="who">
        <h3>{r.name}{agencies.map((a) => <span key={a} className={`tag ${a}`}>{a}</span>)}</h3>
        <div className="where">{r.institution}{r.city ? ` · ${r.city}, ${r.state}` : ''}</div>
        {r.why?.terms?.length > 0 && (
          <div className="why"><span className="lbl">Matches</span>
            {r.why.terms.slice(0, 4).map((t, k) => <span key={t} className="term" style={{ animationDelay: `${d + 300 + k * 70}ms` }}>{t}</span>)}
          </div>
        )}
        {r.why?.grant && <div className="grant"><b>{r.why.grant_agency}</b>{r.why.grant}</div>}
        <div className="facts">
          {r.active_amount > 0 && <span className="fact money"><i />{fmtMoney(r.active_amount)} active</span>}
          {r.funded_until && <span className={`fact${soon ? ' soon' : ''}`}>funded through {fmtDate(r.funded_until)}</span>}
          {r.n_pubs_recent > 0 && <span className="fact">{r.n_pubs_recent} papers since 2023</span>}
        </div>
      </div>
      <div className="right">
        <Dial value={r.score} delay={d} />
        <div className="sig" aria-label="Signals">
          <div className="f">fit<span className="bar"><i style={{ width: `${r.signals.fit * 100}%`, animationDelay: `${d + 200}ms` }} /></span></div>
          <div className="m">funding<span className="bar"><i style={{ width: `${r.signals.funding * 100}%`, animationDelay: `${d + 280}ms` }} /></span></div>
          <div className="r">output<span className="bar"><i style={{ width: `${r.signals.recency * 100}%`, animationDelay: `${d + 360}ms` }} /></span></div>
        </div>
      </div>
      <div className="more"><div>
        <div className="cols">
          <div><h4>Active grants</h4>
            {r.grants?.length ? <ul>{r.grants.map((g) => <li key={g.src}>{g.title}<small>{g.agency} {g.src} · {fmtMoney(g.amount || 0)} · to {fmtDate(g.end)}</small></li>)}</ul> : <p className="empty">None active</p>}
          </div>
          <div><h4>Recent papers</h4>
            {r.papers?.length ? <ul>{r.papers.map((p) => <li key={p}>{p}</li>)}</ul> : <p className="empty">No grant-linked papers yet</p>}
          </div>
        </div>
      </div></div>
    </li>
  );
}

export default function Home() {
  const [q, setQ] = useState('');
  const [ph, setPh] = useState(0);
  const [stats, setStats] = useState(null);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);
  const [open, setOpen] = useState(null);
  const [asked, setAsked] = useState('');
  const labs = useCount(stats?.funded_labs || 0);

  useEffect(() => { fetch('/api/stats').then((r) => r.json()).then(setStats).catch(() => {}); }, []);
  useEffect(() => { const id = setInterval(() => setPh((p) => (p + 1) % PLACEHOLDERS.length), 2800); return () => clearInterval(id); }, []);

  const run = async (text) => {
    const query = (text ?? q).trim();
    if (!query || busy) return;
    setQ(query); setBusy(true); setOpen(null); setAsked(query); setRes(null);
    try {
      const r = await fetch('/api/match', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ query, k: 10 }) });
      const j = await r.json();
      setRes(j.results || []);
    } catch { setRes([]); }
    setBusy(false);
  };

  const toggleTheme = () => {
    const el = document.documentElement;
    const dark = el.dataset.theme ? el.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    el.dataset.theme = dark ? 'light' : 'dark';
  };

  const compact = busy || res;
  return (
    <>
      <Topo />
      <header className="top">
        <div className="wrap">
          <div className="brand">
            <svg width="26" height="24" viewBox="0 0 24 24" aria-hidden="true"><circle className="ring a" cx="9" cy="12" r="7" /><circle className="ring b" cx="15" cy="12" r="7" /><circle cx="12" cy="12" r="1.7" fill="var(--ochre)" /></svg>
            LabMatch
          </div>
          <span className="grow1" />
          <div className="src" title="Data sources"><span>OpenAlex</span><span>NIH</span><span>NSF</span></div>
          <button className="iconbtn" onClick={toggleTheme} aria-label="Switch light or dark theme">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></svg>
          </button>
        </div>
      </header>

      <main className={`wrap${compact ? ' compact' : ''}`}>
        <section className="hero">
          <span className="live"><i />{labs ? `${labs.toLocaleString()} funded AI labs` : 'Live grants'}</span>
          <h1>
            <span className="w">Find</span> <span className="w" style={{ animationDelay: '.08s' }}>the</span> <span className="w" style={{ animationDelay: '.16s' }}>lab</span>{' '}
            <em><span className="w" style={{ animationDelay: '.26s' }}>that</span> <span className="w" style={{ animationDelay: '.34s' }}>fits</span> <span className="w" style={{ animationDelay: '.42s' }}>you.</span></em>
          </h1>
          <p className="sub">Describe your research. LabMatch ranks US labs by how close their work is to yours, whether they have money right now, and how much they publish.</p>

          <form className="ask" onSubmit={(e) => { e.preventDefault(); run(); }}>
            <div className="box">
              <label htmlFor="q" className="sr">Your research interests or a paper abstract</label>
              <textarea id="q" rows={2} value={q} placeholder={PLACEHOLDERS[ph]} onChange={(e) => setQ(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey || !e.shiftKey)) { e.preventDefault(); run(); } }} />
              <div className="boxbar">
                <span className="hint"><kbd>Enter</kbd> to match · <kbd>Shift</kbd>+<kbd>Enter</kbd> new line</span>
                <button className={`go${busy ? ' busy' : ''}`} disabled={!q.trim() || busy} type="submit">
                  {busy ? 'Matching' : 'Match'}
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
                </button>
              </div>
            </div>
            <div className="ex">{EXAMPLES.map((e) => <button key={e} type="button" onClick={() => run(e)}>{e}</button>)}</div>
          </form>
        </section>

        {busy && (
          <div className="scan" aria-live="polite">
            <div className="msg">Reading {stats ? stats.labs.toLocaleString() : ''} lab profiles…</div>
            <div className="ghost" /><div className="ghost" /><div className="ghost" />
          </div>
        )}

        {res && !busy && (
          <>
            <div className="meta"><h2>{res.length ? 'Your best-fit labs' : 'No funded labs matched'}</h2><span>{res.length} of {stats?.funded_labs?.toLocaleString() || '—'} funded labs</span></div>
            <ol className="list">{res.map((r, i) => <Card key={r.inv_id} r={r} i={i} open={open === r.inv_id} onToggle={() => setOpen(open === r.inv_id ? null : r.inv_id)} />)}</ol>
            <p className="foot">Ranked from <b>NIH</b> and <b>NSF</b> awards and <b>OpenAlex</b> papers. Click a lab for its grants and papers.</p>
          </>
        )}
      </main>
      <Chat context={res ? `Current search results: ${res.slice(0, 5).map((r, i) => `${i + 1}. ${r.name} (${r.institution})`).join('; ')}` : 'No search yet.'} />
    </>
  );
}
