'use client';
import { useEffect, useRef, useState } from 'react';

// Help chat. Answers come from /api/chat (Groq, key kept on the server).
// If the server has no key, it falls back to built-in answers so the widget never breaks.
const SUG = ['What is LabMatch?', 'What does the score mean?', 'Which labs work on robot learning?'];
const FAQ = [
  [/what is|about|who.*for/, 'LabMatch helps students and researchers find a research lab. Describe what you want to work on and it ranks US AI/ML labs with active NIH or NSF grants, with the reasons each one fits.'],
  [/score|100|odds/, 'The best match for your search is 100 and the others are scored relative to it. The model behind it is calibrated on held-out papers.'],
  [/data|source/, 'Grants come from NIH RePORTER and NSF Awards, papers from OpenAlex.'],
];
const faq = (q) => (FAQ.find(([re]) => re.test(q.toLowerCase()))?.[1]) || 'Ask me about the scores, the data, or which labs work on a topic.';

export default function Chat({ context = '' }) {
  const [open, setOpen] = useState(false);
  const [log, setLog] = useState([{ role: 'assistant', content: 'Hi! Ask me anything about LabMatch or your matches.' }]);
  const [busy, setBusy] = useState(false);
  const [q, setQ] = useState('');
  const end = useRef(null);
  useEffect(() => end.current?.scrollIntoView({ block: 'end' }), [log, busy]);

  async function ask(text) {
    text = text.trim(); if (!text || busy) return; setQ('');
    const next = [...log, { role: 'user', content: text }]; setLog(next); setBusy(true);
    let answer;
    try {
      const r = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ messages: next.slice(1).slice(-10), context }) });
      answer = r.ok ? (await r.json()).answer : faq(text);
    } catch { answer = faq(text); }
    setLog([...next, { role: 'assistant', content: answer || faq(text) }]); setBusy(false);
  }

  return (
    <div className={`chat${open ? ' open' : ''}`}>
      {open && (
        <div className="chatp" role="dialog" aria-label="LabMatch helper">
          <div className="chath"><div><b>Ask LabMatch</b><span>Questions about this page, your matches or the data</span></div><button className="iconbtn" onClick={() => setOpen(false)} aria-label="Close helper">×</button></div>
          <div className="chatlog" aria-live="polite">
            {log.map((m, i) => <div key={i} className={`msg ${m.role === 'user' ? 'me' : 'bot'}`}>{m.content}</div>)}
            {busy && <div className="msg bot typing"><i /><i /><i /></div>}
            <div ref={end} />
          </div>
          <div className="chatsug">{SUG.map((s) => <button key={s} type="button" onClick={() => ask(s)}>{s}</button>)}</div>
          <form className="chatf" onSubmit={(e) => { e.preventDefault(); ask(q); }}>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask anything…" aria-label="Your question" />
            <button className="go" type="submit" aria-label="Send">→</button>
          </form>
        </div>
      )}
      <button className="chatb" type="button" aria-expanded={open} onClick={() => setOpen(!open)}>Ask</button>
    </div>
  );
}
