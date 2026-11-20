// gta6-trophy-rarity-forecast-01 — forecast each GTA VI trophy's Exophase earn rate at the target.
// The page changes only at Exophase refreshes, so reads collapse to distinct page states.
// players_tracked and each trophy's earners (rate*players/100) follow saturating curves
// A - B*exp(-t/tau); a trophy's tau shrinks toward its tier's pooled tau, its ceiling is capped by
// the players ceiling. The median blends, in logit space, that earners/players view with two damped
// logit-vs-ln(1+t) trends (last day of states; all states). The spread grows with horizon, fit
// residual, extrapolation distance, shrinkage and ensemble disagreement. Thin/flat trophies take
// their last rate plus their rate band's pooled logit growth. Deterministic, stdlib only, offline.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const IN = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const S0 = 0.22, SHRINK = 4;                  // base logit vol / sqrt-day; tau shrinkage pseudo-count
const WDEC = 0.25, WTREC = 0.55, WTALL = 0.2; // weights: decomposition, recent trend, all-states trend
const TWIN = 24, DAMP = 0.85;                 // recent-trend window (h); damping of a trend's extension
const TAUS = [3, 6, 12, 24, 48, 96, 192, 384, 768, 1536]; // exp-approach time constants (h)
const clampPct = (x: number) => Math.min(99.99, Math.max(0.01, x));
const logit = (p: number) => { const q = clampPct(p); return Math.log(q / (100 - q)); };
const invLogit = (l: number) => 100 / (1 + Math.exp(-l));
const round4 = (x: number) => Math.round(x * 10000) / 10000;
const median = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : 0; };
interface Fit { A: number; B: number; tau: number; sse: number }
// Least squares for y = A - B*exp(-t/tau) at a fixed tau, with B >= 0 and A >= max(y).
function fitAtTau(ts: number[], ys: number[], tau: number): Fit {
  const n = ts.length, k = 1 / tau, xs = new Array<number>(n);
  let sx = 0, sxx = 0, sy = 0, sxy = 0, ymax = -Infinity;
  for (let i = 0; i < n; i++) { const x = Math.exp(-k * ts[i]); xs[i] = x; sx += x; sxx += x * x; sy += ys[i]; sxy += x * ys[i]; if (ys[i] > ymax) ymax = ys[i]; }
  const det = n * sxx - sx * sx;
  let A = sy / n, B = 0;
  if (Math.abs(det) > 1e-12) { A = (sxx * sy - sx * sxy) / det; B = (sx * sy - n * sxy) / det; }
  if (B < 0) { B = 0; A = sy / n; }
  if (A < ymax) { A = ymax; let num = 0, den = 0; for (let i = 0; i < n; i++) { num += (A - ys[i]) * xs[i]; den += xs[i] * xs[i]; } B = den > 0 ? Math.max(0, num / den) : 0; }
  let sse = 0;
  for (let i = 0; i < n; i++) { const r = ys[i] - (A - B * xs[i]); sse += r * r; }
  return { A, B, tau, sse };
}
function fitExp(ts: number[], ys: number[]): Fit {
  let best: Fit | null = null;
  for (const tau of TAUS) { const f = fitAtTau(ts, ys, tau); if (!best || f.sse < best.sse) best = f; }
  return best as Fit;
}
type Read = { slot?: string; at: string; players_tracked: number; rates: Array<number | null> };
type Case = { levels: number[]; horizon_minutes: number; trophies: Array<{ id: number; tier: string }>; history: Read[] };
type Pt = [number, number, number]; // [hours since first read, rate, players]
function forecast(c: Case) {
  const trophies = c.trophies, hist = c.history;
  const t0 = Date.parse(hist[0].slot ?? hist[0].at);
  const hrs = (r: Read) => (Date.parse(r.slot ?? r.at) - t0) / 3600000;
  const tLast = hrs(hist[hist.length - 1]), tTgt = tLast + c.horizon_minutes / 60, hDays = c.horizon_minutes / 1440;
  const pp: Array<[number, number]> = [];
  for (const r of hist) if (!pp.length || pp[pp.length - 1][1] !== r.players_tracked) pp.push([hrs(r), r.players_tracked]);
  let pT = hist[hist.length - 1].players_tracked, pCeil = pT;
  if (pp.length >= 3) {
    const f = fitExp(pp.map(p => p[0]), pp.map(p => p[1]));
    pCeil = Math.max(f.A, pT);
    pT = Math.min(pCeil, f.A - f.B * Math.exp(-tTgt / f.tau));
    if (!(pT >= hist[hist.length - 1].players_tracked)) pT = hist[hist.length - 1].players_tracked;
  }
  const series: Pt[][] = trophies.map((_, i) => {
    const pts: Pt[] = [];
    for (const r of hist) {
      const v = r.rates[i];
      if (v === null || v === undefined) continue;
      const last = pts[pts.length - 1];
      if (!last || last[1] !== v || last[2] !== r.players_tracked) pts.push([hrs(r), v, r.players_tracked]);
    }
    return pts;
  });
  const fits = series.map(pts => {
    if (pts.length < 3) return null;
    const ts = pts.map(p => p[0]), es = pts.map(p => (p[1] * p[2]) / 100);
    if (Math.max(...es) <= 0) return null;
    const f = fitExp(ts, es);
    return { ...f, resid: Math.sqrt(f.sse / Math.max(1, ts.length)) / Math.max(1, es[es.length - 1]), n: pts.length, span: ts[ts.length - 1] - ts[0] };
  });
  const tierTau: Record<string, number> = {};
  for (const tier of ['platinum', 'gold', 'silver', 'bronze']) {
    const taus = fits.filter((f, i) => f && trophies[i].tier === tier).map(f => (f as Fit).tau);
    if (taus.length) tierTau[tier] = median(taus);
  }
  const allTaus = fits.filter(Boolean).map(f => (f as Fit).tau);
  const globalTau = allTaus.length ? median(allTaus) : 96;
  const bandOf = (r: number) => (r < 0.5 ? 0 : r < 2 ? 1 : r < 8 ? 2 : 3);
  const bg: number[][] = [[], [], [], []];
  fits.forEach((f, i) => {
    if (!f) return;
    const lastRate = series[i][series[i].length - 1][1];
    const eT = Math.min(pCeil, f.A - f.B * Math.exp(-tTgt / f.tau));
    bg[bandOf(lastRate)].push(logit((100 * Math.max(eT, 0)) / pT) - logit(lastRate));
  });
  const bd = bg.map(g => (g.length ? median(g) : 0));
  const trendAt = (pts: Pt[], winH: number): number | null => {
    const win = winH === Infinity ? pts : pts.filter(p => p[0] >= tLast - winH);
    if (win.length < 3) return null;
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (const p of win) { const x = Math.log(1 + Math.max(0, p[0])), y = logit(p[1]); sx += x; sy += y; sxx += x * x; sxy += x * y; }
    const det = win.length * sxx - sx * sx;
    if (Math.abs(det) < 1e-9) return null;
    const slope = (win.length * sxy - sx * sy) / det, inter = (sy - slope * sx) / win.length;
    const atLast = inter + slope * Math.log(1 + Math.max(0, tLast));
    return atLast + DAMP * (inter + slope * Math.log(1 + Math.max(0, tTgt)) - atLast);
  };
  const rows = trophies.map((tr, i) => {
    const pts = series[i], lastRate = pts.length ? pts[pts.length - 1][1] : 0.01, f = fits[i];
    let medLogit: number, sigma: number;
    if (f) {
      const tauPool = tierTau[tr.tier] ?? globalTau, w = f.n / (f.n + SHRINK);
      const tauS = Math.exp(w * Math.log(f.tau) + (1 - w) * Math.log(tauPool));
      const ts = pts.map(p => p[0]), es = pts.map(p => (p[1] * p[2]) / 100);
      const fs = fitAtTau(ts, es, tauS);
      const ceil = Math.min(Math.max(fs.A, f.A * w + Math.min(fs.A, pCeil) * (1 - w)), pCeil);
      let eT = Math.min(pCeil, ceil - fs.B * Math.exp(-tTgt / tauS));
      eT = Math.max(eT, es[es.length - 1]);
      const cd: Array<[number, number]> = [[WDEC, logit((100 * eT) / pT)]];
      const tR = trendAt(pts, TWIN); if (tR !== null && isFinite(tR)) cd.push([WTREC, tR]);
      const tA = trendAt(pts, Infinity); if (tA !== null && isFinite(tA)) cd.push([WTALL, tA]);
      let wsum = 0, msum = 0;
      for (const [cw, cv] of cd) { wsum += cw; msum += cw * cv; }
      medLogit = msum / wsum;
      let disagree = 0;
      for (const [, cv] of cd) disagree = Math.max(disagree, Math.abs(cv - medLogit));
      const extrap = 1 + (tTgt - tLast) / Math.max(12, f.span);
      sigma = Math.sqrt((S0 * Math.sqrt(hDays)) ** 2 + f.resid ** 2 * extrap + (1 - w) ** 2 * 0.09 + (0.5 * disagree) ** 2);
    } else {
      medLogit = logit(lastRate) + 0.8 * (bd[bandOf(lastRate)] || 0);
      sigma = Math.sqrt((S0 * Math.sqrt(hDays)) ** 2 + 0.16);
    }
    sigma = Math.min(2.2, Math.max(0.15, sigma));
    const q = Z.map(z => round4(invLogit(medLogit + z * sigma)));
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    return { id: tr.id, q };
  });
  return { trophies: rows };
}
mkdirSync(OUT, { recursive: true });
for (const file of readdirSync(IN).filter(f => f.endsWith('.in')).sort()) {
  const c = JSON.parse(readFileSync(join(IN, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}`);
  if (!c.history.length) throw new Error(`${file} has no history`);
  writeFileSync(join(OUT, `${file.slice(0, -3)}.out`), `${JSON.stringify(forecast(c))}\n`);
}
