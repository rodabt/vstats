function parseCSV(text) {
  return text
    .split(/[\s,]+/)
    .map(s => s.trim())
    .filter(s => s.length > 0)
    .map(s => {
      const n = parseFloat(s);
      if (isNaN(n)) throw new Error(`"${s}" is not a number`);
      return n;
    });
}

async function apiFetch(endpoint, payload) {
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);
  return data;
}

function fmt(n, decimals = 4) {
  if (n === null || n === undefined) return '—';
  return Number(n).toFixed(decimals);
}

function pct(n, decimals = 2) {
  return (Number(n) * 100).toFixed(decimals) + '%';
}
