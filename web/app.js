const $ = (sel) => document.querySelector(sel);
const PER_PAGE = 100;

// Remembers the current result view so the pager can re-fetch the same query
// at a different offset. `url` is the endpoint WITHOUT limit/offset.
let currentView = { url: "/api/photos", label: "All photos" };

async function loadStatus() {
  const s = await (await fetch("/api/status")).json();
  $("#status").textContent =
    `${s.photos} photos · ${s.faces} faces · ${s.people} people` +
    (s.last_scan ? ` · last scan ${s.last_scan}` : "");
}

async function loadView(url, label, offset = 0) {
  currentView = { url, label };
  const sep = url.includes("?") ? "&" : "?";
  const data = await (await fetch(`${url}${sep}limit=${PER_PAGE}&offset=${offset}`)).json();
  renderResults(data, label);
}

function makePager(data) {
  const { total = 0, offset = 0, limit = PER_PAGE } = data;
  const pager = document.createElement("div");
  pager.className = "pager";
  const from = total ? offset + 1 : 0;
  const to = Math.min(offset + limit, total);

  const prev = document.createElement("button");
  prev.type = "button"; prev.textContent = "‹ Prev";
  prev.disabled = offset <= 0;
  prev.onclick = () => loadView(currentView.url, currentView.label, Math.max(0, offset - limit));

  const info = document.createElement("span");
  info.textContent = `${from}–${to} of ${total}`;

  const next = document.createElement("button");
  next.type = "button"; next.textContent = "Next ›";
  next.disabled = offset + limit >= total;
  next.onclick = () => loadView(currentView.url, currentView.label, offset + limit);

  pager.append(prev, info, next);
  return pager;
}

function renderResults(data, heading) {
  const results = data.results || [];
  const grid = $("#results");
  grid.innerHTML = "";

  const head = document.createElement("div");
  head.className = "results-heading";
  head.textContent = heading;
  grid.appendChild(head);
  grid.appendChild(makePager(data));

  if (!results.length) {
    const empty = document.createElement("div");
    empty.textContent = "No matches.";
    grid.appendChild(empty);
    return;
  }

  for (const r of results) {
    const cell = document.createElement("div");
    cell.className = "cell";

    const link = document.createElement("a");
    link.href = `/api/photo/${r.photo_id}`;   // full-res original, opens in new tab
    link.target = "_blank";
    link.rel = "noopener";
    const img = document.createElement("img");
    img.src = `/api/thumb/${r.photo_id}`;
    const score = r.score != null ? `\nscore ${r.score.toFixed(3)}` : "";
    img.title = `${r.path}${score}`;
    img.loading = "lazy";
    link.appendChild(img);

    const reveal = document.createElement("button");
    reveal.type = "button";
    reveal.className = "reveal";
    reveal.textContent = "Reveal in Finder";
    reveal.addEventListener("click", async () => {
      reveal.textContent = "Revealing…";
      const resp = await fetch(`/api/photo/${r.photo_id}/reveal`, { method: "POST" });
      reveal.textContent = resp.ok ? "Revealed ✓" : "Error";
      setTimeout(() => (reveal.textContent = "Reveal in Finder"), 1200);
    });

    cell.appendChild(link);
    cell.appendChild(reveal);
    grid.appendChild(cell);
  }

  grid.appendChild(makePager(data)); // bottom pager too
}

async function loadPeople() {
  const people = await (await fetch("/api/people")).json();
  const panel = $("#people");
  panel.innerHTML = "<h2>People</h2>";
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent = "Click a face group to see its photos. Type a name to label it.";
  panel.appendChild(hint);

  for (const p of people) {
    const row = document.createElement("div");
    row.className = "person";

    const view = document.createElement("button");
    view.type = "button";
    view.className = "person-view";
    const label = p.photos === 1 ? "1 photo" : `${p.photos} photos`;
    const heading = p.name ? `${p.name} (${label})` : `Unnamed (${label})`;
    view.textContent = heading;
    view.addEventListener("click", () =>
      loadView(`/api/people/${p.id}/photos`, heading, 0));

    const input = document.createElement("input");
    input.value = p.name || "";
    input.placeholder = "name this person…";

    const status = document.createElement("span");
    status.className = "person-status";

    input.addEventListener("change", async () => {
      const resp = await fetch(`/api/people/${p.id}/name`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: input.value }),
      });
      status.textContent = resp.ok ? "✓ saved" : "✗ error";
      setTimeout(loadPeople, 800); // refresh labels/counts after save
    });

    row.appendChild(view);
    row.appendChild(input);
    row.appendChild(status);
    panel.appendChild(row);
  }
}

function doSearch(q) {
  if (!q) return;
  loadView(`/api/search?q=${encodeURIComponent(q)}`, `Results for “${q}”`, 0);
}

$("#search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  doSearch($("#q").value.trim());
});
$("#browse-all").addEventListener("click", () => loadView("/api/photos", "All photos", 0));

loadStatus();
loadPeople();
loadView("/api/photos", "All photos", 0);
