const $ = (sel) => document.querySelector(sel);

async function loadStatus() {
  const s = await (await fetch("/api/status")).json();
  $("#status").textContent =
    `${s.photos} photos · ${s.faces} faces · ${s.people} people` +
    (s.last_scan ? ` · last scan ${s.last_scan}` : "");
}

function renderResults(results, heading) {
  const grid = $("#results");
  grid.innerHTML = "";
  if (heading) {
    const h = document.createElement("div");
    h.className = "results-heading";
    h.textContent = heading;
    grid.appendChild(h);
  }
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
    view.textContent = p.name ? `${p.name} (${label})` : `Unnamed (${label})`;
    view.addEventListener("click", async () => {
      const res = await (await fetch(`/api/people/${p.id}/photos`)).json();
      renderResults(res.results, view.textContent);
    });

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

async function doSearch(q) {
  if (!q) return;
  const res = await (await fetch(`/api/search?q=${encodeURIComponent(q)}`)).json();
  renderResults(res.results, `Results for “${q}”`);
}

async function browseAll() {
  const res = await (await fetch("/api/photos")).json();
  renderResults(res.results, `All photos (${res.results.length})`);
}

$("#search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  doSearch($("#q").value.trim());
});
$("#browse-all").addEventListener("click", browseAll);

loadStatus();
loadPeople();
browseAll();
