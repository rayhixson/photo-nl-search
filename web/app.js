const $ = (sel) => document.querySelector(sel);

async function loadStatus() {
  const s = await (await fetch("/api/status")).json();
  $("#status").textContent =
    `${s.photos} photos · ${s.faces} faces · ${s.people} people` +
    (s.last_scan ? ` · last scan ${s.last_scan}` : "");
}

async function loadPeople() {
  const people = await (await fetch("/api/people")).json();
  $("#people").innerHTML = "<h2>People</h2>";
  for (const p of people) {
    const row = document.createElement("div");
    row.className = "person";
    const input = document.createElement("input");
    input.value = p.name || "";
    input.placeholder = `Unnamed (${p.count})`;
    input.addEventListener("change", async () => {
      await fetch(`/api/people/${p.id}/name`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: input.value }),
      });
    });
    row.appendChild(input);
    $("#people").appendChild(row);
  }
}

async function doSearch(q) {
  const res = await (await fetch(`/api/search?q=${encodeURIComponent(q)}`)).json();
  const grid = $("#results");
  grid.innerHTML = "";
  for (const r of res.results) {
    const img = document.createElement("img");
    img.src = `/api/thumb/${r.photo_id}`;
    img.title = `${r.path}\nscore ${r.score.toFixed(3)}`;
    img.loading = "lazy";
    grid.appendChild(img);
  }
  if (!res.results.length) grid.textContent = "No matches.";
}

$("#search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  doSearch($("#q").value.trim());
});

loadStatus();
loadPeople();
