async function fetchJson(url, options = {}) {
  const resp = await fetch(url, options);
  const contentType = resp.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await resp.json() : await resp.text();
  if (!resp.ok) {
    throw new Error(typeof body === "string" ? body : JSON.stringify(body));
  }
  return body;
}

function setStatus(elId, message, isError = false) {
  const el = document.getElementById(elId);
  el.textContent = message;
  el.style.color = isError ? "#b91c1c" : "#0f5132";
}

function createCollapsibleSection(title, content, defaultOpen = true) {
  const details = document.createElement("details");
  details.className = "collapsible-section";
  if (defaultOpen) details.open = true;
  
  const summary = document.createElement("summary");
  summary.textContent = title;
  summary.className = "section-header";
  
  const contentDiv = document.createElement("pre");
  contentDiv.className = "section-content";
  contentDiv.textContent = content;
  
  details.appendChild(summary);
  details.appendChild(contentDiv);
  
  return details;
}

function updateTaskInputs() {
  const task = document.getElementById("task-type").value;
  const queryGroup = document.getElementById("query-group");
  const levelGroup = document.getElementById("level-group");

  queryGroup.classList.add("hidden");
  levelGroup.classList.add("hidden");

  if (task === "ask") {
    queryGroup.classList.remove("hidden");
  }
  if (task === "explain") {
    levelGroup.classList.remove("hidden");
  }
}

function setToolStatus(message, isError = false) {
  const el = document.getElementById("tools-status");
  if (!el) {
    return;
  }
  setStatus("tools-status", message, isError);
}

async function runGapFinder() {
  setToolStatus("Finding research gaps...");
  try {
    const data = await fetchJson("/api/gaps", { method: "POST" });
    if (data.error) {
      setToolStatus(`Error: ${data.error.message}`, true);
      return;
    }
    renderGapsResult(data);
    setToolStatus(`Found ${data.gaps.length} gaps across ${data.paper_count} paper(s).`);
  } catch (err) {
    setToolStatus(`Failed: ${err.message}`, true);
  }
}

function renderGapsResult(data) {
  const readable = document.getElementById("result-readable");
  readable.innerHTML = "";

  const header = document.createElement("div");
  header.className = "explanation-header";
  header.innerHTML = `<strong>Research Gaps</strong> - ${data.paper_count} paper(s) analyzed`;
  readable.appendChild(header);

  const sections = [
    { title: "Unanswered Questions", items: data.gaps },
    { title: "Missing Experiments", items: data.missing_experiments },
    { title: "Follow-up Directions", items: data.followup_directions },
  ];

  for (const section of sections) {
    if (!section.items || !section.items.length) continue;
    const content = section.items.map((item, i) => `${i + 1}. ${item}`).join("\n");
    readable.appendChild(createCollapsibleSection(section.title, content, true));
  }

  if (data.synthesis && !data.gaps.length) {
    const pre = document.createElement("pre");
    pre.style.cssText = "white-space:pre-wrap;padding:12px;font-size:0.85rem";
    pre.textContent = data.synthesis;
    readable.appendChild(pre);
  }

  document.getElementById("result-json").textContent = JSON.stringify(data, null, 2);
}

async function runHypotheses() {
  setToolStatus("Generating hypotheses...");
  try {
    const data = await fetchJson("/api/hypotheses", { method: "POST" });
    if (data.error) {
      setToolStatus(`Error: ${data.error.message}`, true);
      return;
    }
    renderHypothesesResult(data);
    setToolStatus(`Generated ${data.hypotheses.length} hypotheses from ${data.paper_count} paper(s).`);
  } catch (err) {
    setToolStatus(`Failed: ${err.message}`, true);
  }
}

function renderHypothesesResult(data) {
  const readable = document.getElementById("result-readable");
  readable.innerHTML = "";

  const header = document.createElement("div");
  header.className = "explanation-header";
  header.innerHTML = `<strong>Research Hypotheses</strong> - ${data.paper_count} paper(s)`;
  readable.appendChild(header);

  data.hypotheses.forEach((h, idx) => {
    const content = [
      h.description ? `Description:\n${h.description}` : "",
      h.rationale ? `Rationale:\n${h.rationale}` : "",
      h.testability ? `How to test:\n${h.testability}` : "",
    ]
      .filter(Boolean)
      .join("\n\n");

    readable.appendChild(createCollapsibleSection(`Hypothesis ${idx + 1}: ${h.title}`, content, true));
  });

  document.getElementById("result-json").textContent = JSON.stringify(data, null, 2);
}

async function runDigest() {
  setToolStatus("Generating digest...");
  try {
    const data = await fetchJson("/digest", { method: "POST" });
    if (data.error) {
      setToolStatus(`Error: ${data.error.message}`, true);
      return;
    }
    renderDigestResult(data);
    setToolStatus(`Digest generated for ${data.paper_count} paper(s).`);
  } catch (err) {
    setToolStatus(`Failed: ${err.message}`, true);
  }
}

function renderDigestResult(data) {
  const readable = document.getElementById("result-readable");
  readable.innerHTML = "";

  const header = document.createElement("div");
  header.className = "explanation-header";
  header.innerHTML = `<strong>Session Digest</strong> - ${data.paper_count} paper(s): ${(data.paper_titles || []).join(", ")}`;
  readable.appendChild(header);

  const sections = [
    { title: "Full Digest", content: data.digest, open: true },
    { title: "Key Themes", content: (data.key_themes || []).join("\n"), open: true },
    { title: "Notable Results", content: (data.notable_results || []).join("\n"), open: true },
    { title: "Open Questions", content: (data.open_questions || []).join("\n"), open: false },
  ];

  for (const section of sections) {
    if (section.content && section.content.trim()) {
      readable.appendChild(createCollapsibleSection(section.title, section.content, section.open));
    }
  }

  document.getElementById("result-json").textContent = JSON.stringify(data, null, 2);
}

function renderResult(data) {
  const readable = document.getElementById("result-readable");
  const pre = document.getElementById("result-json");
  pre.textContent = JSON.stringify(data, null, 2);

  const topTask = data?.task;
  const maybeWrapped = data?.result;
  const innerResult = maybeWrapped?.result;

  // Handle both shapes:
  // 1) { task, selected_papers, result: { task, selected_papers, result, meta } }
  // 2) { task, selected_papers, result }
  // 3) direct raw result object
  let task = topTask || maybeWrapped?.task || "";
  let payload = {};

  if (innerResult && typeof innerResult === "object") {
    payload = innerResult;
  } else if (maybeWrapped && typeof maybeWrapped === "object") {
    payload = maybeWrapped;
  } else if (data && typeof data === "object") {
    payload = data;
  }

  if (!task && payload.task && typeof payload.task === "string") {
    task = payload.task;
  }

  if (task === "analyse") {
    const analyses = Array.isArray(payload.analyses) ? payload.analyses : [];
    const reports = Array.isArray(payload.reports) ? payload.reports : [];
    if (!analyses.length && !reports.length) {
      readable.textContent = "No analysis content available for the selected papers.";
      return;
    }

    const parts = analyses.length
      ? analyses.map((item, idx) => {
          const title = item.paper_name ? `Paper ${idx + 1}: ${item.paper_name}` : `Paper ${idx + 1}`;
          return (
            `${title}\n\n` +
            `Summary\n${item.summary || "N/A"}\n\n` +
            `Methodology\n${item.methodology || "N/A"}\n\n` +
            `Key Idea\n${item.key_idea || "N/A"}\n\n` +
            `Results\n${item.results || "N/A"}\n\n` +
            `Citation Insight\n${item.citation_insight || "N/A"}`
          );
        })
      : reports.map((item, idx) => {
          const title = item.paper_name ? `Paper ${idx + 1}: ${item.paper_name}` : `Paper ${idx + 1}`;
          const review = item.paper_review || {};
          const exp = item.explanation || {};
          const strengths = Array.isArray(review.strengths) ? review.strengths.map((s) => `- ${s}`).join("\n") : "- N/A";
          const weaknesses = Array.isArray(review.weaknesses) ? review.weaknesses.map((s) => `- ${s}`).join("\n") : "- N/A";
          const suggestions = Array.isArray(review.suggestions) ? review.suggestions.map((s) => `- ${s}`).join("\n") : "- N/A";

          return (
            `${title}\n\n` +
            `Paper Review\n\n` +
            `1. Strengths\n${strengths}\n\n` +
            `2. Weaknesses\n${weaknesses}\n\n` +
            `3. Suggestions\n${suggestions}\n\n` +
            `Explanation\n` +
            `- Beginner: ${exp.beginner || "N/A"}\n\n` +
            `- Intermediate: ${exp.intermediate || "N/A"}\n\n` +
            `- Expert: ${exp.expert || "N/A"}\n\n` +
            `Citation Insight\n${item.citation_insight || "N/A"}`
          );
        });

    parts.push(`Comparative Analysis\n\n${payload.comparison || payload.comparative_analysis || "N/A"}`);

    readable.textContent = parts.join("\n\n------------------------------\n\n");
    return;
  }

  if (task === "ask") {
    readable.textContent = payload.answer || "No answer generated.";
    return;
  }

  if (task === "gaps") {
    runGapFinder();
    return;
  }

  if (task === "hypotheses") {
    runHypotheses();
    return;
  }

  if (task === "explain") {
    const level = payload.level || "unknown";
    const explanations = Array.isArray(payload.explanations) ? payload.explanations : [];
    
    // Create enhanced display with collapsible sections
    readable.innerHTML = "";
    
    // Add level indicator
    const levelHeader = document.createElement("div");
    levelHeader.className = "explanation-header";
    levelHeader.innerHTML = `<strong>Explanation Level:</strong> ${level.charAt(0).toUpperCase() + level.slice(1)} | <strong>Papers:</strong> ${explanations.length}`;
    readable.appendChild(levelHeader);
    
    if (!explanations.length) {
      const noData = document.createElement("p");
      noData.textContent = "No explanations generated.";
      readable.appendChild(noData);
      return;
    }
    
    // Render each paper's explanation
    explanations.forEach((item, idx) => {
      const paperHeader = document.createElement("div");
      paperHeader.className = "paper-separator";
      paperHeader.innerHTML = `<h3>📄 Paper ${idx + 1}: ${item.paper_name || item.paper_id || "Unknown"}</h3>`;
      readable.appendChild(paperHeader);
      
      const explanation = item.explanation || "No explanation generated.";
      const diagram = item.diagram || "";
      
      // Add explanation section
      if (explanation) {
        const explSection = createCollapsibleSection("Explanation", explanation, true);
        readable.appendChild(explSection);
      }
      
      // Add diagram section if exists
      if (diagram) {
        const diagramSection = createCollapsibleSection("Diagram / Detailed View", diagram, true);
        readable.appendChild(diagramSection);
      }
    });
    
    return;
  }

  if (task === "review") {
    const reviews = Array.isArray(payload.reviews) ? payload.reviews : [];
    if (!reviews.length) {
      readable.textContent = "No review content available.";
      return;
    }

    const text = reviews
      .map((r, idx) => {
        const strengths = (r.strengths || []).join("; ");
        const weaknesses = (r.weaknesses || []).join("; ");
        const suggestions = (r.suggestions || []).join("; ");
        return (
          `Paper ${idx + 1}\n\n` +
          `Strengths: ${strengths || "N/A"}\n\n` +
          `Weaknesses: ${weaknesses || "N/A"}\n\n` +
          `Suggestions: ${suggestions || "N/A"}`
        );
      })
      .join("\n\n------------------------------\n\n");

    readable.textContent = payload.comparison ? `${text}\n\nComparative Review\n\n${payload.comparison}` : text;
    return;
  }

  if (task === "citations") {
    const papers = Array.isArray(payload.papers) ? payload.papers : [];
    if (papers.length) {
      const blocks = papers.map((paper, pIdx) => {
        const citations = Array.isArray(paper.citations) ? paper.citations : [];
        const title = `Paper ${pIdx + 1}: ${paper.paper_name || "unknown.pdf"}`;
        if (!citations.length) {
          return `${title}\n\nNo citations detected.`;
        }

        const lines = citations
          .slice(0, 8)
          .map((c, i) => `${i + 1}. ${c.raw_text}\nType: ${c.type}\n${c.context}\nInsight: ${c.insight}`)
          .join("\n\n");
        return `${title}\n\n${lines}`;
      });

      readable.textContent = blocks.join("\n\n------------------------------\n\n");
      return;
    }

    // Backward-compatible flat shape
    const citations = Array.isArray(payload.citations) ? payload.citations : [];
    if (!citations.length) {
      readable.textContent = "No citations detected.";
      return;
    }

    readable.textContent = citations
      .slice(0, 10)
      .map((c, i) => `${i + 1}. ${c.raw_text}\nType: ${c.type}\n${c.context}\nInsight: ${c.insight}`)
      .join("\n\n");
    return;
  }

  readable.textContent = "Result generated. Expand raw JSON to inspect full details.";
}

async function refreshPapers() {
  const list = document.getElementById("papers-list");
  list.innerHTML = "";
  try {
    const papers = await fetchJson("/papers");
    if (!papers.length) {
      const li = document.createElement("li");
      li.textContent = "No papers ingested yet.";
      list.appendChild(li);
      updateSelectionCount(0);
      return;
    }

    for (const paper of papers) {
      const li = document.createElement("li");
      li.className = "paper-item";
      
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.id = `paper-${paper.paper_id}`;
      checkbox.className = "paper-checkbox";
      checkbox.checked = paper.selected;
      checkbox.dataset.paperId = paper.paper_id;
      checkbox.addEventListener("change", onPaperSelectionChange);
      
      const label = document.createElement("label");
      label.htmlFor = `paper-${paper.paper_id}`;
      label.textContent = paper.filename;
      
      li.appendChild(checkbox);
      li.appendChild(label);
      list.appendChild(li);
    }
    
    updateSelectionCount(papers.filter(p => p.selected).length);
  } catch (err) {
    const li = document.createElement("li");
    li.textContent = `Failed to fetch papers: ${String(err.message || err)}`;
    list.appendChild(li);
  }
}

function updateSelectionCount(count) {
  const countEl = document.getElementById("selection-count");
  countEl.textContent = `${count} selected`;
}

async function onPaperSelectionChange() {
  const checkboxes = document.querySelectorAll(".paper-checkbox");
  const selectedIds = Array.from(checkboxes)
    .filter(cb => cb.checked)
    .map(cb => cb.dataset.paperId);
  
  try {
    const result = await fetchJson("/papers/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paper_ids: selectedIds }),
    });
    updateSelectionCount(result.selected_count);
  } catch (err) {
    console.error("Failed to update selection:", err);
  }
}

async function selectAllPapers() {
  const checkboxes = document.querySelectorAll(".paper-checkbox");
  const allIds = Array.from(checkboxes).map(cb => cb.dataset.paperId);
  
  try {
    const result = await fetchJson("/papers/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paper_ids: allIds }),
    });
    await refreshPapers();
  } catch (err) {
    console.error("Failed to select all:", err);
  }
}

async function deselectAllPapers() {
  try {
    const result = await fetchJson("/papers/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paper_ids: [] }),
    });
    await refreshPapers();
  } catch (err) {
    console.error("Failed to deselect all:", err);
  }
}

async function onUploadSubmit(event) {
  event.preventDefault();
  const filesInput = document.getElementById("pdf-files");
  if (!filesInput.files || filesInput.files.length === 0) {
    setStatus("upload-status", "Select at least one PDF file.", true);
    return;
  }

  const formData = new FormData();
  for (const file of filesInput.files) {
    formData.append("files", file);
  }

  setStatus("upload-status", "Uploading and processing...");
  try {
    const data = await fetchJson("/ingest", { method: "POST", body: formData });
    setStatus("upload-status", `Ingested ${data.papers.length} paper(s).`);
    renderResult(data);
    await refreshPapers();
  } catch (err) {
    setStatus("upload-status", `Upload failed: ${String(err.message || err)}`, true);
  }
}

async function onTaskSubmit(event) {
  event.preventDefault();
  const task = document.getElementById("task-type").value;
  const query = document.getElementById("task-query").value.trim();
  const level = document.getElementById("task-level").value;

  const payload = { task };
  if (task === "ask") {
    if (!query) {
      setStatus("task-status", "Question is required for ask task.", true);
      return;
    }
    payload.question = query;
  }
  if (task === "explain") {
    payload.level = level;
  }

  if (task === "gaps") {
    setStatus("task-status", "Running task...");
    await runGapFinder();
    setStatus("task-status", "Task completed.");
    return;
  }

  if (task === "hypotheses") {
    setStatus("task-status", "Running task...");
    await runHypotheses();
    setStatus("task-status", "Task completed.");
    return;
  }

  setStatus("task-status", "Running task...");

  try {
    const data = await fetchJson("/task", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderResult(data);
    setStatus("task-status", "Task completed.");
  } catch (err) {
    setStatus("task-status", `Task failed: ${String(err.message || err)}`, true);
  }
}

function collapseAll() {
  const sections = document.querySelectorAll(".collapsible-section");
  sections.forEach(section => section.open = false);
}

function expandAll() {
  const sections = document.querySelectorAll(".collapsible-section");
  sections.forEach(section => section.open = true);
}

window.addEventListener("DOMContentLoaded", async () => {
  document.getElementById("upload-form").addEventListener("submit", onUploadSubmit);
  document.getElementById("task-form").addEventListener("submit", onTaskSubmit);
  document.getElementById("refresh-papers").addEventListener("click", refreshPapers);
  document.getElementById("select-all-papers").addEventListener("click", selectAllPapers);
  document.getElementById("deselect-all-papers").addEventListener("click", deselectAllPapers);
  document.getElementById("task-type").addEventListener("change", updateTaskInputs);
  document.getElementById("collapse-all").addEventListener("click", collapseAll);
  document.getElementById("expand-all").addEventListener("click", expandAll);
  updateTaskInputs();
  await refreshPapers();
});
