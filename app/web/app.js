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
            `Text Diagram\n${item.text_diagram || "N/A"}\n\n` +
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
    parts.push(`Insight Summary\n${payload.insight_summary || "N/A"}`);
    parts.push(`Why It Matters\n${payload.why_it_matters || "N/A"}`);
    parts.push(`Evolution\n${payload.evolution || "N/A"}`);

    readable.textContent = parts.join("\n\n------------------------------\n\n");
    return;
  }

  if (task === "ask") {
    readable.textContent = payload.answer || "No answer generated.";
    return;
  }

  if (task === "explain") {
    const level = payload.level || "unknown";
    const explanation = payload.explanation || "No explanation generated.";
    const diagram = payload.diagram || "";
    readable.textContent = diagram
      ? `Level: ${level}\n\n${explanation}\n\nDiagram\n${diagram}`
      : `Level: ${level}\n\n${explanation}`;
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
      return;
    }

    for (const paper of papers) {
      const li = document.createElement("li");
      li.textContent = paper.filename;
      list.appendChild(li);
    }
  } catch (err) {
    const li = document.createElement("li");
    li.textContent = `Failed to fetch papers: ${String(err.message || err)}`;
    list.appendChild(li);
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

window.addEventListener("DOMContentLoaded", async () => {
  document.getElementById("upload-form").addEventListener("submit", onUploadSubmit);
  document.getElementById("task-form").addEventListener("submit", onTaskSubmit);
  document.getElementById("refresh-papers").addEventListener("click", refreshPapers);
  document.getElementById("task-type").addEventListener("change", updateTaskInputs);
  updateTaskInputs();
  await refreshPapers();
});
