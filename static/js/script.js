(function () {
  const textarea = document.getElementById("posting-input");
  const charCount = document.getElementById("char-count");
  const analyzeBtn = document.getElementById("analyze-btn");
  const errorMsg = document.getElementById("error-msg");

  const verdictSection = document.getElementById("verdict-section");
  const verdictStamp = document.getElementById("verdict-stamp");
  const verdictText = document.getElementById("verdict-text");
  const confidenceFill = document.getElementById("confidence-fill");
  const confidenceValue = document.getElementById("confidence-value");
  const fraudProbValue = document.getElementById("fraud-prob-value");
  const reasoningText = document.getElementById("reasoning-text");
  const redFlagsList = document.getElementById("red-flags-list");
  const evidenceTags = document.getElementById("evidence-tags");
  const annotatedText = document.getElementById("annotated-text");

  textarea.addEventListener("input", () => {
    charCount.textContent = `${textarea.value.length.toLocaleString()} characters`;
  });

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function highlightText(rawText, terms) {
    // Escape first, then wrap whole-word (case-insensitive) matches of each
    // term/phrase in <mark>. Longer phrases go first so they win over
    // shorter substrings they might contain.
    let html = escapeHtml(rawText);
    const sorted = [...terms].sort((a, b) => b.length - a.length);

    sorted.forEach((term) => {
      if (!term || term.length < 2) return;
      const escapedTerm = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
      const re = new RegExp(`(?<![\\w<])(${escapedTerm})(?![\\w>])`, "gi");
      html = html.replace(re, (match) => `<mark>${match}</mark>`);
    });

    return html;
  }

  async function analyze() {
    const text = textarea.value.trim();
    errorMsg.hidden = true;

    if (text.length < 30) {
      errorMsg.textContent = "Paste the full posting first \u2014 need at least a few sentences to work with.";
      errorMsg.hidden = false;
      return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.querySelector(".stamp-btn-label").textContent = "Examining\u2026";

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Something went wrong.");
      }

      renderVerdict(data, text);
    } catch (err) {
      errorMsg.textContent = err.message || "Couldn't reach the examiner. Is the server running?";
      errorMsg.hidden = false;
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.querySelector(".stamp-btn-label").textContent = "Examine Posting";
    }
  }

  function renderVerdict(data, rawText) {
    const isFraud = data.label === "fraudulent";

    // restart stamp animation
    verdictStamp.classList.remove("fraud");
    verdictStamp.style.animation = "none";
    void verdictStamp.offsetWidth;
    verdictStamp.style.animation = "";

    verdictText.textContent = isFraud ? "FRAUDULENT SCAM" : "LEGITIMATE POSTING";
    if (isFraud) verdictStamp.classList.add("fraud");

    confidenceFill.style.width = "0%";
    requestAnimationFrame(() => {
      confidenceFill.style.width = `${data.confidence}%`;
    });
    confidenceValue.textContent = `${data.confidence}%`;
    fraudProbValue.textContent = `${data.fraud_probability}%`;

    reasoningText.textContent = data.reasoning || "No additional notes provided.";

    redFlagsList.innerHTML = "";
    const flags = data.red_flags || [];
    if (flags.length === 0) {
      const li = document.createElement("li");
      li.className = "no-flags";
      li.textContent = "No specific red flags identified in this posting.";
      redFlagsList.appendChild(li);
    } else {
      flags.forEach(({ reason, quote }) => {
        const li = document.createElement("li");
        const reasonEl = document.createElement("span");
        reasonEl.className = "flag-reason";
        reasonEl.textContent = reason;
        li.appendChild(reasonEl);
        if (quote) {
          const quoteEl = document.createElement("span");
          quoteEl.className = "flag-quote";
          quoteEl.textContent = `"${quote}"`;
          li.appendChild(quoteEl);
        }
        redFlagsList.appendChild(li);
      });
    }

    const terms = (data.top_terms || []).map((t) => t.term);

    evidenceTags.innerHTML = "";
    if (terms.length === 0) {
      evidenceTags.innerHTML = `<span class="index-card-footnote">No specific terms singled out for this posting.</span>`;
    } else {
      terms.forEach((term) => {
        const tag = document.createElement("span");
        tag.className = "evidence-tag";
        tag.textContent = term;
        evidenceTags.appendChild(tag);
      });
    }

    const highlightTargets = [...terms, ...flags.map((f) => f.quote).filter(Boolean)];
    annotatedText.innerHTML = highlightText(rawText, highlightTargets);

    verdictSection.hidden = false;
    verdictSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  analyzeBtn.addEventListener("click", analyze);
  textarea.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
  });
})();
