document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("pipeline-form");
    const queryInput = document.getElementById("query-input");
    const topkSlider = document.getElementById("param-topk");
    const topkVal = document.getElementById("val-topk");
    const k1Slider = document.getElementById("param-k1");
    const k1Val = document.getElementById("val-k1");
    const bSlider = document.getElementById("param-b");
    const bVal = document.getElementById("val-b");
    
    const samplesList = document.getElementById("samples-list");
    const emptyState = document.getElementById("empty-state");
    const resultsContent = document.getElementById("results-content");
    const stage1Box = document.getElementById("stage1-detail-box");
    const termTags = document.getElementById("term-tags");
    
    const extractedSpanText = document.getElementById("extracted-span-text");
    const spanSourceRef = document.getElementById("span-source-ref");
    const spanSentenceIdx = document.getElementById("span-sentence-idx");
    const spanMatchedTerms = document.getElementById("span-matched-terms");
    const breakdownTbody = document.getElementById("breakdown-tbody");
    const chunksList = document.getElementById("chunks-list");
    
    const badgeConfidence = document.getElementById("badge-confidence");
    const badgeLatency = document.getElementById("badge-total-latency");
    const hdrLatency = document.getElementById("hdr-latency");
    const pipelineStatusText = document.getElementById("pipeline-status-text");

    // Slider value synchronization
    topkSlider.addEventListener("input", (e) => topkVal.textContent = e.target.value);
    k1Slider.addEventListener("input", (e) => k1Val.textContent = e.target.value);
    bSlider.addEventListener("input", (e) => bVal.textContent = e.target.value);

    // Fetch Samples
    async function loadSamples() {
        try {
            const res = await fetch("/api/pipeline/samples");
            const data = await res.json();
            if (data.samples && data.samples.length > 0) {
                samplesList.innerHTML = "";
                data.samples.forEach((s) => {
                    const chip = document.createElement("div");
                    chip.className = "sample-chip";
                    chip.innerHTML = `
                        <span>${s.question}</span>
                        <span class="chip-tag">${s.ground_truth ? s.ground_truth : s.doc_id}</span>
                    `;
                    chip.addEventListener("click", () => {
                        queryInput.value = s.question;
                        executePipeline(s.question);
                    });
                    samplesList.appendChild(chip);
                });
            }
        } catch (err) {
            samplesList.innerHTML = `<div class="loading-text">Không tải được mẫu test: ${err.message}</div>`;
        }
    }

    // Set Stepper State
    function updateStepper(stages) {
        stages.forEach((s) => {
            const node = document.getElementById(`node-stage-${s.stage_id}`);
            const lat = document.getElementById(`lat-stage-${s.stage_id}`);
            if (node && lat) {
                node.className = "step-node completed";
                lat.textContent = `${s.latency_ms} ms`;
            }
        });
    }

    function resetStepper() {
        for (let i = 1; i <= 4; i++) {
            const node = document.getElementById(`node-stage-${i}`);
            const lat = document.getElementById(`lat-stage-${i}`);
            if (node && lat) {
                node.className = "step-node active";
                lat.textContent = "running...";
            }
        }
    }

    // Highlight keywords in document text
    function highlightTerms(text, terms) {
        if (!terms || terms.length === 0) return text;
        let escaped = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
        terms.forEach(term => {
            if (term.length > 1) {
                const regex = new RegExp(`\\b(${term})\\b`, "gi");
                escaped = escaped.replace(regex, `<span class="highlight-kw">$1</span>`);
            }
        });
        return escaped;
    }

    // Execute Pipeline
    async function executePipeline(query) {
        if (!query.trim()) return;
        
        resetStepper();
        pipelineStatusText.textContent = "Executing 4 pipeline stages...";
        
        try {
            const tStart = performance.now();
            const res = await fetch("/api/pipeline/execute", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: query,
                    top_k: parseInt(topkSlider.value),
                    k1: parseFloat(k1Slider.value),
                    b: parseFloat(bSlider.value)
                })
            });

            if (!res.ok) {
                throw new Error(await res.text());
            }

            const data = await res.json();
            const clientLatency = (performance.now() - tStart).toFixed(1);

            // Update Stepper with actual stage latencies
            updateStepper(data.stages);
            pipelineStatusText.textContent = `Completed in ${data.telemetry.total_pipeline_latency_ms} ms (100% Deterministic)`;
            
            // Header stats
            hdrLatency.textContent = `${data.telemetry.total_pipeline_latency_ms} ms`;
            badgeLatency.textContent = `Latency: ${data.telemetry.total_pipeline_latency_ms} ms (API: ${clientLatency}ms)`;
            badgeConfidence.textContent = `Confidence: ${(data.result.confidence * 100).toFixed(0)}%`;

            // Stage 1 Detail
            const s1 = data.stages[0];
            termTags.innerHTML = "";
            s1.term_analysis.forEach(item => {
                const tag = document.createElement("span");
                tag.className = `term-tag ${item.is_stopword ? "stopword" : "keyword"}`;
                tag.textContent = `${item.term} (IDF: ${item.idf})`;
                termTags.appendChild(tag);
            });
            stage1Box.style.display = "block";

            // Stage 3 Span Extractor Result
            const s3 = data.stages[2];
            extractedSpanText.innerHTML = highlightTerms(data.result.top_snippet, s1.active_terms);
            spanSourceRef.textContent = `Chunk #${data.result.grounded_source_chunk} • Doc: ${data.result.grounded_doc_id}`;
            spanSentenceIdx.textContent = s3.sentence_index > 0 ? s3.sentence_index : "1";
            spanMatchedTerms.textContent = s3.matched_terms.join(", ") || "None";

            // Score Breakdown Table
            breakdownTbody.innerHTML = "";
            const topChunk = data.result.top_chunks[0];
            if (topChunk && topChunk.score_breakdown) {
                topChunk.score_breakdown.forEach(item => {
                    const row = document.createElement("tr");
                    row.innerHTML = `
                        <td><strong>${item.term}</strong></td>
                        <td>${item.tf}</td>
                        <td>${item.idf}</td>
                        <td style="color: var(--accent); font-weight: bold;">+${item.term_score}</td>
                    `;
                    breakdownTbody.appendChild(row);
                });
            }

            // Chunks List
            chunksList.innerHTML = "";
            data.result.top_chunks.forEach(chunk => {
                const chunkDiv = document.createElement("div");
                chunkDiv.className = `chunk-item ${chunk.rank === 1 ? "top-rank" : ""}`;
                chunkDiv.innerHTML = `
                    <div class="chunk-item-header">
                        <span class="chunk-badge">#${chunk.rank} &bull; Chunk ID: ${chunk.chunk_id} &bull; Doc: ${chunk.doc_id}</span>
                        <span class="chunk-score">BM25 Score: ${chunk.score}</span>
                    </div>
                    <div class="chunk-body">${highlightTerms(chunk.text, s1.active_terms)}</div>
                `;
                chunksList.appendChild(chunkDiv);
            });

            // Show Results
            emptyState.style.display = "none";
            resultsContent.style.display = "block";

        } catch (err) {
            pipelineStatusText.textContent = `Pipeline Error: ${err.message}`;
            alert(`Lỗi thực thi pipeline: ${err.message}`);
        }
    }

    form.addEventListener("submit", (e) => {
        e.preventDefault();
        executePipeline(queryInput.value);
    });

    // Enter key shortcuts
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            executePipeline(queryInput.value);
        }
    });

    loadSamples();
});
