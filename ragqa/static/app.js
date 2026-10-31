document.addEventListener("DOMContentLoaded", () => {
    "use strict";
    const $ = (id) => document.getElementById(id);
    const form = $("pipeline-form"), query = $("query-input"), samplesList = $("samples-list");
    const runRagCheckbox = $("param-run-rag");
    const number = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 3 });
    const percent = new Intl.NumberFormat("vi-VN", { style: "percent", maximumFractionDigits: 0 });
    const sliders = ["topk", "k1", "b"].map((name) => ({ input: $(`param-${name}`), output: $(`val-${name}`) }));
    const views = ["empty-state", "loading-state", "no-results", "error-state", "results-content"];
    let busy = false, samples = [], showAll = false, completed = null, lastRequest = null;
    let completedLabel = "Đã hoàn tất";
    let infoTimer = null;

    function node(tag, className, text) {
        const el = document.createElement(tag);
        if (className) el.className = className;
        if (text !== undefined) el.textContent = String(text);
        return el;
    }

    function view(id) {
        views.forEach((name) => {
            const el = $(name);
            if (el) el.hidden = name !== id;
        });
    }

    function payload() {
        return {
            query: query.value.trim(),
            top_k: Number($("param-topk").value),
            k1: Number($("param-k1").value),
            b: Number($("param-b").value),
            run_rag: runRagCheckbox ? runRagCheckbox.checked : true,
            retrieval_mode: $("param-retrieval").value
        };
    }

    function equal(a, b) {
        return Boolean(a && b && a.query === b.query && a.top_k === b.top_k && a.k1 === b.k1 && a.b === b.b && a.run_rag === b.run_rag && a.retrieval_mode === b.retrieval_mode);
    }

    function setBusy(value) {
        busy = value;
        const disableList = [
            query, $("btn-run"), $("reset-params"), $("samples-toggle"), $("sample-retry"), $("retry-button"),
            runRagCheckbox, $("param-retrieval"), $("load-rag-button"), $("sample-filter"), ...sliders.map((x) => x.input), ...samplesList.querySelectorAll("button")
        ].filter(Boolean);
        disableList.forEach((el) => { el.disabled = value; });
        $("run-label").textContent = value ? "Đang xử lý…" : "So sánh & Tìm bằng chứng";
        $("results-panel").setAttribute("aria-busy", String(value));
    }

    function updateSelection() {
        samplesList.querySelectorAll("button[data-question]").forEach((button) => {
            const selected = button.dataset.question === query.value.trim();
            button.classList.toggle("is-selected", selected);
            button.setAttribute("aria-pressed", String(selected));
        });
    }

    function edited() {
        updateSelection();
        const stale = Boolean(completed && !equal(payload(), completed));
        $("stale-notice").hidden = !stale;
        if (completed) $("result-status").textContent = stale ? "Kết quả của lần tìm trước" : completedLabel;
    }

    function highlight(target, text, terms) {
        const value = String(text ?? ""), keywords = new Set((terms || []).map((x) => String(x).toLowerCase()));
        const fragment = document.createDocumentFragment();
        let cursor = 0;
        for (const match of value.matchAll(/\b[a-zA-Z0-9_\-']+\b/g)) {
            fragment.append(document.createTextNode(value.slice(cursor, match.index)));
            fragment.append(keywords.has(match[0].toLowerCase()) ? node("mark", "keyword", match[0]) : document.createTextNode(match[0]));
            cursor = match.index + match[0].length;
        }
        fragment.append(document.createTextNode(value.slice(cursor)));
        target.replaceChildren(fragment);
    }

    async function json(url, options = {}) {
        const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 60000);
        try {
            const response = await fetch(url, { ...options, signal: controller.signal });
            let data;
            try { data = await response.json(); }
            catch { throw new Error("Máy chủ trả về dữ liệu không hợp lệ. Vui lòng thử lại."); }
            if (!response.ok) {
                if ([400, 422].includes(response.status)) throw new Error("Câu hỏi hoặc tham số chưa hợp lệ. Hãy kiểm tra lại rồi tìm lần nữa.");
                throw new Error("Máy chủ chưa xử lý được yêu cầu. Hãy kiểm tra ứng dụng đang chạy và thử lại.");
            }
            return data;
        } catch (error) {
            if (error.name === "AbortError") throw new Error("Yêu cầu quá thời gian chờ (60s). Hãy thử lại.");
            if (error instanceof TypeError) throw new Error("Không kết nối được máy chủ. Hãy kiểm tra ứng dụng đang chạy và thử lại.");
            throw error;
        } finally { clearTimeout(timer); }
    }

    async function loadInfo() {
        clearTimeout(infoTimer);
        try {
            const data = await json("/api/pipeline/info");
            $("hdr-corpus").textContent = number.format(data.total_chunks ?? 0);
            $("hdr-vocab").textContent = number.format(data.vocabulary_size ?? 0);
            $("bm25-state").textContent = `${number.format(data.total_chunks ?? 0)} chunks`;
            $("retrieval-help").textContent = "Hybrid kết hợp xếp hạng và ưu tiên từ khóa hiếm. Chọn Dense để đối chiếu."
                + (data.demo_only_corpus ? " Corpus demo có nguồn bổ sung; không dùng để chấm benchmark." : "");

            const ragDot = $("rag-dot");
            const ragState = $("rag-state");
            const ragHint = $("rag-engine-hint");
            const hdrDevice = $("hdr-device");
            const loadButton = $("load-rag-button");
            loadButton.hidden = data.rag_status === "ready" || data.rag_status === "loading" || !data.rag_available;
            loadButton.disabled = busy;
            loadButton.textContent = data.rag_status === "error" ? "Thử nạp lại RAG" : "Nạp mô hình RAG";

            if (data.rag_status === "ready") {
                ragDot.className = "status-dot online";
                ragState.textContent = `Sẵn sàng (${data.rag_device || "CUDA"})`;
                ragHint.textContent = `Mô hình nơ-ron hoạt động trên ${data.rag_device || "CUDA"}`;
                hdrDevice.textContent = `Thiết bị: ${data.rag_device || "CUDA"}`;
            } else if (data.rag_status === "loading") {
                ragDot.className = "status-dot loading";
                ragState.textContent = "Đang nạp mô hình…";
                ragHint.textContent = "Đang tải T5 và FAISS vào bộ nhớ…";
                hdrDevice.textContent = "Thiết bị: Đang khởi động…";
                infoTimer = setTimeout(loadInfo, 3000);
            } else if (data.rag_status === "error") {
                ragDot.className = "status-dot offline";
                ragState.textContent = "Nạp mô hình thất bại";
                ragHint.textContent = data.rag_error || "Kiểm tra môi trường ML và mô hình local, sau đó thử nạp lại.";
                hdrDevice.textContent = "Thiết bị: RAG chưa sẵn sàng";
            } else {
                ragDot.className = "status-dot offline";
                ragState.textContent = data.rag_available ? "Sẵn sàng nạp khi chạy" : "Chưa cài đặt";
                ragHint.textContent = data.rag_available ? "Nạp mô hình trước khi so sánh. BM25 có thể chạy ngay." : "Thiếu index, metadata hoặc checkpoint T5. Bạn vẫn có thể dùng BM25.";
                hdrDevice.textContent = `Thiết bị: ${data.rag_device || "N/A"}`;
            }
        } catch {
            $("bm25-state").textContent = "Mất kết nối";
            $("rag-state").textContent = "Mất kết nối";
            $("hdr-corpus").textContent = "—";
            $("hdr-vocab").textContent = "—";
        }
    }

    function renderSamples() {
        samplesList.replaceChildren();
        if (!samples.length) {
            samplesList.append(node("p", "muted", "Chưa có câu hỏi mẫu. Bạn có thể nhập câu hỏi tiếng Anh ở trên."));
            return;
        }
        const filter = $('sample-filter').value;
        const filtered = filter === 'all' ? samples : samples.filter((sample) => sample.category === filter);
        (showAll ? filtered : filtered.slice(0, 5)).forEach((sample) => {
            const button = node("button", "sample-button");
            button.type = "button";
            button.dataset.question = sample.question;
            button.disabled = busy;
            const questionText = node("span", "sample-question", sample.question);
            const arrow = node("span", "sample-arrow", "↑");
            questionText.lang = "en";
            arrow.setAttribute("aria-hidden", "true");
            button.append(questionText, arrow);
            if (sample.case_note) button.append(node('span', 'sample-case-note', sample.case_note));
            button.addEventListener("click", () => {
                if (busy) return;
                query.value = sample.question;
                query.setCustomValidity("");
                edited();
                query.focus();
            });
            samplesList.append(button);
        });
        $("samples-toggle").hidden = filtered.length <= 5;
        $("samples-toggle").textContent = showAll ? "Thu gọn câu hỏi" : `Xem tất cả ${filtered.length} câu hỏi mẫu`;
        $("samples-toggle").setAttribute("aria-expanded", String(showAll));
        updateSelection();
    }

    async function loadSamples() {
        $("sample-retry").hidden = true;
        $("samples-toggle").hidden = true;
        samplesList.replaceChildren(node("p", "muted", "Đang tải câu hỏi mẫu…"));
        try {
            const data = await json("/api/pipeline/samples");
            samples = (Array.isArray(data.samples) ? data.samples : []).filter((s) => typeof s.question === "string" && s.question.trim());
            renderSamples();
        } catch {
            samplesList.replaceChildren(node("p", "muted", "Chưa tải được câu hỏi mẫu."));
            $("sample-retry").hidden = false;
        }
    }

    function stepper(state, stages = []) {
        for (let id = 1; id <= 4; id++) {
            const stage = stages.find((s) => s.stage_id === id);
            const el = $(`node-stage-${id}`);
            const lat = $(`lat-stage-${id}`);
            if (el) el.className = `step-node${state ? ` ${state}` : ""}`;
            if (lat) {
                lat.textContent = stage ? `${number.format(stage.latency_ms)} ms` : (state === "active" ? "Đang xử lý" : (state === "failed" ? "Lỗi" : "—"));
            }
        }
    }

    function renderChunkItem(chunk, terms, isDense = false) {
        const details = node("details", `chunk-item${chunk.rank === 1 ? " top-rank" : ""}`);
        const summary = node("summary");
        const heading = node("span", "chunk-heading");
        const content = node("div", "chunk-body");

        heading.append(
            node("span", "chunk-title", `${chunk.rank}. ${chunk.title || chunk.doc_id || "Tài liệu nguồn"}`),
            node("span", "chunk-meta", `Tài liệu ${chunk.doc_id ?? "—"} · Đoạn ${chunk.chunk_id ?? "—"}`)
        );

        const scoreLabel = isDense
            ? `Cosine: ${number.format(chunk.score)}${chunk.matched_anchor ? ' · Khớp từ hiếm' : ''}`
            : `BM25: ${number.format(chunk.score)}`;
        summary.append(heading, node("span", "chunk-score", scoreLabel));
        content.lang = "en";
        highlight(content, chunk.text, terms);
        if (typeof chunk.source_url === 'string' && /^https:\/\/en\.wikipedia\.org\//.test(chunk.source_url)) {
            const source = node('a', 'chunk-source-link', 'Xem tài liệu gốc trên Wikipedia');
            source.href = chunk.source_url;
            source.target = '_blank';
            source.rel = 'noopener noreferrer';
            content.append(node('br'), source);
        }
        details.append(summary, content);
        return details;
    }

    function renderResult(data, request, elapsed) {
        const result = data.result, stages = data.stages, rag = data.rag;
        const s1 = stages.find((s) => s.stage_id === 1) || {}, s3 = stages.find((s) => s.stage_id === 3) || {};
        const terms = s1.active_terms || [];
        const bm25Chunks = result.top_chunks || [];
        const faissChunks = (rag && rag.retrieved_chunks) ? rag.retrieved_chunks : [];
        const latency = number.format(data.request_latency_ms ?? data.telemetry.total_pipeline_latency_ms);
        const hybrid = rag?.retrieval_method === 'hybrid_rrf_anchor';
        const method = hybrid ? "Hybrid" : "FAISS Dense";
        $("rag-method-label").textContent = `${method} + T5-base`;
        $("neural-source-label").textContent = method;
        $("neural-source-help").textContent = (hybrid
            ? `Xếp hạng kết hợp Dense và BM25 (RRF)${rag.anchor_term ? `, ưu tiên từ hiếm “${rag.anchor_term}”` : ''}. Điểm hiển thị là Cosine, không phải điểm xếp hạng kết hợp.`
            : "Xếp hạng theo Cosine của FAISS.")
            + " T5 dùng tối đa hai đoạn đầu, giới hạn 512 token. Độ tương đồng không bảo đảm đúng đáp án.";

        $("result-query").textContent = request.query;
        $("badge-total-latency").textContent = `Máy chủ: ${latency} ms · Toàn yêu cầu: ${number.format(elapsed)} ms`;
        $("badge-device").textContent = rag?.available ? (rag.device_name || "CPU") : "BM25: CPU";

        if (rag && rag.latency) {
            $("badge-rag-latency").hidden = false;
            $("badge-rag-latency").textContent = `T5 Sinh câu: ${number.format(rag.latency.generation_ms)} ms`;
        } else {
            $("badge-rag-latency").hidden = true;
        }

        const coverage = Number(s3.grounding_ratio);
        $("badge-coverage").textContent = Number.isFinite(coverage) ? `${percent.format(Math.min(1, Math.max(0, coverage)))} phủ từ BM25` : "—";
        $("pipeline-status-text").textContent = `BM25 · ${number.format(data.telemetry.total_pipeline_latency_ms)} ms`;
        stepper("completed", stages);
        $("stale-notice").hidden = true;

        // 1. KHỐI RAG
        if (rag && rag.available) {
            $("rag-answer-box").textContent = rag.answer || "Không thể sinh đáp án.";
            $("rag-gen-only-box").textContent = rag.generator_only_answer || "—";
            $("rag-extractive-box").textContent = rag.extractive_answer || (rag.extractive_available ? "Không có span đáp án đạt ngưỡng" : "Chưa nạp mô hình DistilBERT");
            $("rag-explanation").textContent = rag.missing_terms?.length
                ? `Cảnh báo: kho không có từ khóa ${rag.missing_terms.join(', ')}. T5 vẫn sinh đáp án, nhưng cần kiểm tra nguồn và đáp án tham chiếu.`
                : `Cosine ${number.format(rag.top_score)} · Ngưỡng ${number.format(rag.relevance_threshold ?? 0.25)}. Độ tương đồng không bảo đảm đáp án đúng.`;
            $("rag-latency-stat").textContent = `⏱ T5: ${number.format(rag.latency.generation_ms)} ms`;
            $("rag-retrieval-stat").textContent = `${method}: ${number.format(rag.latency.retrieval_ms)} ms`;
        } else {
            $("rag-answer-box").textContent = request.run_rag ? (rag?.error || "RAG chưa sẵn sàng.") : "RAG đã tắt cho truy vấn này.";
            $("rag-explanation").textContent = rag?.is_loading ? "Chờ trạng thái Sẵn sàng, sau đó chạy lại câu hỏi để so sánh." : "Kết quả BM25 vẫn có thể kiểm tra độc lập.";
            $("rag-gen-only-box").textContent = "—";
            $("rag-extractive-box").textContent = "—";
            $("rag-latency-stat").textContent = "—";
            $("rag-retrieval-stat").textContent = "—";
        }

        // 2. KHỐI BM25
        if (bm25Chunks.length) {
            const snippet = String(result.top_snippet ?? "");
            highlight($("bm25-answer-box"), snippet, terms);
            $("bm25-source-ref").textContent = `Tài liệu ${result.grounded_doc_id ?? "—"} · Đoạn ${result.grounded_source_chunk ?? "—"}`;
            $("bm25-coverage-stat").textContent = `📊 Phủ từ: ${percent.format(Math.min(1, Math.max(0, coverage)))}`;
            $("bm25-latency-stat").textContent = `⏱ BM25: ${number.format(stages.find((s) => s.stage_id === 2)?.latency_ms ?? 0)} ms`;
        } else {
            $("bm25-answer-box").textContent = "Không tìm thấy đoạn phù hợp.";
            $("bm25-source-ref").textContent = "—";
            $("bm25-coverage-stat").textContent = "—";
            $("bm25-latency-stat").textContent = "—";
        }

        // 3. KHỐI GROUND TRUTH
        const reference = data.reference || samples.find((s) => s.question.trim().toLowerCase() === request.query.trim().toLowerCase());
        const matchBadge = $("gt-match-badge");
        if (reference) {
            const gtAnswer = reference.ground_truth || "";
            $("gt-answer-box").textContent = gtAnswer;
            $("gt-doc-box").textContent = reference.doc_id || "—";
            
            // This is only a normalized text comparison, not a dataset EM score.
            const ragAnsNorm = (rag?.answer || "").trim().toLowerCase();
            const gtNorm = gtAnswer.trim().toLowerCase();
            if (!rag?.available || !rag.is_grounded) {
                matchBadge.textContent = "Chưa đối chiếu T5";
                matchBadge.className = "model-badge lexical-badge";
            } else if (ragAnsNorm && gtNorm && ragAnsNorm === gtNorm) {
                matchBadge.textContent = "Khớp chuỗi đáp án";
                matchBadge.className = "model-badge match-success";
            } else {
                matchBadge.textContent = "Khác chuỗi tham chiếu";
                matchBadge.className = "model-badge match-diff";
            }
        } else {
            $("gt-answer-box").textContent = "Câu hỏi tự nhập (Chưa có nhãn test)";
            $("gt-doc-box").textContent = "—";
            matchBadge.textContent = "Tự nhập";
            matchBadge.className = "model-badge lexical-badge";
        }

        // 4. DANH SÁCH CHUNKS: FAISS & BM25
        const faissList = $("faiss-chunks-list");
        faissList.replaceChildren();
        $("faiss-chunks-count").textContent = String(faissChunks.length);
        if (faissChunks.length) {
            faissChunks.forEach((c) => faissList.append(renderChunkItem(c, terms, true)));
        } else {
            faissList.append(node("p", "muted", "Không có đoạn truy hồi FAISS."));
        }

        const bm25List = $("bm25-chunks-list");
        bm25List.replaceChildren();
        $("bm25-chunks-count").textContent = String(bm25Chunks.length);
        if (bm25Chunks.length) {
            bm25Chunks.forEach((c) => bm25List.append(renderChunkItem(c, terms, false)));
        } else {
            bm25List.append(node("p", "muted", "Không có đoạn truy hồi BM25."));
        }
        selectSource(faissChunks.length ? "faiss" : "bm25");

        // 5. TECHNICAL DETAILS
        $("span-matched-terms").textContent = (s3.matched_terms || []).join(", ") || "Không có";
        $("term-tags").replaceChildren(...(s1.term_analysis || []).map((item) =>
            node("span", `term-tag${item.is_stopword ? " stopword" : ""}`, `${item.term} · IDF ${number.format(item.idf)}`)
        ));

        $("breakdown-tbody").replaceChildren();
        if (bm25Chunks.length && bm25Chunks[0].score_breakdown) {
            for (const item of bm25Chunks[0].score_breakdown) {
                const row = node("tr"), term = node("th", "", item.term);
                term.scope = "row";
                row.append(term, node("td", "", number.format(item.tf)), node("td", "", number.format(item.idf)), node("td", "score-value", number.format(item.term_score)));
                $("breakdown-tbody").append(row);
            }
        }

        completed = { ...request };
        completedLabel = rag?.available ? "Đã hoàn tất so sánh" : "Đã có kết quả BM25";
        $("result-status").textContent = completedLabel;
        view("results-content");
        if (window.matchMedia("(max-width: 900px)").matches) $("results-heading").focus();
    }

    async function execute(request = payload()) {
        if (busy) return;
        if (!request.query) {
            query.setCustomValidity("Vui lòng nhập câu hỏi trước khi tìm.");
            query.reportValidity();
            query.focus();
            return;
        }
        query.setCustomValidity("");
        lastRequest = { ...request };
        completed = null;
        setBusy(true);
        view("loading-state");
        $("result-status").textContent = request.run_rag ? "Đang xử lý RAG & BM25…" : "Đang truy hồi BM25…";
        $("loading-text").textContent = request.run_rag ? "Đang truy hồi và đối chiếu; thời gian phụ thuộc CPU/GPU." : "Đang tìm đoạn nguồn theo từ khóa BM25…";
        $("pipeline-status-text").textContent = "Đang xử lý truy vấn…";
        $("stale-notice").hidden = true;
        stepper("active");

        try {
            const start = performance.now();
            const data = await json("/api/pipeline/execute", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(request)
            });
            if (!data.result || !Array.isArray(data.stages)) throw new Error("Kết quả từ máy chủ chưa đúng định dạng.");
            renderResult(data, request, performance.now() - start);
            loadInfo();
        } catch (error) {
            $("error-message").textContent = error.message;
            $("result-status").textContent = "Chưa hoàn tất";
            $("pipeline-status-text").textContent = "Truy vấn chưa hoàn tất";
            stepper("failed");
            view("error-state");
            loadInfo();
        } finally {
            setBusy(false);
        }
    }

    // Source Tabs handling
    const tabFaiss = $("tab-btn-faiss");
    const tabBm25 = $("tab-btn-bm25");
    const paneFaiss = $("tab-pane-faiss");
    const paneBm25 = $("tab-pane-bm25");

    function selectSource(name) {
        const faiss = name === "faiss";
        for (const [tab, selected] of [[tabFaiss, faiss], [tabBm25, !faiss]]) {
            tab.classList.toggle("active", selected);
            tab.setAttribute("aria-selected", String(selected));
            tab.tabIndex = selected ? 0 : -1;
        }
        paneFaiss.hidden = !faiss;
        paneBm25.hidden = faiss;
    }

    if (tabFaiss && tabBm25 && paneFaiss && paneBm25) {
        tabFaiss.addEventListener("click", () => selectSource("faiss"));
        tabBm25.addEventListener("click", () => selectSource("bm25"));
        [tabFaiss, tabBm25].forEach((tab) => tab.addEventListener("keydown", (event) => {
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
            event.preventDefault();
            const next = event.key === "Home" ? tabFaiss : event.key === "End" ? tabBm25 : tab === tabFaiss ? tabBm25 : tabFaiss;
            selectSource(next === tabFaiss ? "faiss" : "bm25");
            next.focus();
        }));
    }

    runRagCheckbox.addEventListener("change", edited);
    $("param-retrieval").addEventListener("change", edited);
    $("load-rag-button").addEventListener("click", async () => {
        $("load-rag-button").disabled = true;
        try {
            await json("/api/rag/load", { method: "POST" });
            await loadInfo();
        } catch (error) {
            $("rag-engine-hint").textContent = error.message;
        } finally { $("load-rag-button").disabled = false; }
    });

    sliders.forEach(({ input, output }) => {
        output.textContent = number.format(Number(input.value));
        input.addEventListener("input", () => {
            output.textContent = number.format(Number(input.value));
            edited();
        });
    });

    $("reset-params").addEventListener("click", () => {
        if (busy) return;
        sliders.forEach(({ input, output }, i) => {
            input.value = String([5, 1.5, 0.75][i]);
            output.textContent = number.format(Number(input.value));
        });
        edited();
    });

    query.addEventListener("input", () => { query.setCustomValidity(""); edited(); });
    query.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.isComposing) {
            event.preventDefault();
            form.requestSubmit();
        }
    });

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        execute();
    });

    $("retry-button").addEventListener("click", () => execute(equal(payload(), lastRequest) ? lastRequest : payload()));
    $("samples-toggle").addEventListener("click", () => { if (!busy) { showAll = !showAll; renderSamples(); } });
    $('sample-filter').addEventListener('change', () => { showAll = false; renderSamples(); });
    $("sample-retry").addEventListener("click", () => { if (!busy) loadSamples(); });

    view("empty-state");
    loadInfo();
    loadSamples();
});
