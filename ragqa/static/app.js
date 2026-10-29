document.addEventListener("DOMContentLoaded", () => {
    "use strict";
    const $ = (id) => document.getElementById(id);
    const form = $("pipeline-form"), query = $("query-input"), samplesList = $("samples-list");
    const number = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 3 });
    const percent = new Intl.NumberFormat("vi-VN", { style: "percent", maximumFractionDigits: 0 });
    const sliders = ["topk", "k1", "b"].map((name) => ({ input: $(`param-${name}`), output: $(`val-${name}`) }));
    const views = ["empty-state", "loading-state", "no-results", "error-state", "results-content"];
    let busy = false, samples = [], showAll = false, completed = null, lastRequest = null, snippet = "";
    let completedLabel = "Đã hoàn tất";

    function node(tag, className, text) {
        const el = document.createElement(tag);
        if (className) el.className = className;
        if (text !== undefined) el.textContent = String(text);
        return el;
    }
    function view(id) { views.forEach((name) => { $(name).hidden = name !== id; }); }
    function payload() {
        return { query: query.value.trim(), top_k: Number($("param-topk").value), k1: Number($("param-k1").value), b: Number($("param-b").value) };
    }
    function equal(a, b) { return Boolean(a && b && a.query === b.query && a.top_k === b.top_k && a.k1 === b.k1 && a.b === b.b); }
    function setBusy(value) {
        busy = value;
        [query, $("btn-run"), $("reset-params"), $("samples-toggle"), $("sample-retry"), $("retry-button"), ...sliders.map((x) => x.input), ...samplesList.querySelectorAll("button")].forEach((el) => { el.disabled = value; });
        $("run-label").textContent = value ? "Đang tìm bằng chứng…" : "Tìm bằng chứng";
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
        $("copy-status").textContent = "";
    }
    function systemStatus(online, text) {
        const state = $("system-state"), wrapper = state.closest(".system-status");
        state.textContent = text;
        state.dataset.state = online ? "online" : "offline";
        if (wrapper) {
            wrapper.classList.toggle("online", online);
            wrapper.classList.toggle("offline", !online);
        }
    }
    function revealResultsOnMobile() {
        if (!window.matchMedia("(max-width: 760px)").matches) return;
        const heading = $("results-heading");
        if (!heading) return;
        heading.focus({ preventScroll: true });
        heading.scrollIntoView({
            behavior: "auto",
            block: "start"
        });
    }
    // Corpus content is always text. Matching tokens become safe <mark> nodes.
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
        const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 15000);
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
            if (error.name === "AbortError") throw new Error("Yêu cầu quá 15 giây. Hãy kiểm tra kết nối và thử lại.");
            if (error instanceof TypeError) throw new Error("Không kết nối được máy chủ. Hãy kiểm tra ứng dụng đang chạy và thử lại.");
            throw error;
        } finally { clearTimeout(timer); }
    }
    async function loadInfo() {
        try {
            const data = await json("/api/pipeline/info"), online = data.status === "online";
            systemStatus(online, online ? "Sẵn sàng" : "Chưa sẵn sàng");
            $("hdr-corpus").textContent = number.format(data.total_chunks ?? 0);
            $("hdr-vocab").textContent = number.format(data.vocabulary_size ?? 0);
        } catch {
            systemStatus(false, "Mất kết nối");
            $("hdr-corpus").textContent = "—";
            $("hdr-vocab").textContent = "—";
        }
    }
    function renderSamples() {
        samplesList.replaceChildren();
        if (!samples.length) samplesList.append(node("p", "muted", "Chưa có câu hỏi mẫu. Bạn vẫn có thể nhập câu hỏi tiếng Anh ở trên."));
        (showAll ? samples : samples.slice(0, 4)).forEach((sample) => {
            const button = node("button", "sample-button");
            button.type = "button";
            button.dataset.question = sample.question;
            button.disabled = busy;
            const questionText = node("span", "sample-question", sample.question), arrow = node("span", "sample-arrow", "↑");
            questionText.lang = "en";
            arrow.setAttribute("aria-hidden", "true");
            button.append(questionText, arrow);
            button.addEventListener("click", () => {
                if (busy) return;
                query.value = sample.question;
                query.setCustomValidity("");
                edited();
                query.focus();
            });
            samplesList.append(button);
        });
        $("samples-toggle").hidden = samples.length <= 4;
        $("samples-toggle").textContent = showAll ? "Thu gọn" : `Xem tất cả ${samples.length} câu hỏi`;
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
            samplesList.replaceChildren(node("p", "muted", "Chưa tải được câu hỏi mẫu. Bạn có thể nhập câu hỏi hoặc tải lại."));
            $("sample-retry").hidden = false;
        }
    }
    function stepper(state, stages = []) {
        for (let id = 1; id <= 4; id++) {
            const stage = stages.find((s) => s.stage_id === id);
            $(`node-stage-${id}`).className = `step-node${state ? ` ${state}` : ""}`;
            $(`lat-stage-${id}`).textContent = stage ? `${number.format(stage.latency_ms)} ms` : state === "active" ? "Đang xử lý" : state === "failed" ? "Chưa hoàn tất" : "Chờ truy vấn";
        }
    }
    function renderChunks(chunks, terms) {
        $("chunks-list").replaceChildren();
        $("chunks-count").textContent = `${chunks.length} đoạn`;
        chunks.forEach((chunk) => {
            const details = node("details", `chunk-item${chunk.rank === 1 ? " top-rank" : ""}`);
            const summary = node("summary"), heading = node("span", "chunk-heading"), content = node("div", "chunk-body");
            heading.append(node("span", "chunk-title", `${chunk.rank}. ${chunk.title || chunk.doc_id || "Tài liệu nguồn"}`), node("span", "chunk-meta", `Tài liệu ${chunk.doc_id ?? "—"} · Đoạn ${chunk.chunk_id ?? "—"}`));
            summary.append(heading, node("span", "chunk-score", `BM25 ${number.format(chunk.score)}`));
            content.lang = "en";
            highlight(content, chunk.text, terms);
            details.append(summary, content);
            $("chunks-list").append(details);
        });
    }
    function renderResult(data, request, elapsed) {
        const result = data.result, stages = data.stages;
        const s1 = stages.find((s) => s.stage_id === 1) || {}, s3 = stages.find((s) => s.stage_id === 3) || {};
        const terms = s1.active_terms || [], chunks = result.top_chunks;
        const latency = number.format(data.telemetry.total_pipeline_latency_ms);
        $("result-query").textContent = request.query;
        $("badge-total-latency").textContent = `${latency} ms xử lý`;
        $("badge-total-latency").title = `Cả yêu cầu: ${number.format(elapsed)} ms, bao gồm kết nối và truyền dữ liệu.`;
        $("pipeline-status-text").textContent = `Hoàn tất · ${latency} ms`;
        stepper("completed", stages);
        $("stale-notice").hidden = true;
        $("copy-status").textContent = "";
        $("technical-details").open = false;
        $("reference-details").open = false;
        const reference = samples.find((s) => s.question.trim() === request.query);
        $("reference-details").hidden = !reference;
        $("reference-answer").textContent = reference?.ground_truth || "Chưa có đáp án tham chiếu.";
        $("reference-doc").textContent = reference ? reference.doc_id || "—" : "";
        completed = { ...request };
        if (!chunks.length) {
            snippet = "";
            completedLabel = "Không có kết quả phù hợp";
            $("result-status").textContent = completedLabel;
            view("no-results");
            return;
        }
        snippet = String(result.top_snippet ?? "");
        highlight($("extracted-span-text"), snippet, terms);
        $("span-source-ref").textContent = `Tài liệu ${result.grounded_doc_id ?? "—"} · Đoạn ${result.grounded_source_chunk ?? "—"}`;
        $("span-sentence-idx").textContent = s3.sentence_index > 0 ? String(s3.sentence_index) : "—";
        $("span-matched-terms").textContent = (s3.matched_terms || []).join(", ") || "Không có";
        const coverage = Number(s3.grounding_ratio);
        $("badge-coverage").textContent = Number.isFinite(coverage) ? `${percent.format(Math.min(1, Math.max(0, coverage)))} độ phủ từ khóa` : "Chưa có độ phủ từ khóa";
        $("term-tags").replaceChildren(...(s1.term_analysis || []).map((item) => node("span", `term-tag${item.is_stopword ? " stopword" : ""}`, `${item.term} · IDF ${number.format(item.idf)}`)));
        $("breakdown-tbody").replaceChildren();
        for (const item of chunks[0].score_breakdown || []) {
            const row = node("tr"), term = node("th", "", item.term);
            term.scope = "row";
            row.append(term, node("td", "", number.format(item.tf)), node("td", "", number.format(item.idf)), node("td", "score-value", number.format(item.term_score)));
            $("breakdown-tbody").append(row);
        }
        renderChunks(chunks, terms);
        completedLabel = "Đã hoàn tất";
        $("result-status").textContent = completedLabel;
        view("results-content");
    }
    async function execute(request = payload()) {
        if (busy) return;
        if (!request.query) {
            query.setCustomValidity("Vui lòng nhập câu hỏi trước khi tìm bằng chứng.");
            query.reportValidity();
            query.focus();
            return;
        }
        query.setCustomValidity("");
        lastRequest = { ...request };
        completed = null;
        setBusy(true);
        view("loading-state");
        $("result-status").textContent = "Đang truy hồi tài liệu…";
        $("pipeline-status-text").textContent = "Đang xử lý truy vấn…";
        $("stale-notice").hidden = true;
        stepper("active");
        try {
            const start = performance.now();
            const data = await json("/api/pipeline/execute", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
            if (!data.result || !Array.isArray(data.result.top_chunks) || !Array.isArray(data.stages) || !data.telemetry) throw new Error("Kết quả từ máy chủ chưa đúng định dạng. Vui lòng thử lại.");
            renderResult(data, request, performance.now() - start);
            if ($("system-state").dataset.state !== "online") loadInfo();
            systemStatus(true, "Sẵn sàng");
        } catch (error) {
            snippet = "";
            $("error-message").textContent = error.message;
            $("result-status").textContent = "Chưa hoàn tất";
            $("pipeline-status-text").textContent = "Truy vấn chưa hoàn tất";
            stepper("failed");
            view("error-state");
            loadInfo();
        } finally {
            setBusy(false);
            revealResultsOnMobile();
        }
    }
    sliders.forEach(({ input, output }) => {
        output.textContent = number.format(Number(input.value));
        input.addEventListener("input", () => { output.textContent = number.format(Number(input.value)); edited(); });
    });
    $("reset-params").addEventListener("click", () => {
        if (busy) return;
        sliders.forEach(({ input, output }, i) => { input.value = String([5, 1.5, 0.75][i]); output.textContent = number.format(Number(input.value)); });
        edited();
    });
    query.addEventListener("input", () => { query.setCustomValidity(""); edited(); });
    query.addEventListener("invalid", () => {
        if (!query.value.trim()) query.setCustomValidity("Vui lòng nhập câu hỏi trước khi tìm bằng chứng.");
    });
    query.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.isComposing) { event.preventDefault(); form.requestSubmit(); }
    });
    form.addEventListener("submit", (event) => { event.preventDefault(); execute(); });
    $("retry-button").addEventListener("click", () => execute(equal(payload(), lastRequest) ? lastRequest : payload()));
    $("samples-toggle").addEventListener("click", () => { if (!busy) { showAll = !showAll; renderSamples(); } });
    $("sample-retry").addEventListener("click", () => { if (!busy) loadSamples(); });
    $("copy-result").addEventListener("click", async () => {
        if (!snippet) return;
        $("copy-result").disabled = true;
        try {
            if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
            await navigator.clipboard.writeText(snippet);
            $("copy-status").textContent = "Đã sao chép đoạn trích.";
        } catch { $("copy-status").textContent = "Chưa sao chép được. Hãy chọn đoạn trích và nhấn Ctrl+C (hoặc ⌘C)."; }
        finally { $("copy-result").disabled = false; }
    });
    view("empty-state");
    loadInfo();
    loadSamples();
});
