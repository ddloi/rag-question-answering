'use strict';
const $ = id => document.getElementById(id);
const labels = {correct:'Đúng theo canonical EM',corpus_answer_string_missing:'Thiếu chuỗi đáp án trong corpus',retrieval_answer_string_missing:'Retrieval chưa tìm thấy đáp án',context_selection_or_truncation:'Context bị chọn thiếu / cắt',generation_wrong_with_answer_string_in_context:'Sinh sai dù context có chuỗi đáp án',hybrid_rescue:'Hybrid sửa được lỗi Dense',hybrid_regression:'Hybrid kém hơn Dense',official_canonical_disagreement:'Official và canonical chấm khác nhau'};
let ready=false, busy=false, cases=[], last=null, polling=null;
async function request(url, options={}) {
  const response=await fetch(url,options);
  // Proxies and crashed servers can answer with HTML; never surface a JSON parse error.
  let data=null;
  try { data=await response.json(); } catch { data=null; }
  if(!response.ok) throw new Error(typeof data?.detail==='string'?data.detail:`Máy chủ trả lỗi ${response.status}. Kiểm tra log máy chủ rồi thử lại.`);
  if(data===null) throw new Error('Máy chủ trả dữ liệu không đọc được. Tải lại trang rồi thử lại.');
  return data;
}
function controls() { $('run').disabled=!ready||busy; $('load').disabled=busy||ready; ['question','method','generator','case-filter'].forEach(id=>$(id).disabled=busy); document.querySelectorAll('#cases button').forEach(b=>b.disabled=busy); }
function showCases() {
  const box=$('cases'); box.replaceChildren();
  const selected=cases.filter(c=>$('case-filter').value==='all'||c.category===$('case-filter').value);
  for(const item of selected) { const b=document.createElement('button'); b.type='button'; b.textContent=labels[item.category]||item.category; b.title=item.question; b.addEventListener('click',()=>{$('question').value=item.question; stale(); $('question').focus();}); box.append(b); }
  if(!selected.length) box.textContent='Chưa có run dev đầy đủ cho tình huống này.';
}
async function refresh() {
  try {
    const data=await request('/api/research/info'); ready=data.status==='ready';
    $('status').textContent=data.status==='loading'?'Đang nạp MiniLM, hai T5 và DistilBERT. Chờ trạng thái sẵn sàng…':ready?`Ba hệ sẵn sàng · ${data.device}. Câu hỏi tự nhập không có nhãn sẽ không được tự chấm đúng/sai.`:data.error|| (data.available?'Artifact đã đủ. Nạp ba hệ để bắt đầu.':'Model nghiên cứu chưa hoàn tất. Có thể xem bố cục; chạy suy luận sau khi artifact đã đủ.');
    $('load').disabled=busy||ready||data.status==='loading'||!data.available;
    $('run').disabled=busy||!ready;
    const opt=data.options||{}; const rr=$('method').querySelector('option[value=rerank]'); rr.hidden=rr.disabled=!opt.rerank;
    if(!opt.rerank&&$('method').value==='rerank') $('method').value='dense';
    $('generator-label').hidden=!opt.retrieved_generator; if(!opt.retrieved_generator) $('generator').value='v1';
    const c=data.counts; $('corpus').replaceChildren();
    for(const [key,value] of [['Documents',c.documents],['Chunks',c.chunks],['Ứng viên train / dev / test',c.train?`${c.train} / ${c.dev} / ${c.test}`:null],['QA train thực dùng',data.training_qa]]) { const dt=document.createElement('dt'),dd=document.createElement('dd'); dt.textContent=key; dd.textContent=value==null?'—':typeof value==='number'?value.toLocaleString('vi-VN'):value; $('corpus').append(dt,dd); }
    if(JSON.stringify(cases)!==JSON.stringify(data.cases)) { cases=data.cases; const filter=$('case-filter'); filter.replaceChildren(new Option('Tất cả tình huống','all')); [...new Set(cases.map(c=>c.category))].forEach(k=>filter.add(new Option(labels[k]||k,k))); showCases(); }
    if(data.status==='loading') { if(!polling) polling=setInterval(refresh,3000); }
    else if(polling) { clearInterval(polling); polling=null; }
  } catch(e) { $('status').textContent=`Không đọc được trạng thái: ${e.message} Tải lại trang để thử lại.`; }
}
function stale() { $('stale').hidden=!last||($('question').value.trim()===last.query&&$('method').value===last.retrieval_mode&&$('generator').value===(last.generator||'v1')); }
const methodNames={dense:'Dense',hybrid:'Hybrid RRF',rerank:'Dense + Reranker'};
const rankingNames={dense:'Dense · điểm cosine',hybrid:'Hybrid · điểm RRF',rerank:'Dense + Reranker · điểm cross-encoder'};
function render(data) {
  last=data; $('result-query').hidden=false; $('result-query').textContent=data.query;
  $('rag-label').textContent=`${methodNames[data.retrieval_mode]} + T5 ${data.generator==='retrieved'?'train với context truy hồi':'v1'}`;
  for(const [arm,result] of Object.entries(data.systems)) {
    $(`answer-${arm}`).textContent=result.answer||'Không có span / đáp án đạt điều kiện.';
    const score=$(`score-${arm}`); score.className='score';
    if(result.scores) { const c=result.scores.canonical,o=result.scores.official; score.textContent=`Canonical EM ${(100*c.em).toFixed(0)}% · F1 ${(100*c.f1).toFixed(1)}% | Official EM ${(100*o.em).toFixed(0)}% · F1 ${(100*o.f1).toFixed(1)}%`; score.classList.add(c.em?'correct':'incorrect'); }
    else score.textContent='Chưa có nhãn dev để tính EM/F1.';
    if(arm!=='generator_only') {
      const container=$(`context-${arm}`); container.replaceChildren();
      for(const p of result.used_passages) {
        const section=document.createElement('div'),name=document.createElement('strong'),text=document.createElement('p');
        name.textContent=`${(p.visible_heading||p.title).trim()} · ${p.chunk_id}`;
        text.textContent=p.text||'(Chỉ thấy phần tiêu đề trước giới hạn token)';
        section.append(name,text); container.append(section);
      }
      if(!result.used_passages.length) container.textContent='Không có context thực dùng.';
    }
  }
  $('reference').textContent=data.reference?`Canonical: ${data.reference.answer}. Split: dev. Official aliases: ${data.reference.official_answer_aliases.join('; ')}.`:'Câu hỏi tự nhập chưa có nhãn dev. Không suy ra đúng/sai từ điểm retrieval.';
  if(data.reference?.audit_note) $('reference').textContent=`Lưu ý: ${data.reference.audit_note} ${data.reference.audit_source}. ${$('reference').textContent}`;
  const sources=$('sources'); sources.replaceChildren();
  $('ranking-label').textContent=rankingNames[data.retrieval_mode];
  const used=new Set(data.systems.rag.used_passages.map(p=>p.chunk_id));
  for(const p of data.retrieval[data.retrieval_mode]) { const d=document.createElement('details'),s=document.createElement('summary'),id=document.createElement('code'),text=document.createElement('p'); s.textContent=`${p.rank}. ${p.title} · ${p.score.toFixed(4)}${used.has(p.chunk_id)?' · T5 đã thấy một phần nguồn này':''}`; id.textContent=p.chunk_id; text.textContent=p.text; d.append(s,id,text); sources.append(d); }
  stale();
}
$('load').addEventListener('click',async()=>{ $('load').disabled=true; try{await request('/api/research/load',{method:'POST'}); await refresh();}catch(e){$('status').textContent=e.message;$('load').disabled=false;} });
$('query-form').addEventListener('submit',async event=>{
  event.preventDefault();
  if(busy||!ready||!$('question').value.trim())return;
  busy=true; controls(); $('error').hidden=true; $('results').classList.add('loading');
  $('latency').textContent='Đang chạy ba hệ…';
  $('status').textContent=last?'Đang chạy câu hỏi mới. Kết quả trước chưa được cập nhật.':'Đang chạy ba hệ cho câu hỏi này…';
  const start=performance.now();
  try {
    const data=await request('/api/research/execute',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:$('question').value.trim(),retrieval_mode:$('method').value,generator:$('generator').value})});
    render(data);
    $('latency').textContent=`Toàn bộ yêu cầu: ${((performance.now()-start)/1000).toFixed(1)} giây`;
    $('status').textContent='Đã chạy đủ ba hệ. Mở context và passage để kiểm tra nguồn.';
    if(innerWidth<680)$('result-title').focus();
  } catch(e) {
    $('error').hidden=false; $('error').textContent=e.message;
    $('latency').textContent='Yêu cầu chưa hoàn tất';
    $('status').textContent='Yêu cầu chưa hoàn tất. Kiểm tra thông báo lỗi và thử lại.';
  } finally {
    busy=false; $('results').classList.remove('loading'); controls();
  }
});
$('question').addEventListener('input',stale);$('method').addEventListener('change',stale);$('generator').addEventListener('change',stale);$('case-filter').addEventListener('change',showCases);
$('question').addEventListener('keydown',event=>{if(event.key==='Enter'&&(event.ctrlKey||event.metaKey)){event.preventDefault();$('query-form').requestSubmit();}});
showCases();refresh();
