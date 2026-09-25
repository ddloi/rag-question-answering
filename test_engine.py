from pipeline_engine import DeterministicPipelineEngine

e = DeterministicPipelineEngine()
res = e.execute_pipeline('director of Jaws')
print(Telemetry:, res[telemetry])
print(Top snippet:, res[result][top_snippet])
print(Grounding Doc ID:, res[result][grounded_doc_id])
print(Score breakdown for rank 1:)
for item in res[result][top_chunks][0][score_breakdown]:
    print( , item)
