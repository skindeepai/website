"""Separate paired timing of old/new tiny models; no test-driven model selection."""
from chat_refinement import *


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    p,old,splits,qt,bt=load()
    path=OUT/'tiny-timing-protocol.json'
    if not path.exists():
        write(path.name,dict(source_sha256=sha(Path(__file__)),parent_protocol_sha256=sha(OUT/'protocol.json'),
            fit_sha256=sha(OUT/'fit.json'),paths=['old_bert','new_bert'],repetitions=3,
            purpose='Additional timing for BOTH standalone tiny variants, regardless of their fresh quality. No model/threshold selection.',
            boundary='Same raw text, bounding tokenizer, BERT input, two layers, head, softmax and trace hooks as main study; loading/warmup excluded.',
            comparison='Separate paired timing of tiny models; not paired with the full Qwen timing run.'))
        print('Prepared standalone tiny timing protocol.');return
    guard('tiny-benchmark.json')
    sealed=json.loads(path.read_text(encoding='utf-8'))
    assert sealed['source_sha256']==sha(Path(__file__))
    assert sealed['parent_protocol_sha256']==sha(OUT/'protocol.json')
    assert sealed['fit_sha256']==sha(OUT/'fit.json')
    expected={(r['id'],r['path']):r for r in read('results/chat-refinement/predictions.json')}
    model_old=load_weights(Specialist(),ROOT/'results/chat-smoke-specialist/specialist.npz')
    model_new=load_weights(Specialist(),OUT/'specialist.npz')
    oldfit=read('results/chat-smoke-specialist/result.json');newfit=read('results/chat-refinement/fit.json')
    def execute(row,name):
        start=time.perf_counter();text=bounded(row['text'],qt)
        model=model_old if name=='old_bert' else model_new
        trace=[]
        hooks=[layer.register_forward_hook(lambda m,i,o,d=d:trace.append(d)) for d,layer in enumerate(model.encoder.encoder.layer,1)]
        try:prob=float(model(bt(text,return_tensors='pt',truncation=True,max_length=512)).softmax(1)[0,1])
        finally:
            for hook in hooks:hook.remove()
        threshold=oldfit['decision_threshold'] if name=='old_bert' else newfit['standalone_threshold']
        prediction=int(prob>=threshold);seconds=time.perf_counter()-start
        assert trace==[1,2] and prediction==expected[(row['id'],name)]['prediction']
        return dict(id=row['id'],path=name,prediction=prediction,bert_layers=trace,seconds=seconds)
    records=[]
    with torch.inference_mode():
        for name in sealed['paths']:execute(splits['evaluation'][0],name)
        for rep in range(3):
            for i,row in enumerate(splits['evaluation']):
                paths=sealed['paths'] if (rep+i)%2==0 else list(reversed(sealed['paths']))
                for name in paths:records.append(dict(repetition=rep,**execute(row,name)))
    totals={name:[sum(r['seconds'] for r in records if r['path']==name and r['repetition']==rep) for rep in range(3)] for name in sealed['paths']}
    write('tiny-timings.json',records)
    write('tiny-benchmark.json',dict(totals_seconds=totals,calls=len(records),n=100,
        protocol_sha256=sha(path),evaluation_predictions_sha256=sha(OUT/'predictions.json'),
        parity='All 600 predictions and two-layer traces match the quality study.',
        comparison=sealed['comparison']))
    print(json.dumps(totals))


if __name__=='__main__':main()
