import './search-demo-core.js';
const S=self.SearchDemo,ROOT=new URL('../results/search-ranking/',import.meta.url);
let corpus,postings,manifest,vectors,extractor,ranker,busy=false;
const setup={};
const send=(type,data)=>postMessage({type,...data});
async function get(name,hash){
 const response=await fetch(new URL(name,ROOT));if(!response.ok)throw new Error('Could not download '+name+': HTTP '+response.status);
 const bytes=await response.arrayBuffer();
 if(hash){const actual=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');if(actual!==hash)throw new Error(name+' failed its integrity check.');}
 return bytes;
}
async function loadCorpus(){
 if(corpus)return;
 const start=performance.now();send('status',{message:'Loading the small research-paper collection.'});
 manifest=JSON.parse(new TextDecoder().decode(await get('manifest.json')));
 if(manifest.model!=='Xenova/all-MiniLM-L6-v2'||manifest.revision!=='751bff37182d3f1213fa05d7196b954e230abad9'||manifest.dimensions!==384)throw new Error('Unexpected search model manifest.');
 const data=JSON.parse(new TextDecoder().decode(await get('corpus.json',manifest.corpusSHA256)));
 if(data.documents.length!==manifest.count)throw new Error('Corpus count does not match.');
 corpus=data.documents;postings=S.index(corpus);setup.corpusMs=performance.now()-start;
}
async function loadModel(){
 if(extractor)return;
 const start=performance.now();send('status',{message:'Loading MiniLM (about 23 MB) and the precomputed document vectors.'});
 vectors=new Float32Array(await get('embeddings.f32',manifest.embeddingsSHA256));
 ranker=JSON.parse(new TextDecoder().decode(await get('learned/model.json','e8589d2d6c38b30da75ce45bdfe194e50a60904fcdd5bc60e3eb00537a712ad7')));
 if(vectors.length!==corpus.length*384)throw new Error('Document vectors do not match the corpus.');
 const {pipeline,env}=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js');
 env.allowLocalModels=false;env.backends.onnx.wasm.numThreads=1;
 let previous=-1;
 extractor=await pipeline('feature-extraction',manifest.model,{revision:manifest.revision,dtype:'q8',device:'wasm',progress_callback:event=>{
  const percent=Math.round(event.progress);if(event.status==='progress'&&event.file?.endsWith('.onnx')&&percent!==previous){previous=percent;send('status',{message:'Downloading MiniLM: '+percent+'%.'});}
 }});
 setup.modelMs=performance.now()-start;
}
self.onmessage=async({data})=>{
 if(data.type!=='search'||busy)return;busy=true;
 try{
  if(!crypto?.subtle)throw new Error('Use HTTPS or localhost so the dataset integrity check can run.');
  const query=String(data.query||'').trim();if(!query||query.length>1000)throw new Error('Enter a query of 1 to 1,000 characters.');
  await loadCorpus();
  if(data.compare)await loadModel();
  const start=performance.now(),lexicalScores=S.keywordScores(query,postings,corpus.length),lexical=S.rank(lexicalScores),lexicalMs=performance.now()-start;
  const result={query,candidates:corpus.length,bm25:{ms:lexicalMs,results:S.top(lexical,lexicalScores,corpus)},setup:{...setup},model:manifest.model,modelRevision:manifest.revision};
  if(data.compare){
   send('status',{message:'Encoding your query locally and comparing it with every document.'});
   const started=performance.now();
   const inputTokens=extractor.tokenizer.encode(query,{add_special_tokens:true}).length;
   const embedding=await extractor(query,{pooling:'mean',normalize:true,truncation:true,max_length:256});
   try{
    const scores=S.denseScores(embedding.data,vectors,corpus.length),order=S.rank(scores);
    result.minilm={ms:performance.now()-started,inputTokens,truncated:inputTokens>256,results:S.top(order,scores,corpus)};
    const fusionStarted=performance.now();const fusion=S.fuse(lexical,order);
    result.hybrid={ms:lexicalMs+result.minilm.ms+performance.now()-fusionStarted,results:S.top(fusion,null,corpus)};
    const learned=S.learnedScores(lexicalScores,scores,lexical,order,ranker);
    result.learned={results:S.top(S.rank(learned),learned,corpus)};
   }finally{embedding.dispose?.();}
  }
  send('result',{result});
 }catch(error){send('error',{message:error.message||String(error)});}
 finally{busy=false;}
};
