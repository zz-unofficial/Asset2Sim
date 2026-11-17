const section = document.querySelector('.gen-results');
const base = section.dataset.base;
const viewer = section.querySelector('#result-viewer');
const image = section.querySelector('#input-image');
const state = section.querySelector('#model-state');
const cases = {
  metal:{label:'Chrome stool',group:'abo6',view:'012'}, wood:{label:'Wood tables',group:'abo6',view:'012'},
  ceramic:{label:'Ceramic vase',group:'abo6',view:'012'}, plastic:{label:'Shredder',group:'abo6',view:'012'},
  leather:{label:'Leather chair',group:'abo6',view:'012'}, fabric:{label:'Fabric chair',group:'abo6',view:'012'},
  basket:{label:'Open basket',group:'gso32',view:'002'}, bottle:{label:'Bottle',group:'gso32',view:'008'},
  dinosaur:{label:'Dinosaur',group:'gso32',view:'012'}
};
const models = {sf3d:'SF3D',spar3d:'SPAR3D',trellis2:'TRELLIS.2'};
let object = 'metal', model = 'sf3d';
function showModel() {
  const item=cases[object], src=`${base}/research/${item.group}/${object}/${model}.glb`;
  if (viewer.getAttribute('src')!==src) {state.textContent='Loading 3D model…';viewer.style.visibility='hidden';viewer.setAttribute('src',src);}
  viewer.setAttribute('alt',`${models[model]} reconstruction of ${item.label}`);
  section.querySelector('#model-name').textContent=models[model];
  section.querySelector('#object-name').textContent=item.label;
  section.querySelector('#case-meta').innerHTML=`${item.group.toUpperCase()} · view ${item.view}<br>Same input for all three models`;
  section.querySelector('.explorer-top .index').textContent=`${String(Object.keys(cases).indexOf(object)+1).padStart(2,'0')} — 09`;
  image.src=`${base}/research/${item.group}/${object}/input.png`;image.alt=`Input image of ${item.label}`;
  section.querySelectorAll('[data-model]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.model===model)));
  section.querySelectorAll('[data-case]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.case===object)));
}
section.querySelectorAll('[data-case]').forEach(button=>button.addEventListener('click',()=>{object=button.dataset.case;showModel()}));
section.querySelectorAll('[data-model]').forEach(button=>button.addEventListener('click',()=>{model=button.dataset.model;showModel()}));
viewer.addEventListener('load',()=>{viewer.style.visibility='visible';state.textContent='3D model ready'});
viewer.addEventListener('error',()=>{viewer.style.visibility='visible';state.textContent='Unable to load this model. Please try again.'});

const data=JSON.parse(section.querySelector('#research-metrics').textContent);
const metrics=[
 ['generation_chamfer_l1','Chamfer-L1 ↓','Geometry · lower is better'],
 ['generation_fscore@0.02','F-score @ 2% ↑','Geometry · higher is better'],
 ['texture_novel_lpips','LPIPS ↓','Flat RGB · novel views · lower is better'],
 ['generation_hausdorff95','HD95 ↓','Geometry · lower is better'],
 ['generation_fscore@0.01','F-score @ 1% ↑','Geometry · higher is better'],
 ['generation_fscore@0.05','F-score @ 5% ↑','Geometry · higher is better'],
 ['texture_novel_mask_iou','Silhouette IoU ↑','Novel views · higher is better'],
 ['texture_novel_psnr','PSNR ↑','Flat RGB · dB · higher is better'],
 ['texture_novel_ssim','SSIM ↑','Flat RGB · higher is better'],
 ['texture_novel_lpips_union_crop','Cropped LPIPS ↓','Flat RGB · foreground crop · lower is better'],
 ['texture_novel_psnr_foreground','Foreground PSNR ↑','Flat RGB · dB · higher is better'],
 ['pbr_novel_lpips','PBR LPIPS ↓','Fixed HDRI · lower is better'],
 ['pbr_novel_psnr','PBR PSNR ↑','Fixed HDRI · dB · higher is better'],
 ['pbr_novel_ssim','PBR SSIM ↑','Fixed HDRI · higher is better'],
 ['pbr_novel_lpips_union_crop','PBR cropped LPIPS ↓','Fixed HDRI · foreground crop · lower is better'],
 ['pbr_novel_psnr_foreground','PBR foreground PSNR ↑','Fixed HDRI · dB · higher is better']
];
const format=value=>value===''||value==null?'—':Number(value).toFixed(5);
let selectedDataset='GSO32';
const carousel=section.querySelector('#metric-chart');
const previous=section.querySelector('#metric-prev'), next=section.querySelector('#metric-next');
function navState(){previous.disabled=carousel.scrollLeft<2;next.disabled=carousel.scrollLeft+carousel.clientWidth>=carousel.scrollWidth-2}
function scrollMetric(direction){const card=carousel.querySelector('.metric-card');if(card)carousel.scrollBy({left:direction*(card.getBoundingClientRect().width+14),behavior:'smooth'});}
previous.addEventListener('click',()=>scrollMetric(-1));next.addEventListener('click',()=>scrollMetric(1));carousel.addEventListener('scroll',navState,{passive:true});window.addEventListener('resize',navState);
function node(tag,cls,text){const el=document.createElement(tag);if(cls)el.className=cls;if(text!==undefined)el.textContent=text;return el}
function activeMetrics(){return metrics.filter(([key])=>selectedDataset==='ABO6'||!key.startsWith('pbr_'))}
function render(){
 const set=data[selectedDataset], available=activeMetrics();
 section.querySelectorAll('[data-dataset]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.dataset===selectedDataset)));
 carousel.replaceChildren();
 const summary=section.querySelector('#summary-table');summary.replaceChildren();
 for(const [key,title,description] of available){
  const rows=Object.keys(models).map(m=>set.medians.find(r=>r.model===m&&r.metric===key));
  if(rows.some(r=>!r||r.median===''))continue;
  const values=rows.map(r=>Number(r.median));const max=Math.max(...values)*1.1||1;
  const card=node('article','metric-card');card.append(node('h3','',title),node('p','metric-context',description));
  rows.forEach((r,i)=>{const stat=node('div','model-stat'),label=node('div','stat-label');label.append(node('span','',models[r.model]),node('strong','',format(r.median)));const track=node('div','bar-track'),fill=node('div','bar-fill');fill.style.width=`${values[i]/max*100}%`;fill.style.background=['#a9b8ad','#7b9b83','#315d42'][i];track.append(fill);stat.append(label,track);card.append(stat)});
  const axis=node('div','axis');axis.append(node('span','','0'),node('span','',format(max)));card.append(axis);carousel.append(card);
  const tr=node('tr');[title,`${rows[0].n_valid_objects} / ${rows[0].n_planned_objects}`,...rows.map(r=>format(r.median))].forEach(v=>tr.append(node('td','',v)));summary.append(tr);
 }
 const table=section.querySelector('#object-table');table.replaceChildren();const thead=node('thead'),header=node('tr');['Object','Model',...available.map(([,title])=>title)].forEach(v=>header.append(node('th','',v)));thead.append(header);table.append(thead);const tbody=node('tbody');[...set.objects].sort((a,b)=>a.object_id.localeCompare(b.object_id)||a.model.localeCompare(b.model)).forEach(r=>{const tr=node('tr');[r.object_id,models[r.model],...available.map(([key])=>format(r[key]))].forEach(v=>tr.append(node('td','',v)));tbody.append(tr)});table.append(tbody);

 carousel.scrollLeft=0;requestAnimationFrame(navState);
}
section.querySelectorAll('[data-dataset]').forEach(button=>button.addEventListener('click',()=>{selectedDataset=button.dataset.dataset;render()}));
render();
