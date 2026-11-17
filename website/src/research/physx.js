const section=document.querySelector('.physx-section');
const base=section.dataset.base;
const cases=JSON.parse(section.querySelector('#physx-cases').textContent);
const names={gso_box_v004:'Box',gso_mug_v005:'Mug',gso_hammer_v003:'Hammer',partnet_microwave_7236:'Microwave',partnet_foldingchair_100526:'Folding chair',epfl_Lr_bottom_edge_0629:'Cloth 01',epfl_Lr_top_edge_2_0656:'Cloth 02'};
const labels={rigid:'Rigid',articulated:'Articulated',deformable:'Deformable'};
// Observations distinguish semantics and shape from file-level export checks.
const reviews={
 gso_hammer_v003:{ok:true,items:[['ok','Mallet head and handle identified; strongest geometry match among the seven cases.']]},
 gso_box_v004:{ok:false,items:[['error','Identified as Furniture – Table, with a tabletop and four legs; category and part structure were incorrect before decoding.']]},
 gso_mug_v005:{ok:false,items:[['ok','Cup Body and Handle identified.'],['error','The cup opening is closed, the handle is distorted, and an extra rod-like structure appears.']]},
 partnet_microwave_7236:{ok:false,items:[['error','Identified as a wooden Cabinet / Furniture with two doors and two revolute joints; the reference is a microwave with one movable revolute joint.']]},
 partnet_foldingchair_100526:{ok:false,items:[['ok','Identified as a folding chair, with Seat and Base Body.'],['error','Reconstructed shape differs substantially from the reference (F-score @ 2%: 9.8%).']]},
 epfl_Lr_bottom_edge_0629:{ok:false,items:[['error','Cloth input identified as a Bed with frame, pillow, mattress and blanket before decoding.'],['error','The original export contains degenerate mesh faces.']]},
 epfl_Lr_top_edge_2_0656:{ok:false,items:[['error','Cloth input identified as Table with Tablecloth, with Tablecloth+desktop and Board; all groups are fixed.'],['error','The deformable export branch was not activated.']]}
};
const viewer=section.querySelector('#physx-viewer'),state=section.querySelector('#physx-state');
function selectCase(id){
 const item=cases.find(c=>c.id===id);if(!item)return;
 const src=`${base}/research/physx/${id}/prediction.glb`;
 if(viewer.getAttribute('src')!==src){viewer.style.visibility='hidden';state.textContent='Loading 3D asset…';viewer.setAttribute('src',src)}
 viewer.setAttribute('alt',`PhysX-Omni decoded parts for ${names[id]}`);
 section.querySelector('#physx-image').src=`${base}/research/physx/${id}/input.png`;
 section.querySelector('#physx-image').alt=`Input image for ${names[id]}`;
 section.querySelector('#physx-category').textContent=labels[item.category];
 section.querySelector('#physx-case-name').textContent=names[id];
 section.querySelector('#physx-dataset').textContent=item.dataset;
 section.querySelector('#physx-predicted-name').textContent=id==='gso_box_v004' ? `${item.predicted_name} · Furniture – Table` : item.predicted_name;
 section.querySelector('#physx-part-count').textContent=`${item.decoded} decoded parts`;
 const parts=section.querySelector('#physx-parts');parts.replaceChildren();item.part_names.forEach((name,i)=>{const row=document.createElement('li'),part=document.createElement('span'),material=document.createElement('span');part.className='part-name';part.textContent=name;material.className='part-material';material.textContent=item.materials[i];row.append(part,material);parts.append(row)});
 const propertyTable=section.querySelector('#physx-property-table');propertyTable.replaceChildren();item.part_names.forEach((name,i)=>{const row=document.createElement('tr'),props=item.properties[i]||{}; const young=props.young!==undefined&&props.young!==''?(Number(props.young)/1e9).toLocaleString('en-US',{maximumFractionDigits:4}):'—';[name,item.materials[i],props.density||'—',young,props.poisson||'—'].forEach(value=>{const cell=document.createElement('td');cell.textContent=value;row.append(cell)});propertyTable.append(row)});
 const review=section.querySelector('#physx-review');review.replaceChildren();for(const [kind,text] of reviews[id].items){const line=document.createElement('p');line.className=`physx-review-line ${kind}`;const icon=document.createElement('span');icon.className='physx-review-icon';icon.textContent=kind==='ok'?'✓':'×';icon.setAttribute('aria-label',kind==='ok'?'Consistent: ':'Issue: ');const content=document.createElement('span');content.textContent=text;line.append(icon,content);review.append(line)}
 section.querySelectorAll('[data-physx]').forEach(button=>{const review=reviews[button.dataset.physx];button.setAttribute('aria-pressed',String(button.dataset.physx===id));button.classList.toggle('case-ok',review.ok);button.classList.toggle('case-error',!review.ok);});
}
section.querySelectorAll('[data-physx]').forEach(button=>button.addEventListener('click',()=>selectCase(button.dataset.physx)));
viewer.addEventListener('load',()=>{viewer.style.visibility='visible';state.textContent='3D asset ready'});
viewer.addEventListener('error',()=>{viewer.style.visibility='visible';state.textContent='Unable to load this 3D asset'});
function makeNode(tag,cls,text){const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n}
function renderChart(id,key,scale,format){const container=section.querySelector(id);const best=key==='cd'?Math.min(...cases.map(item=>item[key])):Math.max(...cases.map(item=>item[key]));for(const item of cases){const row=makeNode('div','physx-row');row.dataset.category=item.category;const header=makeNode('div','physx-row-label');header.append(makeNode('span','',names[item.id]),makeNode('strong','',format(item[key])));const track=makeNode('div','physx-bar'),fill=makeNode('div','physx-bar-fill');fill.style.width=`${Math.min(100,item[key]/scale*100)}%`;fill.classList.toggle('is-best',item[key]===best);track.append(fill);row.append(header,track);container.append(row)}}
renderChart('#physx-cd-chart','cd',.12,value=>value.toFixed(4));renderChart('#physx-fs-chart','fscore',1,value=>`${(value*100).toFixed(1)}%`);
const table=section.querySelector('#physx-table');for(const item of cases){const row=makeNode('tr');[names[item.id],labels[item.category],`${item.decoded} / ${item.parts}`,item.cd.toFixed(4),`${(item.fscore*100).toFixed(1)}%`,reviews[item.id].items.map(([kind,text])=>`${kind==='ok'?'✓':'×'} ${text}`).join('  /  ')].forEach(value=>row.append(makeNode('td','',value)));table.append(row)}
selectCase('gso_hammer_v003');
