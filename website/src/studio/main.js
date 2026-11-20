import loadMujoco from '@mujoco/mujoco';
import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const viewport = document.querySelector('#viewport');
const status = document.querySelector('#status');
const base = `${import.meta.env.BASE_URL.replace(/\/$/, "")}/studio-scene/`;

const measurements = [];
document.querySelector("#download").addEventListener("click", () => { const blob = new Blob([JSON.stringify(measurements, null, 2)], {type: "application/json"}); const url = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = "asset2sim-studio-measurements.json"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); });
async function fetchResource(url, read, timeout = 120_000) {
  for (let attempt = 0; ; attempt++) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(timeout) });
      if (!response.ok) {
        if (response.status < 500 && response.status !== 429) {
          throw new Error(`HTTP ${response.status}: ${url}`, { cause: 'permanent' });
        }
        throw new Error(`HTTP ${response.status}`);
      }
      return await read(response);
    } catch (error) {
      if (attempt === 1 || error.cause === 'permanent') {
        throw new Error(`${url.split('/').pop()} · ${error.message}`, { cause: error });
      }
      await new Promise(resolve => setTimeout(resolve, 700));
    }
  }
}

async function unpack(url) {
  return fetchResource(url, async response => {
    const bytes = new Uint8Array(await response.arrayBuffer());
    // Fetch may already have decoded a server-supplied Content-Encoding layer.
    if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  });
}

const clamp = (x,a,b) => Math.max(a, Math.min(b,x));

export async function start() {
  const started = performance.now();
  const packed = typeof DecompressionStream !== 'undefined';
  status.textContent = 'Downloading MuJoCo and scene assets…';
  const rawRuntime = () => fetchResource(wasmUrl, response => response.arrayBuffer())
    .then(buffer => loadMujoco({ wasmBinary: new Uint8Array(buffer) }));
  const runtime = packed
    ? unpack(base + 'packed/mujoco-gzip.bin')
      .then(wasmBinary => loadMujoco({ wasmBinary }))
      .catch(error => {
        console.warn('Using the original MuJoCo WASM:', error);
        return rawRuntime();
      })
    : rawRuntime();
  const meshPack = packed
    ? unpack(base + 'packed/meshes-gzip.bin').catch(error => {
      console.warn('Loading individual scene meshes:', error);
      return null;
    })
    : null;
  const [mj, sceneXml, files, assets, pack] = await Promise.all([
    runtime,
    fetchResource(base + 'scene.xml', response => response.text()),
    fetchResource(base + 'files.json', response => response.json()),
    fetchResource(base + 'demo-assets.json', response => response.json()),
    meshPack,
  ]);
  const researchBase = base.replace(/studio-scene\/$/, 'research/physx/');
  const loadVisual = key => fetchResource(researchBase + assets[key].id + '/prediction.glb', response => response.arrayBuffer());
  const defaultVisual = loadVisual('hammer').then(buffer => ({ buffer }), error => ({ error }));
  let xml = sceneXml;
  const names = files.filter(name => !name.startsWith('object_'))
    .concat(Object.values(assets).flatMap(asset => asset.collision));
  const entries = Object.entries(assets);
  const meshDefs = entries.flatMap(([key, asset]) => asset.collision.map((file, i) => `<mesh name="demo_${key}_${i}" file="${file}"/>`)).join('');
  const maxPerAsset=3;
  function createObjectBody(key, asset, index, slot) {
    const name = `object_${key}_${slot}`;
    const initial = key === 'hammer' && slot === 0;
    const positions = initial ? [.55, -.16, assets.hammer.drop_height_m] : [5 + index * maxPerAsset + slot, 0, 2];
    const joints = asset.joints || [];
    const geoms = asset.collision.map((_, i) => {
      const geom = `<geom type="mesh" mesh="demo_${key}_${i}" density="${asset.density_kg_m3[i]}" friction="1 .005 .0001" rgba="0 0 0 0"/>`;
      const joint = joints.find(item => item.part_label === asset.part_labels[i]);
      if (!joint) return geom;
      const [x,y,z] = joint.anchor.map((value, axis) => (value - asset.center[axis]) * asset.scale);
      const [ax,ay,az] = joint.axis;
      const range = joint.range_pi.map(value => value * Math.PI).join(' ');
      return `<body name="${name}_part_${joint.part_label}"><joint name="${name}_hinge_${joint.part_label}" type="hinge" pos="${x} ${-z} ${y}" axis="${ax} ${-az} ${ay}" range="${range}" damping="0.001"/>${geom}</body>`;
    }).join('');
    return `<body name="${name}" pos="${positions.join(' ')}"><freejoint name="${name}_free"/>${geoms}</body>`;
  }
  const bodies = entries.flatMap(([key, asset], index) =>
    Array.from({length:maxPerAsset}, (_, slot) => createObjectBody(key, asset, index, slot))
  ).join('');
  xml = xml.replace(/<mesh name="object_visual_mesh"[\s\S]*?<\/asset>/, `${meshDefs}</asset>`).replace(/<body name="object"[\s\S]*?<\/body><\/worldbody>/, `${bodies}</worldbody>`);
  const gltfLoader = new GLTFLoader();
  const hammer = await defaultVisual;
  if (hammer.error) throw hammer.error;
  const hammerModel = (await gltfLoader.parseAsync(hammer.buffer, '')).scene;
  const vfs = new mj.MjVFS();
  let model;
  try {
    if (pack) {
      const view = new DataView(pack.buffer, pack.byteOffset, pack.byteLength);
      let offset = 0;
      for (const name of names) {
        const size = view.getUint32(offset, true);
        offset += 4;
        if (offset + size > pack.length) throw new Error('Incomplete scene mesh pack');
        vfs.addBuffer(name, pack.subarray(offset, offset + size));
        offset += size;
      }
      if (offset !== pack.length) throw new Error('Scene mesh pack does not match its manifest');
    } else {
      // Limit competing downloads in browsers without native gzip decompression.
      for (let offset = 0; offset < names.length; offset += 4) {
        const batch = names.slice(offset, offset + 4);
        const buffers = await Promise.all(batch.map(name =>
          fetchResource(base + 'assets/' + encodeURIComponent(name), response => response.arrayBuffer())));
        buffers.forEach((buffer, i) => vfs.addBuffer(batch[i], new Uint8Array(buffer)));
      }
    }
    status.textContent = 'Compiling robot model…';
    await new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));
    model = mj.MjModel.from_xml_string(xml, vfs);
  } finally {
    vfs.delete();
  }
  const data = new mj.MjData(model), scratch = new mj.MjData(model);
  const site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE.value, 'grip_pinch');
  const slots = Object.fromEntries(entries.map(([key]) => [key,Array.from({length:maxPerAsset},(_,index)=>{const body=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY.value,`object_${key}_${index}`);const joint=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT.value,`object_${key}_${index}_free`);return {key,index,body,q:model.jnt_qposadr[joint],v:model.jnt_dofadr[joint],used:key==='hammer'&&index===0,visual:null};})]));
  let active=slots.hammer[0], object=active.body, generation=0;
  const geomId = n => mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM.value, n);
  const table = geomId('table'), pads = [geomId('grip_left_pad1'),geomId('grip_left_pad2'),geomId('grip_right_pad1'),geomId('grip_right_pad2')];
  if (site < 0 || object < 0 || pads.some(x => x < 0)) throw Error('Expected scene elements are missing');
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0xe8edf1);
  const camera = new THREE.PerspectiveCamera(55,1,.01,20); camera.up.set(0,0,1); const azimuth=-55*Math.PI/180, elevation=25*Math.PI/180, distance=1.15; const initialCamera=new THREE.Vector3(.35+distance*Math.cos(elevation)*Math.cos(azimuth),distance*Math.cos(elevation)*Math.sin(azimuth),.18+distance*Math.sin(elevation)); camera.position.copy(initialCamera);
  const renderer = new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.domElement.style.width="100%"; renderer.domElement.style.height="100%"; renderer.domElement.style.display="block"; viewport.appendChild(renderer.domElement);
  const orbit = new OrbitControls(camera,renderer.domElement); orbit.target.set(.35,0,.18); orbit.update(); document.querySelector('#camera-reset').addEventListener('click',()=>{camera.position.copy(initialCamera);orbit.target.set(.35,0,.18);orbit.update();});
  scene.add(new THREE.HemisphereLight(0xffffff,0x667788,2)); const light=new THREE.DirectionalLight(0xffffff,2); light.position.set(1,-2,3); scene.add(light);
  const visual=[];
  const assetModels = { hammer: hammerModel };
  const pendingModels = new Map();
  async function loadAsset(key) {
    if (assetModels[key]) return assetModels[key];
    if (!pendingModels.has(key)) {
      pendingModels.set(key, loadVisual(key)
        .then(buffer => gltfLoader.parseAsync(buffer, ''))
        .then(gltf => (assetModels[key] = gltf.scene))
        .finally(() => pendingModels.delete(key)));
    }
    return pendingModels.get(key);
  }
  function attachVisual(slot, source) {
    const spec = assets[slot.key], group = new THREE.Group(), objectMesh = source.clone(true);
    slot.hinges = [];
    for (const joint of spec.joints || []) {
      const part = objectMesh.getObjectByName(`${joint.part_label}_geometry_0`);
      if (!part || !part.parent) throw Error(`Missing visual part ${joint.part_label} for ${slot.key}`);
      const pivot = new THREE.Group();
      pivot.position.fromArray(joint.anchor);
      part.parent.add(pivot);
      pivot.attach(part);
      const id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT.value, `object_${slot.key}_${slot.index}_hinge_${joint.part_label}`);
      slot.hinges.push({ pivot, axis: new THREE.Vector3(...joint.axis).normalize(), q: model.jnt_qposadr[id], v: model.jnt_dofadr[id] });
    }
    objectMesh.position.set(...spec.center.map(x => -x * spec.scale));
    objectMesh.scale.setScalar(spec.scale);
    group.add(objectMesh); scene.add(group); slot.visual = group;
  }
  attachVisual(active, hammerModel);
  async function addAsset(key,point) {
    const slot=slots[key].find(item=>!item.used&&!item.loading);
    if(!slot){status.textContent=`Maximum ${maxPerAsset} ${key} objects. Reset the simulation to start again.`;return;}
    slot.loading=true;const currentGeneration=generation;
    try{const source=await loadAsset(key);if(currentGeneration!==generation)return;const furniture=key==='cabinet'||key==='chair';if(furniture){const occupied=Object.values(slots).flat().filter(item=>item.used).map(item=>({x:data.xpos[3*item.body],y:data.xpos[3*item.body+1],half:assets[item.key].half_width_m}));const half=assets[key].half_width_m;const offsets=[0,.16,-.16,.32,-.32,.48,-.48,.64,-.64,.8,-.8,1,-1,1.2,-1.2].flatMap(x=>[0,.16,-.16,.32,-.32,.48,-.48,.64,-.64,.8,-.8,1,-1,1.2,-1.2].map(y=>[x,y])).sort((a,b)=>Math.hypot(...a)-Math.hypot(...b));const free=offsets.map(([x,y])=>new THREE.Vector2(point.x+x,point.y+y)).find(p=>Math.abs(p.x)+half[0]<1.45&&Math.abs(p.y)+half[1]<1.45&&(Math.abs(p.x)>half[0]+.20||Math.abs(p.y)>half[1]+.20)&&occupied.every(item=>Math.abs(p.x-item.x)>half[0]+item.half[0]+.02||Math.abs(p.y-item.y)>half[1]+item.half[1]+.02));if(!free){status.textContent='No clear space nearby; choose another position.';return;}point=free;}attachVisual(slot,source);slot.used=true;data.qpos.set([point.x,point.y,furniture?assets[key].table_height_m:assets[key].drop_height_m,1,0,0,0],slot.q);data.qvel.fill(0,slot.v,slot.v+6);for(const hinge of slot.hinges || []){data.qpos[hinge.q]=0;data.qvel[hinge.v]=0;}active=slot;object=slot.body;document.querySelectorAll('[data-demo-asset]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.demoAsset===key)));mj.mj_forward(model,data);}
    catch(error){console.error(error);status.textContent=`Unable to load asset: ${error.message}`;}finally{slot.loading=false;}
  }
  const assetButtons=document.querySelectorAll('[data-demo-asset]');
  assetButtons.forEach(button=>{button.disabled=false;button.addEventListener('dragstart',event=>{event.dataTransfer.setData('text/plain',button.dataset.demoAsset);event.dataTransfer.effectAllowed='copy';});button.addEventListener('click',()=>addAsset(button.dataset.demoAsset,new THREE.Vector2(.55,-.16)));});
  viewport.addEventListener('dragover',event=>{event.preventDefault();event.dataTransfer.dropEffect='copy';viewport.classList.add('drag-target');});
  viewport.addEventListener('dragleave',()=>viewport.classList.remove('drag-target'));
  viewport.addEventListener('drop',event=>{event.preventDefault();viewport.classList.remove('drag-target');const key=event.dataTransfer.getData('text/plain');if(!slots[key])return;const rect=renderer.domElement.getBoundingClientRect(),pointer=new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-((event.clientY-rect.top)/rect.height)*2+1),ray=new THREE.Raycaster();ray.setFromCamera(pointer,camera);const point=new THREE.Vector3();if(ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0,0,1),0),point))addAsset(key,new THREE.Vector2(clamp(point.x,-1.2,1.2),clamp(point.y,-1.2,1.2)));});
  for(let i=0;i<model.ngeom;i++) {
    const type=model.geom_type[i], alpha=model.geom_rgba[4*i+3];
    if(alpha<0.01 || (model.geom_group[i] !== 2 && i !== table && model.geom_bodyid[i] !== object)) continue;
    let geometry;
    if(type===7) {
      const id=model.geom_dataid[i], va=model.mesh_vertadr[id], vn=model.mesh_vertnum[id], fa=model.mesh_faceadr[id], fn=model.mesh_facenum[id];
      geometry=new THREE.BufferGeometry();
      geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(model.mesh_vert.slice(3*va,3*(va+vn))),3));
      geometry.setIndex(Array.from(model.mesh_face.slice(3*fa,3*(fa+fn)))); geometry.computeVertexNormals();
    } else if(type===0) geometry=new THREE.PlaneGeometry(3,3);
    else if(type===6) geometry=new THREE.BoxGeometry(...Array.from(model.geom_size.slice(3*i,3*i+3), x=>x*2));
    else continue;
    const color=new THREE.Color(model.geom_rgba[4*i],model.geom_rgba[4*i+1],model.geom_rgba[4*i+2]);
    const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color,side:THREE.DoubleSide})); scene.add(mesh); visual.push([i,mesh]);
  }
  const home=[-3.44395664,-1.69972233,1.89918708,-1.77026108,-1.57079633,-.30236398];
  let goal=home.slice(), commands=Array.from(data.qpos.slice(0,6)), target=[.45,0,.251], angles=[0,0,0], grip=0, homing=true, lastKey=0, latency=null;
  const keys=new Set();
  let demoVisible = false;
  new IntersectionObserver(entries => { demoVisible = entries[0].isIntersecting; if (!demoVisible) keys.clear(); }, {threshold: 0.15}).observe(document.querySelector('#robot-demo')); 
  function reset(){ mj.mj_resetData(model,data); commands=Array.from(data.qpos.slice(0,6)); goal=home.slice(); target=[.45,0,.251]; angles=[0,0,0]; grip=0; homing=true; keys.clear(); generation++;for(const group of Object.values(slots))for(const slot of group){if(slot.visual){scene.remove(slot.visual);slot.visual=null;}slot.used=false;}active=slots.hammer[0];object=active.body;active.used=true;attachVisual(active,assetModels.hammer);document.querySelectorAll('[data-demo-asset]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.demoAsset==='hammer')));mj.mj_forward(model,data); }
  window.addEventListener('keydown',e=>{const k=e.key.toLowerCase(); const editable=e.target instanceof HTMLElement && (e.target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)); if(demoVisible && !editable && !e.ctrlKey && !e.metaKey && !e.altKey && 'wasdrfuioph ljk0'.replaceAll(' ','').includes(k) && k.length===1){e.preventDefault(); if(!keys.has(k)) lastKey=performance.now(); keys.add(k); if(k==='0'&&!e.repeat) reset();}});
  window.addEventListener('keyup',e=>keys.delete(e.key.toLowerCase())); window.addEventListener('blur',()=>keys.clear()); document.addEventListener('visibilitychange',()=>{if(document.hidden) keys.clear();});
  // Numerical damped least squares IK on a separate data instance; only actuator targets reach the live simulation.
  function pose(q){scratch.qpos.set(data.qpos); scratch.qpos.set(q,0); mj.mj_forward(model,scratch); return [Array.from(scratch.site_xpos.slice(site*3,site*3+3)),Array.from(scratch.site_xmat.slice(site*9,site*9+9))];}
  function residual(q,p,angles){const [x,R]=pose(q); const current=new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().set(R[0],R[1],R[2],0,R[3],R[4],R[5],0,R[6],R[7],R[8],0,0,0,0,1)); const desired=new THREE.Quaternion().setFromEuler(new THREE.Euler(...angles,'ZYX')).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),Math.PI)); const err=desired.multiply(current.invert()).normalize(); if(err.w<0) err.set(-err.x,-err.y,-err.z,-err.w); const v=Math.hypot(err.x,err.y,err.z); const factor=v>1e-9 ? 2*Math.atan2(v,err.w)/v : 2; return [3*(p[0]-x[0]),3*(p[1]-x[1]),3*(p[2]-x[2]),factor*err.x,factor*err.y,factor*err.z];}
  function solve(A,b){for(let i=0;i<6;i++){let pivot=i; for(let k=i+1;k<6;k++) if(Math.abs(A[k][i])>Math.abs(A[pivot][i])) pivot=k; [A[i],A[pivot]]=[A[pivot],A[i]]; [b[i],b[pivot]]=[b[pivot],b[i]]; if(Math.abs(A[i][i])<1e-10)return null; const d=A[i][i]; for(let j=i;j<6;j++) A[i][j]/=d; b[i]/=d; for(let k=0;k<6;k++) if(k!==i){const f=A[k][i]; for(let j=i;j<6;j++) A[k][j]-=f*A[i][j]; b[k]-=f*b[i];}} return b;}
  function ik(p,orientation){let q=goal.slice(); for(let it=0;it<18;it++){const r=residual(q,p,orientation); if(Math.hypot(...r)<.003)return q; const J=Array.from({length:6},()=>Array(6)); for(let j=0;j<6;j++){const t=q.slice();t[j]+=.0001;const rr=residual(t,p,orientation); for(let i=0;i<6;i++) J[i][j]=(rr[i]-r[i])/.0001;} const A=Array.from({length:6},(_,i)=>Array.from({length:6},(_,j)=>J.reduce((s,row)=>s+row[i]*row[j],0)+(i===j?.003:0))); const b=Array.from({length:6},(_,i)=>-J.reduce((s,row,k)=>s+row[i]*r[k],0)); const delta=solve(A,b);if(!delta)return null; q=q.map((x,i)=>clamp(x+clamp(delta[i],-.2,.2),model.jnt_range[2*i],model.jnt_range[2*i+1]));} return null;}
  let previous=performance.now(), accum=0, statsTime=previous, frames=0, stepCount=0, stepMs=0, ikClock=0; const loadSeconds=(performance.now()-started)/1000;
  function frame(now){
    if (!demoVisible || document.hidden) {
      previous = statsTime = now;
      frames = stepCount = stepMs = 0;
      requestAnimationFrame(frame);
      return;
    }
    const wall=Math.min((now-previous)/1000,.05);previous=now;accum+=wall;ikClock+=wall;
    if(homing && Math.max(...goal.map((x,i)=>Math.abs(x-data.qpos[i])))<.08) homing=false;
    if(!homing && ikClock>=.05){const dir=[+keys.has('w')-keys.has('s'),+keys.has('a')-keys.has('d'),+keys.has('r')-keys.has('f')]; const norm=Math.hypot(...dir); const angular=[+keys.has('u')-keys.has('o'),+keys.has('i')-keys.has('p'),+keys.has('h')-keys.has('l')]; const angularNorm=Math.hypot(...angular); if(norm||angularNorm){const min=[.05,-.5,-.01],max=[.85,.5,.65];const next=target.map((v,i)=>clamp(v+(norm?dir[i]/norm*.06*ikClock:0),min[i],max[i])); const nextAngles=angles.map((v,i)=>clamp(v+(angularNorm?angular[i]/angularNorm*.4*ikClock:0),-Math.PI,Math.PI));const result=ik(next,nextAngles);if(result){goal=result;target=next;angles=nextAngles;if(lastKey) {latency=performance.now()-lastKey;lastKey=0;}}}ikClock=0;}
    if(!homing) grip=clamp(grip+(+keys.has('j')-keys.has('k'))*160*wall,0,255);
    const t0=performance.now(); let steps=0; while(accum>=.002 && steps<25){for(let i=0;i<6;i++){commands[i]+=clamp(goal[i]-commands[i],-.65*.002,.65*.002);data.ctrl[i]=commands[i];}data.ctrl[6]=grip;mj.mj_step(model,data);accum-=.002;steps++;}stepMs+=performance.now()-t0;stepCount+=steps;
    for(const [i,mesh] of visual){mesh.position.fromArray(data.geom_xpos,3*i); const a=data.geom_xmat, n=9*i; const m=new THREE.Matrix4().set(a[n],a[n+1],a[n+2],0,a[n+3],a[n+4],a[n+5],0,a[n+6],a[n+7],a[n+8],0,0,0,0,1);mesh.quaternion.setFromRotationMatrix(m);}
    for(const group of Object.values(slots))for(const slot of group){if(!slot.used||!slot.visual)continue;for(const hinge of slot.hinges || [])hinge.pivot.quaternion.setFromAxisAngle(hinge.axis,data.qpos[hinge.q]);slot.visual.position.fromArray(data.xpos,3*slot.body);const q=data.xquat;slot.visual.quaternion.set(q[4*slot.body+1],q[4*slot.body+2],q[4*slot.body+3],q[4*slot.body]).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),Math.PI/2));}
    const w=viewport.clientWidth,h=viewport.clientHeight;if(renderer.domElement.width!==Math.round(w*renderer.getPixelRatio())||renderer.domElement.height!==Math.round(h*renderer.getPixelRatio())){renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}orbit.update();renderer.render(scene,camera); frames++;
    if(now-statsTime>=1000){let contacts=['no','no','no'];for(let i=0;i<data.ncon;i++){const c=data.contact.get(i); const a=c.geom1,b=c.geom2; const belongs=id=>{let body=model.geom_bodyid[id];while(body>0&&body!==object)body=model.body_parentid[body];return body===object};if(belongs(a)||belongs(b)){const other=belongs(a)?b:a;if(pads.slice(0,2).includes(other))contacts[0]='yes';if(pads.slice(2).includes(other))contacts[1]='yes';if(other===table)contacts[2]='yes';}c.delete();}const height=data.xpos[3*object+2]; measurements.push({wall_s:(now-started)/1000,simulation_s:data.time,load_s:loadSeconds,fps:frames*1000/(now-statsTime),step_ms:stepMs/Math.max(stepCount,1),input_to_ik_ms:latency,object:active.key,object_count:Object.values(slots).flat().filter(slot=>slot.used).length,object_z_m:height,left_contact:contacts[0]==='yes',right_contact:contacts[1]==='yes',table_contact:contacts[2]==='yes',gripper_control:grip, target_position_m:target.slice(), target_angles_rad:angles.slice(), pinch_rotation_matrix:Array.from(data.site_xmat.slice(site*9,site*9+9))}); status.textContent=`${homing?'Homing':'Ready'} | ${active.key} height ${height.toFixed(3)} m | Objects ${Object.values(slots).flat().filter(slot=>slot.used).length} | Load ${loadSeconds.toFixed(2)} s | FPS ${(frames*1000/(now-statsTime)).toFixed(1)} | mj_step ${(stepMs/Math.max(stepCount,1)).toFixed(3)} ms/step\nInput to accepted IK: ${latency===null?'—':latency.toFixed(1)+' ms'} | Contact left / right / table: ${contacts.join(' / ')} | sim ${data.time.toFixed(1)} s`;statsTime=now;frames=stepCount=stepMs=0;}
    requestAnimationFrame(frame);
  }
  mj.mj_forward(model,data);requestAnimationFrame(frame);
}
