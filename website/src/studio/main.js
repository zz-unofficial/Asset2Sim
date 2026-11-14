import loadMujoco from '@mujoco/mujoco';
import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const viewport = document.querySelector('#viewport');
const status = document.querySelector('#status');
const base = `${import.meta.env.BASE_URL.replace(/\/$/, "")}/studio-scene/`;
const started = performance.now();
const measurements = [];
document.querySelector("#download").addEventListener("click", () => { const blob = new Blob([JSON.stringify(measurements, null, 2)], {type: "application/json"}); const url = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = "asset2sim-studio-measurements.json"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); });
const fetchOk = async url => { const r = await fetch(url, {signal: AbortSignal.timeout(30000)}); if (!r.ok) throw Error(`${r.status} ${url}`); return r; };
const clamp = (x,a,b) => Math.max(a, Math.min(b,x));

async function start() {
  status.textContent = '正在初始化 MuJoCo WASM…';
  const mj = await loadMujoco({locateFile: path => path.endsWith('.wasm') ? wasmUrl : path});
  const [xmlResponse, listResponse] = await Promise.all([fetchOk(base+'scene.xml'), fetchOk(base+'files.json')]);
  const xml = await xmlResponse.text();
  const names = await listResponse.json();
  const vfs = new mj.MjVFS();
  let loaded = 0;
  status.textContent = `正在加载场景资源 0 / ${names.length}…`;
  await Promise.all(names.map(async name => { const r = await fetchOk(base+'assets/'+encodeURIComponent(name)); vfs.addBuffer(name,new Uint8Array(await r.arrayBuffer())); loaded++; status.textContent = `正在加载场景资源 ${loaded} / ${names.length}…`; }));
  status.textContent = '正在编译机器人模型，请稍候…';
  await new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));
  const model = mj.MjModel.from_xml_string(xml,vfs);
  const data = new mj.MjData(model), scratch = new mj.MjData(model);
  const site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE.value, 'grip_pinch');
  const object = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, 'object');
  const geomId = n => mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM.value, n);
  const table = geomId('table'), pads = [geomId('grip_left_pad1'),geomId('grip_left_pad2'),geomId('grip_right_pad1'),geomId('grip_right_pad2')];
  if (site < 0 || object < 0 || pads.some(x => x < 0)) throw Error('Expected scene elements are missing');
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0xe8edf1);
  const camera = new THREE.PerspectiveCamera(55,1,.01,20); camera.up.set(0,0,1); const azimuth=-55*Math.PI/180, elevation=25*Math.PI/180, distance=1.15; const initialCamera=new THREE.Vector3(.35+distance*Math.cos(elevation)*Math.cos(azimuth),distance*Math.cos(elevation)*Math.sin(azimuth),.18+distance*Math.sin(elevation)); camera.position.copy(initialCamera);
  const renderer = new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.domElement.style.width="100%"; renderer.domElement.style.height="100%"; renderer.domElement.style.display="block"; viewport.appendChild(renderer.domElement);
  const orbit = new OrbitControls(camera,renderer.domElement); orbit.target.set(.35,0,.18); orbit.update(); document.querySelector('#camera-reset').addEventListener('click',()=>{camera.position.copy(initialCamera);orbit.target.set(.35,0,.18);orbit.update();});
  scene.add(new THREE.HemisphereLight(0xffffff,0x667788,2)); const light=new THREE.DirectionalLight(0xffffff,2); light.position.set(1,-2,3); scene.add(light);
  const visual=[];
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
  function reset(){ mj.mj_resetData(model,data); commands=Array.from(data.qpos.slice(0,6)); goal=home.slice(); target=[.45,0,.251]; angles=[0,0,0]; grip=0; homing=true; keys.clear(); }
  window.addEventListener('keydown',e=>{const k=e.key.toLowerCase(); const editable=e.target instanceof HTMLElement && (e.target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)); if(demoVisible && !editable && !e.ctrlKey && !e.metaKey && !e.altKey && 'wasdrfuioph ljk0'.replaceAll(' ','').includes(k) && k.length===1){e.preventDefault(); if(!keys.has(k)) lastKey=performance.now(); keys.add(k); if(k==='0'&&!e.repeat) reset();}});
  window.addEventListener('keyup',e=>keys.delete(e.key.toLowerCase())); window.addEventListener('blur',()=>keys.clear()); document.addEventListener('visibilitychange',()=>{if(document.hidden) keys.clear();});
  // Numerical damped least squares IK on a separate data instance; only actuator targets reach the live simulation.
  function pose(q){scratch.qpos.set(data.qpos); scratch.qpos.set(q,0); mj.mj_forward(model,scratch); return [Array.from(scratch.site_xpos.slice(site*3,site*3+3)),Array.from(scratch.site_xmat.slice(site*9,site*9+9))];}
  function residual(q,p,angles){const [x,R]=pose(q); const current=new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().set(R[0],R[1],R[2],0,R[3],R[4],R[5],0,R[6],R[7],R[8],0,0,0,0,1)); const desired=new THREE.Quaternion().setFromEuler(new THREE.Euler(...angles,'ZYX')).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),Math.PI)); const err=desired.multiply(current.invert()).normalize(); if(err.w<0) err.set(-err.x,-err.y,-err.z,-err.w); const v=Math.hypot(err.x,err.y,err.z); const factor=v>1e-9 ? 2*Math.atan2(v,err.w)/v : 2; return [3*(p[0]-x[0]),3*(p[1]-x[1]),3*(p[2]-x[2]),factor*err.x,factor*err.y,factor*err.z];}
  function solve(A,b){for(let i=0;i<6;i++){let pivot=i; for(let k=i+1;k<6;k++) if(Math.abs(A[k][i])>Math.abs(A[pivot][i])) pivot=k; [A[i],A[pivot]]=[A[pivot],A[i]]; [b[i],b[pivot]]=[b[pivot],b[i]]; if(Math.abs(A[i][i])<1e-10)return null; const d=A[i][i]; for(let j=i;j<6;j++) A[i][j]/=d; b[i]/=d; for(let k=0;k<6;k++) if(k!==i){const f=A[k][i]; for(let j=i;j<6;j++) A[k][j]-=f*A[i][j]; b[k]-=f*b[i];}} return b;}
  function ik(p,orientation){let q=goal.slice(); for(let it=0;it<18;it++){const r=residual(q,p,orientation); if(Math.hypot(...r)<.003)return q; const J=Array.from({length:6},()=>Array(6)); for(let j=0;j<6;j++){const t=q.slice();t[j]+=.0001;const rr=residual(t,p,orientation); for(let i=0;i<6;i++) J[i][j]=(rr[i]-r[i])/.0001;} const A=Array.from({length:6},(_,i)=>Array.from({length:6},(_,j)=>J.reduce((s,row)=>s+row[i]*row[j],0)+(i===j?.003:0))); const b=Array.from({length:6},(_,i)=>-J.reduce((s,row,k)=>s+row[i]*r[k],0)); const delta=solve(A,b);if(!delta)return null; q=q.map((x,i)=>clamp(x+clamp(delta[i],-.2,.2),model.jnt_range[2*i],model.jnt_range[2*i+1]));} return null;}
  let previous=performance.now(), accum=0, statsTime=previous, frames=0, stepCount=0, stepMs=0, ikClock=0; const loadSeconds=(performance.now()-started)/1000;
  function frame(now){const wall=Math.min((now-previous)/1000,.05);previous=now;accum+=wall;ikClock+=wall;
    if(homing && Math.max(...goal.map((x,i)=>Math.abs(x-data.qpos[i])))<.08) homing=false;
    if(!homing && ikClock>=.05){const dir=[+keys.has('w')-keys.has('s'),+keys.has('a')-keys.has('d'),+keys.has('r')-keys.has('f')]; const norm=Math.hypot(...dir); const angular=[+keys.has('u')-keys.has('o'),+keys.has('i')-keys.has('p'),+keys.has('h')-keys.has('l')]; const angularNorm=Math.hypot(...angular); if(norm||angularNorm){const min=[.2,-.3,.045],max=[.65,.3,.45];const next=target.map((v,i)=>clamp(v+(norm?dir[i]/norm*.06*ikClock:0),min[i],max[i])); const nextAngles=angles.map((v,i)=>clamp(v+(angularNorm?angular[i]/angularNorm*.4*ikClock:0),-1.2,1.2));const result=ik(next,nextAngles);if(result){goal=result;target=next;angles=nextAngles;if(lastKey) {latency=performance.now()-lastKey;lastKey=0;}}}ikClock=0;}
    if(!homing) grip=clamp(grip+(+keys.has('j')-keys.has('k'))*160*wall,0,255);
    const t0=performance.now(); let steps=0; while(accum>=.002 && steps<25){for(let i=0;i<6;i++){commands[i]+=clamp(goal[i]-commands[i],-.65*.002,.65*.002);data.ctrl[i]=commands[i];}data.ctrl[6]=grip;mj.mj_step(model,data);accum-=.002;steps++;}stepMs+=performance.now()-t0;stepCount+=steps;
    for(const [i,mesh] of visual){mesh.position.fromArray(data.geom_xpos,3*i); const a=data.geom_xmat, n=9*i; const m=new THREE.Matrix4().set(a[n],a[n+1],a[n+2],0,a[n+3],a[n+4],a[n+5],0,a[n+6],a[n+7],a[n+8],0,0,0,0,1);mesh.quaternion.setFromRotationMatrix(m);}
    const w=viewport.clientWidth,h=viewport.clientHeight;if(renderer.domElement.width!==Math.round(w*renderer.getPixelRatio())||renderer.domElement.height!==Math.round(h*renderer.getPixelRatio())){renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}orbit.update();renderer.render(scene,camera); frames++;
    if(now-statsTime>=1000){let contacts=['no','no','no'];for(let i=0;i<data.ncon;i++){const c=data.contact.get(i); const a=c.geom1,b=c.geom2; if(model.geom_bodyid[a]===object||model.geom_bodyid[b]===object){const other=model.geom_bodyid[a]===object?b:a;if(pads.slice(0,2).includes(other))contacts[0]='yes';if(pads.slice(2).includes(other))contacts[1]='yes';if(other===table)contacts[2]='yes';}c.delete();}const height=data.xpos[3*object+2]; measurements.push({wall_s:(now-started)/1000,simulation_s:data.time,load_s:loadSeconds,fps:frames*1000/(now-statsTime),step_ms:stepMs/Math.max(stepCount,1),input_to_ik_ms:latency,object_z_m:height,left_contact:contacts[0]==='yes',right_contact:contacts[1]==='yes',table_contact:contacts[2]==='yes',gripper_control:grip, target_angles_rad:angles.slice(), pinch_rotation_matrix:Array.from(data.site_xmat.slice(site*9,site*9+9))}); status.textContent=`${homing?'归位中，请稍候':'可以操作'} | 方块高度 ${height.toFixed(3)} m | 加载 ${loadSeconds.toFixed(2)} s | FPS ${(frames*1000/(now-statsTime)).toFixed(1)} | mj_step ${(stepMs/Math.max(stepCount,1)).toFixed(3)} ms/step\n按键到接受 IK: ${latency===null?'—':latency.toFixed(1)+' ms'} | 接触 左 / 右 / 桌面: ${contacts.map(value=>value==='yes'?'是':'否').join(' / ')} | sim ${data.time.toFixed(1)} s`;statsTime=now;frames=stepCount=stepMs=0;}
    requestAnimationFrame(frame);
  }
  mj.mj_forward(model,data);requestAnimationFrame(frame);
}
start().catch(e=>{console.error(e);status.textContent=`仿真加载失败: ${e.message}`;});
