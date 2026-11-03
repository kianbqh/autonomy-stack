// Focused full07 regression only. Passive GL observations; no FPS/performance measurement.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {execFileSync} from 'node:child_process';
const base=import.meta.dirname,work=path.join(base,'execution-fit-20261103'),out=path.join(work,'focused-attempt2'),repo='G:/ResumeProject/platforms/autonomy-stack',viz=path.join(repo,'viz');
fs.mkdirSync(out);
const expectedSource='a5139856c457eb61cdc8359d2432f12268336ccb';
const input=path.join(base,'execution-20261103-3case/inputs/full07/replay.json'),bytes=fs.readFileSync(input),run=JSON.parse(bytes);
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
assert.equal(hash(bytes),'22789ab85de9a5fbf4100f5896e743d7c8efb1b99a37383af4ef325d3ae86122');
assert.equal(run.provenance.git_sha,'61d485e79d8f1cd4ef5791f148c853555f4db390');assert.equal(run.accepted_frames,1101);
assert.equal(execFileSync('git',['-C',repo,'rev-parse','HEAD'],{encoding:'utf8'}).trim(),expectedSource);
assert.equal(execFileSync('git',['-C',repo,'status','--porcelain=v1'],{encoding:'utf8'}).trim(),'');
const sceneHash=hash(fs.readFileSync(path.join(viz,'src/slam/KittiReplay.tsx')));
assert.equal(sceneHash,'a7fbb68010adce5de8a471d66a4bde7de62d4cecd4ead75b41fcced0753577aa');
const require=createRequire(path.join(viz,'package.json')),{chromium}=require('playwright-core'),THREE=require('three');
const server=JSON.parse(fs.readFileSync(path.join(work,'server.json'),'utf8'));
const report={status:'started',utc:new Date().toISOString(),viewer_git_sha:expectedSource,native_git_sha:run.provenance.git_sha,replay_sha256:hash(bytes),ui_scene_sha256:sceneHash,scope:'Focused full07 framing/fidelity only; one headless SwiftShader context; no FPS/performance result',views:[]};
const save=()=>fs.writeFileSync(path.join(out,'fit-regression.json'),JSON.stringify(report,null,2)+'\n');save();let context;
try{
  context=await chromium.launchPersistentContext(path.join(out,'profile'),{executablePath:'G:/ResumeProject/.runtime/consolidation/f3/browsers/chromium-1248/chrome-win64/chrome.exe',headless:true,viewport:{width:1440,height:1000},args:['--use-angle=swiftshader','--enable-unsafe-swiftshader',`--disk-cache-dir=${path.join(out,'browser-cache')}`]});
  const page=context.pages()[0]??await context.newPage();page.setDefaultTimeout(30000);const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.addInitScript(()=>{
    window.__fitBuffers=[];window.__fitDraws=[];const original=HTMLCanvasElement.prototype.getContext,seen=new WeakSet();
    HTMLCanvasElement.prototype.getContext=function(...a){const gl=original.apply(this,a);if(gl&&/^webgl/.test(String(a[0]))&&!seen.has(gl)){
      seen.add(gl);const locs=new WeakMap(),matrices=new WeakMap();let program;
      const get=gl.getUniformLocation;gl.getUniformLocation=function(p,n){const l=get.call(this,p,n);if(l)locs.set(l,{program:p,name:n});return l;};
      const use=gl.useProgram;gl.useProgram=function(p){program=p;return use.call(this,p);};
      const uniform=gl.uniformMatrix4fv;gl.uniformMatrix4fv=function(...b){const info=locs.get(b[0]);if(info&&['projectionMatrix','modelViewMatrix'].includes(info.name)){const m=matrices.get(info.program)??{};m[info.name]=Array.from(b[2]);matrices.set(info.program,m);}return uniform.apply(this,b);};
      const upload=gl.bufferData;gl.bufferData=function(...b){if(b[0]===gl.ARRAY_BUFFER&&b[1] instanceof Float32Array){window.__fitBuffers.push(Array.from(b[1]));if(window.__fitBuffers.length>128)window.__fitBuffers.shift();}return upload.apply(this,b);};
      const draw=gl.drawArrays;gl.drawArrays=function(...b){if(b[0]===gl.LINE_STRIP&&b[2]===1101){const m=matrices.get(program);if(m?.projectionMatrix&&m?.modelViewMatrix){window.__fitDraws.push({...m,width:gl.drawingBufferWidth,height:gl.drawingBufferHeight,count:b[2]});if(window.__fitDraws.length>24)window.__fitDraws.shift();}}return draw.apply(this,b);};
    }return gl;};
  });
  const url=`http://127.0.0.1:${server.port}/?view=slam&replay=/data/slam/full07/replay.json`;
  const responsePromise=page.waitForResponse(r=>r.url()===`http://127.0.0.1:${server.port}/data/slam/full07/replay.json`);
  await page.goto(url,{waitUntil:'networkidle'});const response=await responsePromise;assert.equal(response.status(),200);assert.equal(hash(await response.body()),hash(bytes));
  assert.match(await page.getByTestId('slam-status').innerText(),/1101\/1101.*optimization available/);
  await page.waitForFunction(()=>window.__fitBuffers.length>0&&window.__fitDraws.length>=3);
  const xyz={};for(const track of ['odom','optimized','truth'])xyz[track]=run.frames.flatMap(f=>[f[track][3],-f[track][7],-f[track][11]]).map(Math.fround);
  const buffers=await page.evaluate(()=>window.__fitBuffers);
  for(const [track,expected] of Object.entries(xyz))assert(buffers.some(b=>b.length===expected.length&&b.every((v,i)=>v===expected[i])),`${track} actual buffer mismatch`);
  report.track_points={odom:1101,optimized:1101,truth:1101};report.actual_track_buffers_match=true;
  report.renderer=await page.locator('canvas').evaluate(c=>{const gl=c.getContext('webgl2'),d=gl.getExtension('WEBGL_debug_renderer_info');return {renderer:gl.getParameter(d?d.UNMASKED_RENDERER_WEBGL:gl.RENDERER),vendor:gl.getParameter(d?d.UNMASKED_VENDOR_WEBGL:gl.VENDOR)};});assert.match(report.renderer.renderer,/SwiftShader/i);
  report.browser_version=context.browser()?.version();
  await page.getByLabel('SLAM time').focus();await page.getByLabel('SLAM time').press('End');await page.waitForFunction(()=>document.querySelector('[data-testid="slam-frame"]')?.getAttribute('data-frame-id')==='1100');
  for(const [name,width,height] of [['desktop',1440,1000],['narrow',480,1000]]){
    await page.evaluate(()=>{window.__fitDraws=[];});await page.setViewportSize({width,height});await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
    await page.waitForFunction(()=>window.__fitDraws.length>=3);
    const draw=(await page.evaluate(()=>window.__fitDraws)).at(-1),matrix=new THREE.Matrix4().fromArray(draw.projectionMatrix).multiply(new THREE.Matrix4().fromArray(draw.modelViewMatrix));
    let maximum=0;const extrema={x:[Infinity,-Infinity],y:[Infinity,-Infinity],z:[Infinity,-Infinity]};let count=0;
    for(const values of Object.values(xyz))for(let i=0;i<values.length;i+=3){const p=new THREE.Vector3(values[i],values[i+1],values[i+2]).applyMatrix4(matrix);for(const axis of ['x','y','z']){extrema[axis][0]=Math.min(extrema[axis][0],p[axis]);extrema[axis][1]=Math.max(extrema[axis][1],p[axis]);}maximum=Math.max(maximum,Math.abs(p.x),Math.abs(p.y));assert(Math.abs(p.x)<1&&Math.abs(p.y)<1&&p.z>-1&&p.z<1,'Actual rendered track point outside clip volume');count++;}
    await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(out,`${name}-end.png`),fullPage:true});
    report.views.push({name,viewport:[width,height],canvas:[draw.width,draw.height],actual_gl_line_count:draw.count,track_points_checked:count,max_abs_ndc_xy:maximum,ndc_extrema:extrema,all_actual_track_points_inside:true,projection_matrix:draw.projectionMatrix,model_view_matrix:draw.modelViewMatrix});save();
  }
  report.page_errors=errors;assert.deepEqual(errors,[]);report.status='passed';
}catch(e){report.status='failed';report.failure={type:e.name,message:String(e),stack:e.stack};throw e;}
finally{if(context)await context.close();report.finished_utc=new Date().toISOString();save();}
