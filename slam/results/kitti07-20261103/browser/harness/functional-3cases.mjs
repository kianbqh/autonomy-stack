// Preparation only. Root must release an exact export, private build and serve path.
// Uses existing G: packages/browser. No download, install, benchmark or source mutation.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';

const args = Object.fromEntries(process.argv.slice(2).map((v,i,a) => v.startsWith('--') ? [v.slice(2),a[i+1]] : null).filter(Boolean));
for (const key of ['base-url','replay-json','run-root','truth','real-failed-json','real-failed-run-root','real-failed-truth','real-failed-sha','negative-json','negative-run-root','negative-sha','repo-root','viz-root','browser-executable','output','expected-sha']) assert(args[key], `--${key} required`);
const out=path.resolve(args.output), allowed=path.resolve(import.meta.dirname);
assert(out.startsWith(allowed+path.sep), 'Fresh output must stay inside the assigned acceptance directory');
assert(!fs.existsSync(out), 'Never overwrite an earlier acceptance attempt');
fs.mkdirSync(out);
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const hash=p=>sha(fs.readFileSync(p));
const base=new URL(args['base-url']); assert.equal(base.hostname,'127.0.0.1');
const report={status:'started',utc:new Date().toISOString(),command:process.argv,node:process.version,
  scope:'Recorded SLAM functional browser acceptance; headless SwiftShader; shared host; no FPS or performance claim',
  headless:true,renderer_mode:'software SwiftShader',cases:[],errors:[]};
const save=()=>fs.writeFileSync(path.join(out,'browser-acceptance.json'),JSON.stringify(report,null,2)+'\n');
save(); let context;
const floats=p=>fs.readFileSync(p,'utf8').trim().split(/\s+/).map(Number);
const positions=poses=>poses.flatMap(p=>[p[3],-p[7],-p[11]]).map(Math.fround);
const same=(a,b)=>a.length===b.length && a.every((v,i)=>Object.is(v,b[i]) || v===b[i]);

function verifyRaw(run,file,runRoot,full,truthPath,nativeSourceSha,realFailed=false) {
  assert.equal(run.schema,'kitti-slam-replay/v1');
  assert.equal(run.provenance.git_sha,nativeSourceSha);
  assert.equal(read(path.join(runRoot,'run.json')).git_sha,nativeSourceSha);
  assert.equal(run.frames.length,run.accepted_frames);
  if(full) { assert.equal(run.kind,'kitti_full'); assert.equal(run.sequence,'07'); assert.equal(run.status,'complete'); assert.equal(run.accepted_frames,1101); assert.equal(run.expected_frames,1101); }
  else {
    assert.equal(run.status,'failed');assert.equal(run.optimized_available,false);assert.equal(run.failure.failed_frame_pose,null);assert.equal(run.failure.optimized_trajectory,null);
    assert.deepEqual(run.failure,read(path.join(runRoot,'native','failure.json')));assert.equal(run.failure.accepted_prefix_frames,run.accepted_frames);
    if(realFailed){assert.equal(run.kind,'kitti_full');assert.equal(run.sequence,'07');assert.equal(run.expected_frames,1101);assert.equal(run.accepted_frames,27);assert.match(run.dataset,/KITTI/i);assert.doesNotMatch(run.dataset,/synthetic/i);assert.match(run.attribution,/KITTI/i);}
    else {assert(!run.kind.startsWith('kitti'),'The requested negative fixture must retain its actual synthetic label');assert.equal(run.accepted_frames,2);assert.match(run.dataset,/synthetic/i);assert.match(run.attribution,/synthetic/i);}
  }
  const times=floats(path.join(runRoot,'native',full?'timestamps.txt':'timestamps_partial.txt'));
  const odom=floats(path.join(runRoot,'native',full?'poses_odom.txt':'poses_odom_partial.txt'));
  const optimized=full?floats(path.join(runRoot,'native','poses_slam.txt')):null;
  assert.equal(times.length,run.frames.length); assert.equal(odom.length,run.frames.length*12);
  if(full)assert.equal(optimized.length,run.frames.length*12);
  run.frames.forEach((f,i)=>{assert.equal(f.frame_id,i);assert.equal(f.timestamp_s,times[i]);assert(Number.isFinite(f.timestamp_s) && (i===0 || f.timestamp_s>times[i-1]));assert.deepEqual(f.odom,odom.slice(i*12,(i+1)*12));if(full)assert.deepEqual(f.optimized,optimized.slice(i*12,(i+1)*12));else assert.equal(f.optimized,null);});
  const artifacts=run.provenance.source_artifacts_sha256;
  for(const [name,digest] of Object.entries(artifacts)) {
    if(name==='external_ground_truth') { assert(truthPath,'Exact external truth path required by this export'); assert.equal(hash(truthPath),digest); continue; }
    const artifact=path.resolve(runRoot,name); assert(artifact.startsWith(path.resolve(runRoot)+path.sep));assert.equal(hash(artifact),digest,`Source artifact changed: ${name}`);
  }
  if(artifacts.external_ground_truth) {
    const gt=floats(truthPath);assert.equal(gt.length,run.expected_frames*12);const first=gt.slice(0,12);
    run.frames.forEach((f,i)=>{assert(f.truth);const current=gt.slice(i*12,(i+1)*12),normalized=[];
      for(let r=0;r<3;r++){for(let c=0;c<3;c++)normalized.push([0,1,2].reduce((s,k)=>s+first[k*4+r]*current[k*4+c],0));normalized.push([0,1,2].reduce((s,k)=>s+first[k*4+r]*(current[k*4+3]-first[k*4+3]),0));}
      f.truth.forEach((v,k)=>assert(Math.abs(v-normalized[k])<1e-8,'Initial-camera0 truth normalization mismatch'));
    });
  } else assert(run.frames.every(f=>f.truth===null),'Unexpected unproven truth track');
  assert.equal(run.coordinates.pose,'T_camera0_initial_camera0_i row-major 3x4');
  assert.equal(run.coordinates.alignment,'initial-frame only; no ATE/scale alignment');
  assert.deepEqual(run.coordinates.T_three_camera0_initial,[1,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,1]);
  assert.equal(run.point_cloud.status,'not_exported');
  return {file:path.resolve(file),sha256:hash(file),kind:run.kind,data_class:full?'complete recorded KITTI':realFailed?'real recorded KITTI failed prefix':'synthetic failed fixture',native_git_sha:nativeSourceSha,status:run.status,accepted_frames:run.accepted_frames,failure:run.failure,
    expected_frames:run.expected_frames,all_original_ids_times_poses_match_native:true,source_artifact_hashes_match:true,
    gt_source_sha256:artifacts.external_ground_truth??null,truth_normalization_matches:artifacts.external_ground_truth?true:'No truth in actual export',loop_counts:{accepted:run.loops.filter(l=>l.accepted).length,rejected:run.loops.filter(l=>!l.accepted&&l.reason!=='temporal_exclusion').length,temporal:run.loops.filter(l=>l.reason==='temporal_exclusion').length},checks:{}};
}

try {
  report.git_sha=execFileSync(args.git??'git',['-C',args['repo-root'],'rev-parse','HEAD'],{encoding:'utf8'}).trim();
  assert.equal(report.git_sha,args['expected-sha']);
  assert.equal(execFileSync(args.git??'git',['-C',args['repo-root'],'status','--porcelain=v1'],{encoding:'utf8'}).trim(),'');
  const sources=['src/main.tsx','src/slam/entry.ts','src/slam/replay.ts','src/slam/KittiReplay.tsx'];
  report.ui_source_sha256=Object.fromEntries(sources.map(f=>[f,hash(path.join(args['viz-root'],f))]));
  // Coordinate/heading contract is checked in the actual unmodified source as well as rendered buffers.
  const scene=fs.readFileSync(path.join(args['viz-root'],'src/slam/KittiReplay.tsx'),'utf8');
  assert(scene.includes('new THREE.Matrix4().makeScale(1,-1,-1)'));
  assert(scene.includes('markers[k].matrix.multiplyMatrices(transform,poseMatrix(p))'));
  report.camera0_heading_source_contract='marker model = diag(1,-1,-1,1) * complete row-major camera0 SE3; source hashes retained';
  const require=createRequire(path.join(path.resolve(args['viz-root']),'package.json'));
  const {chromium}=require('playwright-core');
  // One browser context; complete trajectory once, then the two original failed attempts.
  // Historical failures keep their own accepted native checkpoint after the full-run fix.
  const cases=[['full07',args['replay-json'],args['run-root'],true,args.truth,args['expected-sha'],false],['real-kitti-failed27',args['real-failed-json'],args['real-failed-run-root'],false,args['real-failed-truth'],args['real-failed-sha'],true],['synthetic-negative',args['negative-json'],args['negative-run-root'],false,args['negative-truth'],args['negative-sha'],false]];
  const prepared=cases.map(([name,file,root,full,truth,nativeSourceSha,realFailed])=>({name,file,run:read(file),record:verifyRaw(read(file),file,root,full,truth,nativeSourceSha,realFailed)}));
  prepared.forEach(c=>report.cases.push({name:c.name,...c.record})); save();
  context=await chromium.launchPersistentContext(path.join(out,'profile'),{
    executablePath:args['browser-executable'],headless:true,viewport:{width:1440,height:1000},
    args:['--use-angle=swiftshader','--enable-unsafe-swiftshader',`--disk-cache-dir=${path.join(out,'browser-cache')}`]});
  report.browser_version=context.browser()?.version()??'persistent browser; see user agent';
  for(let n=0;n<prepared.length;n++) {
    const {name,file,run}=prepared[n], item=report.cases[n],dir=path.join(out,name);fs.mkdirSync(dir);
    const page=await context.newPage();page.setDefaultTimeout(30000);const errors=[],consoleLog=[];
    page.on('pageerror',e=>errors.push(String(e)));page.on('console',m=>consoleLog.push({type:m.type(),text:m.text()}));
    // Small passive hook: copy actual Float32 ARRAY_BUFFER uploads; call the original GL method unchanged.
    await page.addInitScript(()=>{
      window.__slamBuffers=[];const original=HTMLCanvasElement.prototype.getContext,seen=new WeakSet();
      HTMLCanvasElement.prototype.getContext=function(...a){const gl=original.apply(this,a);if(gl && /^webgl/.test(String(a[0])) && !seen.has(gl)){seen.add(gl);const upload=gl.bufferData;gl.bufferData=function(...b){if(b[0]===gl.ARRAY_BUFFER && b[1] instanceof Float32Array && b[1].length<=1000000){window.__slamBuffers.push(Array.from(b[1]));if(window.__slamBuffers.length>256)window.__slamBuffers.shift();}return upload.apply(this,b);};}return gl;};
    });
    const replayPath=`/data/slam/${name}/replay.json`,replayUrl=new URL(replayPath,base).href;
    const url=new URL(base);url.searchParams.set('view','slam');url.searchParams.set('replay',replayPath);item.url=url.href;
    const responsePromise=page.waitForResponse(r=>r.url()===replayUrl);
    await page.goto(url.href,{waitUntil:'networkidle'});const response=await responsePromise;
    assert.equal(response.status(),200);const served=await response.body();assert.equal(sha(served),hash(file));
    fs.writeFileSync(path.join(dir,'served-replay.json'),served);item.served_sha256=sha(served);
    const servedJson=JSON.parse(served.toString('utf8'));assert.deepEqual(servedJson.frames,run.frames);
    const status=await page.getByTestId('slam-status').innerText();assert(status.includes(`${run.accepted_frames}/${run.expected_frames}`));
    item.status_text=status;item.checks.exact_browser_json_all_original_frames=true;
    if(run.status==='failed'){
      const title=await page.locator('h2').innerText();if(name==='synthetic-negative')assert.match(title,/synthetic/i);else {assert.match(title,/KITTI/i);assert.doesNotMatch(title,/synthetic/i);}
      assert.match(status,/Failed attempt; accepted prefix only/);assert.match(status,/optimization unavailable/);const failureText=await page.getByRole('status').innerText();assert.match(failureText,/failed-frame pose is unknown/);assert(failureText.includes(String(run.failure.message??'see raw journal')));item.failure_text=failureText;
    }
    assert(run.frames.length>0,'The selected actual negative prefix must contain accepted frames');
    await page.getByTestId('slam-viewport').locator('canvas').waitFor();
    item.renderer=await page.locator('canvas').evaluate(c=>{const gl=c.getContext('webgl2'),d=gl?.getExtension('WEBGL_debug_renderer_info');return gl?{renderer:gl.getParameter(d?d.UNMASKED_RENDERER_WEBGL:gl.RENDERER),vendor:gl.getParameter(d?d.UNMASKED_VENDOR_WEBGL:gl.VENDOR),version:gl.getParameter(gl.VERSION),width:c.width,height:c.height}:{available:false};});
    assert.match(item.renderer.renderer,/SwiftShader/i);item.user_agent=await page.evaluate(()=>navigator.userAgent);
    await page.waitForFunction(()=>window.__slamBuffers.length>0);
    const buffers=await page.evaluate(()=>window.__slamBuffers);
    item.tracks={};for(const track of ['odom','optimized','truth']){const poses=run.frames.flatMap(f=>f[track]?[f[track]]:[]);if(poses.length){assert(buffers.some(b=>same(b,positions(poses))),`Actual WebGL ${track} trajectory buffer differs from camera0 export`);item.tracks[track]={points:poses.length,actual_float32_xyz_upload_matches:true};}else item.tracks[track]={points:0,available:false};}
    item.checks.actual_3d_camera0_track_buffers=true;
    const slider=page.getByLabel('SLAM time');item.scrub=[];
    for(const [key,index] of [['Home',0],['End',run.frames.length-1]]){
      await slider.focus();await slider.press(key);await page.waitForFunction(i=>document.querySelector('[data-testid="slam-frame"]')?.getAttribute('data-frame-id')===String(i),index);
      const text=await page.getByTestId('slam-frame').innerText();assert(text.includes(run.frames[index].timestamp_s.toFixed(6)));
      item.scrub.push({key,frame_id:index,raw_timestamp_s:run.frames[index].timestamp_s,display:text,slider_value:Number(await slider.inputValue())});
      await page.screenshot({path:path.join(dir,`${key.toLowerCase()}.png`),fullPage:true});
    }
    const expectedRows=run.loops.filter(l=>l.query_frame<=run.frames.length-1 && l.reason!=='temporal_exclusion').slice(-12);
    const rows=await page.locator('[data-testid="slam-loop-decisions"] tbody tr').evaluateAll(nodes=>nodes.map(r=>({id:Number(r.getAttribute('data-proposal-id')),cells:[...r.querySelectorAll('td')].map(c=>c.textContent)})));
    assert.deepEqual(rows,expectedRows.map(l=>({id:l.proposal_id,cells:[`${l.query_frame} → ${l.candidate_frame}`,l.accepted?'geometry accepted':'rejected',l.reason]})));
    item.visible_decision_rows=rows;item.checks.actual_loop_decisions_and_reasons=true;item.layers=[];
    await page.getByLabel('SLAM loop layer').selectOption('none');
    await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    for(const mode of ['accepted','rejected','temporal','none']){
      await page.evaluate(()=>{window.__slamBuffers=[];});await page.getByLabel('SLAM loop layer').selectOption(mode);
      assert.equal(await page.getByLabel('SLAM loop layer').inputValue(),mode);
      await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
      const selected=run.loops.filter(l=>l.endpoints_available && l.query_frame<=run.frames.length-1 && (mode==='accepted'?l.accepted:mode==='rejected'?!l.accepted&&l.reason!=='temporal_exclusion':mode==='temporal'?l.reason==='temporal_exclusion':false));
      if(selected.length){const expected=selected.flatMap(l=>positions([run.frames[l.candidate_frame].odom,run.frames[l.query_frame].odom]));await page.waitForFunction(()=>window.__slamBuffers.length>0);assert((await page.evaluate(()=>window.__slamBuffers)).some(b=>same(b,expected)),`Actual ${mode} edge buffer differs from recorded decisions`);}
      item.layers.push({mode,recorded_edges:selected.length,buffer_matches:selected.length?true:'No such edges in actual data; none manufactured'});
      if(mode==='rejected')await page.screenshot({path:path.join(dir,'rejected.png'),fullPage:true});
    }
    item.page_errors=errors;assert.deepEqual(errors,[]);assert.equal(await page.getByRole('alert').count(),0);
    fs.writeFileSync(path.join(dir,'console.json'),JSON.stringify(consoleLog,null,2)+'\n');item.status='passed';save();await page.close();
  }
  report.status='passed';
}catch(e){report.status='failed';report.failure={type:e.name,message:String(e),stack:e.stack};throw e;}
finally{if(context)await context.close();report.finished_utc=new Date().toISOString();save();}
