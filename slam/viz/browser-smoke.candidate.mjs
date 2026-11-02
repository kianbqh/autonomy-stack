// Explicit later-stage UI acceptance. NOT invoked during private preparation.
// Reuses the viewer's existing Playwright installation; no managed-browser download.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
const args=Object.fromEntries(process.argv.slice(2).map((v,i,a)=>v.startsWith('--')?[v.slice(2),a[i+1]]:null).filter(Boolean));
for(const key of ['url','replay-json','viz-root','repo-root','browser-executable','output'])if(!args[key])throw new Error(`--${key} required`);
const out=path.resolve(args.output);fs.mkdirSync(out,{recursive:false});
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const json=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const run=json(args['replay-json']),start=performance.now();
const receipt={status:'started',utc:new Date().toISOString(),command:process.argv,node:process.version,replay_sha256:hash(args['replay-json']),native_status:run.status,checks:{},timing_claim:'functional browser acceptance only; not FPS/latency benchmark'};
const save=()=>fs.writeFileSync(path.join(out,'browser-smoke.json'),JSON.stringify(receipt,null,2)+'\n');save();
let context;
try{
  if(run.schema!=='kitti-slam-replay/v1')throw new Error('Wrong schema');
  const git=args.git??'git';receipt.git_sha=execFileSync(git,['-C',args['repo-root'],'rev-parse','HEAD'],{encoding:'utf8'}).trim();
  if(receipt.git_sha!==run.provenance.git_sha)throw new Error('UI/replay native Git checkpoint mismatch');
  const require=createRequire(path.join(path.resolve(args['viz-root']),'package.json'));
  const {chromium}=require('playwright-core');
  receipt.ui_source_sha256=Object.fromEntries(['KittiReplay.tsx','replay.ts'].map(f=>[f,hash(path.join(args['viz-root'],'src/slam',f))]));
  // Persistent profile is explicitly inside the requested G-backed output.
  context=await chromium.launchPersistentContext(path.join(out,'browser-profile'),{executablePath:args['browser-executable'],headless:true,viewport:{width:1440,height:1000}});
  const page=context.pages()[0]??await context.newPage();const errors=[];
  page.on('pageerror',e=>errors.push(String(e)));await page.goto(args.url,{waitUntil:'networkidle'});
  await page.getByTestId('slam-status').waitFor();const status=await page.getByTestId('slam-status').innerText();
  if(!status.includes(`${run.accepted_frames}/${run.expected_frames}`))throw new Error('UI frame count differs from raw export');
  receipt.checks.status_count=true;
  if(run.frames.length){
    const slider=page.getByLabel('SLAM time');await slider.focus();await slider.press('Home');
    await page.waitForFunction(()=>document.querySelector('[data-testid="slam-frame"]')?.getAttribute('data-frame-id')==='0');
    if(await page.getByTestId('slam-frame').getAttribute('data-frame-id')!=='0')throw new Error('Start scrub mismatch');
    await slider.press('End');
    await page.waitForFunction(i=>document.querySelector('[data-testid="slam-frame"]')?.getAttribute('data-frame-id')===String(i),run.frames.length-1);
    if(await page.getByTestId('slam-frame').getAttribute('data-frame-id')!==String(run.frames.length-1))throw new Error('End scrub mismatch');
    receipt.checks.original_frame_scrub=true;await page.getByTestId('slam-viewport').locator('canvas').waitFor();
    receipt.renderer=await page.locator('canvas').evaluate(canvas=>{const gl=canvas.getContext('webgl2');if(!gl)return {available:false};const dbg=gl.getExtension('WEBGL_debug_renderer_info');return {available:true,renderer:dbg?gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER),vendor:dbg?gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL):gl.getParameter(gl.VENDOR)};});
    await page.screenshot({path:path.join(out,'end.png'),fullPage:true});
    const rejected=run.loops.filter(l=>!l.accepted && l.reason!=='temporal_exclusion' && l.endpoints_available).at(-1);
    await page.getByLabel('SLAM loop layer').selectOption('rejected');
    receipt.checks.rejected_layer=true;
    if(rejected){const row=page.locator(`[data-proposal-id="${rejected.proposal_id}"]`);await row.waitFor();if(!(await row.innerText()).includes(rejected.reason))throw new Error('Rejected-loop reason mismatch');receipt.checks.actual_rejected_reason=true;}
    else receipt.checks.actual_rejected_reason='not present in this actual run; no decision invented';
    await page.screenshot({path:path.join(out,'rejected-loops.png'),fullPage:true});
  }else receipt.checks.empty_prefix='no accepted trajectory; failure text only';
  if(run.status==='failed' && !status.includes('Failed attempt'))throw new Error('Failure hidden by UI');
  receipt.checks.failure_state=run.status==='failed'?'actual failed attempt displayed':'not applicable to complete run; use an actual failed attempt separately';
  receipt.page_errors=errors;if(errors.length)throw new Error('Browser page errors');receipt.status='passed';
}catch(e){receipt.status='failed';receipt.failure={type:e?.name??'Error',message:String(e)};throw e;}
finally{if(context)await context.close();receipt.finished_utc=new Date().toISOString();receipt.wall_s=(performance.now()-start)/1000;save();}
