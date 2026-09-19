// Run sequentially against a local preview. Dependencies are supplied by path.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs=require('fs');
const path=require('path');
const root=path.resolve(__dirname,'..');
const pages=Object.keys(JSON.parse(fs.readFileSync(path.join(root,'content/pages.json'))));
const origin=process.env.AUDIT_ORIGIN || 'http://127.0.0.1:8768';
const axe=process.env.AXE_CORE_PATH || require.resolve('axe-core/axe.min.js');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--renderer-process-limit=2']});
 const checks=[];
 try {
  const page=await browser.newPage({reducedMotion:'reduce'});
  await page.route('**/*',r=>r.request().url().startsWith(origin)||r.request().url().startsWith('blob:')?r.continue():r.abort());
  async function audit(name,state,width){
   const result=await page.evaluate(async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}}));
   const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
   const violations=result.violations.map(v=>({id:v.id,impact:v.impact,nodes:v.nodes.map(n=>({target:n.target,reason:n.failureSummary}))}));
   checks.push({page:name,state,width,overflow,violations,manualReview:result.incomplete.map(v=>({id:v.id,targets:v.nodes.map(n=>n.target)}))});
   if(overflow||violations.length) console.log(JSON.stringify({page:name,state,width,overflow,violations}));
  }
  for(const width of [320,1440]){
   await page.setViewportSize({width,height:1000});
   for(const name of pages){
    await page.goto(origin+'/'+name);
    await page.addScriptTag({path:axe});
    await audit(name,'initial',width);
    await page.locator('main details').evaluateAll(items=>items.forEach(item=>item.open=true));
    await audit(name,'expanded',width);
    await page.addStyleTag({content:'html{font-size:200%!important} *{line-height:1.5!important;letter-spacing:.12em!important;word-spacing:.16em!important} p{margin-bottom:2em!important}'});
    const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
    checks.push({page:name,state:'expanded with enlarged text and spacing',width,overflow});
    if(overflow)console.log(JSON.stringify(checks.at(-1)));
   }
  }
  await page.setViewportSize({width:390,height:900});
  await page.goto(origin+'/demo.html');
  await page.addScriptTag({path:axe});
  for(let i=0;i<12;i++)await page.locator(i%2?'#like-btn':'#pass-btn').click();
  await page.locator('main details').evaluateAll(items=>items.forEach(item=>item.open=true));
  await audit('demo.html','trained and expanded',390);
  for(let i=0;i<8;i++)await page.locator(i%2?'#like-btn':'#pass-btn').click();
  await audit('demo.html','suggestion dialog',390);
  await page.keyboard.press('Escape');
  await page.setViewportSize({width:320,height:900});
  await page.addStyleTag({content:'html{font-size:200%!important} *{line-height:1.5!important;letter-spacing:.12em!important;word-spacing:.16em!important} p{margin-bottom:2em!important}'});
  await audit('demo.html','trained with enlarged text and spacing',320);
  await page.goto(origin+'/adaptive.html');
  await page.addScriptTag({path:axe});
  await page.locator('.lab-menu').click();
  await audit('adaptive.html','mobile navigation open',320);
  await page.keyboard.press('Escape');
  await page.emulateMedia({forcedColors:'active'});
  const forcedColors=await page.locator('.processing-steps i').evaluateAll(nodes=>nodes.map(n=>({border:getComputedStyle(n).borderStyle,fill:getComputedStyle(n).backgroundColor})));
  if(forcedColors.slice(0,6).some(n=>n.border!=='solid')||forcedColors.slice(6).some(n=>n.border!=='dashed'))throw Error('Early-stop distinction lost in forced colors');
  await page.screenshot({path:path.join(root,'results/ui/adaptive-forced-colors.png'),fullPage:true});
  await page.emulateMedia({forcedColors:'none'});
  function luminance(hex){const rgb=hex.match(/[\da-f]{2}/gi).map(v=>parseInt(v,16)/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4);return .2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2];}
  const contrastPairs=[['body','#182438','#f4f7fb'],['secondary text','#43536a','#f4f7fb'],['used blocks','#315d8e','#ffffff'],['skipped outlines','#596b80','#ffffff'],['control borders','#718198','#ffffff'],['focus ring','#204bb8','#f4f7fb'],['menu glyph','#182438','#ffffff'],['diagram arrows','#182438','#eaf0fc'],['hit marks','#37694a','#e3eee7'],['miss marks','#914f3d','#f6e8e2'],['card chips worst-case white underlay','#ffffff','#525864'],['ring label worst-case black underlay','#1e293b','#d9d9d9']].map(([name,foreground,background])=>({name,foreground,background,ratio:Number(((Math.max(luminance(foreground),luminance(background))+.05)/(Math.min(luminance(foreground),luminance(background))+.05)).toFixed(2))}));
  const report={date:new Date().toISOString(),browser:await browser.version(),axeVersion:await page.evaluate(()=>axe.version),pages:pages.length,checks,contrastPairs,forcedColors:'Filled solid outlines and empty dashed outlines remain distinct.',limitations:['Automated checks do not establish WCAG conformance.','Screen reader, physical devices, and Safari have not been tested.','Archived historical pages are outside this current-site audit.','Axe flags decorative glyphs and overlays for manual contrast review; palette and worst-case composited colors are recorded separately.']};
  const target=process.env.AUDIT_OUTPUT || path.join(root,'results/ui/accessibility.json');
  fs.writeFileSync(target,JSON.stringify(report,null,2)+'\n');
  const failures=checks.filter(c=>c.overflow||c.violations?.length);
  console.log(JSON.stringify({checks:checks.length,failedChecks:failures.length,output:target}));
  if(failures.length)process.exitCode=1;
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
