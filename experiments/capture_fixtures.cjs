// PLAYWRIGHT_MODULE may point to an existing playwright-core installation.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('fs'); const path = require('path');
(async () => {
    const root = path.resolve(__dirname, '..');
    const browser = await chromium.launch({headless:true,args:['--renderer-process-limit=2']});
    const records=[];
    for (const [name,width,height] of [['desktop',1000,720],['mobile',390,780]]) {
        const page=await browser.newPage({viewport:{width,height},deviceScaleFactor:1});
        await page.goto('file:///' + path.join(__dirname,'fixtures/interface.html').replaceAll('\\','/'));
        const file=`experiments/fixtures/${name}.png`;
        await page.screenshot({path:path.join(root,file)});
        for (const [id,instruction] of [['new','Create a new experiment'],['results','View the results of the preference pilot'],['archive','Archive the project']]) {
            records.push({id:`${name}-${id}`,image:file,width,height,instruction,box:await page.locator('#'+id).boundingBox()});
        }
        records.push({id:`${name}-absent`,image:file,width,height,instruction:'Click the Delete account button',box:null});
        await page.close();
    }
    fs.writeFileSync(path.join(__dirname,'fixtures/coordinates.json'),JSON.stringify(records,null,2)+'\n');
    fs.copyFileSync(path.join(__dirname,'fixtures/desktop.png'),path.join(root,'images/coordinate-fixture.png'));
    await browser.close();console.log('Saved 8 coordinate fixtures (6 targets, 2 absent).');
})().catch(e=>{console.error(e);process.exitCode=1});
