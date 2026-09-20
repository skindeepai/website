"""One focused page for the fresh visual grounding/refusal stress test."""
import json
import html
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def update(pages):
    result_path = ROOT / 'results/visual-next/result.json'
    if not result_path.exists():
        return
    result = json.loads(result_path.read_text(encoding='utf-8'))
    protocol = json.loads((ROOT / 'results/visual-next/protocol.json').read_text(encoding='utf-8'))
    predictions = json.loads((ROOT / 'results/visual-next/predictions.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT / 'results/visual-next/example.png').read_bytes()).hexdigest() == protocol['samples'][0]['image_sha256']
    methods = result['methods']
    labels = {'max_patch': 'Highest-scoring patch', 'connected_region': 'Combine nearby patches', 'peak_gate': 'Combine + confidence cutoff', 'relative_peak_gate': 'Combine + grid-adjusted cutoff'}
    rows = ''.join('<tr><th scope="row">' + labels[key] + '</th>' + ''.join('<td>' + str(row[field]) + '</td>' for field in ['correct_clicks', 'wrong_present_clicks', 'absent_clicks', 'withheld_present', 'refused_absent']) + '</tr>' for key, row in methods.items())
    main_rows = ''.join('<tr><th scope="row">' + label + '</th>' + ''.join('<td>' + str(methods[key][field]) + '</td>' for field in ['correct_clicks', 'absent_clicks', 'withheld_present']) + '</tr>' for key, label in [('connected_region', 'Always return a point'), ('peak_gate', 'Hold low-confidence clicks')])
    gate = methods['peak_gate']
    crop = result['crop_study']
    body = '''<p><a href="@/coordinates.html">Back to finding where to click</a> &middot; <a href="@/screenshot-demo.html">Explore the earlier screenshot demo</a></p>
<p>Sometimes the requested button is not on the screen. We tested <strong>50 new real screenshots</strong>: 25 with a target and 25 requests that the dataset authors label infeasible.</p>
<figure class="small-example"><svg viewBox="0 0 360 105" role="img" aria-label="Screenshot and instruction pass through the vision model. A confidence check either returns a point or holds the click." style="width:100%;max-width:360px;height:auto"><g fill="none" stroke="#395775" stroke-width="2"><rect x="2" y="28" width="88" height="48" rx="5"/><path d="M90 52h20"/><rect x="110" y="28" width="98" height="48" rx="5"/><path d="M208 52h20V22h14M228 52v31h14"/></g><g fill="#152c45" font-family="system-ui,sans-serif" font-size="18" text-anchor="middle"><text x="46" y="48">Screen</text><text x="46" y="66">+ request</text><text x="159" y="48">Vision</text><text x="159" y="66">model</text><text x="296" y="28">Click a point</text><text x="296" y="89">Hold click</text></g></svg><figcaption>The model returns numbers. The cutoff can withhold an action.</figcaption></figure>
<div class="table-scroll" role="region" tabindex="0" aria-label="Clicks and withheld requests"><table><thead><tr><th scope="col">Method</th><th scope="col">Correct clicks / 25</th><th scope="col">Infeasible but clicked / 25</th><th scope="col">Present but withheld</th></tr></thead><tbody>''' + main_rows + '</tbody></table></div>'
    body += f'<p>The confidence cutoff allowed <strong>{gate["correct_clicks"]} correct clicks</strong>, {gate["wrong_present_clicks"]} wrong clicks on present targets, and <strong>{gate["absent_clicks"]} clicks on infeasible requests</strong>. It also withheld {gate["withheld_present"]} requests with a real target, including {gate["withheld_correct_present"]} points the model had already located correctly.</p>'
    body += '<p><strong>Holding a click is not proof that a target is missing.</strong> The output is UNCERTAIN; a withheld present request remains unfinished. This is a balanced stress test, not a production success rate.</p>'
    example = predictions[0]
    point = example['points']['connected_region']
    output = 'CLICK' if example['peak_probability'] >= protocol['gates']['peak_gate']['cutoff'] else 'UNCERTAIN'
    body += f'''<details><summary>See the first test case</summary><p><strong>Request:</strong> {html.escape(protocol['samples'][0]['instruction'])}</p><div style="position:relative;max-width:480px"><img src="@/results/visual-next/example.png" alt="Ubuntu Files window. The Show Applications button is at the bottom left; the requested Hide applications control is not shown." loading="lazy" style="display:block;width:100%;height:auto"><span aria-hidden="true" style="position:absolute;left:{point[0]*100:.5f}%;top:{point[1]*100:.5f}%;width:12px;height:12px;border:2px solid white;outline:2px solid #1743ae;background:#1743ae;border-radius:50%;transform:translate(-50%,-50%)"></span></div><p>The blue point is the model\u2019s proposed click. The dataset labels this request infeasible. The confidence rule returned <strong>{output}</strong>. This is a saved result, not live inference.</p><p class="small">First request in the sealed order, not selected by its outcome. <a href="@/results/visual-next/example.png">Open the original screenshot</a>.</p></details>'''
    body += f'<h2>Does a closer look help?</h2><p>For 10 requests chosen before inference, we cropped around the model\u2019s predicted point and ran it again. Correct locations changed from <strong>{crop["before_correct"]} / 10 to {crop["after_correct"]} / 10</strong>: {crop["corrected"]} corrected and {crop["lost"]} lost. This costs a second model pass; the crop never uses the correct target.</p>'
    body += '<details><summary>All four readout variants</summary><div class="table-scroll" role="region" tabindex="0" aria-label="Detailed readout variants"><table><thead><tr><th scope="col">Method</th><th scope="col">Correct clicks</th><th scope="col">Wrong target</th><th scope="col">Infeasible but clicked</th><th scope="col">Present but withheld</th><th scope="col">Infeasible and withheld</th></tr></thead><tbody>' + rows + '</tbody></table></div></details>'
    body += '''<details><summary>What actually ran?</summary><p>GUI-Actor-2B-Qwen2-VL processed every screenshot and instruction on a CPU. Its trained pointer head scores image patches and returns a location without generating text. All 28 language-model layers ran, along with the vision encoder. No layers were skipped.</p><p>The two cutoffs were selected using the earlier 30 ScreenSpot examples, before seeing any new OSWorld-G model outputs. The new benchmark includes Ubuntu desktop screens outside that gate-development set; model-training overlap is unknown. There is no dedicated NOT_FOUND class in this pointer head.</p><p>Inputs use the authors\u2019 refined instructions and a maximum of 576 visual tokens. Timing was diagnostic while other experiments ran, so we make no speed claim. These 50 requests do not constitute a full benchmark evaluation.</p></details>
<p class="small"><a href="@/docs/visual-next.md">Method, dataset, every result and limitations</a> &middot; <a href="@/results/visual-next/protocol.json">Protocol fixed before inference</a> &middot; <a href="https://github.com/xlang-ai/OSWorld-G">OSWorld-G dataset</a></p>'''
    pages['visual-refusal-results.html'] = {'title': 'Click, or hold the action?', 'description': 'A real-screen test that includes infeasible requests.', 'status': 'Fresh benchmark stress test', 'styles': ['results.css'], 'body': body}
    link = f'<p>A harder desktop test found <strong>{methods["connected_region"]["correct_clicks"]} of 25 targets</strong> with the same model. <a href="@/visual-refusal-results.html">See the new test, including infeasible requests</a>.</p>'
    old_link = '<p><a href="@/visual-refusal-results.html">New test: 50 real screens, including infeasible requests</a></p>'
    section = '<section id="visual-fresh-followup">' + link + '</section>'
    for name in ['coordinates.html', 'screenshot-demo.html']:
        if name in pages:
            body = pages[name]['body'].replace(old_link, '').replace(link, '')
            if '<section id="visual-fresh-followup">' in body:
                body = re.sub(r'<section id="visual-fresh-followup">.*?</section>', lambda _: section, body, flags=re.S)
            else:
                body += section
            pages[name]['body'] = body
