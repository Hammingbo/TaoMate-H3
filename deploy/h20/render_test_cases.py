"""Create a private static review page alongside the benchmark artifacts."""
import argparse
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    groups = []
    for source in range(1, 10):
        cards = []
        entries = [entry for entry in manifest['cases'] if entry['source_index'] == source]
        for entry in entries:
            case = json.loads(Path(entry['case_file']).read_text())
            segment = case['segments'][0]
            case_id = entry['case_id']
            start = entry['segment_index']*5
            cards.append(f'''<article data-case="{html.escape(case_id)}">
              <h3>片段 {entry['segment_index']+1:02d} · 参考 {start}–{start+5}s</h3>
              <p class="state">等待生成</p>
              <video controls playsinline preload="none" data-src="renders/{html.escape(case_id)}/delivery/video.mp4"></video>
              <p>{html.escape(segment['speech_text'])}</p>
              <details><summary>查看输入 prompt</summary><pre>{html.escape(segment['prompt'])}</pre></details>
              </article>''')
        anchor = Path(case['initial_conditions'][0]['uri']).resolve().relative_to(root).as_posix()
        groups.append(f'''<section><header><img loading="lazy" src="{html.escape(anchor)}" alt="输入首帧">
           <div><h2>任务 {source:02d} · {html.escape(case['title'])}</h2>
           <p>12 段独立生成 · 每段 5 秒 · 共用本任务的编辑首帧</p></div></header>
           <div class="grid">{''.join(cards)}</div></section>''')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>TaoMate-H3 · 108 段测试</title>
    <style>body{margin:0;background:#11151d;color:#eaf0fa;font:15px system-ui,sans-serif}
    main{max-width:1500px;margin:auto;padding:28px}h1{margin:0}h2{font-size:20px}
    .intro{padding:24px 0;border-bottom:1px solid #364052}section{margin:40px 0}
    header{display:flex;gap:20px;align-items:center;margin-bottom:20px}header img{height:130px;border-radius:8px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
    article{background:#1b2330;border:1px solid #354153;border-radius:12px;padding:16px}
    h3{font-size:16px}video{display:block;width:100%;aspect-ratio:768/1376;background:#080b10;max-height:400px}
    .state{color:#a9b5c7}.passed{color:#73e6ae}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
    summary{cursor:pointer;color:#a8cdff}p{line-height:1.7}</style><main>
    <div class="intro"><h1>TaoMate-H3 · 9 组 × 12 段</h1>
    <p>首帧单图条件 · 无尾帧约束 · 768×1376 · 24 fps · 交付时长严格 5 秒</p>
    <strong id="progress">正在读取生成状态…</strong></div>''' + ''.join(groups) + '''</main>
    <script>async function refresh(){try{const response=await fetch('renders/status.json',{cache:'no-store'});
    if(!response.ok)return;const status=await response.json();
    document.getElementById('progress').textContent=`已通过 ${status.passed}/108 段`;
    for(const item of status.cases){const card=document.querySelector(`[data-case="${item.case_id}"]`);
    if(!card)continue;const label=card.querySelector('.state');
    label.textContent=item.passed?'已生成 · 完整解码通过':item.teacher_complete?'等待视频生成':'等待生成';
    label.classList.toggle('passed',item.passed);const video=card.querySelector('video');
    if(item.passed&&!video.getAttribute('src'))video.src=video.dataset.src;}}
    catch(error){document.getElementById('progress').textContent='等待生成状态';}}
    document.addEventListener('play',event=>{if(event.target.tagName==='VIDEO')
    document.querySelectorAll('video').forEach(video=>{if(video!==event.target)video.pause();});},true);
    refresh();setInterval(refresh,15000);</script></html>'''
    (root/'index.html').write_text(page)
    print(f'REVIEW_PAGE_READY {root / "index.html"}')


if __name__ == '__main__':
    main()
