"""Isolated native-window check; contains no chat or case data."""
from pathlib import Path
import re
import webview
root=Path(__file__).resolve().parents[1]
css=(root/'local_model_app/static/styles.css').read_text(encoding='utf-8')
html=(root/'local_model_app/static/index.html').read_text(encoding='utf-8')
js=(root/'local_model_app/static/app.js').read_text(encoding='utf-8')
# Exercise the production DOM, styles, renderer, and event handlers. All network
# calls are stubbed: this window cannot fetch any real chat or case information.
stub="""window.fetch=async function(path){return {ok:true,status:200,json:async()=>
 path.includes('health')?{loaded:false}:path.includes('activity')?{active:false,events:[]}:[]};};"""
html=re.sub(r'<link rel="stylesheet"[^>]*>',lambda m:'<style>'+css+'</style>',html)
html=re.sub(r'<script src="[^"]*app.js[^"]*"></script>',lambda m:'<script>'+stub+js+"\nsetTimeout(()=>renderMessages([{role:'assistant',content:'Synthetic response: select these words normally.'}]),1000);</script>",html)
webview.create_window('Local Model UI selection check',html=html,width=1280,height=820,text_select=True)
webview.start()
