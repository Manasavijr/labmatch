import os
css = open('/home/user/labmatch/web/app/globals.css').read()
css = css.replace("min-height:100vh;", "").replace('--muted:#675B4E', '--muted:#574B3F').replace('--muted:#A99C8B', '--muted:#C4B7A5')  # MUTED brighter small text
css = open('tokens.css').read() + css + """
h1{text-wrap:balance}
.fact{font-variant-numeric:tabular-nums}
""" + open('shell.css').read()
data = open('/home/user/labmatch/data/build/demo.json').read().replace('</', '<\\/')
ev = '/home/user/labmatch/data/build/eval.json'
evs = open(ev).read().replace('</', '<\\/') if os.path.exists(ev) else 'null'
t = open('template.html').read().replace('/*CSS*/', css).replace('/*DATA*/', data).replace('/*EVAL*/', evs)
open('labmatch.html', 'w').write(t)
print(len(t))
