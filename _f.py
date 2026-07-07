import urllib.request, urllib.parse, sys, html, re
def get(url, timeout=50):
    req=urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            return r.read().decode('utf-8','ignore')
    except Exception as ex:
        return "ERR:"+str(ex)
if __name__=="__main__":
    url=sys.argv[1]
    t=get(url)
    # strip tags
    txt=re.sub(r'<[^>]+>',' ',t)
    txt=html.unescape(txt)
    txt=re.sub(r'\s+',' ',txt)
    kw=sys.argv[2] if len(sys.argv)>2 else 'reward'
    # find windows around keyword occurrences
    for m in re.finditer(kw, txt, re.I):
        s=max(0,m.start()-160); e=min(len(txt),m.start()+320)
        print("...",txt[s:e],"...")
        print("----")
