import urllib.request, urllib.parse, re, html, sys
def search(q, n=8, sort='submittedDate', order='descending'):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": q, "max_results": n, "sortBy": sort, "sortOrder": order})
    with urllib.request.urlopen(url, timeout=45) as r:
        t = r.read().decode('utf-8','ignore')
    entries = re.findall(r'<entry>(.*?)</entry>', t, re.S)
    out=[]
    for e in entries:
        idm=re.search(r'<id>(.*?)</id>',e); ti=re.search(r'<title>(.*?)</title>',e,re.S)
        pm=re.search(r'<published>(.*?)</published>',e)
        sm=re.search(r'<summary>(.*?)</summary>',e,re.S)
        out.append((idm.group(1) if idm else "",
            html.unescape((ti.group(1) if ti else "").strip()),
            (pm.group(1) if pm else "")[:10],
            html.unescape((sm.group(1) if sm else "").strip())))
    return out
if __name__=="__main__":
    q=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 6
    for id_,ti,pu,sm in search(q,n):
        print("###",pu,id_); print("T:",ti)
        print("S:",sm[:680].replace(chr(10),' ')); print()
