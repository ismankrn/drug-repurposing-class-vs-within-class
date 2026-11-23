"""Ambil bagian INDICATIONS AND USAGE (LOINC 34067-9) dari SPL DailyMed."""
import re, sys, json, time, urllib.request
import xml.etree.ElementTree as ET

BASE="https://dailymed.nlm.nih.gov/dailymed/services/v2/spls"
LOINC_IND="34067-9"

def fetch_xml(setid, version=None, tries=3):
    url=f"{BASE}/{setid}.xml" + (f"?version={version}" if version else "")
    for i in range(tries):
        try:
            req=urllib.request.Request(url, headers={"User-Agent":"research-script"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except Exception as e:
            print(f"   [retry {i+1}] {type(e).__name__}: {e}", file=sys.stderr); time.sleep(3*(i+1))
    return None

def strip_ns(tag): return tag.split('}')[-1]

def indications_text(xml_bytes):
    """Kembalikan teks bagian indikasi (string), atau '' bila tidak ada."""
    try: root=ET.fromstring(xml_bytes)
    except ET.ParseError as e: return f"__PARSE_ERROR__ {e}"
    out=[]
    for sec in root.iter():
        if strip_ns(sec.tag)!='section': continue
        code=None
        for ch in sec:
            if strip_ns(ch.tag)=='code': code=ch.get('code'); break
        if code!=LOINC_IND: continue
        buf=[]
        for node in sec.iter():
            if node.text and node.text.strip(): buf.append(node.text.strip())
            if node.tail and node.tail.strip(): buf.append(node.tail.strip())
        out.append(' '.join(buf))
    return '\n'.join(out)

def norm_sentences(txt):
    txt=re.sub(r'\s+',' ',txt)
    parts=re.split(r'(?<=[.;:])\s+|•', txt)
    return [p.strip() for p in parts if len(p.strip())>25]

if __name__=='__main__':
    setid, v_old, v_new = sys.argv[1], sys.argv[2], sys.argv[3]
    a=fetch_xml(setid, v_old); b=fetch_xml(setid, v_new)
    if not a or not b: print("gagal fetch"); sys.exit(1)
    ta, tb = indications_text(a), indications_text(b)
    print(f"versi {v_old}: {len(ta)} char indikasi")
    print(f"versi {v_new}: {len(tb)} char indikasi")
    sa, sb = set(norm_sentences(ta)), set(norm_sentences(tb))
    added = [s for s in norm_sentences(tb) if s not in sa]
    print(f"\n=== KALIMAT BARU di versi {v_new} ({len(added)}) ===")
    for s in added[:20]: print(f"  + {s[:160]}")
