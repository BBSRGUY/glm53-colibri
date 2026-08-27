"""Delimiter / literal sanity check for a C file, with a control comparison.

No compiler is installed on this box, so this is the only structural verification
available. It is NOT a substitute for -fsyntax-only: it catches unbalanced braces
and unterminated string literals (the failure modes of scripted patching), not
type errors, missing declarations, or bad calls.
"""
import sys

BS = chr(92)


def strip(src):
    """Remove comments and the contents of string/char literals, char by char."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n and src[i + 1] == '*':
            j = src.find('*/', i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == '/' and i + 1 < n and src[i + 1] == '/':
            j = src.find('\n', i)
            i = n if j < 0 else j
            continue
        if c in '"\'':
            quote = c
            i += 1
            while i < n:
                if src[i] == BS:
                    i += 2
                    continue
                if src[i] == quote:
                    i += 1
                    break
                if src[i] == '\n':          # unterminated literal
                    return ''.join(out), False
                i += 1
            out.append('""')
            continue
        out.append(c)
        i += 1
    return ''.join(out), True


def report(path, label):
    src = open(path, encoding='utf-8', errors='surrogateescape').read()
    t, ok = strip(src)
    print(f'--- {label} ({len(src.splitlines())} lines) ---')
    print(f'  string literals terminated: {ok}')
    bad = 0
    for o, c, name in (('{', '}', 'braces'), ('(', ')', 'parens'), ('[', ']', 'brackets')):
        d = t.count(o) - t.count(c)
        bad += (d != 0)
        print(f'  {name:9} open={t.count(o):6} close={t.count(c):6} delta={d:+d}')
    return bad == 0 and ok


good = report(sys.argv[1], 'TARGET')
ctrl = report(sys.argv[2], 'CONTROL (unmodified)') if len(sys.argv) > 2 else True
print()
print('RESULT:', 'structurally clean' if good else 'STRUCTURAL PROBLEM')
sys.exit(0 if good else 1)
