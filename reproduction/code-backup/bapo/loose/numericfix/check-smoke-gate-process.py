import pathlib,json
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric');print('receipt', (root/'logs/smoke2-3094-acceptance.json').exists())
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  a=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'verify_smoke_receipt-gpu64.py' in a and a.startswith(str(root/'env/bin/python')):print(p.name,a,(p/'stat').read_text().split(') ',1)[1].split()[19])
 except (FileNotFoundError,PermissionError):pass
