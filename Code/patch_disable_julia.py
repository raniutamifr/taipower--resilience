f = r'C:\reXplan-repo\Project Taipower\CODE\9 OPF_Solver.py'
lines = open(f, encoding='utf-8').readlines()

fixed = []
skip_next = 0
for i, line in enumerate(lines):
    if skip_next > 0:
        skip_next -= 1
        continue
    
    # Line 153: orphan continuation after raise Exception (2 lines to skip)
    if i == 152:  # 0-indexed = line 153
        fixed.append('        raise Exception("Julia disabled")\n')
        skip_next = 1  # skip line 154
        continue
    
    # Line 189: orphan continuation after raise Exception (3 lines to skip)
    if i == 188:  # 0-indexed = line 189
        fixed.append('            raise Exception("Julia disabled")\n')
        skip_next = 2  # skip lines 190, 191
        continue
    
    fixed.append(line)

open(f, 'w', encoding='utf-8').write(''.join(fixed))
print('Fixed! Verifikasi:')
lines2 = open(f, encoding='utf-8').readlines()
for i in range(150, 158):
    print(f'{i+1}: {lines2[i]}', end='')
print('...')
for i in range(186, 194):
    print(f'{i+1}: {lines2[i]}', end='')