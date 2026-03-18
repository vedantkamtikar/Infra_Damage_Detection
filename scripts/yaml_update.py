
from pathlib import Path
yaml_path = Path(r'C:\Users\LOQ\Desktop\Infra_Damage_Detection\datasets\ENIM\data.yaml')
base = Path(r'C:\Users\LOQ\Desktop\Infra_Damage_Detection\datasets\ENIM')
content = f'''train: {(base / 'train' / 'images').as_posix()}
val: {(base / 'valid' / 'images').as_posix()}
test: {(base / 'test' / 'images').as_posix()}
nc: 1
names: ['crack']
'''
yaml_path.write_text(content)
print('Done')
print(content)
