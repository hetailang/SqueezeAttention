import transformers
import os, shutil
install_path = transformers.__file__
source_path = './modeling_mistral.py'
target_path = '/'.join(install_path.split('/')[:-1])

target_path_mistral = os.path.join(target_path, 'models/mistral/modeling_mistral.py')
shutil.copy(source_path, target_path_mistral)

source_path = './modeling_llama.py'
target_path_llama = os.path.join(target_path, 'models/llama/modeling_llama.py')
shutil.copy(source_path, target_path_llama)
