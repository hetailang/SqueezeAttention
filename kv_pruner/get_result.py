import os
import json
import sys

def process_folder(folder_path, result_data):
    result_json_path = os.path.join(folder_path, 'result.json')

    if os.path.exists(result_json_path):
        with open(result_json_path, 'r') as json_file:
            try:
                data = json.load(json_file)
                folder_name = os.path.basename(folder_path)
                result_data[folder_name] = {
                   # 'multi_news': data.get('multi_news'),
                   # 'qmsum': data.get('qmsum'),
                    'samsum': data.get('samsum'),
                   # 'qasper': data.get('qasper'),
                   # 'narrativeqa': data.get('narrativeqa'),
                   # 'multifieldqa_en': data.get('multifieldqa_en'),
                   # 'hotpotqa': data.get('hotpotqa'),
                   # '2wikimqa': data.get('2wikimqa'),
                   # 'musique': data.get('musique'),
                   # 'gov_report': data.get('gov_report'),
                   # 'trec': data.get('trec'),
                   # 'triviaqa': data.get('triviaqa'),
                   # 'passage_count': data.get('passage_count'),
                   # 'passage_retrieval_en': data.get('passage_retrieval_en'),
                   # 'lcc': data.get('lcc'),
                   # 'repobench-p': data.get('repobench-p')
                }
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {result_json_path}")

def main(root_path):
    result_data = {}

    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        if os.path.isdir(folder_path):
            process_folder(folder_path, result_data)

    output_json_path = 'result.json'
    with open(output_json_path, 'w') as result_json_file:
        json.dump(result_data, result_json_file, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_result.py /path/to/root/folder")
    else:
        root_folder_path = sys.argv[1]
        main(root_folder_path)

