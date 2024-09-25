from datasets import load_dataset

# English only
for i in range(20):
    en = load_dataset("allenai/c4", data_files={'train': f'en/c4-train.{str(i).zfill(5)}-of-01024.json.gz'}, cache_dir='./assets/data', split='train')
    print(len(en))

    # save as json files
    en.to_json(f'./assets/data/en/c4-train.{str(i).zfill(5)}-of-01024.json', orient='records', lines=True)