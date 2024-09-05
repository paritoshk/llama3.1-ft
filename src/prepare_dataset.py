from datasets import load_dataset

def prepare_dataset():
    # Load the MIMIC-III dataset
    dataset = load_dataset("dmacres/mimiciii-hospitalcourse-meta")
    
    # We'll use the 'train' split if available, otherwise use the default split
    if 'train' in dataset:
        dataset = dataset['train']
    
    # Save the dataset locally
    dataset.save_to_disk("./fine_tuning_dataset")
    print(f"Dataset prepared with {len(dataset)} examples.")
    
    # Print some information about the dataset
    print("\nDataset columns:")
    for col in dataset.column_names:
        print(f"- {col}: {dataset[col].dtype}")
    
    print("\nSample entry:")
    print(dataset[0])

if __name__ == "__main__":
    prepare_dataset()