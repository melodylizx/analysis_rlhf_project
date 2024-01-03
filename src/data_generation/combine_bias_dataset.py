import pandas as pd
import worker_modeling

#axis validationidation
comparisons_validation_df = pd.read_pickle('../../data/overlap_comp_val.pkl')

directory_path = '../../data/bias/'
comparisons_validation_df['worker_label'] = comparisons_validation_df['choice']

bias_perfect = worker_modeling.to_parquet(comparisons_validation_df, directory_path,"train", "small_perfect")

train_accuracy = pd.read_parquet('../../data/bias/accuracy/all_accuracy.parquet')
train_coherence = pd.read_parquet('../../data/bias/coherence/all_coherence.parquet')
train_coverage = pd.read_parquet('../../data/bias/coverage/all_coverage.parquet')
train_small_perfect = pd.read_parquet('../../data/bias/small_perfect/train_small_perfect.parquet')
train_perfect = pd.read_parquet('../../data/reliability/perfect/train_perfect.parquet')
validation_perfect = pd.read_parquet('../../data/reliability/perfect/validation_perfect.parquet')

def combine_datasets(train_bias, train_perfect, validation_perfect):
    # Calculate the split size for the train_bias dataset
    split_size = int(0.5 * len(train_bias)) + 1

    # Split the train_bias dataset into two parts
    train_bias_split_1 = train_bias.iloc[:split_size]
    train_bias_split_2 = train_bias.iloc[split_size:]

    # Calculate the perfect size for train_perfect and validation_perfect datasets
    perfect_size = int(split_size / 0.2 - split_size)

    # Combine the datasets to create the training and validation sets
    combined_train = pd.concat([train_bias_split_1, train_perfect]).drop_duplicates()
    combined_valid = pd.concat([train_bias_split_2, validation_perfect]).drop_duplicates()

    # Adjust the size of the combined datasets to match the required size
    combined_train = combined_train.head(split_size + perfect_size)
    combined_valid = combined_valid.head(split_size + perfect_size)
    
    combined_train = combined_train.reset_index(drop=True)
    combined_valid = combined_valid.reset_index(drop=True)

    return combined_train, combined_valid


new_train_accuracy , new_valid_accuracy = combine_datasets(train_accuracy, train_perfect, validation_perfect)
new_train_coherence , new_valid_coherence = combine_datasets(train_coherence, train_perfect, validation_perfect)
new_train_coverage , new_valid_coverage = combine_datasets(train_coverage, train_perfect, validation_perfect)
new_train_perfect, new_valid_perfect = combine_datasets(train_small_perfect, train_perfect, validation_perfect)

new_train_accuracy.to_parquet('../../data/bias/accuracy/train_accuracy.parquet')
new_valid_accuracy.to_parquet('../../data/bias/accuracy/validation_accuracy.parquet')

new_train_coherence.to_parquet('../../data/bias/coherence/train_coherence.parquet')
new_valid_coherence.to_parquet('../../data/bias/coherence/validation_coherence.parquet')

new_train_coverage.to_parquet('../../data/bias/coverage/train_coverage.parquet')
new_valid_coverage.to_parquet('../../data/bias/coverage/validation_coverage.parquet')

new_train_perfect.to_parquet('../../data/bias/small_perfect/train_small_perfect.parquet')
new_valid_perfect.to_parquet('../../data/bias/small_perfect/validation_small_perfect.parquet')