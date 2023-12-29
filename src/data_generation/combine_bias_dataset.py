import pandas as pd

train_accuracy = pd.read_parquet('../../data/bias/accuracy/accuracy.parquet')
train_coherence = pd.read_parquet('../../data/bias/coherence/coherence.parquet')
train_coverage = pd.read_parquet('../../data/bias/coverage/coverage.parquet')
train_perfect = pd.read_parquet('../../data/reliability/perfect/train_perfect.parquet')
validation_perfect = pd.read_parquet('../../data/reliability/perfect/validation_perfect.parquet')

def combine_datasets(train_bias, train_perfect, validation_perfect):
    # Calculate the split size for the train_accuracy dataset
    split_size = int(0.5 * len(train_bias)) + 1

    # Split the train_accuracy dataset into two parts
    train_accuracy_split_1 = train_accuracy.iloc[:split_size]
    train_accuracy_split_2 = train_accuracy.iloc[split_size:]

    # Calculate the perfect size for train_perfect and validation_perfect datasets
    perfect_size = int(split_size / 0.3 - split_size)

    # Trim the train_perfect and validation_perfect datasets to the perfect size
    train_perfect = train_perfect.iloc[:perfect_size]
    validation_perfect = validation_perfect.iloc[:perfect_size]

    # Combine the datasets to create the training and validation sets
    combined_train = pd.concat([train_accuracy_split_1, train_perfect])
    combined_valid = pd.concat([train_accuracy_split_2, validation_perfect])

    return combined_train, combined_valid

new_train_accuracy , new_valid_accuracy = combine_datasets(train_accuracy, train_perfect, validation_perfect)
new_train_coherence , new_valid_coherence = combine_datasets(train_coherence, train_perfect, validation_perfect)
new_train_coverage , new_valid_coverage = combine_datasets(train_coverage, train_perfect, validation_perfect)

new_train_perfect =train_perfect.iloc[:1213]
new_validation_perfect =validation_perfect.iloc[:1212]
new_train_perfect.to_parquet('../../data/bias/small_perfect/train_small_perfect.parquet')
new_validation_perfect.to_parquet('../../data/bias/small_perfect/validation_small_perfect.parquet')
