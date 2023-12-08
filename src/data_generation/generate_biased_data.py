import pandas as pd
import worker_modeling
from utils import create_directory
# Read data from PKL files
comp_val_df = pd.read_pickle('../../data/comp_val.pkl')
comp_train_df = pd.read_pickle('../../data/comp_train.pkl')
directory_path = '../../data/bias'
create_directory(directory_path)

#bias towards 0
bias_towards_0 = True    #Bias towards class 0 over class 1
bias_towards_1 = False  #Bias towards class 1 over class 0
train_bias_0 = worker_modeling.introduce_bias_class(comp_train_df, bias_towards_0, bias_towards_1)

#bias towards 1
bias_towards_0 = False   #Bias towards class 0 over class 1
bias_towards_1 = True  #Bias towards class 1 over class 0
train_bias_1 = worker_modeling.introduce_bias_class(comp_train_df, bias_towards_0, bias_towards_1)

train_bias_0 = worker_modeling.to_parquet(train_bias_0, "train", "bias_0")
train_bias_1 = worker_modeling.to_parquet(train_bias_1, "train", "bias_1")

#axis validation
overlap_axis_val = pd.read_pickle('../../data/overlap_axis_val.pkl')
comparisons_val_df = pd.read_pickle('../../data/overlap_comp_val.pkl')
# # Group and average the evaluation scores
aggregated_eval = overlap_axis_val[['summary_id', 'overall', 'accuracy', 'coverage', 'coherence']].groupby('summary_id').mean().reset_index()


#assign bias
#has to use comparisons_val_df_overlap before generating data
bias_accuracy, bias_coverage, bias_coherence = True,False,False
#bias towards accuracy
bias_accuracy_df = worker_modeling.introduce_bias_aspect(comparisons_val_df, aggregated_eval, bias_accuracy, bias_coverage, bias_coherence)

bias_accuracy, bias_coverage, bias_coherence = False,True,False
#bias towards coverage
bias_coverage_df = worker_modeling.introduce_bias_aspect(comparisons_val_df, aggregated_eval, bias_accuracy, bias_coverage, bias_coherence)

#bias towards coherence
bias_accuracy, bias_coverage, bias_coherence = False,False,True
bias_coherence_df = worker_modeling.introduce_bias_aspect(comparisons_val_df, aggregated_eval, bias_accuracy, bias_coverage, bias_coherence)

bias_accuracy_ = worker_modeling.to_parquet(bias_accuracy_df, directory_path+"/train", "accuracy")
bias_coverage_ = worker_modeling.to_parquet(bias_coverage_df, directory_path+"/train", "coverage")
bias_coherence_ = worker_modeling.to_parquet(bias_coherence_df, directory_path+"/train", "coherence")
