import pdb
import random
import pandas as pd
import numpy as np
import random
from utils import create_directory

def assign_instances(num_comparisons, worker_assignments):
    assignments = {}
    worker_assignments.sort_values(by='n_assignments', ascending=False)
    pairs_w = pd.DataFrame(0,index=range(num_comparisons), columns=["count_w"])
    pair_worker = np.array([[],[]])
    pool = range(1, num_comparisons)
    for worker in worker_assignments.worker_id:
        worker_info = worker_assignments[worker_assignments['worker_id']==worker]
        w_assignments = worker_info.n_assignments.iloc[0]
        #randomly assign w_assignments pairs to worker
        assignments_worker = random.sample(pool, min(len(pool),w_assignments))
        assignments[worker] = assignments_worker
        pairs_w["count_w"].iloc[assignments_worker] += 1
        worker_pair = np.array([assignments_worker, [worker]*len(assignments_worker)])
        pair_worker = np.append(pair_worker,worker_pair,axis=1)
        if pairs_w[pairs_w["count_w"]>3].shape[0]>0:
            pool = list(pairs_w[pairs_w["count_w"]<3].index)

    return pair_worker,assignments,pairs_w


def assign_rel_workers(pair_worker_assignment,dict_rel):
    pair_worker_assignment['worker_label'] = -1
    for worker in dict_rel:
        w_rel = dict_rel[worker]
        w_pairs = pair_worker_assignment[pair_worker_assignment['worker'] == worker]
        w_pairs_rel = fixed_reliability(w_rel, w_pairs)
        pair_worker_assignment['worker_label'][w_pairs_rel.index] = w_pairs_rel.worker_label
    return pair_worker_assignment

def fixed_reliability(reliability, comp_original_df):
    comp_df = comp_original_df.copy()
    # Sample reliability % of the DataFrame
    sampled_df = comp_df['choice'].sample(frac=1-reliability, random_state=1)
    # Invert the values in the 'binary_column'
    sampled_df = sampled_df.replace({0: 1, 1: 0})
    # Update the original DataFrame with the sampled and inverted values
    comp_df['worker_label']= comp_df['choice']
    comp_df['worker_label'].update(sampled_df)
    return comp_df

def n_assignments(num_comparisons,total_num_workers,assignments):
    workers_index = list(range(total_num_workers))
    num_assignments = num_comparisons * 3
    random_nums = np.random.randint(assignments['min'],assignments['max'],total_num_workers)
    random_nums_sum = np.round(num_assignments * random_nums / sum(random_nums)).astype(int)
    if sum(random_nums_sum) < num_assignments:
        random_nums_sum[-1] += num_assignments-sum(random_nums_sum)
    elif sum(random_nums_sum) > num_assignments:
        random_nums_sum[-1] += num_assignments-sum(random_nums_sum)
    assert sum(random_nums_sum) == num_assignments
    np.random.shuffle(workers_index)
    worker_assignments = pd.DataFrame(np.array([workers_index,random_nums_sum]).transpose(),columns=['worker_id','n_assignments'])
    return worker_assignments

def assign_reliability(worker_assignments,scenario):
    worker_assignments['reliability'] = 0
    maj_lim = round(0.8*len(worker_assignments))
    quarter = round(0.25*len(worker_assignments))
    if scenario == 'majority_low':
        worker_assignments['reliability'].iloc[:maj_lim] = 0.2
        worker_assignments['reliability'][maj_lim:] = 0.8
    elif scenario == 'majority_high':
        worker_assignments['reliability'].iloc[:maj_lim] = 0.8
        worker_assignments['reliability'][maj_lim:] = 0.2
    elif scenario == 'quarter_reliabilities':
        worker_assignments['reliability'].iloc[:quarter] = 0.25
        worker_assignments['reliability'][quarter:2*quarter] = 0.5
        worker_assignments['reliability'].iloc[2*quarter:3*quarter] = 0.75
        worker_assignments['reliability'][3*quarter:] = 1
    elif scenario == 'random':
        worker_assignments['reliability'] = np.random.rand(len(worker_assignments))
    return worker_assignments


def create_worker_answer_pairs(num_comparisons,total_num_workers,assignments,scenario):
    pairs_index = list(range(num_comparisons))
    worker_assignments = n_assignments(num_comparisons, total_num_workers, assignments)
    worker_assignments_rel = assign_reliability(worker_assignments, scenario)

class Worker:
    def __init__(self, id, reliability):
        self.id = id
        self.reliability = reliability



class Summary:
    def __init__(self, id, overall, accuracy, coverage, coherence):
        self.id = id
        self.overall = overall
        self.accuracy = accuracy
        self.coverage = coverage
        self.coherence = coherence






def assign_answer(reliability, num_comparisons):
    if num_comparisons == 1:
        # If there is only one comparison, assign 'correct' based on reliability probability
        return ['correct'] if random.random() < reliability else ['incorrect']

    correct_answers = int(reliability * num_comparisons)  # Number of correct answers based on worker's reliability
    incorrect_answers = num_comparisons - correct_answers  # Number of incorrect answers
    # Create a list of correct and incorrect answers
    answers = ['correct'] * correct_answers + ['incorrect'] * incorrect_answers
    random.shuffle(answers)

    return answers

def get_labels(worker ,ground_truth_labels):

    num_labels = len(ground_truth_labels)
    #calls the assign_answer function to generate answers
    #and then creates labels by mapping the correct answers to the corresponding ground truth labels
    #and flipping the incorrect answers
    answers = assign_answer(worker.reliability, num_labels)
    labels = [0] * num_labels
    for i in range(num_labels):
        if answers[i] == "correct":
            labels[i] = ground_truth_labels[i]
        else:
            labels[i] = 1 - ground_truth_labels[i]
    return labels


def randomly_assign_workers(total_num_workers, num_generated, num_comparisons,comparisons_per_worker):
    #create the list with workers id
    worker_comp_id_list = []

    for worker_id in range(total_num_workers):
        if len(worker_comp_id_list) < num_generated:
            remaining_comparisons = num_generated - len(worker_comp_id_list)
            num_assignments = min(comparisons_per_worker, remaining_comparisons)
            worker_comp_id_list.extend([worker_id] * num_assignments)

    random.shuffle(worker_comp_id_list)  # Shuffle the list for random assignment

    return worker_comp_id_list

def match_worker_with_summary(num_generated, worker_comp_id_list, comparison_ds):
    if (num_generated==len(worker_comp_id_list)):  #same size as the orginal dataset
          comparison_ds['worker_id'] = worker_comp_id_list

    return comparison_ds

def get_new_comp_labels_df(match_df, worker):
    # Assuming match_df has a column 'assigned' to track if a summary has been assigned
    if 'assigned' not in match_df.columns:
        match_df['assigned'] = False

    # Filter for unassigned summaries for the current worker
    worker_unassigned_summaries = match_df.loc[match_df['assigned']==False]

    # Assign labels to these unassigned summaries
    ground_truth_labels = worker_unassigned_summaries['choice'].tolist()
    labels = get_labels(worker, ground_truth_labels)

    # Update the DataFrame with generated labels and mark them as assigned
    worker_unassigned_summaries['worker_label'] = labels
    worker_unassigned_summaries['assigned'] = True

    # Update the original DataFrame
    match_df.update(worker_unassigned_summaries)

    return worker_unassigned_summaries

def get_reliability_value(worker_id, total_num_workers, reliability):
    if reliability == "extreme":
        return 0  
    elif reliability == "low":
        return 0.2
    elif reliability == "medium":
        return 0.5
    elif reliability == "high":
        return 0.8
    elif reliability == "half_low_half_high":
        return 0.2 if worker_id < total_num_workers // 2 else 0.8
    elif reliability == "quarter_reliabilities":
        quarter_size = total_num_workers / 4
        quarter_index = int(worker_id // quarter_size)
        return [0.2, 0.4, 0.6, 0.8][quarter_index]
    elif reliability == "perfect":
        return 1.0
    elif reliability == "random":
        return random.uniform(0.0, 1.0)

def get_the_generated_df_for_comp(total_num_workers, num_generated, reliability, num_comaprisons, comparisons_per_worker, comparison_ds):
    comparison_ds_copy = comparison_ds.copy()

    # Assign workers to comparisons randomly based on specified parameters
    worker_comp_id_list = randomly_assign_workers(total_num_workers, num_generated, num_comaprisons, comparisons_per_worker)

    # Match workers with summary information based on the assignment
    match_df = match_worker_with_summary(num_generated, worker_comp_id_list, comparison_ds_copy)
    generated_df = pd.DataFrame()

    for worker_id in set(worker_comp_id_list):

        reliability_value = get_reliability_value(worker_id, total_num_workers, reliability)
        worker = Worker(worker_id, reliability_value)
        # Get the new comparison labels for the worker and match them with corresponding data
        matching_rows = get_new_comp_labels_df(match_df, worker)
        generated_df = pd.concat([generated_df, matching_rows])

    return generated_df

def match_id_with_eval(summary_id, criteria, unique_eval):
    filtered_data =unique_eval[unique_eval['summary_id'] == summary_id]
    return filtered_data[criteria].values[0]


def compare_eval_metrics(row, metric, unique_eval):
    metric_0 = float(match_id_with_eval(row["id_0"], metric, unique_eval))
    metric_1 = float(match_id_with_eval(row["id_1"], metric, unique_eval))
    if metric_0 > metric_1:
        return 0
    elif metric_0 < metric_1:
        return 1
    else:
        overall_0 = unique_eval[unique_eval['summary_id']==row["id_0"]].overall.iloc[0]
        overall_1 = unique_eval[unique_eval['summary_id']==row["id_1"]].overall.iloc[0]
        if overall_0 > overall_1:
            return 0
        else:
            return 1

def introduce_bias_class(comp_df_input, bias_towards_0, bias_towards_1):
    comp_df_biased = comp_df_input.copy()
    if bias_towards_0:
        comp_df_biased["worker_label"] = 0
    elif bias_towards_1:
        comp_df_biased["worker_label"] = 1
    return comp_df_biased

def introduce_bias_aspect(comp_df_input, unique_eval, bias_accuracy, bias_coverage, bias_coherence):
    biased_input = comp_df_input.copy()
    for index, row in biased_input.iterrows():
        if bias_accuracy:
            label = compare_eval_metrics(row, "accuracy", unique_eval)
        elif bias_coverage:
            label = compare_eval_metrics(row, "coverage", unique_eval)
        elif bias_coherence:
            label = compare_eval_metrics(row, "coherence", unique_eval)

        if label is not None:
            biased_input.at[index, "worker_label"] = label
    biased_input["worker_label"] = biased_input["worker_label"].astype("int")
    return biased_input

def get_chosen_and_rejected(row):
    chosen_idx = row['worker_label']
    rejected_idx = 1 - chosen_idx  # Switch between 0 and 1
    return row[f'summary_text_{chosen_idx}'], row[f'summary_text_{rejected_idx}']


def to_parquet(generated_df, directory_path, split, title):
    # Create a copy of the dataframe to avoid modifying the input
    create_directory(directory_path+'/'+title)
    output_df = generated_df.copy()

    # Convert to the format for training
    output_df['chosen'], output_df['rejected'] = zip(*output_df.apply(get_chosen_and_rejected, axis=1))
    output_df = output_df[['prompt', 'chosen', 'rejected']]

    # Add prefixes to the prompt, chosen, and rejected columns
    output_df['chosen'] = "TL;DR: " + output_df['chosen']
    output_df['rejected'] = "TL;DR: " + output_df['rejected']

    # Save to parquet
    output_df.to_parquet(directory_path+'/'+title+'/'+split + "_" + title + '.parquet',index=False)

    return output_df
