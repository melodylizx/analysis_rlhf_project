import random
import pandas as pd
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

def assign_labels(worker ,ground_truth_labels):

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
    labels = assign_labels(worker, ground_truth_labels)

    # Update the DataFrame with generated labels and mark them as assigned
    worker_unassigned_summaries['worker_label'] = labels
    worker_unassigned_summaries['assigned'] = True

    # Update the original DataFrame
    match_df.update(worker_unassigned_summaries)

    return worker_unassigned_summaries

def get_the_generated_df_for_comp(total_num_workers, num_generated, reliability, num_comaprisons,  comparisons_per_worker ,comparison_ds):
    comparison_ds_copy = comparison_ds.copy()
    worker_comp_id_list = randomly_assign_workers(total_num_workers, num_generated ,num_comaprisons, comparisons_per_worker)
    match_df = match_worker_with_summary(num_generated, worker_comp_id_list, comparison_ds_copy)
    generated_df = pd.DataFrame()

    for worker_id in set(worker_comp_id_list):
        if reliability == "low":
            worker = Worker(worker_id, 0.2)  # Set low reliability for all workers
            matching_rows = get_new_comp_labels_df(match_df, worker)
        elif reliability == "medium":
            worker = Worker(worker_id, 0.5)  # Set 50% reliability for all workers
            matching_rows = get_new_comp_labels_df(match_df, worker)
        elif reliability == "high":
            worker = Worker(worker_id, 0.8)  # Set 80% reliability for all workers
            matching_rows = get_new_comp_labels_df(match_df, worker)
        elif reliability == "half_low_half_high":
            # Assign 20% reliability to half of the workers, and 80% reliability to the other half
            if worker_id < total_num_workers // 2:
                worker = Worker(worker_id, 0.2)
                matching_rows = get_new_comp_labels_df(match_df, worker)
            else:
                worker = Worker(worker_id, 0.8)
                matching_rows = get_new_comp_labels_df(match_df, worker)
        elif reliability == "quarter_reliabilities":
            # Assign different reliabilities to each 25% of workers
            num_quarters = 4
            reliability_options = [0.2, 0.4, 0.6, 0.8]
            quarter_size = total_num_workers / num_quarters
            quarter_index = int(worker_id // quarter_size)
            worker = Worker(worker_id, reliability_options[quarter_index])
            matching_rows = get_new_comp_labels_df(match_df, worker)
        elif reliability == "perfect":
            worker = Worker(worker_id, 1.0)  # Set perfect reliability for all workers
            matching_rows = get_new_comp_labels_df(match_df, worker)
        elif reliability == "random":
            # Handle random scenarios here
            rj = random.uniform(0.0, 1.0)
            worker = Worker(worker_id, rj)
            matching_rows = get_new_comp_labels_df(match_df, worker)

        generated_df = pd.concat([generated_df, matching_rows])
        #comparison_ds = comparison_ds.drop('assigned', axis=1)
        #generated_df = generated_df.drop('assigned', axis=1)

    return generated_df

def match_id_with_eval(summary_id, cirteria, unique_eval):
    filtered_data =unique_eval[unique_eval['summary_id'] == summary_id]
       return filtered_data[cirteria].values[0]

def introduce_bias(comp_df_input, bias_accuracy, bias_coverage, bias_coherence, bias_towards_0, bias_towards_1,unique_eval ):
    if bias_accuracy:
        for i in range(len(comp_df_input)):
            accuracy_0 = float(match_id_with_eval(comp_df_input.loc[i, "id_0"], "accuracy",unique_eval))
            accuracy_1 = float(match_id_with_eval(comp_df_input.loc[i, "id_1"], "accuracy",unique_eval))
            if accuracy_0 > accuracy_1:
                comp_df_input.loc[i, "worker_label"] = 0
            elif accuracy_0 < accuracy_1:
                comp_df_input.loc[i, "worker_label"] = 1

    if bias_coverage:
        for i in range(len(comp_df_input)):
            coverage_0 = float(match_id_with_eval(comp_df_input.loc[i, "id_0"], "coverage",unique_eval))
            coverage_1 = float(match_id_with_eval(comp_df_input.loc[i, "id_1"], "coverage",unique_eval))
            if coverage_0 > coverage_1:
                comp_df_input.loc[i, "worker_label"] = 0
            elif coverage_0 < coverage_1:
                comp_df_input.loc[i, "worker_label"] = 1

    if bias_coherence:
        for i in range(len(comp_df_input)):
            coherence_0 = float(match_id_with_eval(comp_df_input.loc[i, "id_0"], "coherence",unique_eval))
            coherence_1 = float(match_id_with_eval(comp_df_input.loc[i, "id_1"], "coherence",unique_eval))
            if coherence_0 > coherence_1:
                comp_df_input.loc[i, "worker_label"] = 0
            elif coherence_0 < coherence_1:
                comp_df_input.loc[i, "worker_label"] = 1

    if bias_towards_0:
        for i in range(len(comp_df_input)):
               comp_df_input.loc[i, "worker_label"] = 0

    if bias_towards_1:
        for i in range(len(comp_df_input)):
               comp_df_input.loc[i, "worker_label"] = 1


    return comp_df_input

def get_chosen_and_rejected(row):
    chosen_idx = row['worker_label']
    rejected_idx = 1 - chosen_idx  # Switch between 0 and 1
    return row[f'summary_text_{chosen_idx}'], row[f'summary_text_{rejected_idx}']

def to_parquet(generated_df, split, title):
    # Create a copy of the dataframe to avoid modifying the input
    output_df = generated_df.copy()

    # Convert to the format for training
    output_df['prompt'] = output_df['post']
    output_df['chosen'], output_df['rejected'] = zip(*output_df.apply(get_chosen_and_rejected, axis=1))
    output_df = output_df[['prompt', 'chosen', 'rejected']]

    # Add prefixes to the prompt, chosen, and rejected columns
    output_df['prompt'] = "POST: " + output_df['prompt']
    output_df['chosen'] = "TL;DR: " + output_df['chosen']
    output_df['rejected'] = "TL;DR: " + output_df['rejected']

    # Save to parquet
    output_df.to_parquet(split + "_" + title + '.parquet')

    return output_df