import pandas as pd

def create_dict_comparision(comparisons):
    comparisons_df = pd.DataFrame(comparisons["info"])
    comparisons_dict = dict(zip(comparisons_df.id, comparisons_df.post))
    return comparisons_dict

def create_post2comparisons(comparisons_ds):
    comparisons_df = pd.DataFrame(comparisons_ds)
    info_df = pd.json_normalize(comparisons_df['info'])
    summary_df = pd.json_normalize(comparisons_df['summaries'])

def filter_comparisons(comparisons_ds, comparisons_filtered,summary_t2id,filtered_ids):
    for i in range(len(comparisons_ds)):
        post_id = comparisons_ds[i]['info']["id"]
        summary_text_0 = comparisons_ds[i]['summaries'][0]["text"]
        summary_text_1 = comparisons_ds[i]['summaries'][1]["text"]
        id_0, id_1 = -1, -1
        if summary_text_0 in summary_t2id:
            id_0 = summary_t2id[summary_text_0]
        if summary_text_1 in summary_t2id:
            id_1 = summary_t2id[summary_text_1]
        worker_choice = comparisons_ds[i]['choice']
        worker_id = comparisons_ds[i]['worker']
        if (id_0 != -1) and (id_1 != -1):
            filtered_ids.append(id_0)
            filtered_ids.append(id_1)
            comp_dict = {'id_0': id_0,
                         'id_1': id_1,
                         'summary_text_0': summary_text_0,
                         "summary_text_1": summary_text_1,
                         "worker_choice": worker_choice,
                         "worker_id": worker_id}
            if post_id in comparisons_filtered:
                comparisons_filtered[post_id].append(comp_dict)
            else:
                comparisons_filtered[post_id] = [comp_dict]
    return comparisons_filtered, filtered_ids

# used for the validation set
# initially you have
# summary 1, summary 2, preference score
# post content, summary, accuracy, coverage, coherence
# extract the summary pairs that have the aspects scores
# summary 1, summary 2, preference score that have accuracy, coverage, coherence
def create_post2summaries_ratings(axis):
    axis_df = pd.DataFrame(axis)
    info_df = pd.json_normalize(axis_df['info'])
    summary_df = pd.json_normalize(axis_df['summary'])
    axis_df = pd.concat([axis_df.drop(['info'], axis=1), info_df], axis=1)
    axis_df = pd.concat([axis_df.drop(['summary'], axis=1), summary_df], axis=1)
    repeated_summaries = axis_df.shape[0] - len(axis_df.text.unique())
    summary_text2id = dict(zip(axis_df.text.unique(), range(len(axis_df.text.unique()))))
    axis_df ["summary_id"] = axis_df["text"].map(summary_text2id)
    axes_list = ['axes.overall', 'axes.accuracy',
       'axes.coverage', 'axes.coherence', 'axes.compatible']
    axes_list.extend(["summary_id","text","worker"])
    unique_post_ids = axis_df.id.unique()
    post2summaries = {}
    for post_id in unique_post_ids:
        summaries_post_id = axis_df[axis_df['id']==post_id]
        axis_dict = summaries_post_id[axes_list].to_dict('index')
        post2summaries[post_id] = list(axis_dict.values())
    return post2summaries, summary_text2id, repeated_summaries



def update_summary_t2id(comparisons_ds, summary_t2id):
    complete_summary_t2id = summary_t2id.copy()
    # Iterate over each row in the DataFrame
    for i in range(len(comparisons_ds)):
        summary_text_0 = comparisons_ds[i]['summaries'][0]["text"]
        summary_text_1 = comparisons_ds[i]['summaries'][1]["text"]

        # Check if summary_0 is not in the summary_t2id dictionary
        if summary_text_0 not in complete_summary_t2id:
            # Generate a new id for the summary
            new_id = len(complete_summary_t2id) + 1
            # Add the summary and its id to the summary_t2id dictionary
            complete_summary_t2id[summary_text_0] = new_id

        # Check if summary_1 is not in the summary_t2id dictionary
        if summary_text_1 not in complete_summary_t2id:
            # Generate a new id for the summary
            new_id = len(complete_summary_t2id) + 1
            # Add the summary and its id to the summary_t2id dictionary
            complete_summary_t2id[summary_text_1] = new_id

    # Return the updated summary_t2id dictionary
    return complete_summary_t2id

def filter_biased_comp(comparisons_ds, comparisons_filtered_biased,summary_t2id,complete_summary_t2id,filtered_ids):
    for i in range(len(comparisons_ds)):
        post_id = comparisons_ds[i]['info']["id"]
        summary_text_0 = comparisons_ds[i]['summaries'][0]["text"]
        summary_text_1 = comparisons_ds[i]['summaries'][1]["text"]
        #id_0, id_1 = -1, -1
        if summary_text_0 in summary_t2id and summary_text_1 in summary_t2id:
            id_0, id_1 = -1, -1
        else:
            id_0 = complete_summary_t2id[summary_text_0]
            id_1 = complete_summary_t2id[summary_text_1]
        worker_choice = comparisons_ds[i]['choice']
        worker_id = comparisons_ds[i]['worker']
        if (id_0 != -1) and (id_1 != -1):
            filtered_ids.append(id_0)
            filtered_ids.append(id_1)
            comp_dict = {'id_0': id_0,
                         'id_1': id_1,
                         'summary_text_0': summary_text_0,
                         "summary_text_1": summary_text_1,
                         "worker_choice": worker_choice,
                         "worker_id": worker_id}
            if post_id in comparisons_filtered_biased:
                comparisons_filtered_biased[post_id].append(comp_dict)
            else:
                comparisons_filtered_biased[post_id] = [comp_dict]
    return comparisons_filtered_biased