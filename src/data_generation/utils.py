import pandas as pd
import pdb
import os

def create_directory(directory_path):
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        try:
            # Create the directory
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except OSError as e:
            print(f"Error creating directory '{directory_path}': {e}")
    else:
        print(f"Directory '{directory_path}' already exists.")

def create_dict_comparision(comparisons):
    comparisons_df = pd.DataFrame(comparisons["info"])
    comparisons_dict = dict(zip(comparisons_df.id, comparisons_df.post))
    return comparisons_dict

def create_post2comparisons(comparisons_ds):
    comparisons_df = pd.DataFrame(comparisons_ds)
    info_df = pd.json_normalize(comparisons_df['info'])
    summary_df = pd.json_normalize(comparisons_df['summaries'])

def process_comparisons_df(comparisons_ds,summary_text2id):
    comparisons_ds_df = pd.DataFrame(comparisons_ds)
    comparisons_ds_df['summary_text_0'] = comparisons_ds_df.summaries.map(lambda x: x[0]['text'])
    comparisons_ds_df['summary_text_1'] = comparisons_ds_df.summaries.map(lambda x: x[1]['text'])
    comparisons_ds_df['id_0'] = comparisons_ds_df.summaries.map(lambda x: summary_text2id[x[0]['text']] if x[0]['text'] in summary_text2id else -1)
    comparisons_ds_df['id_1'] = comparisons_ds_df.summaries.map(lambda x: summary_text2id[x[1]['text']] if x[1]['text'] in summary_text2id else -1)
    comparisons_ds_df['conf'] = comparisons_ds_df.extra.map(lambda x: x['confidence'])
    comparisons_ds_df = comparisons_ds_df.drop(['summaries'], axis=1)
    comparisons_ds_df_first = comparisons_ds_df[comparisons_ds_df['id_0']!=-1]
    process_val_df = comparisons_ds_df_first[comparisons_ds_df_first['id_1']!=-1]
    process_val_df = process_val_df.reset_index(drop=True)
    info_df = pd.json_normalize(process_val_df['info'])
    process_val_df = pd.concat([process_val_df.drop(['info'], axis=1), info_df], axis=1)
    process_val_df = process_val_df.dropna(subset=['post'])
    process_val_df.fillna(0, inplace=True)
    process_val_df = process_val_df.reset_index()
    columns_list = ['choice', 'worker', 'summary_text_0',
                    'summary_text_1', 'id_0', 'id_1', 'id', 'prompt','conf']
    process_val_df['prompt'] ="SUBREDDIT: " + process_val_df['subreddit'] + " TITLE: " + process_val_df['title'] + " POST: " + process_val_df['post']
    return process_val_df[columns_list]


# used for the validation set
# initially you have
# summary 1, summary 2, preference score
# post content, summary, accuracy, coverage, coherence
# extract the summary pairs that have the aspects scores
# summary 1, summary 2, preference score that have accuracy, coverage, coherence


def create_post2summaries_ratings(axis):
    post2summaries_df = pd.DataFrame(axis)
    info_df = pd.json_normalize(post2summaries_df['info'])
    summary_df = pd.json_normalize(post2summaries_df['summary'])
    post2summaries_df = pd.concat([post2summaries_df.drop(['info'], axis=1), info_df], axis=1)
    post2summaries_df = pd.concat([post2summaries_df.drop(['summary'], axis=1), summary_df], axis=1)
    repeated_summaries = post2summaries_df.shape[0] - len(post2summaries_df.text.unique())
    summary_text2id = dict(zip(post2summaries_df.text.unique(), range(len(post2summaries_df.text.unique()))))
    post2summaries_df ["summary_id"] = post2summaries_df["text"].map(summary_text2id)
    axes_list = ['axes.overall', 'axes.accuracy', 'axes.coverage', 'axes.coherence']
    rename_dict = {col: col.split('.', 1)[1] if 'axes.' in col else col for col in axes_list}
    post2summaries_df.rename(columns=rename_dict, inplace=True)
    list_columns = ["overall","accuracy","coverage","coherence","text","worker","id","summary_id"]
    return post2summaries_df[list_columns], repeated_summaries, summary_text2id


def update_summary_t2id(comparisons_ds, summary_t2id):
    complete_summary_t2id = summary_t2id.copy()
    comparisons_ds_df = pd.DataFrame(comparisons_ds)

    comparisons_ds_df['summary_text_0'] = comparisons_ds_df.summaries.map(lambda x: x[0]['text'])
    comparisons_ds_df['summary_text_1'] = comparisons_ds_df.summaries.map(lambda x: x[1]['text'])
    all_summaries = pd.concat([comparisons_ds_df['summary_text_0'],comparisons_ds_df['summary_text_1']] )

    # Create a set of unique summaries that are not already in complete_summary_t2id
    unique_new_summaries = set(all_summaries) - set(complete_summary_t2id.keys())

    for summary in unique_new_summaries:
        complete_summary_t2id[summary] = len(complete_summary_t2id) + 1
    return complete_summary_t2id


def filter_duplicates_disagreement(comp):
    comp.loc[:, 'pair'] = comp['id_0'].astype(str) + '_' + comp['id_1'].astype(str)
    # remove duplicates
    comp_drop_duplicates = comp.drop_duplicates(subset=['choice', 'pair', 'id'])
    # remove conflicting cases
    mask = comp_drop_duplicates[comp_drop_duplicates.duplicated(subset=['pair'])].pair
    comp_rem_confl = comp_drop_duplicates[~comp_drop_duplicates['pair'].isin(mask)]
    # remove rows where the summaries are too short
    remove_short_sum_0 = comp_rem_confl[comp_rem_confl['summary_text_0'].str.len() > 5]
    remove_short_sum_1 = remove_short_sum_0[remove_short_sum_0['summary_text_1'].str.len() > 5]
    comp.reset_index(drop=True, inplace=True)
    return remove_short_sum_1
