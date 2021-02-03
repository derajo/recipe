import pandas as pd

in_file_path = "../data/raw/train_file.csv"
out_file_path = "../data/raw/ingredient_phrase_tagger_training_data.csv"
clean_data_out_file_path = "../data/interim/ingredient_phrase_tagger_training_data_cleaned.csv"

def column_names():
	print("Renaming columns...")
	return [
			 'text',
			 'text_index',
			 'text_length',
			 'capital_letter',
			 'parenthesis_flag',
			 'label'
			]

def add_id_to_dataset(df):
	print("Adding an ID to ingredients...")
	df['ID'] = None
	ID_ = 0
	df_len = len(df)
	for i, row in df.iterrows():
	    print(f"{round((i/df_len)*100,3)}%", end='\r')
	    if row['text_index'] == 'I1':
	        ID_ = str(int(ID_)+1).zfill(9)
	    df.loc[i, "ID"] = ID_
	return df

def remove_label_prefix(label):
    """Remove the prefix in labels such as 'B-'
    and 'I-'"""
    if "-" in label: # if statement faster than split()
        return label[2:]
    return label

def replace_INDEX_with_QTY(label):
    """Remove and replace 'INDEX' label with 'QTY'"""
    if "INDEX" in label: # if statement faster than str.replace
        return "QTY"
    return label

def remove_QTY_symbol(text):
    """Remove the '$' symbol from ingredient quantities"""
    if "$" in text: # if faster when there is not a "$", slower when there is
        return text.replace("$", '') 
    return text

def remove_parenthesis(df):
    """Remove parenthesis from training data"""
    df = df[df['parenthesis_flag'] != "YesPAREN"]
    return df

def find_hyphen_texts(text, label, ID):
    """Remove units and qtys with hyphens in them.
    These instances happen in examples like '1-pound'
    and is incorrectly labelled just a qty or just a unit.

    Returns the ID of the entry
    """
    not_hyphen_only = (text != '-')
    hyphen_in = ('-' in text)
    qty = (label == 'QTY')
    unit = (label == 'UNIT')
    qty_or_unit = (qty | unit)
    if not_hyphen_only & (hyphen_in & qty_or_unit):
        return ID

def run_data_cleaning(df):
	"""
	Cleanses training data of text that increases complexity
	or that will cause issues while creating the model.
	"""
	print("Cleaning phrase tagger data...")
	df = remove_parenthesis(df)
	cleaned_data = []
	remove_IDs = []
	df_len = len(df)
	for i, row in df.iterrows():
		print(f"{round((i/df_len)*100,3)}%", end='\r')
		label = remove_label_prefix(row['label'])
		label = replace_INDEX_with_QTY(label)
		text = remove_QTY_symbol(row['text'])
		ID = row['ID']
		remove_IDs.append(find_hyphen_texts(text, label, ID))
		cleaned_data.append([text,label,ID])
	cleaned_df = pd.DataFrame(
		cleaned_data, columns=['text', 'labels', 'ID']
	)
	return cleaned_df[~cleaned_df['ID'].isin(remove_IDs)]

print("Reading in ingredient_phrase_tagger_training_data...")
df = pd.read_csv(in_file_path, sep='\t', header=None)

df.columns = column_names()

df = add_id_to_dataset(df)
print(df.head(5))

print(f"Writing formatted dataset to {out_file_path}")
df.to_csv(out_file_path, index=False)

cleaned_df = run_data_cleaning(df)
print(f"Writing cleaned dataset to {clean_data_out_file_path}")
cleaned_df.to_csv(clean_data_out_file_path, index=False)
