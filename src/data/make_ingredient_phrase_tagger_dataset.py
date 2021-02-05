import pandas as pd

in_file_path = "../data/raw/train_file.csv"
out_file_path = "../data/raw/ingredient_phrase_tagger_training_data.csv"
clean_data_out_file_path = "../data/interim/ingredient_phrase_tagger_training_data_cleaned_v2.csv"

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
        return text.replace("$", ' ') 
    return text

def remove_parenthesis(input_df):
    """Remove parenthesis from training data"""
    df = input_df.copy()
    return df.loc[df['parenthesis_flag'] != "YesPAREN",:]

def ingredient_has_name(d):
    """
    Ensure the ingredient has a name within the text
    """
    return any(d.label.str.match('^NAME$'))

def find_hyphen_ingredients(d):
    """Find units and qtys with hyphens in them.
    These instances happen in examples like '1-pound'
    and is incorrectly labelled just a qty or just a unit.

    Returns False if this occurs in the ingredient
    """
    hyphen_in = d.text.str.contains("-")
    not_hyphen_only = ~d.text.str.match('^-$')
    qty = d.label.str.match('^QTY$') 
    unit = d.label.str.match('^UNIT$')
    qty_or_unit = (qty | unit)
    return any(not_hyphen_only & (hyphen_in & qty_or_unit))

def or_to_comment(df):
    """
    Check that the first NAME comes before the first "or".
    If True, it typically means the text following the or
    is an alternative to the first NAME. If this is False,
    typically it means there are two comments to the NAME
    e.g. chicken or beef stock - where "stock" is the NAME
    and "chicken or beef" is the COMMENT. This scenario is
    fine for the parser, however it may be worth it to
    `split` this after the parser is applied.
    
    If there are two or more 'or's, we will see which ones
    come after the first or and replace all text after them
    as a comment.
    """
    # make a copy of the df
    or_ingr = df.copy()
    or_index = or_ingr[or_ingr['text'] ==  "or"].index
    first_name_index = or_ingr[or_ingr['label'] ==  "NAME"].index[0]

    or_replacement_index = or_index[or_index > first_name_index]
    if or_replacement_index.values.size > 0:
        or_ingr.loc[or_replacement_index[0]:, "label"] = "COMMENT"
    return or_ingr

def run_data_cleaning(input_df):
	"""
	Cleanses training data of text that increases complexity
	or that will cause issues while creating the model.
	"""
	print("Cleaning phrase tagger data...")
	df = remove_parenthesis(input_df)
	df.loc[:, 'label'] = df.loc[:, 'label'].apply(remove_label_prefix)
	df.loc[:, 'label'] = df.loc[:, 'label'].apply(replace_INDEX_with_QTY)
	df.loc[:, 'text']  = df.loc[:, 'text'].apply(remove_QTY_symbol)

	df_len = len(df.ID.unique())
	cleaned_df = []
	for d in iter(df.groupby("ID")):
		print(f"{round((int(d[0])/df_len)*100,3)}%", end='\r')
		# Select the df and the columns we need
		ingredient_df = d[1].loc[:, ('text', 'label', 'ID')]
		# If no hyphen errors and ingredient has a NAME
		if ((not find_hyphen_ingredients(ingredient_df))
				& (ingredient_has_name(ingredient_df))):
			# Handle when there is an or in the ingredient
			if any(ingredient_df.text.str.match('^or$')):
				ingredient_df = or_to_comment(ingredient_df)
			cleaned_df.append(ingredient_df)

	return pd.concat(cleaned_df)

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
