import pandas as pd
in_file_path = "../data/raw/train_file.csv"
out_file_path = "../data/raw/nyt_ingredient_phrase_tagger_training_data.csv"

print("Reading in ingredient_phrase_tagger_training_data...")
df = pd.read_csv(
	in_file_path, sep = '\t', header = None
)

print("Renaming columns...")
df.columns = [
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

df = add_id_to_dataset(df)

print(df.head(5))

print(f"Writing file to {out_file_path}")
df.to_csv(
	out_file_path, index = False
)
print("Success!")