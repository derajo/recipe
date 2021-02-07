import hashlib
import pandas as pd
from functools import reduce

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

def generate_id(s):
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()

def add_id_to_dataset(input_df):
    print("Adding ID to dataset")
    df = input_df.copy()
    df['ID'] = None
    new_ingredient_loc = df.text_index.str.match("^I1$")
    ingredient_count = new_ingredient_loc.sum()
    df.loc[new_ingredient_loc, "ID"] = list(map(generate_id, range(1,ingredient_count+1)))
    return df.fillna(method = 'ffill')

def remove_label_prefix(df):
    """Remove the prefix in labels such as 'B-'
    and 'I-'"""
    df.loc[:, 'label'] = df.label.replace('^(.-)', '', regex= True)
    return df

def replace_index_with_qty(df):
    """Remove and replace 'INDEX' label with 'QTY'"""
    df.loc[:, 'label'] = df.label.replace('INDEX', 'QTY')
    return df

def remove_qty_symbol(df):
    """Remove the '$' symbol from ingredient quantities"""
    df.loc[:, 'text'] = df.text.replace('\$', ' ', regex = True)
    return df

def parenthesis_correcting(input_df):
    """Turns parenthesis into COMMENT"""
    df = input_df.copy()

    # remove instances where there are imbalanced parenthesis
    open_p = df.loc[(df.text.str.match('^\($'))].groupby("ID").count()[['text']].rename(columns = {"text":"("}).reset_index(drop = False)
    closed_p = df.loc[(df.text.str.match('^\)$'))].groupby("ID").count()[['text']].rename(columns = {"text":")"}).reset_index(drop = False)
    matches = open_p.merge(closed_p, on = "ID", how = "outer")
    remove_ids = list(matches[matches["("]!=matches[")"]]['ID'])
    df = df.loc[~df.ID.isin(remove_ids)]

    opened = df.loc[df.text.str.match('^\($')].index
    closed = df.loc[df.text.str.match('^\)$')].index
    parenthesis_groups = zip(opened,closed)
    # intialize
    p = next(parenthesis_groups)
    arr = []
    for i in range(len(df)):
        if i > p[1]:
            p = next(parenthesis_groups, (0,0))
        if (i >= p[0]) & (i <= p[1]):
            arr.append(True)
        else:
            arr.append(False)
    # make parenthesis comments
    # if removing parenthsis, switch True and False then filter out.
    df.loc[arr,"label"] = "COMMENT"
    return df

def ingredient_has_name(df):
    """
    Ensure the ingredient has a name within the text
    """
    return any(df.label.str.match('^NAME$'))

def remove_hyphen_ingredients(df):
    """Find and rmeove units and qtys with hyphens in them.
    These instances happen in examples like '1-pound'
    and is incorrectly labelled just a qty or just a unit.
    """
    d = df.copy()
    hyphen_in = d.text.str.contains("-")
    not_hyphen_only = ~d.text.str.match('^-$')
    qty = d.label.str.match('^QTY$') 
    unit = d.label.str.match('^UNIT$')
    qty_or_unit = (qty | unit)
    filter_ids = d.loc[(not_hyphen_only & (hyphen_in & qty_or_unit))].ID
    return d[~d['ID'].isin(filter_ids)].reset_index(drop=True)

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
    or_ingr = df.copy()
    first_name_ingredients = or_ingr[or_ingr['label'] ==  "NAME"].drop_duplicates("ID")
    ingredient_with_name_id = list(first_name_ingredients.ID)
    or_ingr = or_ingr[or_ingr['ID'].isin(ingredient_with_name_id)].reset_index(drop = True) # filter out ingredients with no names in them.
    or_index = or_ingr[or_ingr['text'] ==  "or"].index # get index of 'or'
    first_name_index = first_name_ingredients.index # index of first names for each ingredient
    new_id_index = or_ingr.drop_duplicates("ID").index  # indexes for when new ingredient starts

    iter_id = iter(new_id_index)
    iter_name = iter(first_name_index)
    iter_or = iter(or_index)
    id_ = next(iter_id) # intialize the ID
    or_ = next(iter_or) # initialize the or index

    arr = []
    for i in range(len(or_ingr)):
        if i == id_: # new ingredient, reset val
            val = False
            id_ = next(iter_id, None) # get next ingredient index
            name_ = next(iter_name, None) # initialize the name

        if val: # if we are commenting out, continue to comment out
            if i == or_:
                or_ = next(iter_or, None) # if more 'or's appear in alternative ingredients, skip them.
            arr.append(True)
            continue

        if i == or_: # if the text is or, evaluate

            if name_ < or_: # if name came before or, comment out the or's
                val = True # initialize the comment out markers
            else:
                val = False
            or_ = next(iter_or, None)

        arr.append(val)

    or_ingr.loc[arr, "label"] = "COMMENT"
    return or_ingr

def run_data_cleaning(df, *funcs):
    """
    Cleanses training data of text that increases complexity
    or that will cause issues while creating the model.
    """
    print("Cleaning phrase tagger data...")
    return reduce(lambda arg, func: func(arg), funcs, df)

print("Reading in ingredient_phrase_tagger_training_data...")
df = pd.read_csv(in_file_path, sep='\t', header=None)

df.columns = column_names()

df = add_id_to_dataset(df)
print(df.head(5))

print(f"Writing formatted dataset to {out_file_path}")
df.to_csv(out_file_path, index=False)

cleaned_df = run_data_cleaning(
    df, 
    remove_label_prefix,
    replace_index_with_qty,
    remove_qty_symbol,
    parenthesis_correcting,
    remove_hyphen_ingredients,
    or_to_comment,
)
print(len(cleaned_df))
print(f"Writing cleaned dataset to {clean_data_out_file_path}")
cleaned_df.to_csv(clean_data_out_file_path, index=False)
