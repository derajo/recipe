import hashlib
import pandas as pd
import numpy as np
from functools import reduce

in_file_path = "../data/raw/train_file.csv"
out_file_path = "../data/raw/ingredient_phrase_tagger_training_data.csv"
clean_data_out_file_path = "../data/interim/ingredient_phrase_tagger_training_data_cleaned_v3.csv"

replacement_dict = {
            "tsp": "teaspoon",
            "tsp.": "teaspoon",
            "oz": "ounce",
            "oz.": "ounce",
            "tbsp": "tablespoon",
            "tbsp.": "tablespoon",
            "lb": "pound",
            "lb.": "pound",
            "ml": "milliliter",
            "ml.": "milliliter",
            "g" : "grams",
            "g." : "grams",
        }


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

def remove_urls(df):
    """remove urls from training data"""
    print("Removing urls from data")
    df = df.loc[~df['text'].str.contains('http')].reset_index(drop = True)
    return df

def remove_label_prefix(df):
    """Remove the prefix in labels such as 'B-'
    and 'I-'"""
    print("Removing label prefixes")
    df.loc[:, 'label'] = df.label.replace('^(.-)', '', regex= True)
    return df

def replace_index_with_qty(df):
    """Remove and replace 'INDEX' label with 'QTY'"""
    print("Replacing INDEX label with QTY label")
    df.loc[:, 'label'] = df.label.replace('INDEX', 'QTY')
    return df

def remove_qty_symbol(df):
    """Remove the '$' symbol from ingredient quantities"""
    print("Remove $ from QTY's")
    df.loc[:, 'text'] = df.text.replace('\$', ' ', regex = True)
    return df

def parenthesis_correcting(input_df):
    """Turns parenthesis into COMMENT"""
    print("Correcting parenthesis label")
    df = input_df.copy()
    
    # remove instances where there are imbalanced parenthesis
    open_p = df.loc[(df.text.str.match('^\($'))].groupby("ID").count()[['text']].rename(columns = {"text":"("}).reset_index(drop = False)
    closed_p = df.loc[(df.text.str.match('^\)$'))].groupby("ID").count()[['text']].rename(columns = {"text":")"}).reset_index(drop = False)
    matches = open_p.merge(closed_p, on = "ID", how = "outer")
    remove_ids = list(matches[matches["("]!=matches[")"]]['ID'])
    df = df.loc[~df.ID.isin(remove_ids)]
    
    opened = iter(df.loc[df.text.str.match('^\($')].index)
    closed = iter(df.loc[df.text.str.match('^\)$')].index)
    id_iter = iter(df.drop_duplicates("ID").index)  # indexes for when new ingredient starts
    # intialize
    o = next(opened)
    c = next(closed)
    id_ = next(id_iter)
    
    # logic to comment out words in parenthesis
    arr = []
    for i in range(len(df)):
        while c < o:
            c = next(closed)
        if i == id_:
            id_ = next(id_iter)
            var = False
        if var:
            if i > c:
                arr.append(False)
                c = next(closed)
            else:
                arr.append(True)
            continue
        if i < o:
            var = False
            arr.append(var)
            continue
        if (i == o) | (i == c):
            var = True
            if i == o:
                o = next(opened)
        arr.append(var)
    # make parenthesis comments
    # if removing parenthsis, switch True and False then filter out.
    df.loc[arr,"label"] = "COMMENT"
    return df

def remove_hyphen_ingredients(df):
    """Find and remove units and qtys with hyphens in them.
    These instances happen in examples like '1-pound'
    and is incorrectly labelled just a qty or just a unit.
    """
    print("Removing hyphened ingredients")
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
    print("Fixing alternative ingredient labels to comments")
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

def bad_qty_unit_entries(input_df):
    print("Removing ingredient entries that need labelling")
    df = input_df.copy()
    bad_qty_and_unit = df.ID.isin(df[(df.text.str.contains("\d", regex = True))
                                     & (df.text.str.contains("-"))
                                     & (df.label == "OTHER")]['ID'].unique())
    return df.loc[~bad_qty_and_unit]
    
def hyphen_replacement(input_df):
    print("Replace hyphens with spaces")
    df = input_df.copy()
    df.loc[:, 'text'] = df.text.replace('-', ' ', regex = True)
    return df

# Write out some bad data to be labelled later
def explode(input_df, lst_col="text", fill_value='', preserve_index=False):
    print("exploding data so one word per line")
    # make sure `lst_cols` is list-alike
    df = input_df.copy()
    df.loc[:, lst_col] = df[lst_col].str.split(" ")
    if (lst_col is not None
        and len(lst_col) > 0
        and not isinstance(lst_col, (list, tuple, np.ndarray, pd.Series))):
        lst_col = [lst_col]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_col)
    # calculate lengths of lists
    lens = df[lst_col[0]].str.len()
    # preserve original index values    
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_col}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:        
        res = res.reset_index(drop=True)
    return res

def fix_other_labels(input_df):
    """Most times that a period appeared it was labelled as OTHER.
    Replace those instances with a COMMENT label. Ensure that when
    the text is just a period to label as OTHER"""
    print("Fixing OTHER labels")
    df = input_df.copy()
    period = df.text.str.contains("\.")
    other = df.label.str.match("OTHER")
    just_period = df.text.str.contains("^\.$")
    df.loc[((period) & (other) & (~just_period)), "label"] = "COMMENT"
    df.loc[just_period, "label"] = "OTHER"
    return df

def fix_commented_units(input_df):
    print("Fix commented UNITs")
    df = input_df.copy()
    regex_string = "$|^".join(replacement_dict.keys())
    regex_string = "^" + regex_string + "$"
    regex_string = regex_string.replace(".", "\.")
    df.loc[df.text.str.lower().str.contains(regex_string, regex=True), "label"] = "UNIT"
    return df

def replace_abbreviated_units(input_df):
    print("Replace abbreviated units with their full name")
    df = input_df.copy()
    def replacement_function(text):
        lowercase_text = text.lower() 
        if lowercase_text in replacement_dict:
              return replacement_dict[lowercase_text]
        else:
            return text
    df.loc[:, "text"] = df.text.apply(replacement_function)
    return df



def run_data_cleaning(df, *funcs):
    """
    Cleanses training data of text that increases complexity
    or that will cause issues while creating the model.
    """
    print("Cleaning phrase tagger data...")
    return reduce(lambda arg, func: func(arg), funcs, df)[['ID', 'text', 'label']]

print("Reading in ingredient_phrase_tagger_training_data...")
df = pd.read_csv(in_file_path, sep='\t', header=None)

df.columns = column_names()

df = add_id_to_dataset(df)
print(df.head(5))

print(f"Writing formatted dataset to {out_file_path}")
df.to_csv(out_file_path, index=False)

cleaned_df = run_data_cleaning(
    df, 
    remove_urls,
    remove_label_prefix,
    replace_index_with_qty,
    remove_qty_symbol,
    remove_hyphen_ingredients,
    bad_qty_unit_entries,
    hyphen_replacement,
    explode,
    fix_other_labels,
    fix_commented_units,
    replace_abbreviated_units,
    parenthesis_correcting,
    or_to_comment,
)
print(f"Writing cleaned dataset to {clean_data_out_file_path}")
cleaned_df.to_csv(clean_data_out_file_path, index=False)
