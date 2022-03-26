import dash
from dash import dash_table, html

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



####################################################################################################################
## Wrangle the data
####################################################################################################################

data_path = 'food_recipes.csv'
data = pd.read_csv("food_recipes.csv")


data.drop(columns=['url', 'record_health', 'vote_count', 'author'], inplace=True)
data.dropna(inplace=True)
data.drop_duplicates(keep='first', inplace=True)
data.reset_index(inplace=True, drop=True)

data['ingredients_xpl'] = data['ingredients']
data['ingredients'] = data['ingredients'].apply(lambda x: x.replace('|', ' '))

data['tags'] = data['tags'].apply(lambda x: x.replace('|', ' '))
data['prep_time'] = data['prep_time'].str.split(' ').str[0].astype(int)
data['cook_time'] = data['cook_time'].str.split(' ').str[0].astype(int)


ingredients_index = data.loc[:,['ingredients_xpl']]
ingredients_index['ingredients'] = ingredients_index['ingredients_xpl'].apply(lambda x: x.split('|')).apply(lambda x:[(str.lower(i)) for i in x])
ingredients_index.drop(columns=['ingredients_xpl'], inplace=True)

ingredients_index['recipe_id'] = data.index
ingredients_index['recipe_name'] = data['recipe_title']
ingredients_index = ingredients_index.explode('ingredients').reset_index(drop=True)


unique_ingredients = pd.DataFrame(ingredients_index['ingredients'].unique().tolist(), 
                                  columns=['ingredients'])

print(ingredients_index.head(2))
######################################################Functions##############################################################



def get_recipe_from_index(df, i):
    return df.iloc[i,:]

def get_top_similar(simx, k, i, df):
    """
    simx = similarity matrix
    k = number of results to return
    i = index of recipe to compare
    """
    similar_recipes = list(enumerate(simx[i]))
    sorted_similar = sorted(similar_recipes, key=lambda x:x[1], reverse=True)
    top_k = sorted_similar[1:k+1]
    
    
    top_k_df = pd.DataFrame(columns=df.columns)
    top_k_scores = []
    top_k_index = []
    for i in top_k:
        top_k_df = top_k_df.append(get_recipe_from_index(df,i[0]), ignore_index=True)
        top_k_scores.append(i[1])
        top_k_index.append(i[0])

    top_k_df['Score'] = top_k_scores
    top_k_df['id'] = top_k_index

    return top_k_df
    
def combined_features(row):
    combi = row['course']+" "+row['cuisine']+" "+row['diet']+" "+row['ingredients']+" "+row['tags']+" "+row['category']
    combi.replace(" Recipes", "")
    return combi

data["combined_features"] = data.apply(combined_features, axis =1)

#get_top_similar(cosine_sim_df, 2, 2493, data)    

cv = CountVectorizer()
count_matrix = cv.fit_transform(data["combined_features"])
#print("Count Matrix:", count_matrix.toarray())

cosine_sim_df = pd.DataFrame(cosine_similarity(count_matrix))


######################################################Data##############################################################


######################################################Interactive Components############################################
#print( unique_ingredients )#[dict(label=ingredient, value=ingredient) for ingredient in unique_ingredients['ingredients']] )

ing_options = [dict(label=ingredient, value=ingredient) for ingredient in unique_ingredients['ingredients']]

dropdown_ingredient = dcc.Dropdown(
        id='ing_drop',
        options=ing_options,
        multi=True
    )


recipe_table = dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": 'Recipe', "id": 'recipe_name', "deletable": False, "selectable": True},
            {"name": 'ID', "id": 'recipe_id', "deletable": False, "selectable": False}

        ],
        #data=df.to_dict('records'),
        editable=False,
        #filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="single",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 1,
        page_size= 10,
    )
    


##################################################APP###################################################################

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([

    html.H1('iDashboard'),

    html.Label('Ingredients In My Pantry:'),
    dropdown_ingredient,

    html.Br(),
    html.Div(id='dd-output-container'),
    html.Br(),

    html.Label('Recipe Choice'),
    recipe_table,

])


######################################################Callbacks#########################################################

@app.callback(
#    Output('dd-output-container', 'children'),
    Output('datatable-interactivity', 'value'),
    Input('ing_drop', 'value')
)
def update_output(value):
    return value


@app.callback(
    Output('datatable-interactivity', 'data'),
    [Input('datatable-interactivity', 'page_current'),
     Input('datatable-interactivity', 'page_size'),
     Input('datatable-interactivity', 'sort_by'),
     Input("ing_drop", "value"),
     ])
def update_table(page_current, page_size, sort_by, filter_string):
    # Filter
    dff = ingredients_index.loc[ingredients_index['ingredients'].isin(filter_string),['recipe_name', 'recipe_id']]
    dff.drop_duplicates(inplace=True)
    print(dff.head(3))


    return dff.iloc[
           page_current * page_size:(page_current + 1) * page_size
           ].to_dict('records')





if __name__ == '__main__':
    app.run_server(debug=True)
