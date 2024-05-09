import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import base64

import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt

import ast
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD

from scipy import sparse
from lightfm import LightFM
from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k, recall_at_k

import pickle

#image_filename = 'skimage.png'
#encoded_image = base64.b64encode(open(image_filename, 'rb').read())

df = pd.read_csv('skindataall.csv', index_col=[0])

with open('mf_model.pkl', 'rb') as f:
    mf_model = pickle.load(f)

def dicts(df, colname):
    vals = list(set(df[colname]))
    l = []
    for i in vals:
        dic = {}
        dic['label'] = i
        dic['value'] = i
        l.append(dic)
    return l

tones_dict = dicts(df, 'Skin_Tone')
types_dict = dicts(df, 'Skin_Type')
eyes_dict = dicts(df, 'Eye_Color')
hair_dict = dicts(df, 'Hair_Color')

products_dictionary = dicts(df, 'Product')

user_dictionary = dicts(df, 'User_id')

def Table(df):
    rows = []
    for i in range(len(df)):
        row = []
        for col in df.columns:
            value = df.iloc[i][col]
            # update this depending on which
            # columns you want to show links for
            # and what you want those links to be
            if col == 'Product':
                cell = html.Td(html.A(href=df.iloc[i]['Product_Url'], children=value))
            elif col == 'Product_Url':
                continue
            else:
                cell = html.Td(children=value)
            row.append(cell)
        rows.append(html.Tr(row))
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in ['V', 'V']])] + rows
        )

def Display(data):
    grid_items = []
    
    for index, row in data.iterrows():
        item_html = html.Div(
            children=[
                html.Img(src=row['Product_img'], style={'width': '100%'}), 
                html.A(
                    row['Product'],  
                    href=row['Product_Url'], 
                    target='_blank',  
                    style={'display': 'block', 'textAlign': 'center'}
                ),
                html.P(
                    f"Rating: {row['Rating_Stars']} stars", 
                    style={'textAlign': 'center'}
                )
            ],
            style={'width': '30%', 'display': 'inline-block', 'boxSizing': 'border-box', 'padding': '10px'}
        )
        grid_items.append(item_html)
    
    grid_div = html.Div(children=grid_items, style={'width': '100%'})
    
    return grid_div

def Display2(data):
    grid_items = []

    for index, row in data.iterrows():
        item_html = html.Div(
            children=[
                html.Div(
                    children=[
                        html.Img(src=row['Product_img'], style={'width': '300px', 'height': '100px'}),  # Product image
                        html.Div(
                            children=[
                                html.A(
                                    row['Product'], 
                                    href=row['Product_Url'],  
                                    target='_blank',  
                                    style={'display': 'block', 'textAlign': 'center', 'fontWeight': 'bold'}
                                ),
                                html.P(
                                    f"Rating: {row['Rating']}", 
                                    style={'textAlign': 'center'}
                                )
                            ],
                            style={'width': '200px'}
                        ),
                        html.Div(
                            row['Ing_Tfidf'],  
                            style={'marginLeft': '20px'}
                        )
                    ],
                    style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}
                )
            ],
            style={'width': '100%', 'display': 'inline-block', 'boxSizing': 'border-box', 'padding': '10px'}
        )
        grid_items.append(item_html)
    
    grid_div = html.Div(children=grid_items, style={'width': '100%'})
    
    return grid_div


separation_string = '''




'''

intro_text = '''
__This simple app makes skincare recommendations! Whether you are new to the world of beauty and self care, or already have your favorite products, one of the recommenders below can help.__
'''


markdown_text_1 = '''
__Based on your features, these are the top products for you:__
'''

markdown_text_2 = '''
__Based on your preference, these are the top products for you:__
'''

markdown_text_3 = '''
__This user may like the following products:__
'''

def create_interaction_matrix(df, user_col, item_col, rating_col, norm= False, threshold = None):
    interactions = df.groupby([user_col, item_col])[rating_col].sum().unstack().reset_index().fillna(0).set_index(user_col)
    if norm:
        interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
    return interactions

interaction_matrix = create_interaction_matrix(df=df, user_col='User_id', item_col = 'Product_id', rating_col='Rating_Stars')

def create_user_dict(interactions):
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_dict

user_dict = create_user_dict(interaction_matrix)

def create_item_dict(df, id_col, name_col):
    item_dict ={}
    for i in df.index:
        item_dict[(df.loc[i, id_col])] = df.loc[i, name_col]
    return item_dict

product_dict = create_item_dict(df = df, id_col = 'Product_id', name_col = 'Product')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

tab_selected_style = {
    'backgroundColor': '#e8e6e3',  # Màu cam đậm cho tab được chọn
    'color': 'black',  # Màu chữ cho tab được chọn
    'font-size': '20px',  # Cỡ chữ cho tab được chọn
}

tab_style = {
    'backgroundColor': '#f5c771',  # Màu cam nhạt cho tab không được chọn
    'color': 'black',  # Màu chữ cho tab không được chọn
    'font-size': '20px',  # Cỡ chữ cho tab không được chọn
}

#https://codepen.io/chriddyp/pen/bWLwgP.css

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



colors = {
    #'background': '#1DB954',
    "text": "#111111",
    "background-image" : "url('/assets/background.png')",
    # "background-image" : "url('/assets/wallpaperskin_retouched.png')",
    "background-color" : "white",
    "background-size": "cover",
}

app.layout = html.Div(
    # style=colors,
    children=[
        html.Div(
            style=colors,
            children=[
                html.H1(
                    children='Skincare Recommendations',
                    style={
                        'textAlign': 'center',
                        'color': colors['text'],
                        'backgroundColor': colors["background-image"],
                        'font-family': 'Bangers',
                        'font-weight': 'Bold'
                    }
                ),
                dcc.Markdown(
                    style={
                        'color': colors['text'],
                        'padding':'30px'
                    },
                    children=intro_text,
                ),
                dcc.Markdown(children=separation_string),
            ]
        ),  

        dcc.Tabs(
            id='tabs',
            children=[
                dcc.Tab(
                    label="SR based on Personal Features",
                    style=tab_style,
                    selected_style=tab_selected_style,
                    children=[
                        html.H2(
                            children='SR based on your personal features',
                            style={
                                'textAlign': 'center',
                                'color': colors['text'],
                                'font-family': 'Bangers'
                            }
                        ),

                        html.Label(
                            'Skin Tone',
                            style={
                                'paddingLeft': '50px',
                            },
                        ),
                        dcc.Dropdown(
                            id='skintone-selector',
                            options=tones_dict,
                            placeholder='Select your skin tone',
                            style={
                                'width': '70%',
                                'paddingLeft': '45px',
                            },
                        ),

                        html.Label('Skin Type',
                                   style={
                                'paddingLeft': '50px',
                            },),
                        dcc.Dropdown(
                            id='skintype-selector',
                            options=types_dict,
                            placeholder='Select your skin type',
                            style={
                                'width': '70%',
                                'paddingLeft': '45px',
                            },
                        ),

                        html.Label('Eye color',
                                   style={
                                'paddingLeft': '50px',
                            },),
                        dcc.Dropdown(
                            id='eyecolor-selector',
                            options=eyes_dict,
                            placeholder='Select your eye color',
                            style={
                                'width': '70%',
                                'paddingLeft': '45px',
                            },
                        ),

                        html.Label('Hair color',
                                   style={
                                'paddingLeft': '50px',
                            },),
                        dcc.Dropdown(
                            id='haircolor-selector',
                            options=hair_dict,
                            placeholder='Select your hair color',
                            style={
                                'width': '70%',
                                'paddingLeft': '45px',
                            },
                        ),

                        dcc.Markdown(children=separation_string),
                        dcc.Markdown(
                            style={
                                'padding': '20px'
                            },
                            children=markdown_text_1
                            ),
                        html.Div(
                            id='output_1',
                            style={
                                'padding': '20px'
                            },
                        ),
                    ]
                ),

                dcc.Tab(
                    label="SR based on Favorite Products",
                    style=tab_style,
                    selected_style=tab_selected_style,
                    children=[
                        html.H2(
                            children='SR based on your favorites',
                            style={
                                'textAlign': 'center',
                                'color': colors['text'],
                                'font-family': 'Bangers'
                            }
                        ),
                        html.Label('Your favorite product!',
                                   style={
                                'paddingLeft': '50px',
                            },),
                        dcc.Dropdown(
                            id='product-selector',
                            options=products_dictionary,
                            placeholder='Select your favorite product',
                            style={
                                'width': '80%',
                                'paddingLeft': '45px',
                            },
                        ),
                        dcc.Markdown(children=markdown_text_2,
                                     style={
                                'padding': '20px'
                            },),
                        html.Div(id='output_2'),
                    ]
                ),

                dcc.Tab(
                    label="SR to Users for Business",
                    style=tab_style,
                    selected_style=tab_selected_style,
                    children=[
                        html.H2(
                            children='SR to users (for business)',
                            style={
                                'textAlign': 'center',
                                'color': colors['text'],
                                'font-family': 'Bangers'
                            }
                        ),
                        html.Label('List of user ids', 
                                   style={
                                'paddingLeft': '50px',
                            },),
                        dcc.Dropdown(
                            id='user-selector',
                            options=user_dictionary,
                            placeholder='Select user',
                            style={
                                'width': '50%',
                                'paddingLeft': '45px',
                            },
                        ),

                        dcc.Markdown(children=markdown_text_3, 
                                     style={
                                'padding': '20px'
                            },),
                        html.Div(id='output_3',
                                 style={
                                'paddingLeft': '50px'
                            },
                        ),
                    ]
                )
            ]
        )
    ]
)



@app.callback(
	Output('output_1', 'children'),
    [Input('skintone-selector', 'value'),
    Input('skintype-selector', 'value'),
    Input('eyecolor-selector', 'value'),
    Input('haircolor-selector', 'value')])

def recommend_products_by_user_features(skintone, skintype, eyecolor, haircolor):
    ddf = df[(df['Skin_Tone'] == skintone) & (df['Hair_Color'] == haircolor) & (df['Skin_Type'] == skintype) & (df['Eye_Color'] == eyecolor)]
    recommendations = ddf[(ddf['Rating_Stars'].notnull())]
    data = recommendations[['Rating_Stars', 'Product_Url', 'Product', 'Product_img']]

    data = data.sort_values('Rating_Stars', ascending=False).head()

    return Display(data)

@app.callback(
	Output('output_2', 'children'),
    [Input('product-selector', 'value')]
    )

def content_recommender(product):
    try:
        df_cont = df[['Product', 'Product_id', 'Ingredients', 'Product_Url', 'Ing_Tfidf', 'Rating', 'Product_img']]
        df_cont.drop_duplicates(inplace=True)
        df_cont = df_cont.reset_index(drop=True)
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
        tfidf_matrix = tf.fit_transform(df_cont['Ingredients'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        titles = df_cont[['Product', 'Ing_Tfidf', 'Rating', 'Product_Url', 'Product_img']]
        indices = pd.Series(df_cont.index, index=df_cont['Product'])
        idx = indices[product]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        product_indices = [i[0] for i in sim_scores]

    except KeyError:
        return None

    return Display2(titles.iloc[product_indices])

@app.callback(
	Output('output_3', 'children'),
    [Input('user-selector', 'value')]
    )

def sample_recommendation_user(user_id, model=mf_model, interactions=interaction_matrix, user_dict=user_dict,
                               item_dict=product_dict, threshold = 4, nrec_items = 10, show = True):

    try:
        n_users, n_items = interactions.shape
        user_x = user_dict[user_id]
        scores = pd.Series(model.predict(user_x,np.arange(n_items)))
        scores.index = interactions.columns
        scores = list(pd.Series(scores.sort_values(ascending=False).index))

        known_items = list(pd.Series(interactions.loc[user_id,:] \
                                     [interactions.loc[user_id,:] > threshold].index).sort_values(ascending=False))
        scores = [x for x in scores if x not in known_items]
        return_score_list = scores[0:nrec_items]
        known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
        scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))

        h_list = []
        for i in scores:
            h_list.append(html.H6(i))

    except KeyError:
        return None

    return h_list

if __name__ == '__main__':
    app.run_server(debug=True)
