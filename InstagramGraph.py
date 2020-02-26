#data, strucuture and maths
import pandas as pd
import numpy as np
import math
import json
import string
import itertools
from  more_itertools import unique_everseen
import random
import glob
from ast import literal_eval

#progress,performance and management
from tqdm import tqdm_notebook
import threading
import os
import ssl
from IPython.display import clear_output

#pre-processing
from sklearn import preprocessing

#time
import datetime as datetime
from time import sleep
import time

#text processing / regex
import regex
import re
import emoji

#dataviz and look/feel
import seaborn as sns
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
sns.set(style="white", context="talk")
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
sns.set_style("whitegrid")

# language & NLP
import spacy
import langdetect as ld

#network libraries and data viz
import networkx as nx
from networkx.algorithms import community
import plotly
import plotly.graph_objs as go
import plotly.express as px

#plotly offline rendering
from plotly.offline import download_plotlyjs, iplot, plot

import warnings
warnings.simplefilter('ignore')


class InstagramGraph():

    def __init__(self,csv,source_col='searched_for',post_col='post',user_col='user'):


        self.df = pd.read_csv(csv,encoding='latin').head(750)

        self.post_col = post_col

        self.user_col = user_col

        self.nlp = spacy.load('en_core_web_lg')

        self.lemma_count = 0

        self.hashtag_count = 0

        self.source = self.df[source_col].unique()

        self.default_stopwords= ['photooftheday','picoftheday','like4likes',
                                'like4like','instagood','likeforlikes',
                                'l4l','likeforlike','instagram','follow4follow',
                                'followforfollow','instadaily','instagrammers',
                                'instalike','follow','likeforfollow',
                                'like4follow','instamood','instafollow',
                                'bestoftheday','like','followme','instapic'
                                'repost','bhfyp']

        self.getLanguage

        self.cleaning

        self.getHashtag

        self.getHashtagLemma

        self.getUserpostcount

        self.getUserhashtagcount

        self.getEdgesNodes

        self.getGraph

    #cleans and formats dataframe
    def cleaning(self,df,col):

        #drop nulls on post column
        self.df.dropna(subset=[self.post_col],inplace=True)

        #convert any posts to string
        self.df[self.post_col] = self.df[self.post_col].map(lambda x: str(x))

        #remove emojis
        self.df[self.post_col] = self.df[self.post_col].map(lambda x: x.encode('ascii', 'ignore').decode('ascii'))

        return self.df

    #extracts hashtags from any string returning list of hashtags
    def getHashtag(self,_string):

        #splits string into list and appends unique hashtags into a new list
        hashtags = [hashtag for hashtag in set([token for token in _string.split() if token.startswith("#")])]

        #if there are hashtags in the string we process them further..
        if len(hashtags) > 0:

            #this will break up any hashtags that haven't been seperated by a space
            hashtags_seperated = [i for i in ''.join(hashtags).strip().split('#') if len(i) > 0]

            #this will remove any punctuation
            hashtags_clean = [hashtag.translate(str.maketrans('', '', string.punctuation)) for hashtag in hashtags_seperated]

            hashtags_clean = [i.lower() for i in hashtags_clean]

            #returns unique, cleaned hashtags without the
            return list(set(hashtags_clean))

        else:
            return np.nan

    #converts list of strings to lemma (if applicable) returning list of lemmas
    def getHashtagLemma(self,hashtags):

        #create a spacy document using hashtags as an argument
        doc = self.nlp(' '.join(hashtags))

        #empty list for lemmas
        tokens = []

        #loop through each token,
        for token in doc:

            if token.lemma_ != '-PRON-':

                tokens.append(token.lemma_)

                if str(token.text) != str(token.lemma_):

                    self.lemma_count+=1

        self.hashtag_count += len(tokens)

        return list(set(tokens))

    #gets a users post count that exists in the data
    def getUserpostcount(self,user):

        return self.user_count_dict[user]

    #gets a users median hashtag use in the data
    def getUserhashtagcount(self,user):

        return self.user_hashtag_count_dict[user]

    #gets the language of the string
    def getLanguage(self,_string):

        try:
            return ld.detect(_string)
        except:
            return np.nan

    def eda(self):


        """
        Language split
        """
        if self.translate == True:

            language_frame = pd.DataFrame(list(self.language_split.items()))

            language_frame.columns = ['language','incidence']

            language_frame.incidence = language_frame.incidence.map(lambda x: x/sum(language_frame.incidence))

            language_low_incidence = language_frame[language_frame.incidence < 0.05]

            language_frame_summary = language_frame.replace(language_low_incidence.language.values,'other')

            language_frame_summary = language_frame_summary.groupby('language')['incidence'].sum()

            language_frame_summary.sort_values(ascending=False,inplace=True)

            fig = go.Figure([go.Bar(x=language_frame_summary.index, y=language_frame_summary.values,name='Secondary Product')])

            fig.update_layout(xaxis_tickangle=-45,

                title="Incidence of language by post for #"+self.source[0],

                xaxis_title="Language",

                yaxis_title="Incidence")

            fig.show()

        def _histogram(metric,metric_label):

            fig = go.Figure(data=[go.Histogram(x=metric,histnorm='probability density')])

            fig.update_layout(

                title=f"Distribution of {metric_label} for #"+self.source[0],

                xaxis_title=f"Count of {metric_label}",

                yaxis_title="Frequency")

            return fig.show()

        _histogram(self.df.user_post_count,'user post frequency')

        _histogram(self.df.hashtag_count,'hashtags by post')


        #get each user's posting frequency
        df_count = pd.DataFrame(self.df.user.value_counts())

        #col labels
        df_count.columns=['post_freq']

        #normalise column
        df_count['post_freq_norm'] = df_count['post_freq'].map(lambda x: int(x)/df_count['post_freq'].sum())

        #cumulative sum on posts
        cum_sum_posts = np.cumsum(df_count['post_freq_norm'])

        users = []
        count=1

        for i in range(self.df.user.nunique()):
            users.append(count)
            count+=1

        #normalise users
        users_ = [i/users[-1] for i in users]

        #growth = pd.DataFrame(zip(users,cum_sum_posts))
        fig = go.Figure(data=go.Scatter(x=users_,y=cum_sum_posts))

        fig.update_layout(title=f'User post contribution for #'+ self.source[0],

                        xaxis_title='Normalised User Base',

                        yaxis_title='Normalised Post Contribution')

        fig.show()

        return

    #gets a list of hashtag lists from the dataset
    def getBatches(self,additional_stopwords=[]):

        #if no extra stopwords are specificed we use the defalut stop word list
        if len(additional_stopwords) == 0:

            self.current_stopwords=self.default_stopwords

        #append new stopwords to default stopword list
        else:
            self.current_stopwords = self.default_stopwords+additional_stopwords

        #function that iterates through list input and removes any stopwords
        def _removestop(words):
            for stop_word in self.current_stopwords:
                try:
                    words.remove(stop_word)
                    words = words
                except:
                    pass

            return words

        #apply function to hashtag column
        df_nostop = self.target[self.target.columns[0]].map(_removestop)

        #create new list of lists containing hashtags
        batch = [[df_nostop.iloc[i]][0] for i in range(len(df_nostop.index))]

        return batch

    #calculates the edges and nodes that exist in the list of hashtag lists
    def getEdgesNodes(self,batches,min_frequency):

        #ranks hashtags in alphabetical order
        def _ranked_topics(batches):

            batches.sort()

            return batches

        #finds all possible unique combinations of topics
        def _unique_combinations(batches):
            return list(itertools.combinations(_ranked_topics(batches), 2))

        #adds each combination to a dictionary, if combination already exists value of key increases by one
        def _add_unique_combinations(_unique_combinations,_dict):

            for combination in _unique_combinations:

                if combination in _dict:

                    _dict[combination]+=1

                else:

                    _dict[combination]=1

            return _dict

        edge_dict = {}

        source = []

        target = []

        edge_frequency = []

        #execute functions as above looping through each list, finding all unique combinations in each list
        #and adding them to dict object
        for batch in batches:

            edge_dict = _add_unique_combinations(_unique_combinations(batch),edge_dict)

        #create edge dataframe
        for key,value in edge_dict.items():

            source.append(key[0])

            target.append(key[1])

            edge_frequency.append(value)

        edge_df = pd.DataFrame({'source':source,'target':target,'edge_frequency':edge_frequency})

        edge_df.sort_values(by='edge_frequency',ascending=False,inplace=True)

        edge_df.reset_index(drop=True,inplace=True)

        #mask edge dataframe, only retinaing edges that occur n times
        edge_df = edge_df[edge_df['edge_frequency'] > min_frequency]

        #create node dataframe
        node_df = pd.DataFrame({'id':list(set(list(edge_df['source'])+list(edge_df['target'])))})

        labels = [i for i in range(len(node_df['id']))]

        node_df['id_code'] = node_df.index

        #create a dictionary of all the nodes
        node_dict = dict(zip(node_df['id'],labels))

        #add relevant id's to each node in the edge dataframe
        edge_df['source_code'] = edge_df['source'].apply(lambda x: node_dict[x])

        edge_df['target_code'] = edge_df['target'].apply(lambda x: node_dict[x])

        #retain some attributes for the instance
        self.edge_df = edge_df

        self.node_df = node_df

        self.node_dict = node_dict

        self.edge_dict = edge_dict

        return

    #build the graph using the edge and node data
    def getGraph(self):

        #function that loops through and appends edge tuples to list
        def _extract_edges(edge_df):

            tuple_out = []

            for i in range(0,len(self.edge_df.index)):

                tuple_out.append((self.edge_df['source_code'][i],self.edge_df['target_code'][i]))

            return tuple_out

        #instantiate an instance of a Networkx graph
        G=nx.Graph()

        #add the nodes to the instance
        G.add_nodes_from(self.node_df.id_code)

        #extract the edges
        edge_tuples = _extract_edges(self.edge_df)

        #loop through and add each edge to the instance
        for i in edge_tuples:
            G.add_edge(i[0],i[1])


        return G


    """
    Pipeline of all methods
    """
    #generate all the features we need
    def getFeatures(self,translate=False):

        self.translate = translate

        if self.translate == True:


            print('Attempting to identify language...')

            #detect language using getLanguage method
            self.df['language'] = self.df[self.post_col].map(self.getLanguage)

            #get language split as a class dictionary attribute
            self.language_split = dict(self.df['language'].value_counts())
            print('Languages identified...')

        #call cleaning method
        self.df = self.cleaning(self.df,self.post_col)
        print('Data cleaned...')

        print('Attempting to extract hashtags...')
        #call getHashtag method to extract hashtags to new column
        self.df['hashtags'] = self.df[self.post_col].map(self.getHashtag)

        #drop any rows in the dataframe that don't have any hashtags
        self.df.dropna(subset=['hashtags'],inplace=True)

        #count of hashtags by post as new columns
        self.df['hashtag_count'] = self.df['hashtags'].map(lambda x: len(x))

        print('Attempting to lemmatise hashtags...')
        #lemmatise any hashtags to new column
        self.df['hashtags_lemma'] = self.df['hashtags'].map(self.getHashtagLemma)

        lemma_conversion = self.lemma_count / self.hashtag_count

        print(f'Of {str(self.hashtag_count)} hashtags, {str(self.lemma_count)} hashtags were successfully lemmatised ({str(lemma_conversion)})')

        #get user post frequency as class attribute
        self.user_count_dict = dict(self.df[self.user_col].value_counts())

        #get median post count for each user as a class attribute
        self.user_hashtag_count_dict = dict(self.df.groupby(self.user_col)['hashtag_count'].median())

        #get user post count as new column
        self.df['user_post_count'] = self.df[self.user_col].map(lambda x: self.getUserpostcount(x))

        #get median user post count as new column
        self.df['user_median_hashtag_count'] = self.df[self.user_col].map(lambda x: self.getUserhashtagcount(x))

        print('Running EDA and generating plots...')

        self.eda()

        return

    #select the data we want to include
    def selectData(self,english=True,remove_verified=True,max_posts=3,lemma=True):

        #retain some attributes
        self._filterenglish = english

        self._filterverified = remove_verified

        self._filterpostcount = max_posts

        self.df_edit = self.df.copy()

        if self.translate == True:
            #filter dataset to only include english if arg is true (default)
            if self._filterenglish == True:

                self.df_edit = self.df_edit[self.df_edit['language']=='en']

        #filter dataset to only include unverified accounts if arg is true (default)
        if self._filterverified == True:

            self.df_edit = self.df_edit[self.df_edit['user_verified_status']== False]

        #filter dataset to only include users who have posted under a threshold number of posts - gets rids of high volume posters
        self.df_edit = self.df_edit[self.df_edit['user_post_count'] <= self._filterpostcount]

        #retains the target column as an attribute of either hashtags that have been lemmatised or not
        if lemma == True:

            self.target = self.df_edit[['hashtags_lemma']]
        else:
            self.target = self.df_edit[['hashtags']]

        print('Data Selected.')

        return

    #create edges and nodes and add these to an instance of a graph object
    def buildGraph(self,additional_stopwords=[],min_frequency=5):

        #call getBatches method passing any contextual stop words as an arg
        batches = self.getBatches(additional_stopwords)

        #call getEdgesNodes mnethod taking max frequency as an arg
        self.getEdgesNodes(batches,min_frequency)

        #call the getGraph method and build the graph
        self.G = self.getGraph()
        print('Graph successfully built.')
        print('Node and Edge dataframes created.')


        """
        save a number of attributes to the instance of the class
        """
        #retain graph object adjacencies
        self.adjacencies = dict(self.G.adjacency())

        #retain graph object node betweeness centrality
        self.betweeness = nx.betweenness_centrality(self.G)

        #retain graph object clustering coefficients
        self.clustering_coeff = nx.clustering(self.G)

        """
        add these attributes as columns on the node dataframe
        """

        self.node_df['adjacency_frequency'] = self.node_df['id_code'].map(lambda x: len(self.adjacencies[x]))

        self.node_df['betweeness_centrality'] = self.node_df['id_code'].map(lambda x: self.betweeness[x])

        self.node_df['clustering_coefficient'] = self.node_df['id_code'].map(lambda x: self.clustering_coeff[x])

        #identify communities in instance of graph object and retain as attribute
        self.communities = community.greedy_modularity_communities(self.G)

        """
        assign each node to its community and add as column to node dataframe
        """
        self.communities_dict = {}

        nodes_in_community = [list(i) for i in self.communities]

        for i in nodes_in_community:

            self.communities_dict[nodes_in_community.index(i)] = i

        def community_allocation(source_val):
            for k,v in self.communities_dict.items():
                if source_val in v:
                    return k

        self.node_df['community'] = self.node_df['id_code'].map(lambda x: community_allocation(x))

        print('Communities calculated.')
        return

    #plot the graph using plotly
    def plotGraph(self,sizing=75,node_size='adjacency_frequency',layout=nx.kamada_kawai_layout,light_theme=True,colorscale='Viridis',community_plot=False):

        #formatting options for plot - dark vs. light theme
        if light_theme:
            back_col = '#ffffff'
            edge_col = '#ece8e8'

        else:
            back_col = '#000000'
            edge_col = '#2d2b2b'

        """
        normalise all graph metrics
        """
        #subset graph metrics
        X = self.node_df[self.node_df.columns[2:5]]

        #get columns labels
        cols = self.node_df.columns[2:5]

        #instantiate instance of MinMaxScaler class
        min_max_scaler = preprocessing.MinMaxScaler()

        #transform graph metrics
        X_scaled = min_max_scaler.fit_transform(X)

        #create new dataframe of scaled metrics
        plot_df = pd.DataFrame(X_scaled)

        plot_df.columns=cols

        for i in plot_df.columns:
            plot_df[i] = plot_df[i].apply(lambda x: x*sizing)


        #extract graph x,y co-ordinates from G instance
        pos = layout(self.G)

        #add position of each node from G to 'pos' key
        for node in self.G.nodes:
            self.G.nodes[node]['pos'] = list(pos[node])



        stack = []

        index = 0

        #add edges to Plotly go.Scatter object
        for edge in self.G.edges:

            x0, y0 = self.G.nodes[edge[0]]['pos']

            x1, y1 = self.G.nodes[edge[1]]['pos']

            weight = 0.5

            trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                               mode='lines',
                               line={'width': weight},
                               marker=dict(color=edge_col),
                               line_shape='spline',
                               opacity=1)

            #append edge traces
            stack.append(trace)

            index = index + 1

        #conditionals for either showing a plot where formatting denotes community or not
        if community_plot == True:

            #make a partly empty dictionary for the nodes
            marker = {'size':[],'line':dict(width=0.5,color=edge_col),'color':[]}

        else:

            #make a partly empty dictionary for the nodes
            marker = {'colorscale':colorscale,'size':[],'line':dict(width=0.5,color=edge_col),'color':[],'colorbar':dict(thickness=15,
                                                                                               title='Node Connections',
                                                                                               xanchor='left',
                                                                                               titleside='right')}


        #initialise a go.Scatter object for the nodes
        node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers', textposition="bottom center",
                                hoverinfo="text", marker=marker)

        index = 0

        #add nodes to Plotly go.Scatter object
        for node in self.G.nodes():

            x, y = self.G.nodes[node]['pos']

            node_trace['x'] += tuple([x])

            node_trace['y'] += tuple([y])

            node_trace['text'] += tuple([self.node_df['id'][index]])

            if community_plot == True:

                node_trace['marker']['color'] += tuple(list(self.node_df.community))

                node_trace['marker']['size'] += tuple([list(plot_df[node_size])[index]])

            else:

                node_trace['marker']['color'] += tuple([list(self.node_df.adjacency_frequency)[index]])

                node_trace['marker']['size'] += tuple([list(self.node_df.adjacency_frequency)[index]])

            index = index + 1

        #append node traces
        stack.append(node_trace)


        #set up axis for plot
        axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
                  zeroline=False,
                  showgrid=False,
                  showticklabels=False,
                  title=''
                  )

        #set up figure for plot
        figure = {
        "data": stack,
        "layout":
        go.Layout(title=str(self.source[0]+' is..'),
                    font= dict(family='Arial',size=20),
                    width=1100,
                    height=1100,
                    autosize=False,
                    showlegend=False,
                    xaxis=axis,
                    yaxis=axis,
                    margin=dict(
                    l=40,
                    r=40,
                    b=85,
                    t=100,
                    pad=0,

            ),
            hovermode='closest',
            plot_bgcolor=back_col, #set background color
            )}

        #retain plot figure as attribute
        self.graph_plot = figure

        #plot the figure
        iplot(self.graph_plot)

        return

    #sunburst that plots communities and relevant hahstags
    def plotCommunity(self,colorscale=False):

        #make copy of node dataframe
        df_temp = self.node_df.copy()

        #change community label to string (needed for plot)
        df_temp['community'] = df_temp['community'].map(lambda x: str(x))

        #conditionals for plot type
        if colorscale == False:

            fig = px.sunburst(df_temp, path=['community', 'id'], values='adjacency_frequency',color='community',hover_name=None,
                          hover_data=None)
        else:
            fig = px.sunburst(df_temp, path=['community', 'id'], values='adjacency_frequency',
                          color='betweeness_centrality', hover_data=None,
                          color_continuous_scale='blugrn',
                          color_continuous_midpoint=np.average(df_temp['betweeness_centrality'], weights=df_temp['betweeness_centrality']))

        #add margin to plot
        fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))

        #retain community plot as attribute
        self.community_plot = fig

        #offline sunburst plot
        iplot(self.community_plot)

        return

    #save map / sunburst plot locally as html file
    def savePlot(self,plot='map'):

        #get current time
        date = str(pd.to_datetime(datetime.datetime.now())).split(' ')[0]


        if plot == 'map':

            plot_save = self.graph_plot

            filename=date+'_'+self.source[0]+'_graph_plot_instagram.html'

            plotly.offline.plot(plot_save, filename=filename)


        elif plot == 'community':

            plot_save = self.community_plot

            filename= date+'_'+self.source[0]+'_community_plot_instagram.html'

            plotly.offline.plot(plot_save, filename=filename)

        return print('Plot saved.')

    #save csv output
    def saveTables(self):

        date = str(pd.to_datetime(datetime.datetime.now())).split(' ')[0]

        self.node_df.to_csv(date+"_node_df_"+str(self.source[0])+".csv",index=False)
        print('Saved nodes')

        self.edge_df.to_csv(date+"_edge_df_"+str(self.source[0])+".csv",index=False)
        print('Saved edges'

        self.df_edit.to_csv(date+"_df_edit_"+str(self.source[0])+".csv",index=False)
        print('Saved edited dataframe')

        self.df.to_csv(date+"_df_"+str(self.source[0])+"_.csv",index=False)
        print('Saved unedited dataframe')

        return
