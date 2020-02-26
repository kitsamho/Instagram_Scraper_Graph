There are two Python files here each containing a custom Instagram class

# InstagramScraper()

This class is made up of a series of methods that allow for the scraping of Instagram post data. The pipeline consists of three main methods that need to be called sequentially.There is no current method to chain the whole pipeline. 

`self.logIn()` : user detail capture, webdriver initialisation, Instagram log in. You will need to have an Instagram account to do this.

`self.getLinks()` : gets n unique links containing <#HASHTAG> using WebDriver. 

`self.getData()` : implements multi-threaded scraping of data from self.getLinks using a combination of Selenium WebDriver and Beautiful Soup. Method returns a pandas DataFrame

# InstagramGraph()

This class is made up of a series of methods that take the DataFrame from InstagramScraper(). As above, the methods below need to be called sequentially.There is no current method to chain the whole pipeline. 

`self.getFeatures(translate=False)` : creates various descriptive metrics from the data and if translate set to True, will access Google Translate API and translate posts. 

`self.selectData(english=True,remove_verified=True,max_posts=3,lemma=True)`:Subsets the data across various variables, filtering out non-English data, removing verified users, limiting post frequency and lemmatising any hashtags

`selfbuildGraph(additional_stopwords=[],min_frequency=5)`: generates edges and nodes and adds them to an instance of a NetworkX graph object. 
*additional_stopwords*: There are default stopwords however extra ones that are relevant to the topic scraped can be added.

*min_frequency*: Refers to an edge frequency threshold. The lower the min_frequency, the longer it takes to build the graph and the more potential noise will exist in the model. The higher the frequency the more 'bigger' picture the ouputs will be - albeit with less detail. Experiment with this.

`self.plotGraph(sizing=75,node_size='adjacency_frequency',layout=nx.kamada_kawai_layout,light_theme=True,colorscale='Viridis',community_plot=False)`:

*sizing*: modify this to change the relative size of all nodes in the Plotly Scatterplot
*node_size*: choose a graph metric to reprent node size - betweeness_centrality,clustering_coefficient are alternatives to the default.
*layout*: choose the nx.layout to plot
*light_theme*: different Plot styles
*colorscale*: colourscalefor gradient colouring
*community_plot*: colours nodes as per community allocation if set True

