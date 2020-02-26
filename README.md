There are two Python files here each containing a custom Instagram class

InstagramScraper()

This class is made up of a series of bespoke and existing methods that allow for the scraping of Instagram post data. The pipeline consists of:

self.logIn() : user detail capture, webdriver initialisation, Instagram log in

self.getLinks() : gets n unique links containing <#HASHTAG> using webdriver. 

self.getData() : implements multi-threaded scraping of data from self.getLinks using a combination of Selenium Webdriver and Beautiful Soup. Method returns a pandas DataFrame

InstagramGraph()

This class is made up of a series of methods that take the DataFrame from InstagramScraper() 

self.getFeatures(translate=False) : creates various descriptive metrics from the data and if translate set to True, will access Google Translate API and tr
