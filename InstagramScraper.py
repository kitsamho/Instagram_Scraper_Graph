#requests
import requests
import urllib
import urllib.request
from urllib.request import urlopen
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup
from datetime import datetime
import requests
from urllib.request import urlopen, Request

#data, strucuture and maths
import pandas as pd
import numpy as np
import math
import json
import string
from  more_itertools import unique_everseen
import random

#progress,performance and management
from tqdm import tqdm_notebook
import threading
import os
import ssl
from IPython.display import clear_output
import platform
import os

# imports used in Selenium
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.keys import Keys

#time
from time import sleep
import time

#text processing / regex
import regex
import re

#make wide
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

#passwords
import getpass



class InstagramScraper():

    """
    Class that allows you to scrape the content of Instagram posts, either
    a profile or a hashtag.

    Initialised with the location of your Chromedriver location

    """

    def __init__(self,driver_loc='/Users/sam/Desktop/Chromedriver/chromedriver'):

        self.driver_loc = driver_loc

    def multithreadCompile(self,thread_count,iteration_list,func):

        """
        This function compiles the batched needed for mult-threadding

        Args:

            thread_count is the number of threads used for multi-threadding

            iteration_list is the source list of urls to iterate over

            func is the function to be used in the multi-thredding process

        Returns:

            The batches that have been allocated to be run using the specified
            function

        """

        jobs = [] #empty list for jobs

        #distribute iteration list to batches and append to jobs list
        batches = [i.tolist() for i in np.array_split(iteration_list,thread_count)]

        for i in range(len(batches)):

            jobs.append(threading.Thread(target=func,args=[batches[i]]))

        return jobs

    def multithreadExecute(self,jobs):

        """

        This function executes the multi-threadding process

        Args:

            The batches that have been appended to a jobs list

        Returns:

            Nothing, merely executes the multi-threadding

        """

        # Start the threads
        for j in jobs:
            print('execute working')
            j.start()

        # Ensure all of the threads have finished
        for j in jobs:

            j.join()

        return

    def getJson(self,url):

        """
        This function exracts a JSON style dictionary from the html for any
        given unique Instagram post

        Args:

            An Instagram post URL

        Returns:

            JSON dictionary ouput

        """

        page = urlopen(url).read() #read url

        data=BeautifulSoup(page, 'html.parser') #get a BeautifulSoup object

        body = data.find('body') #find body element

        script = body.find('script') #find script element

        #some string formatting
        raw = script.text.strip().replace('window._sharedData =', '').replace(';', '')

        #load string
        json_data=json.loads(raw)

        return json_data #return JSON dictonary

    def userDetails(self):

        """
        Functions that capture log in details and logs user into Instagram

        Args:

            None needed

        Returns:

            Nothing

        """
        #capture username
        username = input('Enter username...')

        #capture password
        password = getpass.getpass('Enter password...')

        self._password = password #retain password as attribute

        self._username = username #retain user name as attribute

        return

    def openWebdriver(self):

        """
        Launches Chrome webdriver

        Args:

            None needed

        Returns:

            driver

        """

        #intiate driver
        print("Launching driver...")

        #retain current driver as attribute
        driver = webdriver.Chrome(self.driver_loc)

        return driver

    def closeWebdriver(self,driver):

        """
        Closes Chrome webdriver

        Args:

            webDriver

        Returns:

            Nothing

        """

        driver.close()

        return

    def instagramLogin(self,driver):

        """
        Logs in to Instagram

        Args:

            Current webdriver

        Returns:

            Current webdriver - logged into Instagram

        """

        #base url
        driver.get('https://www.instagram.com/accounts/login/?source=auth_switcher')

        sleep(2) #wait

        #log in
        username_field = driver.find_element_by_xpath('//*[@id="react-root"]/section/main/div/article/div/div[1]/div/form/div[2]/div/label/input')

        username_field.click() #click on username button

        #send username
        username_field.send_keys(self._username)

        #locate element to click
        try:
            password_field = driver.find_element_by_xpath('//*[@id="react-root"]/section/main/div/article/div/div[1]/div/form/div[3]/div/label/input')

        except:
            password_field = driver.find_element_by_xpath('//*[@id="react-root"]/section/main/div/article/div/div[1]/div/form/div[4]/div/label/input')

        password_field.click()

        password_field.send_keys(self._password)

        sleep(2)

        #find log in button
        login_button = driver.find_element_by_xpath('//*[@id="react-root"]/section/main/div/article/div/div[1]/div/form/div[4]')

        login_button.click()

        sleep(3)

        #locate floating window to click and close
        floating_window = driver.find_element_by_class_name('piCib')

        button = floating_window.find_element_by_class_name('mt3GC')

        not_now = button.find_element_by_xpath('/html/body/div[4]/div/div/div[3]/button[2]')

        not_now.click()

        return driver

    def setTarget(self):

        """
        Function that sets either a profile or a hashtag as a target

        Args:

            None

        Returns:

            base url to scrape - either a hashtag page or a profile page

        """

        #tou can choose either hashtag search or a profile to search
        route = input('What do you want to scrape, profile posts or hashtags? (p/h)')

        #if hashtags
        if route == 'h':

            #set hashtag
            hashtag = input('Which hashtag do you want to scrape posts for: ')

            self.target_label = '#'+hashtag #retain hashtag as attribute

            tag_url = 'https://www.instagram.com/explore/tags/' #set base url

            self._target = tag_url+hashtag #set url to scrape from

            return self._target #return url to scrape from

        else:

            profile = input('What profile do you want to scrape posts for: ')

            self.target_label = '@'+profile #retain profile as attribute

            profile_url = 'https://www.instagram.com/' #set base url

            self._target = profile_url+profile #set url to scrape from

            return self._target #return url to scrape from

    def scrapeLinks(self,url):

        """
        Function that scrapes the links needed

        Args:

            target_url

        Returns:

            Nothing - but retains a list of urls to scrape

        """

        #pass url as argument to Selenium webDriver, loads url
        self.activedriver.get(url)

        options = webdriver.ChromeOptions()

        #start maximised
        options.add_argument("--start-maximized")

        #gets scroll height
        last_height = self.activedriver.execute_script("return document.body.scrollHeight")

        #initiate empty list for unique Instagram links
        links = []

        #some lines for user interactivity / selection of link target(n)
        print("\n")
        target = input("How many links do you want to scrape (minimum)?: ")
        print("\n")
        print("Staring Selenium scrape, please keep browser open.")
        print("\n")

        #this loops round until n links achieved or page has ended

        while True:

            source = self.activedriver.page_source

            data= BeautifulSoup(source, 'html.parser')

            body = data.find('body')

            #script = body.find('span')

            for link in body.findAll('a'):

                if re.match("/p", link.get('href')):

                    links.append('https://www.instagram.com'+link.get('href'))

                else:
                    continue

            # Scroll down to bottom
            self.activedriver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(2)

            # Calculate new scroll height and compare with last scroll height
            new_height = self.activedriver.execute_script("return document.body.scrollHeight")

            #if no more content, scrape loop is terminated
            if new_height == last_height:
                break

            last_height = new_height

            #update on successful links scraped
            print("Scraped ", len(links)," links, ", len(set(links)),' are unique')

            #if n target met then while loop breaks
            if len(set(links))>int(target):
                break

        #links are saved as an attribute for the class instance
        self._links = list(unique_everseen(links))

        #clear the screen and provide user feedback on performance
        clear_output()

        print("Finished scraping links. Maxed out at ", len(links)," links, of which ", len(self._links),' are unique.')

        print("\n")

        print("Unique links obtained. Closing driver")

        print("\n")
        # close driver
        self.closeWebdriver(self.activedriver)

        return


    def postDate(self,data):

        """
        Function that gets the date of post
        Args:
            JSON dictionary for post
        Returns:
            datetime of post
        """

        return datetime.utcfromtimestamp(data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['taken_at_timestamp']).strftime('%Y-%m-%d %H:%M:%S')

    def postUser(self,data):

        """
        Function that gets the username of the person who posted
        Args:
            JSON dictionary for post
        Returns:
            username
        """
        return data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['owner']['username']

    def postVerifiedUser(self,data):

        """
        Function gets the verified status of the user
        Args:
            JSON dictionary for post
        Returns:
            verified status
        """
        return data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['owner']['is_verified']


    def postLikes(self,data):

        """
        Function that gets the number of likes the post received
        Args:
            JSON dictionary for post
        Returns:
            number of likes
        """

        return data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['edge_media_preview_like']['count']

    def postVerifiedTags(self,data):

        """
        Function that gets the verified tags that a post contains
        Args:
            JSON dictionary for post
        Returns:
            the verified tags in the post
        """

        tag_end_point = data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['edge_media_to_tagged_user']['edges']

        entities = []

        verif = []

        for i in range(len(tag_end_point)):

            entities.append(tag_end_point[i]['node']['user']['full_name'])

            verif.append(tag_end_point[i]['node']['user']['is_verified'])

        df = pd.DataFrame({'Brand':entities,'Verified':verif})

        df = df[df.Verified == True]

        if len(list(df.Brand)) < 1:

            return np.nan

        else:

            return list(df.Brand)

    def postUnverifiedTags(self,data):

        """
        Function that gets the unverified tags a post contains
        Args:
            JSON dictionary for post
        Returns:
            the unverified tags in the post
        """

        tag_end_point = data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['edge_media_to_tagged_user']['edges']

        tags = [] #emoty list for entities

        verif = [] #empty list for verified status

        #loop through
        for i in range(len(tag_end_point)):

            #append entities
            tags.append(tag_end_point[i]['node']['user']['full_name'])

            #append verified status
            verif.append(tag_end_point[i]['node']['user']['is_verified'])

        #DataFrame of verified / unverified tags
        df = pd.DataFrame({'Tag':tags,'Verified':verif})

        #subset on unverified tags
        df = df[df.Verified == False]

        #if there are unverified tags then return NaN else return unverified tags
        if len(list(df.Tag)) < 1:

            return np.nan

        else:

            return ''.join(list(df.Tag))

    def postComment(self,data):

        return data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['edge_media_to_caption']['edges'][0]['node']['text']

    #get location of post
    def postLocation(self,data):

        """
        Function that gets the post location if available
        Args:
            JSON dictionary for post
        Returns:
            the posts location
        """

        try:

            if len(list(data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['location']['name'])) > 0:

                return data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['location']['name']
        except:

            return np.nan

    #get accessibility  / image data
    def postAccessibility(self,data):

        """
        Function that gets the post accessibility data if available
        Args:
            JSON dictionary for post
        Returns:
            the accessibility data
        """

        try:
            try:
                image = data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['accessibility_caption'].replace('Image may contain: ','').replace(' and ',', ').replace('one or more ','')

                return image

            except:
                image = data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['edge_sidecar_to_children']['edges'][0]['node']['accessibility_caption'].replace('Image may contain: ','').replace(' and ',', ').replace('one or more ','')

                return image
        except:
            return np.nan

    #return original post link
    def postLink(self,data):

        return data


    """
    The three main methods that combine all above
    """
    #get user details, log in and initiate driver
    def logIn(self):

        self.userDetails()

        driver = self.openWebdriver()

        self.activedriver = self.instagramLogin(driver)

        clear_output()

        print('Successfully logged in..ready to scrape')

    #get all the unique links
    def getLinks(self):

        return self.scrapeLinks(self.setTarget())

    #extract data and return dataframe
    def getData(self):

        #create empty lists for posts and comments
        post_date_l = []

        post_user_l = []

        post_verif_l = []

        post_likes_l = []

        post_tags_v_l =[]

        post_tags_u_l = []

        post_l = []

        post_location_l = []

        post_insta_classifier_l = []

        post_link_l = []

        self._listStack = [post_date_l,post_user_l,post_verif_l,post_likes_l,post_tags_v_l,
                      post_tags_u_l,post_l,post_location_l,post_insta_classifier_l,post_link_l]

        self._functionStack = [ self.postDate,
                                    self.postUser,
                                    self.postVerifiedUser,
                                    self.postLikes,
                                    self.postVerifiedTags,
                                    self.postUnverifiedTags,
                                    self.postComment,
                                    self.postLocation,
                                    self.postAccessibility,
                                    self.postLink]

        def extractData(links=self._links):

            #loops through and calls each data collection function on each link
            for i in tqdm_notebook(range(len(links))):

                try:

                    data = self.getJson(links[i])

                    for function in self._functionStack:

                        if function != self._functionStack[-1]:

                            try:

                                self._listStack[self._functionStack.index(function)].append(function(data))

                            except:

                                 self._listStack[self._functionStack.index(function)].append(np.nan)
                        else:
                            self._listStack[-1].append(self._functionStack[-1](links[i]))

                except:
                    pass
            return

        # execute html parsing fuction using multi threading
        print("Attemping multi-threading...")

        print("\n")

        threads = int(input("How many threads?: "))

        print("\n")

        print("Executing...")

        self.multithreadExecute(self.multithreadCompile(threads,self._links,extractData))

        #set up intial data structure
        df = pd.DataFrame({'searched_for':[self.target_label]*len(post_l),
                           'post_link' :post_link_l,
                           'post_date':post_date_l,
                           'post':post_l,
                           'user':post_user_l,
                           'user_verified_status': post_verif_l,
                           'post_likes':post_likes_l,
                           'post_verified_tags':post_tags_v_l,
                           'post_unverified_tags':post_tags_u_l,
                           'post_location':post_location_l,
                           'post_image':post_insta_classifier_l,

                               })

        df.sort_values(by='post_date',ascending=False,inplace=True)

        df.reset_index(drop=True,inplace=True) #reset index

        self._df = df #retain final DataFrame as attribute

        return df
