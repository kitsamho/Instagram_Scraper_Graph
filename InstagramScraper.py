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

    def __init__(self,driver_loc='/Users/sam/Desktop/Chromedriver/chromedriver'):

        self.driver_loc = driver_loc

        self.userDetails

        self.openWebdriver

        self.closeWebdriver

        self.instagramLogin

        self.multithreadCompile

        self.multithreadExecute

        self.getJson

        self.setTarget

        self.scrapeLinks

        self.postDate

        self.postUser

        self.postVerifiedUser

        self.postLikes

        self.postVerifiedTags

        self.postUnverifiedTags

        self.postComment

        self.postLocation

        self.postAccessibility

        self.logIn

        self.getLinks

        self.getData

    """
    Multi threading functions
    """

    def multithreadCompile(self,thread_count,iteration_list,func):

        jobs = []

        #distribute iteration list to batches and append to jobs list
        batches = [i.tolist() for i in np.array_split(iteration_list,thread_count)]

        for i in range(len(batches)):

            jobs.append(threading.Thread(target=func,args=[batches[i]]))


        return jobs

    def multithreadExecute(self,jobs):

        # Start the threads
        for j in jobs:
            print('execute working')
            j.start()

        # Ensure all of the threads have finished
        for j in jobs:
            j.join()
        return

    """
    JSON Functions
    """
    #exracts a JSON style dictionary from the html in any given unique Instagram URL
    def getJson(self,url):

        page = urlopen(url).read()

        data=BeautifulSoup(page, 'html.parser')

        body = data.find('body')

        script = body.find('script')

        raw = script.text.strip().replace('window._sharedData =', '').replace(';', '')

        json_data=json.loads(raw)

        return json_data

    """
    Functions that capture log in details and log user into Instagram
    """
    def userDetails(self):

        #capture username
        username = input('Enter username...')

        #capture password
        password = getpass.getpass('Enter password...')

        self._password = password

        self._username = username

        return

    def openWebdriver(self):

        #intiate driver
        print("Launching driver...")

        driver = webdriver.Chrome(self.driver_loc)

        return driver

    def closeWebdriver(self,driver):

        driver.close()

        return

    def instagramLogin(self,driver):

        driver.get('https://www.instagram.com/accounts/login/?source=auth_switcher')

        sleep(2)

        #log in
        username_field = driver.find_element_by_xpath('//*[@id="react-root"]/section/main/div/article/div/div[1]/div/form/div[2]/div/label/input')

        username_field.click()

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

    """
    Functions that set either a profile or a hashtag as a target and then
    scrapes user specified number of post links
    """
    def setTarget(self):

        route = input('What do you want to scrape, profile posts or hashtags? (p/h)')

        if route == 'h':

            hashtag = input('Which hashtag do you want to scrape posts for: ')

            self.target_label = '#'+hashtag

            tag_url = 'https://www.instagram.com/explore/tags/'

            self._target = tag_url+hashtag

            return self._target

        else:

            profile = input('What profile do you want to scrape posts for: ')

            self.target_label = '@'+profile

            profile_url = 'https://www.instagram.com/'

            self._target = profile_url+profile

            return self._target

    def scrapeLinks(self,url):

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


    """
    Methods that extract various fields of data from Instagram JSON dictionaries
    """
    #get date of post
    def postDate(self,data):

        return datetime.utcfromtimestamp(data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['taken_at_timestamp']).strftime('%Y-%m-%d %H:%M:%S')

    #get user name
    def postUser(self,data):

        return data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['owner']['username']

    #get verified status
    def postVerifiedUser(self,data):

        return data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['owner']['is_verified']

    #get how many likes post has got
    def postLikes(self,data):

        return data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['edge_media_preview_like']['count']

    #get any verified tags
    def postVerifiedTags(self,data):

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

    #get any unverified tags
    def postUnverifiedTags(self,data):

        tag_end_point = data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['edge_media_to_tagged_user']['edges']

        entities = []

        verif = []

        for i in range(len(tag_end_point)):

            entities.append(tag_end_point[i]['node']['user']['full_name'])

            verif.append(tag_end_point[i]['node']['user']['is_verified'])


        df = pd.DataFrame({'Entity':entities,'Verified':verif})

        df = df[df.Verified == False]

        if len(list(df.Entity)) < 1:

            return np.nan

        else:

            return ''.join(list(df.Entity))

    #get post comment
    def postComment(self,data):

        return data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['edge_media_to_caption']['edges'][0]['node']['text']

    #get location of post
    def postLocation(self,data):

        try:

            if len(list(data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['location']['name'])) > 0:

                return data['entry_data']['PostPage'][0]['graphql']['shortcode_media']['location']['name']
        except:

            return np.nan

    #get accessibility  / image data
    def postAccessibility(self,data):

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

#         df['post_hashtags'] = df['post'].map(self.getHashtags)

        df.sort_values(by='post_date',ascending=False,inplace=True)

        df.reset_index(drop=True,inplace=True)

        self._df = df
        return df
