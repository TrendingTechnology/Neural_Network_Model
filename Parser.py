# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup as Bs4
import requests
import time
import os
import re

"""

Created on 09.06.2021

@author: Nikita

    The module is designed for parsing html files, collecting the text of articles in Russian. 
    The output is an array of texts.

"""


class Parser:

    def __init__(self, main_url: str, headers: dict):
        self.main_url = main_url
        self.headers = headers

    """
    
    Parameters
    ----------

    main_url : Site link.

    headers :  Dictionary consisting of HTTP headers (For example: browser, authorization, headers, etc.).
    
    """

    @staticmethod
    def pars_local(file_location: str):

        """

        The function is designed for local parsing of html files of scientific articles on the topic of disaster
        medicine of any year using regular expressions. For each article, the code reads the annotation and writes it
        to the list. The result is an array of abstracts of scientific articles in Russian.

        """

        files = os.listdir(file_location)
        text_articles = []

        for year in files:

            for article in os.listdir(fr'{file_location}\{year}'):

                with open(fr"{file_location}\{year}\{article}", encoding="utf-8") as f:

                    try:

                        text_html = f.read()
                        text_html = Bs4(text_html, 'lxml')
                        text_tag = text_html.find_all('p', {'align': "justify"})
                        reg_rus = re.compile(r'[А-ЯЁа-яё]+\s+\w')
                        text = [text.text for text in text_tag
                                if len(reg_rus.findall(text.text)) > 0]
                        text_articles.append(text[0])

                    except IndexError:
                        del article

        return text_articles

    """
    
    Next, 2 functions are designed to analyze a specific site "https://sci-article.ru", so the code is not quite unified
    for other sites and is hard-coded. By changing the tags and blocks in the functions, I parsed other sites related to
    emergencies (sites: "https://rg.ru", "https://iz.ru" etc.), as well as sites with various topics.
    
    """

    def get_links(self, number_page: int):

        """

        The function is designed to create a list of links to scientific articles within a single page of the site.

        Parameters
        ----------

        number_page : The page number of the site with links to articles.

        """

        for i in range(number_page):
            url_news_link = fr'{self.main_url}&j={i}'

            try:

                responce = requests.get(url_news_link, headers=self.headers)
                soup = Bs4(responce.text, 'lxml')

                links = soup.find_all('div', {'id': "stattext"})
                out_links = [item.find('a').get('href') for item in links]

                return out_links

            except requests.exceptions.ConnectionError:

                time.sleep(40)
                responce = requests.get(url_news_link, headers=self.headers)
                soup = Bs4(responce.text, 'lxml')

                links = soup.find_all('div', {'id': "stattext"})
                out_links = [item.find('a').get('href') for item in links]

                return out_links

    def get_text(self, links: list, texts: list):

        """

        The function is designed to create a list of texts of scientific articles on this site.

        Parameters
        ----------

        links : List of links for which you need to take the text.

        texts : The name of the variable where you want to save the texts of scientific articles.

        """
        for link in links:

            try:

                url = fr"{self.main_url[:23]}{link}"
                responce = requests.get(url, headers=self.headers)
                soup = Bs4(responce.text, 'lxml')

                news = soup.find('div', {'id': "stattext"})
                reg = re.compile(r'[А-ЯЁа-яё]+')
                text = ' '.join(reg.findall(news.text))
                texts.append(text)

            except requests.exceptions.ConnectionError:
                pass
