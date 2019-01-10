import bayes
import feedparser
ny= feedparser.parse('http://www.chinadaily.com.cn/rss/china_rss.xml')
sf=feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
#sf=feedparser.parse('http://www.chinadaily.com.cn/rss/cndy_rss.xml')
print(ny['entries'])
len(ny['entries'])
vocabList,pSF,pNY=bayes.localWords(ny,sf)
