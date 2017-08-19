import urllib2
import urllib
import os

proxies = {}

proxies['http'] = "http://10.3.100.207:8080"
proxies['https'] = "https://10.3.100.207:8080"

# testfile = urllib.URLopener(proxies = proxies)
# testfile.retrieve("https://www.dropbox.com/s/wjl0uelfn7bo5r5/Getting%20Started.pdf?dl=1", "model_cnn.data-00000-of-00001")

# response = urllib.urlopen('https://www.dropbox.com/s/wjl0uelfn7bo5r5/Getting%20Started.pdf?dl=1', proxies = proxies)
# html = response.read()

url = 'https://www.dropbox.com/s/effa1ve9yu2crjg/model_cnn.data-00000-of-00001?dl=1'
from urllib2 import urlopen
u = urlopen(url)
data = u.read()
u.close()
 
with open('model_cnn.data-00000-of-00001', "wb") as f :
    f.write(data)