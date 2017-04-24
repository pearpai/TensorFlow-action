import urllib2

url = 'http://localhost:8080/getVerifyImage'
request = urllib2.Request(url)
html = urllib2.urlopen(request).read()

count = 0
while count < 10000:
    print 'The count is:', count
    request = urllib2.Request(url)
    _html = urllib2.urlopen(request).read()
    count += 1
