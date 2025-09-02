import requests
from bs4 import BeautifulSoup

url = "https://www.omdbapi.com/"

try:

    response = requests.get(url)
    if response.status_code ==200:
        print("Successfully fetched the URL")
        data = BeautifulSoup(response.content,'html.parser')
        print("Data fetched successfully",data.title)
        print("title",data.title.get_text(strip=True))
        #print("head",data.head)
        #print("body",data.body)

        tags = data.a

        print("tags :",tags["class"][0])

        # find all the anchor tags
        anchor_tags = data.find_all('a')
        for index,tag in enumerate(anchor_tags):
            if "http" in tag.get_text():
                print("    ",index," :",tag.get_text())

        print("comment:-",data.p.string)
        [print("previous sibling:",sibling) for sibling in data.p.previous_siblings]
        
        print("next sibling:-",data.p.next_sibling)
        print("parent name:-",data.p.parent.name)
        print("parent attr:-",data.p.parent.attrs)
        print("parent atrr class:-",data.p.parent.attrs['class'])
        print("parent att class 0:-",data.p.parent.attrs['class'][0])
        print("attrs p :-",data.p.attrs)
        print("attrs class:-",data.p.attrs['class'])
              

    else:
        print(f"Failed to fetch the URL, status code: {response.status_code}")
        exit()

except requests.exceptions.RequestException as e:
    print(f"Error fetching the URL: {e}")
    exit()