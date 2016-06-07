import urllib

from lxml import etree, html


def get_fifa_rating(player_name, season):
    url = "http://sofifa.com/players?keyword="+urllib.parse.quote_plus(player_name)+"&v="+season+"&hl=en-US"
    with urllib.request.urlopen(url) as page:
        s = page.read().decode("utf-8")
    tree = html.fromstring(s).getroottree()
    for table_cell in tree.findall('//td'):
        if 'data-title' in table_cell.attrib and table_cell.attrib['data-title'] == "Overall rating":
            rating = table_cell.find('./span').text
            print((url, int(rating)))
            return 1, int(rating)

    print((url, 50))
    return 0, 50
