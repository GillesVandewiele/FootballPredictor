from lxml import etree


def xml_to_tree(path):
    return etree.parse(path)