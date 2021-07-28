"""Intelligently pretty-print HTML/XML with inline tags.
prettify_xml() can be used for any XML text.
prettify_html() is specifically for BeautifulSoup.prettify() output,
as it does not add tag linebreaks.
"""

import re
import xml.dom.minidom as xmldom


class RegExSub:

    """Dict factory for regex and corresponding substitution expression.
    Attributes:
        regex (re.Pattern): Compiled regex to use in re.search()/match()
        replace_with (TYPE): Description
    """

    def __init__(self, pattern, flags=0, replace_with=''):
        """Create RexExSub instance.
        Args:
            pattern (str): String to compile as regex.
            flags (re.RegexFlag, optional): Flags for re.compile().
            replace_with (str): String to replace regex matches. Default
            removes match by replacing with empty string.
        """
        self.regex = re.compile(pattern, flags)
        self.replace_with = replace_with

    def sub(self, string):
        """Perform regex substitution on given string.
        Args:
            string (str): String to be processed.
        Returns:
            str: String after replacements made.
        """
        return self.regex.sub(self.replace_with, string)


def apply_re_subs(string, RegExSub_list, debug=False):
    """Apply multiple regex substitutions to a string.
    Args:
        string (str): String to be processed.
        RegExSub_list (list): List of RegExSub objects to apply.
        debug (bool, optional): Show results of each regexp application.
    Returns:
        str: String after all regex substitutions have been applied.
    """
    processed_string = string
    for regex_obj in RegExSub_list:
        processed_string = regex_obj.sub(processed_string)
        if debug:
            print('========================================================\n')
            print(regex_obj.regex)
            print('--------------------------------------------------------\n')
            print(processed_string)
    return processed_string


def prettify_xml(xml_string, indent=2, debug=False):
    """Prettify XML with intelligent inline tags.
    Args:
        xml_string (str): XML text to prettify.
        indent (int, optional): Set size of XML tag indents.
        debug (bool, optional): Show results of each regexp application.
    Returns:
        str: Prettified XML.
    """
    doc = xmldom.parseString(xml_string)
    indent_str = ' ' * indent
    ugly_xml = doc.toprettyxml(indent=indent_str)

    inline_all_tags = RegExSub(r'>\n\s*([^<>\s].*?)\n\s*</', re.S, r'>\g<1></')

    whitespace_re = RegExSub(r'^[\s\n]*$', re.M)

    empty_tags = RegExSub(r'(<[^/]*?>)(\n|\s)*(</)', re.M, r'\g<1>\g<3>')

    blankline_re = RegExSub(r'(>)\n$', re.M, r'\g<1>')

    regexps = [inline_all_tags, whitespace_re, empty_tags, blankline_re]
    pretty_xml = apply_re_subs(ugly_xml, regexps, debug)
    return pretty_xml


def prettify_html(html_string, debug=False):
    """Restore inline HTML tag format of BeautifulSoup.prettify() output.
    Does not add or remove regular line breaks. Can be used with regular
    HTML if it already has the newlines you want to keep.
    Args:
        html_string (str): HTML string to parse.
        debug (bool, optional): Show results of each regexp application.
    Returns:
        str: Prettified HTML.
    """

    inline_all_tags = RegExSub(r'>\n\s+([^<>\s].*?)\n\s*</', re.S, r'>\g<1></')

    # Newline after <br> tag, not before.
    # Space after <br> keeps content text seperate when stripping tags.
    br = RegExSub(r'\n\s*(<br.*?>)', re.S, r'\g<1> ')

    # Superscripts are always attached to the ends of words.
    sup_start = RegExSub(r'[\n\s]*(<(sup|wbr).*?>)[\n\s]*', re.S | re.M,
                         r'\g<1>')

    # Superscripts are separated from following words with a space.
    sup_end_space = RegExSub(r'[\n\s]*(</sup>)([a-zA-Z0-9])', re.S,
                             r'\g<1> \g<2>')

    # Removes white space between superscript and immediately following tag
    sup_end_tag = RegExSub(r'[\n\s]*(</sup>)[\n\s]*(<)', re.S, r'\g<1> \g<2>')

    # Inlining common inline elements
    inline_start = RegExSub(r'[\n\s]*(<(strong|a|span).*?>)[\n\s]*', re.S,
                            r' \g<1>')

    # Removes whitespace between end of inline tags and beginning of new tag
    inline_end = RegExSub(r'[\n\s]*(</(strong|a|span)>)[\n\s]*(?=<)', re.S,
                          r'\g<1>')

    # Adds a space between the ending inline tags and following words
    inline_space = RegExSub(r'[\n\s]*(</(strong|a|span)>)([a-zA-Z0-9])', re.S,
                            r'\g<1> \g<3>')

    # Removes spaces between nested inline tags
    nested_spaces_start = RegExSub(r'(<[^/]*?>) (?=<)', 0, r'\g<1>')

    # Removes spaces between nested end tags--which don't have attributes
    # so can be differentiated by string only content
    nested_spaces_end = RegExSub(r'(</\w*?>) (?=</)', 0, r'\g<1>')

    # Adds breaks between <html> tags and removes newline before </body>
    # if str(parser) is used instead of parser.prettify()
    html_tag_start = RegExSub(r'(<html>)(?=<)', 0, r'\g<1>\n ')

    html_tag_end = RegExSub(r'(</body>)(</html>)', 0, r' \g<1>\n\g<2>')

    end_body_newline = RegExSub(r'\n\s+(\n\s*(?=</body>))', 0, r'\g<1>')

    # Make empty tags inline
    empty_tags = RegExSub(r'(<(?P<tag_name>.*?)>)[\n\s]+(</(?P=tag_name))', 0,
                          r'\g<1>\g<3>')

    regexps = [inline_all_tags, br, sup_start, sup_end_tag, sup_end_space,
               inline_start, inline_end, inline_space, nested_spaces_start,
               nested_spaces_end, html_tag_start, html_tag_end,
               end_body_newline, empty_tags]

    pretty_html = apply_re_subs(html_string, regexps, debug)
    return pretty_html
