# -*- coding: utf-8 -*-

import pyCGM2



# pyCGM2

from pyCGM2.Utils import files




class readContent_tests():

    @classmethod
    def yamlTest(cls):

        content ="""
            Translators:
                LASI: LeftASI
                RASI: RightASI
                LPSI: LeftPSI
                RPSI: RightPSI
            """
        config = files.readContent(content)
        #import ipdb; ipdb.set_trace()

    @classmethod
    def jsonTest(cls):

        content ="""
        {
        "Translators": {
            "LASI": "example glossary",
            "RASI": "example glossary"
            }
        }
            """
        config = files.readContent(content)
        import ipdb; ipdb.set_trace()



if __name__ == "__main__":



    readContent_tests.yamlTest()
    readContent_tests.jsonTest()
