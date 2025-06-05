class AbstractSchemeProcedure(object):
    def __init__(self):
        pass
      
    def setData(self,categorie,fileLst,targets):
        attr = getattr(self, categorie)
        for file in fileLst:
            attr.append([file, targets])