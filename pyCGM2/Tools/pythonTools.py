# -*- coding: utf-8 -*-


class ordict(dict):
    """Dictionnaire dont les clés restent dans l'ordre d'insertion"""

    # ========================================================================
    def __init__(self, *args, **kwargs):
        dict.__init__(self) # initialisation du parent
        self._keys = [] # conserve les clés dans l'ordre initial d'insertion
        self.update(*args, **kwargs) # initialise avec les arguments donnés

    # ========================================================================
    def __setitem__(self, key, value):
        """initialise la clé avec la valeur"""
        if key not in self._keys:
            self._keys.append(key)
        dict.__setitem__(self, key, value)

    # ========================================================================
    def __delitem__(self, key):
        """supprime la clé key (Except. KeyError si elle n'existe pas)"""
        try:
           dict.__delitem__(self, key)
           if key in self.keys():
               self._keys.remove(key)
        except:
            raise    # redéclenche l'exception pour l'appelant

    # ========================================================================
    def clear(self):
        """remet le dictionnaire à vide"""
        self._keys = []
        dict.clear(self)

    # ========================================================================
    def keys(self):
        """retourne la liste des clés dans l'ordre d'insertion"""
        return self._keys

    # ========================================================================
    def items(self):
        """retourne la liste des couples (clé, valeur) dans l'ordre d'insertion"""
        return [(key, self[key]) for key in self._keys]

    # ========================================================================
    def pop(self, key, value=None):
        """retire la clé key et retourne sa valeur
           si la clé n'existe pas, retourne la valeur par defaut 'value'
        """
        try:
            val = dict.pop(self, key, value)
            if key in self._keys:
                self._keys.remove(key)
        except:
            raise   # redéclenche l'exception KeyError pour l'appelant
        return val

    # ========================================================================
    def popitem(self):
        """retire et retourne l'une des clés avec sa valeur
           si le dico est vide, déclenche une exception
        """
        try:
            key, val = dict.popitem(self)
            self._keys.remove(key)
        except:
            raise   # redéclenche l'exception KeyError pour l'appelant
        return key, val

    # ========================================================================
    def values(self):
        """retourne la liste des valeurs dans l'ordre d'insertion des clés"""
        return [self[key] for key in self._keys]

    # ========================================================================
    def __iter__(self):
        """itérateur renvoyant les clés dans l'ordre d'insertion"""
        for key in self._keys:
            yield key

    # ========================================================================
    def iteritems(self):
        """itérateur renvoyant les tuples (clés, valeur) dans l'ordre d'insertion"""
        for key in self._keys:
            yield (key, self[key])

    # ========================================================================
    def iterkeys(self):
        """itérateur renvoyant les clés dans leur ordre d'insertion"""
        for key in self._keys:
            yield key

    # ========================================================================
    def itervalues(self):
        """itérateur renvoyant les valeurs dans l'ordre d'insertion des clés"""
        for key in self._keys:
            yield self[key]

    # ========================================================================
    def copy(self):
        """retourne la copie superficielle du dictionnaire
           en concervant dans l'ordre des clés
        """
        d = ordict()
        for key in self._keys:
            d[key] = self[key]
        return d

    # ========================================================================
    def __repr__(self):
        """retourne une chaine représentant le dictionnaire ordonné"""
        lst = ['%r: %r' % (key, self[key]) for key in self._keys]
        return type(self).__name__ + '{' + ', '.join(lst) + '}'

    # ========================================================================
    def __str__(self):
        """retourne une chaine représentant le dictionnaire ordonné"""
        return self.__repr__()

    # ========================================================================
    def setdefault(self, key, value=None):
        """si la clé n'existe pas, la crée avec la valeur value
           dans tous les cas, retourne la valeur de la clé
        """
        if key not in self._keys:
            self.__setitem__(key, value)
        return self[key]

    # ========================================================================
    def update(self, *args, **kwargs):
        """ajoute les arguments au dictionnaire dans le bon ordre si possible
           *args est un dictionnaire (dict ou ordict) ou une liste (clé,valeur)
           **kwargs est donné comme un tuple clé=valeur, présentée comme un dict
                            c'est à dire que l'ordre d'appel n'est pas respecté
        """
        if len(args)>0:
            args = args[0]
            if isinstance(args, (dict, ordict)):
                for key, value in args.items():
                    self.__setitem__(key, value)
            elif isinstance(args, (list, tuple)):
                for key, value in args:
                    self.__setitem__(key, value)
        for key in kwargs:
            self.__setitem__(key, kwargs[key])

    # ========================================================================
    def sort(self, cmp=cmp, key=lambda e: e, reverse=False):
        """trie les clés sur place (l'ordre initial d'insertion est perdu!)"""
        self._keys.sort(cmp, key, reverse)

    # ========================================================================
    def sorted(self, cmp=cmp, key=lambda e: e, reverse=False):
        """retourne un nouveau dictionnaire ordonné avec les clés triées"""
        d = self.copy()
        d._keys.sort(cmp, key, reverse)
        return d
