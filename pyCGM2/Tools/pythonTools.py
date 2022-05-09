# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Report
#APIDOC["Draft"]=False
#--end--

class ordict(dict):
    """ordered dictionary"""

    # ========================================================================
    def __init__(self, *args, **kwargs):
        dict.__init__(self)
        self._keys = []
        self.update(*args, **kwargs)

    # ========================================================================
    def __setitem__(self, key, value):
        """init key"""
        if key not in self._keys:
            self._keys.append(key)
        dict.__setitem__(self, key, value)

    # ========================================================================
    def __delitem__(self, key):
        """delete a key (Except. KeyErrorif not exist)"""
        try:
           dict.__delitem__(self, key)
           if key in self.keys():
               self._keys.remove(key)
        except:
            raise    # redéclenche l'exception pour l'appelant

    # ========================================================================
    def clear(self):
        """clear the dictionary"""
        self._keys = []
        dict.clear(self)

    # ========================================================================
    def keys(self):
        """get keys"""
        return self._keys

    # ========================================================================
    def items(self):
        """return items (key,value)"""
        return [(key, self[key]) for key in self._keys]

    # ========================================================================
    def pop(self, key, value=None):
        """remove a key and return its value
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
        """remove a key an return the key an its value  """
        try:
            key, val = dict.popitem(self)
            self._keys.remove(key)
        except:
            raise   # redéclenche l'exception KeyError pour l'appelant
        return key, val

    # ========================================================================
    def values(self):
        """return values"""
        return [self[key] for key in self._keys]

    # ========================================================================
    def __iter__(self):
        """iterator"""
        for key in self._keys:
            yield key

    # ========================================================================
    def iteritems(self):
        """iterator returning  tuples (key, value)"""
        for key in self._keys:
            yield (key, self[key])

    # ========================================================================
    def iterkeys(self):
        """iterator returning  key """
        for key in self._keys:
            yield key

    # ========================================================================
    def itervalues(self):
        """iterator returning value"""
        for key in self._keys:
            yield self[key]

    # ========================================================================
    def copy(self):
        """copy dict
        """
        d = ordict()
        for key in self._keys:
            d[key] = self[key]
        return d

    # ========================================================================
    def __repr__(self):
        """set strings representative of the dictionary"""
        lst = ['%r: %r' % (key, self[key]) for key in self._keys]
        return type(self).__name__ + '{' + ', '.join(lst) + '}'

    # ========================================================================
    def __str__(self):
        """return strings representative of the dictionary"""
        return self.__repr__()

    # ========================================================================
    def setdefault(self, key, value=None):
        """set a default item
        """
        if key not in self._keys:
            self.__setitem__(key, value)
        return self[key]

    # ========================================================================
    def update(self, *args, **kwargs):
        """update the dictionary"""
        if len(args) > 0:
            args = args[0]
            if isinstance(args, (dict, ordict)):
                for key, value in args.items():
                    self.__setitem__(key, value)
            elif isinstance(args, (list, tuple)):
                for key, value in args:
                    self.__setitem__(key, value)
        for key in kwargs:
            self.__setitem__(key, kwargs[key])
