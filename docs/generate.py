
import pyCGM2.Lib.plot


import inspect

# Création d'une liste de noms de classes
classes = inspect.getmembers(pyCGM2.Lib.plot, inspect.isclass)
class_names = [cls for cls in classes if cls[1].__module__ == 'pyCGM2.Lib.plot']

for name, cls in class_names:
    print(f'Classe: {name}')
    
    # Obtenir les méthodes pour chaque classe
    methods = inspect.getmembers(cls, inspect.isfunction)
    method_names = [method[0] for method in methods if method[1].__module__ == cls.__module__]
    
    for method_name in method_names:
        print(f'   Méthod: {method_name}')
    print("----------------------------")


print("---------Fonctions du Module------------")

# Création d'une liste de noms de fonctions
functions = inspect.getmembers(pyCGM2.Lib.plot, inspect.isfunction)
function_names = [func[0] for func in functions if func[1].__module__ == 'pyCGM2.Lib.plot']

for name in function_names:
    print(f'   {name}')

# import ast
# import os

# class DocstringExtractor(ast.NodeVisitor):
#     def __init__(self):
#         self.docstrings = {}

#     def visit_Module(self, node):
#         self.docstrings[node] = ast.get_docstring(node)
#         self.generic_visit(node)

#     def visit_ClassDef(self, node):
#         self.docstrings[node] = ast.get_docstring(node)
#         self.generic_visit(node)

#     def visit_FunctionDef(self, node):
#         self.docstrings[node] = ast.get_docstring(node)
#         self.generic_visit(node)

# class AddParentNodeTransformer(ast.NodeTransformer):
#     def visit(self, node):
#         for child in ast.iter_child_nodes(node):
#             child.parent = node
#         return super().visit(node)

# def process_file(input_path, output_path):
#     with open(input_path, 'r') as file:
#         source_code = file.read()


#     # Analyse le code source et ajoute des références parent
#     tree = ast.parse(source_code)
#     AddParentNodeTransformer().visit(tree)

#     # Extraction des docstrings
#     extractor = DocstringExtractor()
#     extractor.visit(tree)

#     # Recréer le module avec uniquement les docstrings
#     new_source_code = ""
#     for node, docstring in extractor.docstrings.items():
#         if isinstance(node, ast.Module):
#             new_source_code += f'"""{docstring}"""\n\n' if docstring else ''
#         elif isinstance(node, ast.ClassDef):
#             # Commencer la définition de la classe
#             new_source_code += f'class {node.name}:\n'
#             if docstring:
#                 new_source_code += f'    """{docstring}"""\n'
#             # Ajouter 'pass' si la classe n'a pas de méthodes
#             if not any(isinstance(child, ast.FunctionDef) for child in node.body):
#                 new_source_code += '    pass\n'
#         elif isinstance(node, ast.FunctionDef):
#             # Vérifier si la fonction est une méthode d'une classe
#             if isinstance(node.parent, ast.ClassDef):
#                 new_source_code += f'    def {node.name}(self):\n'
#             else:
#                 new_source_code += f'def {node.name}():\n'
#             if docstring:
#                 new_source_code += f'        """{docstring}"""\n'
#             new_source_code += '        pass\n\n'

#     # Écrire le nouveau fichier
#     with open(output_path, 'w') as file:
#         file.write(new_source_code)

   

# # Exemple d'utilisation
# input_path = 'C:/Program Files/Vicon/Nexus2.15/SDK/Win64/Python/viconnexusapi/viconnexusapi/ViconNexus.py'
# output_path = 'C:/Users/fleboeuf/Documents/Programmation/pyCGM2/pyCGM2/pyCGM2/docs/nexusSDK/ViconNexus.py'
# process_file(input_path, output_path)



