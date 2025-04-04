import re
from io import StringIO
import tokenize

def apply_filters(dataset):
    # Step 1: Create a "summary" element
    def create_doc(example):
        first_line = example['func_code_string'].split('\n')[0]
        example['summary'] = first_line + '\n    """' + example['func_documentation_string'] + '\n    """'
        example['prompt'] = remove_comments_and_docstrings(example['whole_func_string'], 'python')
        return example
    
    dataset = dataset.map(create_doc)

    # Step 2: Consider the first sentence in the comment as the function summary
    def extract_first_sentence(example):
        first_sentence = example['summary'].split('.')[0]
        example['func_documentation_tokens'] = first_sentence.split()
        return example

    dataset = dataset.map(extract_first_sentence)

    # Step 3: Remove data where functions are shorter than three lines or comments containing less than 3 tokens
    def filter_by_length(example):
        code_lines = example['func_code_string'].count('\n') + 1
        doc_tokens_count = len(example['func_documentation_tokens'])
        return code_lines >= 3 and doc_tokens_count >= 3

    dataset = dataset.filter(filter_by_length)

    # Step 4: Remove functions whose names contain the substring “test”
    dataset = dataset.filter(lambda example: 'test' not in example['func_name'])

    """# Step 5: Remove duplicates by comparing the Jaccard similarities of the functions
    def remove_duplicates(dataset):
        vectorizer = TfidfVectorizer().fit_transform(dataset['func_code_string'])
        cosine_similarities = linear_kernel(vectorizer, vectorizer)
        duplicate_indices = set()

        for i in range(len(cosine_similarities)):
            for j in range(i + 1, len(cosine_similarities)):
                if cosine_similarities[i][j] > 0.8:  # Threshold for similarity
                    duplicate_indices.add(j)

        return dataset.filter(lambda _, idx: idx not in duplicate_indices, with_indices=True)

    dataset = remove_duplicates(dataset)"""

    # Reset index if needed
    dataset = dataset.add_column('index', list(range(len(dataset))))

    return dataset

single = {'java': '//', 'python': '#', 'c': '//',
          'ruby': '#', 'javascript': '//', 'go': '//',
          'php': ['#', '//'],
          'erlang': '%', 'haskell': '--', 'prolog': '%'}

multi = {'java': ['/*', '*/'], 'python': ['"""','"""','/*','*/'], 'c': ['/*', '*/'],
          'ruby': ['=begin', '=end'], 'javascript': ['/*', '*/'], 'go': ['/*', '*/'],
          'php': ['/*', '*/'],
          'erlang': [] , 'haskell': ['{-','-}'], 'prolog': []}

"""
Function 'remove_comments_and_docstrings()' taken from "Source Code Summarization in the Era of Large Language Models"
https://github.com/wssun/LLM4CodeSummarization/blob/main/util/remove_comments.py

@article{Sun_Miao_Li_Zhang_Fang_Liu_Deng_Liu_Chen_2024, 
    title={Source Code Summarization in the Era of Large Language Models}, 
    url={http://arxiv.org/abs/2407.07959}, DOI={10.48550/arXiv.2407.07959}, 
    number={arXiv:2407.07959}, 
    publisher={arXiv}, 
    author={Sun, Weisong and Miao, Yun and Li, Yuekang and Zhang, Hongyu and Fang, Chunrong and Liu, Yi and Deng, Gelei and Liu, Yang and Chen, Zhenyu}, 
    year={2024}, month=jul 
}
"""
def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns "source" minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        def replacer(match):
            s = match.group(0)
            if s.startswith('#') or s.startswith('=begin'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'#.*?$|=begin.*?=end|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['erlang', 'prolog']:
        def replacer(match):
            s = match.group(0)
            if s.startswith('%'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(r'%.*?$|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',re.DOTALL|re.MULTILINE)
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['haskell']:
        def replacer(match):
            s = match.group(0)
            if s.startswith('--') or s.startswith('{-'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'--.*?$|\{-.*?-}|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['php']:
        def replacer(match):
            s = match.group(0)
            if s.startswith('#') or s.startswith('//') or s.startswith('/*'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'#.*?$|//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)

    elif lang in ['java', 'go', 'javascript', 'c']:
        def replacer(match):
            s = match.group(0)
            if s.startswith('//') or s.startswith('/*'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)

    else:
        print('Unkown language!')
        return source


if __name__ == '__main__':
    code = '''
          end
      # NOTE this is required, since repaint will just not happen otherwise
      # Some components are erroneously repainting all, after setting this to true so it is 
      # working there. 
      @current_component.repaint_required true
      $log.debug " after on_leave STACKFLOW XXX #{@current_component.focussed}   #{@current_component.name}"
      @current_component.repaint
    end
    '''
    print(remove_comments_and_docstrings(code,'php'))