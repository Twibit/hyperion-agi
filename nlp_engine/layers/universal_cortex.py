import re

class UniversalThalamus:
    """The sensory input router, functioning as a profound Recognition Engine."""
    
    def recognize_semantics(self, text: str) -> dict:
        """Analyzes both the structural format and the deep semantic intent of the input."""
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # 1. Detect Structural Domain (Format)
        domain = 'prose'
        multilingual_chars = re.findall(r'[\u4e00-\u9FFF\u0400-\u04FF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7A3]', text)
        if len(multilingual_chars) > len(text) * 0.1:
            domain = 'multilingual'
        else:
            code_chars = set('{}=;()<>[]')
            code_char_count = sum(1 for c in text if c in code_chars)
            code_keywords = {'def', 'function', 'class', 'import', 'return', 'var', 'let', 'const', 'public', 'private', 'struct'}
            if len(text) > 0 and (code_char_count / len(text) > 0.02 or len(words.intersection(code_keywords)) >= 2):
                domain = 'code'
            elif len(words.intersection({'thou', 'hath', 'wherefore', 'alas', 'forsooth', 'thy'})) > 1:
                domain = 'literature'

        # 2. Detect Semantic Intent (Code vs Information vs Logic)
        semantic_type = 'information'
        if domain == 'code':
            semantic_type = 'code'
        else:
            logic_keywords = {'if', 'then', 'because', 'therefore', 'must', 'should', 'calculate', 'process', 'algorithm', 'ensure', 'loop', 'requires', 'depends', 'evaluate'}
            logic_score = len(words.intersection(logic_keywords))
            
            info_keywords = {'is', 'was', 'are', 'were', 'discovered', 'located', 'history', 'studies', 'shown', 'known', 'consists', 'named'}
            info_score = len(words.intersection(info_keywords))
            
            if logic_score > info_score and logic_score >= 2:
                semantic_type = 'logic'
                
        return {'domain': domain, 'semantic_type': semantic_type}
        
    def detect_domain(self, text: str) -> str:
        """Legacy wrapper for compatibility."""
        return self.recognize_semantics(text)['domain']

    def tokenize(self, text: str, domain: str) -> list[str]:
        """Domain-aware tokenization handling universal text and code specifics."""
        
        # New pattern preserves paths (/root), file extensions (out.txt), and bash flags (-la)
        # Matches prefixes and infixes of ., ~, /, - combined with word characters
        # Added Multilingual CJK splitting support
        alpha_cyrillic = r"[A-Za-z\u0400-\u04FF\u00C0-\u017F]"
        word_pattern = rf"(?:[.~/\-]*{alpha_cyrillic}+)+(?:['`\u2019]{alpha_cyrillic}+)?|[\d]+|[\u4e00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7A3]"
        
        if domain == 'code':
            lines = text.split('\n')
            tokens = []
            indent_stack = [0]
            
            multi_char_ops = ['==', '!=', '<=', '>=', '+=', '-=', '*=', '/=', '=>', '->', '::', '//', '/*', '*/', '&&', '||', '<<', '>>', '**']
            multi_char_regex = '|'.join(map(re.escape, multi_char_ops))
            
            # multi_char_regex is checked first so operators like '//' aren't swallowed by word_pattern
            token_regex = re.compile(rf"{multi_char_regex}|{word_pattern}|[^\s\w]")
            
            for line in lines:
                stripped_line = line.lstrip()
                if not stripped_line:
                    continue
                
                indent = len(line) - len(stripped_line)
                
                if indent > indent_stack[-1]:
                    tokens.append('<INDENT>')
                    indent_stack.append(indent)
                elif indent < indent_stack[-1]:
                    while len(indent_stack) > 1 and indent < indent_stack[-1]:
                        tokens.append('<DEDENT>')
                        indent_stack.pop()
                        
                line_tokens = token_regex.findall(stripped_line)
                tokens.extend(line_tokens)
                tokens.append('\n')
                
            while len(indent_stack) > 1:
                tokens.append('<DEDENT>')
                indent_stack.pop()
                
            return tokens
        else:
            text = re.sub(r'\s+', ' ', text).strip()
            return re.findall(rf"{word_pattern}|[^\s\w]", text, re.UNICODE)
