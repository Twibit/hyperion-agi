import re
from typing import Optional

class IndentationRestorer:
    """
    Takes raw spaced-out Python tokens from the N-gram engine and
    reconstructs proper indentation and token spacing before the
    Corpus Callosum wraps it in bash heredoc syntax.
    """
    
    NO_SPACE_BEFORE = {')', ']', '}', ',', ':', ';', '.'}
    NO_SPACE_AFTER = {'(', '[', '{', '.', '~'}
    SPACED_OPERATORS = {
        '=', '==', '!=', '<=', '>=', '+=', '-=', '*=', '/=',
        '%=', '**=', '//=', '&=', '|=', '^=', '<<=', '>>=',
        'and', 'or', 'not', 'in', 'is', 'if', 'else',
        '+', '-', '*', '/', '//', '%', '**', '&', '|', '^',
        '<', '>', '<<', '>>'
    }
    BLOCK_STARTERS = {'def', 'class', 'if', 'elif', 'else', 'for', 'while',
                      'try', 'except', 'finally', 'with', 'async'}
    
    INDENT_TOKEN = '<INDENT>'
    DEDENT_TOKEN = '<DEDENT>'
    INDENT_UNIT = '    '
    
    def __init__(self):
        self.indent_level = 0
    
    def restore(self, raw_output: str) -> str:
        if not raw_output or not raw_output.strip():
            return raw_output
        lines = raw_output.split('\n')
        restored_lines = []
        for line in lines:
            restored_lines.append(self._restore_line(line))
        return self._fix_block_indentation(restored_lines)
    
    def _restore_line(self, line: str) -> str:
        stripped = line.strip()
        if not stripped: return ''
        leading_spaces = len(line) - len(line.lstrip())
        tokens = self._safe_tokenize(stripped)
        if not tokens: return ''
        
        result_parts = []
        for i, token in enumerate(tokens):
            if i == 0:
                result_parts.append(token)
                continue
            prev_token = tokens[i - 1]
            if self._needs_space_between(prev_token, token):
                result_parts.append(' ')
            result_parts.append(token)
            
        indent_prefix = ' ' * leading_spaces
        return indent_prefix + ''.join(result_parts)
    
    def _safe_tokenize(self, text: str) -> list[str]:
        tokens = []
        i = 0
        while i < len(text):
            if text[i] == ' ':
                i += 1
                continue
            if text[i] in ('"', "'"):
                quote_char = text[i]
                if text[i:i+3] in ('"""', "'''"):
                    end = text.find(text[i:i+3], i + 3)
                    if end == -1:
                        tokens.append(text[i:])
                        break
                    tokens.append(text[i:end+3])
                    i = end + 3
                else:
                    end = text.find(quote_char, i + 1)
                    if end == -1:
                        tokens.append(text[i:])
                        break
                    tokens.append(text[i:end+1])
                    i = end + 1
                continue
            two_char = text[i:i+2]
            if two_char in ('==', '!=', '<=', '>=', '+=', '-=', '*=', '/=',
                           '=>', '->', '::', '//', '**', '<<', '>>', '&&', '||'):
                tokens.append(two_char)
                i += 2
                continue
            if text[i] in '()[]{}:;,.@~#':
                tokens.append(text[i])
                i += 1
                continue
            if text[i] in '+-*/%&|^<>=!':
                tokens.append(text[i])
                i += 1
                continue
            match = re.match(r'[\w.]+', text[i:])
            if match:
                tokens.append(match.group())
                i += match.end()
                continue
            tokens.append(text[i])
            i += 1
        return tokens
    
    def _needs_space_between(self, prev: str, curr: str) -> bool:
        if prev in ('(', '[', '{'): return False
        if curr in (')', ']', '}'): return False
        if curr in (',', ':', ';'): return False
        if prev == '.' or curr == '.': return False
        if prev == '@': return False
        if curr == '(' and re.match(r'^[\w]+$', prev): return False
        if curr == '(' and prev in (')', ']'): return False
        if prev == ',': return True
        if prev == ':': return True
        if prev in self.SPACED_OPERATORS or curr in self.SPACED_OPERATORS:
            if curr in ('+', '-') and prev in ('(', ',', '=', 'return'):
                return False
            return True
        if curr == '(' and prev in ('if', 'elif', 'while', 'for', 'return',
                                     'assert', 'except', 'with', 'yield',
                                     'lambda', 'not', 'and', 'or', 'in',
                                     'is', 'del', 'raise', 'print'):
            return True
        if re.match(r'^[\w]', prev) and re.match(r'^[\w]', curr): return True
        return False
    
    def _fix_block_indentation(self, lines: list[str]) -> str:
        self.indent_level = 0
        result_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                result_lines.append('')
                continue
            if self.INDENT_TOKEN in stripped:
                stripped = stripped.replace(self.INDENT_TOKEN, '').strip()
                self.indent_level += 1
            while self.DEDENT_TOKEN in stripped:
                stripped = stripped.replace(self.DEDENT_TOKEN, '', 1).strip()
                self.indent_level = max(0, self.indent_level - 1)
            if not stripped:
                result_lines.append('')
                continue
            if stripped.split()[0] in ('elif', 'else', 'except', 'finally'):
                self.indent_level = max(0, self.indent_level - 1)
            indented_line = self.INDENT_UNIT * self.indent_level + stripped
            result_lines.append(indented_line)
            if stripped.endswith(':') and stripped.split()[0] in self.BLOCK_STARTERS:
                self.indent_level += 1
        return '\n'.join(result_lines)
